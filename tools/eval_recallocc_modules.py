"""Strict native occupancy evaluation, common frame manifest, percentage metrics."""
import argparse
from pathlib import Path
import time
import types

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.utils import compat_cfg, setup_multi_processes
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import patch_config
from diagnose_future_memory_semantics import confusion, summarize
from recallocc_common import ROOT, NEW_BASE_PREFIXES, digest, read_json, write_json, state_dict, validate_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    set_random_seed(0, deterministic=True)
    cfg = patch_config(compat_cfg(Config.fromfile(args.config)))
    setup_multi_processes(cfg)
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    if args.baseline:
        # Original SparseWorld path, no output adapter. Keep the SAME dataset,
        # current image inputs, TASS, thresholds and native voxel decoder.
        cfg.model.pop('future_memory_adapter_version', None)
        cfg.model.query_memory_cfg.update(enabled=False, freeze_base_model=False,
            future_memory_adapter_enabled=False, future_memory_adapter_finetune_mode=False,
            future_memory_adapter=None, future_memory_adapter_v2=None)
    current_only = cfg.model.query_memory_cfg.future_memory_adapter_v2.get('ablation_mode') == 'current_only_task_time'
    if current_only:
        from train_recallocc_modules import forbidden
        from mmdet3d.datasets.pipelines.loading_query_memory import LoadQueryMemoryFromFiles
        LoadQueryMemoryFromFiles.__call__ = forbidden
    dataset = build_dataset(cfg.data.test)
    manifest = read_json(args.manifest)
    for row in manifest:
        info = dataset.data_infos[dataset.temp2nusc_map[row['index']]]
        assert info['token'] == row['token'] and info['scene_name'] == row['scene']
    loader = build_dataloader(torch.utils.data.Subset(dataset, [r['index'] for r in manifest]),
        samples_per_gpu=1, workers_per_gpu=1, num_gpus=1, dist=False, shuffle=False, seed=0)
    cfg.dump(str(output/'effective_config.py'))
    write_json(output/'manifest.json', manifest)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = state_dict(checkpoint)
    allowed_missing = []
    if args.baseline:
        expected = model.state_dict()
        assert not (set(state)-set(expected)), 'Unexpected baseline keys'
        allowed_missing = sorted(set(expected)-set(state))
        assert all(k.startswith(NEW_BASE_PREFIXES) for k in allowed_missing), allowed_missing
        assert not any(k.startswith('future_memory_adapter') for k in state), 'Baseline must be epoch56'
        state = {**expected, **state}
    validate_state(model,state)
    model.load_state_dict(state, strict=True)
    if args.baseline:
        baseline_epoch = checkpoint['meta']['epoch']
    else:
        meta = checkpoint.get('meta',{}).get('recallocc',{})
        assert meta.get('experiment') == cfg.recallocc.experiment, 'Checkpoint experiment mismatch'
    del checkpoint,state
    wrap_fp16_model(model)
    model.CLASSES = dataset.CLASSES
    model.cuda().eval()
    # TASS/RAP masks are plain tensor attributes, not registered buffers.
    # Construct them on the final device; .cuda() cannot move existing masks.
    if args.baseline:
        model.set_epoch(baseline_epoch)
    else:
        model.validate_query_memory_training_setup()
    parallel = MMDataParallel(model, device_ids=[0]).eval()
    captured = []
    horizon_inputs = []
    if current_only:
        model._query_memory_context = forbidden
        model.future_memory_adapter_v2._coarse = forbidden
        model.future_memory_adapter_v2._reshape_memory = forbidden
        def current_input(module, args):
            q, p, logits, memory, horizon = args[:5]
            assert memory is None
            horizon_inputs.append((horizon, p.detach().clone(), logits.detach().clone(), float(q.float().sum())))
        model.future_memory_adapter_v2.register_forward_pre_hook(current_input)
    original = model.forward_backbone
    def capture(this,*a,**kw):
        result = original(*a,**kw)
        captured.append(result)
        return result
    model.forward_backbone = types.MethodType(capture, model)
    histories = np.zeros((len(manifest),4,18,18),dtype=np.int64)
    elapsed = []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.no_grad(), (output/'progress.jsonl').open('x') as progress:
        for sample,batch in enumerate(loader):
            torch.cuda.synchronize()
            ts = time.monotonic()
            results = parallel(return_loss=False,rescale=True,**batch)
            torch.cuda.synchronize()
            elapsed.append(time.monotonic()-ts)
            out = captured.pop()
            if not args.baseline:
                assert torch.equal(out['cls_score'],out['base_cls_score'])
                for step in (0,2,4):
                    assert torch.equal(out['forecast_semantics_list'][step],out['baseline_forecast_semantics_list'][step])
                    assert torch.equal(out['forecast_points_list'][step],out['baseline_forecast_points_list'][step])
            if current_only:
                from mmdet3d.models.sparsedetectors.query_memory import decode_points_metric
                assert [h[0] for h in horizon_inputs] == [0, 1, 2]
                for h, p, logits, qsum in horizon_inputs:
                    step = 2*h+1
                    assert torch.equal(logits, out['baseline_forecast_semantics_list'][step])
                    assert torch.equal(p, decode_points_metric(out['baseline_forecast_points_list'][step], model.pc_range))
                assert len(set(h[3] for h in horizon_inputs)) == 3
                horizon_inputs.clear()
            ni = dataset.temp2nusc_map[manifest[sample]['index']]
            for h,offset in enumerate((0,2,4,6)):
                info = dataset.data_infos[ni+offset]
                assert info['scene_name'] == manifest[sample]['scene']
                with np.load(Path(info['occ_path'])/'labels.npz') as gt:
                    histories[sample,h] = confusion(gt['semantics'],results[f'semantic_occ_{offset}s'][0])
            progress.write(__import__('json').dumps(dict(sample=sample,**manifest[sample],seconds=elapsed[-1]))+'\n')
            progress.flush()
            if sample < 3 or (sample+1)%100 == 0:
                print(f'{sample+1}/{len(manifest)} {elapsed[-1]:.3f}s/forward',flush=True)
            del results,out,batch
    np.savez_compressed(output/'confusions.npz',hist=histories)
    report = dict(model='SparseWorld' if args.baseline else cfg.recallocc.experiment,
        seed=None if args.baseline else cfg.seed, units='percent', horizons=[0,1,2,3],
        samples=len(manifest), full_validation=len(manifest)==len(dataset),
        manifest_sha256=digest(args.manifest), checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=digest(args.checkpoint), allowed_missing_baseline_keys=allowed_missing,
        memory_disabled=current_only, current_horizon_inputs_verified=current_only,
        metrics=[summarize(h) for h in histories.sum(0)],
        elapsed_seconds=time.monotonic()-start, mean_forward_seconds=float(np.mean(elapsed)),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        checkpoint_selection='fixed_epoch56' if args.baseline else cfg.recallocc.checkpoint_selection,
        protocol='existing_epoch56_schema2')
    write_json(output/'metrics.json',report)
    print([(r['miou'],r['iou']) for r in report['metrics']],flush=True)

if __name__ == '__main__':
    main()
