"""Native inference on a fixed manifest; save counts for paired scene CIs."""
import argparse
import json
from pathlib import Path
import time
import types

import numpy as np

from diagnose_future_memory_semantics import confusion, summarize, CLASSES


def write(path, value):
    with Path(path).open('w') as stream:
        json.dump(value, stream, indent=2)


def metrics(hist):
    return [summarize(h) for h in hist]


def comparison(args):
    roots = [Path(p) for p in args.compare]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in roots]
    assert manifests[0] == manifests[1], 'paired evaluation must use identical frames'
    samples = manifests[0]
    hist = np.stack([np.load(p/'confusions.npz')['hist'] for p in roots], axis=1)
    scenes = sorted({x['scene'] for x in samples})
    clustered = np.zeros((len(scenes), 2, 4, 18, 18), dtype=np.float64)
    scene_index = {s: i for i, s in enumerate(scenes)}
    for entry, values in zip(samples, hist):
        clustered[scene_index[entry['scene']]] += values
    from summarize_future_memory_branches import scores
    rng = np.random.default_rng(17)
    counts = rng.multinomial(len(scenes), np.ones(len(scenes))/len(scenes), size=1000)
    bootstrap_hist = (counts @ clustered.reshape(len(scenes), -1)).reshape(1000, 2, 4, 18, 18)
    boot = scores(bootstrap_hist)
    observed = scores(hist.sum(0))
    result = dict(samples=len(samples), scenes=len(scenes), bootstrap_replicates=1000,
                  reference=str(roots[0]), contrast=str(roots[1]),
                  units='percentage points',
                  scope='fixed archived ego/cache protocol; scene sampling CI, not training-seed uncertainty',
                  metrics={}, classes=CLASSES[:17])
    for key in ('iou', 'miou'):
        delta = observed[key][1] - observed[key][0]
        bd = boot[key][:, 1] - boot[key][:, 0]
        result['metrics'][key] = dict(reference=observed[key][0].tolist(),
            contrast=observed[key][1].tolist(), delta=delta.tolist(),
            delta_ci95=np.nanpercentile(bd, [2.5, 97.5], axis=0).T.tolist(),
            future_delta=float(delta[1:].mean()),
            future_delta_ci95=np.nanpercentile(bd[:, 1:].mean(-1), [2.5, 97.5]).tolist())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'paired_summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


def evaluate(args):
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    set_random_seed(17, deterministic=True)
    cfg = patch_config(compat_cfg(Config.fromfile(args.config)))
    setup_multi_processes(cfg)
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    dataset = build_dataset(cfg.data.test)
    manifest = json.loads(Path(args.manifest).read_text())
    for row in manifest:
        info = dataset.data_infos[dataset.temp2nusc_map[row['index']]]
        assert info['token'] == row['token'] and info['scene_name'] == row['scene']
    loader = build_dataloader(
        torch.utils.data.Subset(dataset, [x['index'] for x in manifest]),
        samples_per_gpu=1, workers_per_gpu=1, num_gpus=1, dist=False,
        shuffle=False, seed=17)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'manifest.json', manifest)
    cfg.dump(str(output/'config.py'))
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    wrap_fp16_model(model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model.CLASSES = dataset.CLASSES
    model.cuda().eval()
    parallel = MMDataParallel(model, device_ids=[0]).eval()
    original = model.forward_backbone
    captured = []

    def capture(self, *a, **kw):
        out = original(*a, **kw)
        captured.append(out)
        return out

    model.forward_backbone = types.MethodType(capture, model)
    histories = np.zeros((len(manifest), 4, 18, 18), dtype=np.int64)
    elapsed = []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.no_grad(), (output/'progress.jsonl').open('x') as progress:
        for sample, batch in enumerate(loader):
            captured.clear()
            torch.cuda.synchronize()
            ts = time.monotonic()
            results = parallel(return_loss=False, rescale=True, **batch)
            torch.cuda.synchronize()
            elapsed.append(time.monotonic()-ts)
            out = captured.pop()
            assert torch.equal(out['cls_score'], out['base_cls_score'])
            for step in (0, 2, 4):
                assert torch.equal(out['forecast_semantics_list'][step],
                                   out['baseline_forecast_semantics_list'][step])
                assert torch.equal(out['forecast_points_list'][step],
                                   out['baseline_forecast_points_list'][step])
            ni = dataset.temp2nusc_map[manifest[sample]['index']]
            for h, offset in enumerate((0, 2, 4, 6)):
                info = dataset.data_infos[ni+offset]
                assert info['scene_name'] == manifest[sample]['scene']
                with np.load(Path(info['occ_path'])/'labels.npz') as gt:
                    pred = results[f'semantic_occ_{offset}s'][0]
                    histories[sample, h] = confusion(gt['semantics'], pred)
            diagnostics = []
            for step in (1, 3, 5):
                diag = out['memory_adapter_diagnostics'][step]
                valid = diag['has_candidate']
                row = dict(horizon=diag['horizon_name'], candidate_fraction=float(valid.float().mean()))
                row['history_slot_limit'] = diag.get('history_slot_limit')
                if 'readable_memory_queries' in diag:
                    row['readable_memory_queries'] = int(diag['readable_memory_queries'].sum())
                for key in ('support_reliability', 'support_age', 'support_distance'):
                    if key in diag:
                        row[key] = float(diag[key][valid].float().mean()) if valid.any() else 0.
                row['task_gate_stats'] = {k: float(v) for k, v in diag.get('task_gate_stats', {}).items()}
                diagnostics.append(row)
            progress.write(json.dumps(dict(sample=sample, **manifest[sample],
                forward_seconds=elapsed[-1], diagnostics=diagnostics))+'\n')
            progress.flush()
            del results, out, batch
            print(f'{sample+1}/{len(manifest)} {elapsed[-1]:.3f}s/forward', flush=True)
    model.forward_backbone = original
    np.savez_compressed(output/'confusions.npz', hist=histories)
    report = dict(samples=len(manifest), scenes=len({m['scene'] for m in manifest}),
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  full_validation=len(manifest) == len(dataset),
                  elapsed_seconds=time.monotonic()-start,
                  mean_forward_seconds=float(np.mean(elapsed)),
                  steady_forward_seconds=float(np.mean(elapsed[1:])) if len(elapsed)>1 else None,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  metrics=metrics(histories.sum(0)),
                  zero_s_and_non_target_identity=True,
                  protocol='archived T1 ego/cache protocol; native outputs; no inference ablation')
    write(output/'metrics.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'metrics'}, indent=2))
    for h, row in enumerate(report['metrics']):
        print(h, 'IoU',row['iou'],'mIoU',row['miou'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?')
    parser.add_argument('checkpoint', nargs='?')
    parser.add_argument('--manifest')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--compare', nargs=2)
    args = parser.parse_args()
    if args.compare:
        comparison(args)
    elif args.config and args.checkpoint and args.manifest:
        evaluate(args)
    else:
        parser.error('evaluation requires config, checkpoint and --manifest')


if __name__ == '__main__':
    main()
