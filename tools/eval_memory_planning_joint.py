"""Save native occupancy and planning from the SAME checkpoint/forward pass.

torchrun workers use disjoint, unpadded validation indices and independent
frozen models. Predictions are streamed as uint8 labels, not gathered in RAM.
Offline scoring reuses the existing dataset and Baseline planning evaluators.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def pack_occupancy(result):
    # Existing temporal exporter uses these internal steps for 0/1/2/3s.
    array = np.stack([result[f'semantic_occ_{step}s'][0]
                      for step in (0, 2, 4, 6)], axis=0)
    if array.ndim != 4 or array.dtype.kind not in 'iu' or array.min() < 0 or array.max() > 17:
        raise ValueError(f'Invalid native occupancy labels: {array.shape}, {array.dtype}')
    return array[None].astype(np.uint8)  # Lossless labels 0..17; 8x less disk than int64.


def predict(args):
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    rank, world = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    set_random_seed(0, deterministic=True)
    cfg = patch_config(compat_cfg(Config.fromfile(args.config)))
    setup_multi_processes(cfg)
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    dataset = build_dataset(cfg.data.test)
    manifest = [dict(index=i, token=dataset.data_infos[j]['token'],
                     scene=dataset.data_infos[j]['scene_name'])
                for i, j in enumerate(dataset.temp2nusc_map)]
    if args.limit:
        manifest = manifest[:args.limit]
    if not manifest or len({r['token'] for r in manifest}) != len(manifest):
        raise ValueError('Empty or duplicate validation tokens')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(out/'manifest.json', manifest)
        cfg.dump(str(out/'effective_config.py'))
    part_path = out/f'trajectories_rank_{rank}.pkl'
    if part_path.exists() or (out/f'complete_rank_{rank}.json').exists():
        raise FileExistsError('Refusing to overwrite completed prediction shard')
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict({k.removeprefix('module.'): v
                           for k, v in checkpoint['state_dict'].items()}, strict=True)
    epoch = checkpoint['meta']['epoch']
    del checkpoint
    wrap_fp16_model(model)
    model.CLASSES = dataset.CLASSES
    model.cuda(local_rank).eval()
    model.validate_query_memory_training_setup()
    if model.memory_tuning_trainable_prefixes() != ('memory_planning_fusion.',):
        raise ValueError('Unexpected model/optimizer scope')
    parallel = MMDataParallel(model, device_ids=[local_rank]).eval()
    subset = manifest[rank::world]  # No sampler padding, each token exactly once.
    loader = build_dataloader(torch.utils.data.Subset(dataset, [r['index'] for r in subset]),
                              samples_per_gpu=1, workers_per_gpu=1, num_gpus=1,
                              dist=False, shuffle=False, seed=0)
    source_files = ('mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py',
                    'mmdet3d/models/sparsedetectors/sparseworld_memory_planning.py',
                    'mmdet3d/models/sparsedetectors/memory_planning_fusion.py',
                    'mmdet3d/models/sparsedetectors/query_memory.py',
                    'mmdet3d/datasets/nuscenes_dataset_occ_trajectory.py',
                    'mmdet3d/datasets/pipelines/loading_query_memory.py')
    write_json(out/f'provenance_rank_{rank}.json', dict(
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=digest(args.checkpoint),
        checkpoint_epoch=epoch, config_sha256=digest(args.config), rank=rank, world_size=world,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={p: digest(p) for p in source_files}, exporter_sha256=digest(__file__),
        samples=len(subset), total_samples=len(manifest), command=sys.argv,
        native_model_call='MMDataParallel(return_loss=False,rescale=True)',
        trajectory_format='np.cumsum(native pred_traj increments, axis=1), current-frame XY',
        occupancy_steps=[0, 2, 4, 6], occupancy_dtype='uint8 lossless 0..17'))
    trajectories = {}
    started = time.monotonic()
    with torch.no_grad(), (out/f'progress_rank_{rank}.jsonl').open('w') as progress:
        for i, (record, batch) in enumerate(zip(subset, loader)):
            result = parallel(return_loss=False, rescale=True, **batch)
            raw = result['pred_traj'].detach().float().cpu().numpy()
            if raw.shape != (1, 6, 2) or not np.isfinite(raw).all():
                raise ValueError('Invalid native planning output')
            trajectories[record['token']] = np.cumsum(raw, axis=1)
            path = out/'occupancy'/record['scene']/(record['token']+'.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            np.save(path, pack_occupancy(result))
            progress.write(json.dumps(record)+'\n')
            progress.flush()
            if i < 3 or (i+1) % 100 == 0:
                elapsed = time.monotonic()-started
                print(f'rank={rank} {i+1}/{len(subset)} elapsed={elapsed:.1f}s '
                      f'mean={elapsed/(i+1):.3f}s/sample', flush=True)
            del result, raw, batch
    if len(trajectories) != len(subset):
        raise RuntimeError('Incomplete dataloader iteration')
    with part_path.open('xb') as f:
        pickle.dump(trajectories, f)
    write_json(out/f'complete_rank_{rank}.json', dict(
        samples=len(trajectories), checkpoint_epoch=epoch, world_size=world,
        trajectory_sha256=digest(part_path), elapsed_seconds=time.monotonic()-started))


def merge_shards(output):
    manifest = json.loads((output/'manifest.json').read_text())
    records = json.loads((output/'provenance_rank_0.json').read_text())
    trajectories = {}
    for rank in range(records['world_size']):
        complete = json.loads((output/f'complete_rank_{rank}.json').read_text())
        source = json.loads((output/f'provenance_rank_{rank}.json').read_text())
        for key in ('checkpoint_sha256', 'checkpoint_epoch', 'config_sha256', 'source_sha256', 'exporter_sha256'):
            if records[key] != source[key]:
                raise RuntimeError('Inconsistent prediction workers: '+key)
        part_path = output/f'trajectories_rank_{rank}.pkl'
        if digest(part_path) != complete['trajectory_sha256']:
            raise RuntimeError('Prediction shard changed')
        with part_path.open('rb') as f:
            part = pickle.load(f)
        if set(part) & set(trajectories) or len(part) != complete['samples']:
            raise RuntimeError('Duplicate/missing prediction shard tokens')
        trajectories.update(part)
    if set(trajectories) != {r['token'] for r in manifest}:
        raise RuntimeError('Prediction token coverage mismatch')
    path = output/'trajectories.pkl'
    if path.exists():
        raise FileExistsError(path)
    with path.open('xb') as f:
        pickle.dump({r['token']: trajectories[r['token']] for r in manifest}, f)
    return manifest, records


def score(args):
    import torch
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import patch_config
    from audit_planning_pair import metrics as planning_metrics

    torch.set_num_threads(1)
    out = Path(args.output)
    manifest, provenance = merge_shards(out)
    cfg = patch_config(Config.fromfile(args.config))
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    expected = [dict(index=i, token=dataset.data_infos[j]['token'],
                     scene=dataset.data_infos[j]['scene_name'])
                for i, j in enumerate(dataset.temp2nusc_map)]
    if manifest != expected:
        raise ValueError('Scoring requires the full matching validation manifest')
    predictions = []
    for r in manifest:
        x = np.load(out/'occupancy'/r['scene']/(r['token']+'.npy'), mmap_mode='r')
        if x.ndim != 5 or x.shape[:2] != (1, 4):
            raise ValueError('Occupancy dump shape mismatch')
        predictions.append(x[0])
    occ = dataset.evaluate([predictions], metric=['segm'])
    write_json(out/'occupancy_metrics.json', occ)
    accumulator = dataset.occ_eval_metrics
    np.savez(out/'occupancy_confusion_matrices.npz', **{
        f'{kind}_{t}s':getattr(accumulator, f'{kind}_{t}s')
        for kind in ('hist', 'occ_hist') for t in range(4)})
    planning_metrics(SimpleNamespace(metric_root=str(Path.cwd()),
                     prediction=str(out/'trajectories.pkl'), output=str(out/'planning_metrics.json')))
    plan = json.loads((out/'planning_metrics.json').read_text())
    baseline = json.loads(Path(args.baseline_metrics).read_text())
    if plan['gt_sha256'] != baseline['gt_sha256'] or plan['metric_source_sha256'] != baseline['metric_source_sha256']:
        raise ValueError('Planning evaluator/GT differs from Baseline')
    full = dict(checkpoint=provenance['checkpoint'], checkpoint_sha256=provenance['checkpoint_sha256'],
                epoch=provenance['checkpoint_epoch'], occupancy_samples=len(manifest),
                planning_samples=plan['samples'], planning_filter_total=plan['filter_total'],
                occupancy=occ, planning=plan,
                future_IoU=float(np.mean(occ['IoU'][1:])),
                future_mIoU=float(np.mean(occ['mIoU'][1:])),
                mean_prefix_L2=plan['prefix']['mean_L2'],
                mean_prefix_L2_delta=plan['prefix']['mean_L2']-baseline['prefix']['mean_L2'],
                protocol='Native occupancy evaluator; Baseline ST-P3 prefix planning plus endpoints; collision fractions in raw JSON')
    write_json(out/'all_metrics.json', full)
    write_json(out/'complete.json', dict(status='PASS', epoch=full['epoch'],
               samples=len(manifest), checkpoint_sha256=full['checkpoint_sha256'],
               trajectories_sha256=digest(out/'trajectories.pkl')))
    print(json.dumps(full, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('predict', 'score'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help='Inference-only probe; never scored as full validation')
    parser.add_argument('--baseline-metrics', default='work_dirs/planning-audit-20261115-9M2EFH/baseline/metrics.json')
    args = parser.parse_args()
    if args.stage == 'predict':
        if not args.checkpoint:
            parser.error('predict requires --checkpoint')
        predict(args)
    else:
        score(args)


if __name__ == '__main__':
    main()
