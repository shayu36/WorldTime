"""Native checkpoint trajectory export and legacy ST-P3 planning metrics.

No branch overrides, output clipping, GT trajectory substitution or training.
Only tiny trajectory records are gathered; occupancy grids are discarded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cumulative_trajectory(increments):
    a = np.asarray(increments)
    if a.shape != (1, 6, 2) or not np.isfinite(a).all():
        raise ValueError(f'Expected finite trajectory increments [1,6,2], got {a.shape}')
    # Exactly the conversion in NuScenesDatasetOccpancy4DTraj.evaluate.
    return np.cumsum(a, axis=1)


def merge_trajectories(parts, expected_tokens):
    merged = {}
    for part in parts:
        for token, traj in part:
            if token in merged:
                if not np.allclose(merged[token], traj, atol=1e-5, rtol=1e-5):
                    raise ValueError(f'DDP padded duplicate prediction disagrees: {token}')
            else:
                merged[token] = traj
    if set(merged) != set(expected_tokens) or len(expected_tokens) != len(set(expected_tokens)):
        raise ValueError('Predicted token coverage differs from the full evaluation dataset')
    return {token: merged[token] for token in expected_tokens}


def check(args):
    from mmcv import Config
    config = Config.fromfile(args.config)
    q = config.model.query_memory_cfg
    assert q.future_memory_adapter_version == 'v2' and q.source == 'cache'
    assert q.future_memory_adapter_enabled and q.freeze_base_model
    prior = json.loads(Path(args.occupancy_provenance).read_text())
    assert sha256(args.checkpoint) == prior['sha256'][prior['checkpoint']], 'Wrong checkpoint'
    for path, digest in prior['sha256'].items():
        if path.startswith('mmdet3d/'):
            assert sha256(path) == digest, f'Model code changed since occupancy evaluation: {path}'
    for filename in ('filter_token.pkl', 'stp3_traj_gt.pkl', 'stp3_occupancy.pkl'):
        assert (Path(args.gt_root) / filename).is_file()
    print('T1 checkpoint / model source / planning GT paths: PASS', flush=True)
    return config


def infer(args):
    import torch
    import torch.distributed as dist
    from mmcv.parallel import MMDistributedDataParallel
    from mmcv.runner import init_dist, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    cfg = patch_config(compat_cfg(check(args)))
    setup_multi_processes(cfg)
    init_dist('pytorch', **cfg.dist_params)
    rank, size = dist.get_rank(), dist.get_world_size()
    set_random_seed(0, deterministic=False)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'provenance.json').write_text(json.dumps(dict(
            checkpoint=args.checkpoint, checkpoint_sha256=sha256(args.checkpoint),
            config=args.config, seed=0, world_size=size,
            description='Native simple_test, no branch overrides; pred_traj cumulative sum',
            source_sha256={p: sha256(p) for p in [__file__,
                'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py',
                'AD-MLP/deps/stp3/stp3/planning_metrics.py']}), indent=2))
    dist.barrier()
    cfg.data.test.test_mode = True
    cfg.data.test.load_interval = 1
    dataset = build_dataset(cfg.data.test)
    expected = [dataset.data_infos[i]['token'] for i in dataset.temp2nusc_map]
    if len(expected) != 4219:
        raise ValueError(f'Expected the recorded full 4219-frame split, got {len(expected)}')
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                              dist=True, shuffle=False, seed=0)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model.CLASSES = dataset.CLASSES
    model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()],
                                      broadcast_buffers=False).eval()
    records = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            meta = data['img_metas'][0].data[0][0]
            result = model(return_loss=False, rescale=True, **data)
            records.append((str(meta['sample_idx']), cumulative_trajectory(
                result['pred_traj'].detach().float().cpu().numpy())))
            del result
            if rank == 0 and (i + 1) % 50 == 0:
                print(f'Planning prediction {min((i+1)*size, len(dataset))}/{len(dataset)}', flush=True)
    parts = [None] * size
    dist.all_gather_object(parts, records)
    if rank == 0:
        merged = merge_trajectories(parts, expected)
        with (output / 'trajectories.pkl').open('xb') as f:
            pickle.dump(merged, f)
        (output / 'manifest.json').write_text(json.dumps(expected))
        print(f'Exported {len(merged)} native trajectories', flush=True)
    dist.barrier()
    dist.destroy_process_group()


def metrics(args):
    import torch
    from tqdm import tqdm
    sys.path.insert(0, str(ROOT / 'AD-MLP/deps/stp3'))
    from stp3.planning_metrics import PlanningMetric
    torch.set_num_threads(1)
    root, gt_root = Path(args.output_dir), Path(args.gt_root)
    if (root / 'planning_metrics.json').exists():
        raise FileExistsError('planning_metrics.json already exists; will not overwrite')
    with (root / 'trajectories.pkl').open('rb') as f:
        predictions = pickle.load(f)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert set(predictions) == set(manifest)
    with (gt_root / 'filter_token.pkl').open('rb') as f:
        token_filter = pickle.load(f)
    with (gt_root / 'stp3_traj_gt.pkl').open('rb') as f:
        trajectories = pickle.load(f)
    with (gt_root / 'stp3_occupancy.pkl').open('rb') as f:
        occupancy = pickle.load(f)
    tokens = [t for t in token_filter if t in predictions]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError('Empty or duplicated planning evaluation tokens')
    metric = [PlanningMetric(2 * h).cpu() for h in (1, 2, 3)]
    with torch.no_grad():
        for token in tqdm(tokens):
            final = torch.from_numpy(predictions[token])
            final = torch.cat([final, torch.zeros_like(final[..., :1])], -1)
            gt = torch.tensor(trajectories[token]['gt_trajectory']).unsqueeze(0)
            occ = torch.as_tensor(occupancy[token]).cpu()
            for h, obj in enumerate(metric, 1):
                n = h * 2
                obj.update(final[:, :n].clone(), gt[:, 1:n+1].clone(), occ[:, :n])
    result = dict(predicted_samples=len(predictions), evaluated_samples=len(tokens),
                  filter_total=len(token_filter), filter_not_in_dataset=len(token_filter)-len(tokens),
                  note='Legacy ST-P3 cumulative 0..h horizon mean, not endpoint-only; box rates are fractions.',
                  metrics={}, baseline_reference_source='user-provided rounded baseline',
                  baseline_L2_m=[.1596, .1945, .2393], baseline_box_collision_percent=[.0948, .1007, .1106])
    for h, obj in enumerate(metric, 1):
        raw = {k: float(v.mean()) for k, v in obj.compute().items()}
        result['metrics'][f'{h}s'] = dict(raw, box_collision_percent=raw['obj_box_col']*100,
            delta_L2_m=raw['L2']-result['baseline_L2_m'][h-1],
            delta_box_collision_pp=raw['obj_box_col']*100-result['baseline_box_collision_percent'][h-1])
    result['trajectory_sha256'] = sha256(root / 'trajectories.pkl')
    result['metric_source_sha256'] = sha256(ROOT / 'AD-MLP/deps/stp3/stp3/planning_metrics.py')
    with (root / 'planning_metrics.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'infer', 'metrics'])
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--occupancy-provenance')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--gt-root', default='stp3_val')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    {'check': check, 'infer': infer, 'metrics': metrics}[args.stage](args)
