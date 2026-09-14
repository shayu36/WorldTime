"""Compare baseline (M0R0) and M1R0 intermediate inference outputs.

The comparison uses the same validation dataloader and feeds every batch to
both models.  It reports streaming statistics, so raw query tensors do not
need to be retained for the whole validation set.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import types
from collections import defaultdict

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.apis import set_random_seed
from mmdet.utils import compat_cfg, setup_multi_processes

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import patch_config


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config-baseline', default=(
        'configs/sparseworld/nuscenes-temporal/'
        'sparseworld-traj-memory-ablation-M0R0.py'))
    p.add_argument('--config-m1r0', default=(
        'configs/sparseworld/nuscenes-temporal/'
        'sparseworld-traj-memory-ablation-M1R0.py'))
    p.add_argument('--checkpoint-baseline', default='ckpts/epoch_56.pth')
    p.add_argument('--checkpoint-m1r0', default=(
        'work_dirs/sparseworld-traj-memory-ablation-M1R0/epoch_12.pth'))
    p.add_argument('--max-samples', type=int, default=0,
                   help='0 means the complete validation set')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--parallel-gpus', action='store_true',
                   help='run baseline and M1R0 concurrently on CUDA devices 0/1')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--out', default=(
        'work_dirs/sparseworld-traj-memory-ablation-M1R0/'
        'baseline_vs_m1r0_intermediates.json'))
    return p.parse_args()


def load_cfg(path):
    cfg = Config.fromfile(path)
    cfg = patch_config(compat_cfg(cfg))
    setup_multi_processes(cfg)
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [0]
    return cfg


def build_eval_model(cfg, checkpoint, device_id=0):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.CLASSES = getattr(cfg, 'class_names', None)
    if torch.cuda.is_available():
        model = model.cuda(device_id)
    model = MMDataParallel(model, device_ids=[device_id])
    model.eval()
    return model


def capture_forward(model):
    captured = []
    module = model.module
    original = module.forward_backbone

    def wrapped(self, *args, **kwargs):
        out = original(*args, **kwargs)
        captured.append(out)
        return out

    module.forward_backbone = types.MethodType(wrapped, module)
    return captured


def tensor_stats(acc, name, x, y):
    """Accumulate elementwise difference statistics for two tensors."""
    if x is None or y is None:
        return
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        return
    # Keep the reduction on the device when both models share a device.  In
    # the optional two-GPU mode the tensors live on different devices, so use
    # CPU for the cross-device comparison.
    x = x.detach().float()
    y = y.detach().float()
    if x.device != y.device:
        x = x.cpu()
        y = y.cpu()
    if x.shape != y.shape:
        acc[name]['shape_mismatch'] += 1
        acc[name]['x_shape'] = list(x.shape)
        acc[name]['y_shape'] = list(y.shape)
        return
    d = (y - x).reshape(-1)
    n = d.numel()
    if n == 0:
        return
    a = d.abs()
    acc[name]['n'] += int(n)
    acc[name]['sum_abs'] += float(a.sum().item())
    acc[name]['sum_sq'] += float((d * d).sum().item())
    acc[name]['max_abs'] = max(acc[name]['max_abs'], float(a.max().item()))
    acc[name]['nonzero_1e-7'] += int((a > 1e-7).sum().item())
    acc[name]['nonzero_1e-5'] += int((a > 1e-5).sum().item())


def finalize_tensor_stats(acc):
    result = {}
    for name, s in acc.items():
        out = dict(s)
        n = s.get('n', 0)
        if n:
            out['mean_abs'] = s['sum_abs'] / n
            out['rmse'] = (s['sum_sq'] / n) ** 0.5
            out['frac_gt_1e-7'] = s['nonzero_1e-7'] / n
            out['frac_gt_1e-5'] = s['nonzero_1e-5'] / n
        result[name] = out
    return result


def main():
    args = parse_args()
    os.chdir(ROOT)
    set_random_seed(17, deterministic=True)

    cfg_b = load_cfg(args.config_baseline)
    cfg_m = load_cfg(args.config_m1r0)
    # Build one shared loader from the baseline config.  The two configs use
    # the same validation pipeline/cache; using one iterator guarantees that
    # both models see exactly the same samples in the same order.
    dataset = build_dataset(cfg_b.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False)

    if args.parallel_gpus:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError('--parallel-gpus requires at least two CUDA devices')
        model_b = build_eval_model(cfg_b, args.checkpoint_baseline, device_id=0)
        model_m = build_eval_model(cfg_m, args.checkpoint_m1r0, device_id=1)
    else:
        model_b = build_eval_model(cfg_b, args.checkpoint_baseline, device_id=0)
        model_m = build_eval_model(cfg_m, args.checkpoint_m1r0, device_id=0)
    cap_b = capture_forward(model_b)
    cap_m = capture_forward(model_m)

    diff = defaultdict(lambda: defaultdict(float))
    occ_diff = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(float)
    sample_ids = []

    executor = ThreadPoolExecutor(max_workers=2) if args.parallel_gpus else None
    with torch.no_grad():
        for i, data in enumerate(loader):
            if args.max_samples and i >= args.max_samples:
                break
            cap_b.clear()
            cap_m.clear()
            if executor is None:
                result_b = model_b(return_loss=False, rescale=True, **data)
                result_m = model_m(return_loss=False, rescale=True, **data)
            else:
                fut_b = executor.submit(
                    model_b, return_loss=False, rescale=True, **data)
                fut_m = executor.submit(
                    model_m, return_loss=False, rescale=True, **data)
                result_b = fut_b.result()
                result_m = fut_m.result()
            if len(cap_b) != 1 or len(cap_m) != 1:
                raise RuntimeError(
                    f'Expected one captured forward, got {len(cap_b)} and '
                    f'{len(cap_m)} at sample {i}')
            out_b = cap_b[0]
            out_m = cap_m[0]
            outs_b, outs_m = out_b['outs'], out_m['outs']

            # Final OPUS outputs before the Memory/refiner future-only route.
            raw_cls_b = outs_b['all_cls_scores'][-1]
            raw_cls_m = outs_m['all_cls_scores'][-1]
            raw_pts_b = outs_b['all_refine_pts'][-1]
            raw_pts_m = outs_m['all_refine_pts'][-1]
            tensor_stats(diff, 'raw_logits_all_queries', raw_cls_b, raw_cls_m)
            tensor_stats(diff, 'raw_refine_points_all_queries', raw_pts_b,
                         raw_pts_m)

            mask_b = model_b.module.pts_bbox_head.ind_stamps_all == 0
            mask_m = model_m.module.pts_bbox_head.ind_stamps_all == 0
            if not torch.equal(mask_b.cpu(), mask_m.cpu()):
                raise RuntimeError('Observation query masks differ')
            tensor_stats(diff, 'raw_logits_observation_queries',
                         raw_cls_b[:, mask_b], raw_cls_m[:, mask_m])
            tensor_stats(diff, 'raw_refine_points_observation_queries',
                         raw_pts_b[:, mask_b], raw_pts_m[:, mask_m])
            tensor_stats(diff, 'current_refined_logits', out_b['cls_score'],
                         out_m['cls_score'])
            tensor_stats(diff, 'current_refined_points', out_b['refine_pts'],
                         out_m['refine_pts'])

            # The six future groups are the tensors consumed by get_occ.
            for h, (cls_b, cls_m, pts_b, pts_m) in enumerate(zip(
                    out_b['forecast_semantics_list'],
                    out_m['forecast_semantics_list'],
                    out_b['forecast_points_list'],
                    out_m['forecast_points_list']), 1):
                tensor_stats(diff, f'future_{h}_refined_logits', cls_b, cls_m)
                tensor_stats(diff, f'future_{h}_refined_points', pts_b, pts_m)

            # Final discrete occupancy arrays returned by simple_test.
            for h in range(7):
                key = f'semantic_occ_{h}s'
                a = np.asarray(result_b[key][0])
                b = np.asarray(result_m[key][0])
                if a.shape != b.shape:
                    raise RuntimeError(f'Occupancy shape mismatch at {key}: '
                                       f'{a.shape} vs {b.shape}')
                d = a != b
                s = occ_diff[key]
                s['voxels'] += int(d.size)
                s['changed_voxels'] += int(d.sum())
                s['baseline_nonempty'] += int((a != 17).sum())
                s['m1r0_nonempty'] += int((b != 17).sum())
                s['baseline_sum'] += int(a.sum())
                s['m1r0_sum'] += int(b.sum())

            counts['samples'] += 1
            if data.get('img_metas'):
                try:
                    meta = data['img_metas'][0].data[0][0]
                    sample_ids.append({
                        'sample_idx': str(meta.get('sample_idx')),
                        'scene_name': str(meta.get('scene_name')),
                        'frame_idx': int(meta.get('frame_idx', -1)),
                    })
                except Exception:
                    pass
            if (i + 1) % 100 == 0:
                print(f'processed {i + 1} samples', flush=True)

    if executor is not None:
        executor.shutdown(wait=True)

    occ_result = {}
    for key, s in occ_diff.items():
        out = dict(s)
        out['changed_fraction'] = s['changed_voxels'] / max(1, s['voxels'])
        out['baseline_nonempty_per_sample'] = s['baseline_nonempty'] / max(
            1, counts['samples'])
        out['m1r0_nonempty_per_sample'] = s['m1r0_nonempty'] / max(
            1, counts['samples'])
        occ_result[key] = out

    report = {
        'num_samples': int(counts['samples']),
        'baseline_checkpoint': args.checkpoint_baseline,
        'm1r0_checkpoint': args.checkpoint_m1r0,
        'metrics_definition': {
            'raw_logits': 'pts_bbox_head all_cls_scores[-1]',
            'raw_refine_points': 'pts_bbox_head all_refine_pts[-1]',
            'refined_future': 'forward_backbone forecast_*_list',
            'occupancy': 'simple_test semantic_occ_{0..6}s',
            'occupancy_empty_label': 17,
        },
        'tensor_differences_m1r0_minus_baseline': finalize_tensor_stats(diff),
        'occupancy_differences': occ_result,
        'sample_ids': sample_ids[:10],
    }
    mmcv.mkdir_or_exist(os.path.dirname(args.out) or '.')
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
