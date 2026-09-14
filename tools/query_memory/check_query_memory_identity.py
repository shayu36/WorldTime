#!/usr/bin/env python
"""Run one real-data STAC-QM C0/C1 identity or trained-difference check."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--mode', choices=['zero', 'trained'], default='zero')
    parser.add_argument(
        '--memory-checkpoint', default=None,
        help='required in trained mode; loaded into both ON and OFF models')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument(
        '--sample-index', type=int, default=None,
        help='dataset index; default selects the first sample with every '
             'target-age history slot populated')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--min-future-diff', type=float, default=1e-6)
    return parser.parse_args()


def _tensor_digest(model):
    digest = hashlib.sha256()
    state = [
        (name, value) for name, value in model.state_dict().items()
        if not name.startswith('query_memory.')
    ]
    for name, value in sorted(state, key=lambda item: item[0]):
        value = value.detach().contiguous().cpu()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _snapshot_outputs(model, outputs):
    obs_mask = model.pts_bbox_head.ind_stamps_all == 0

    def cpu(value):
        return value.detach().float().cpu().clone()

    forecast_cls = [cpu(value) for value in outputs['forecast_semantics_list']]
    forecast_points = [cpu(value) for value in outputs['forecast_points_list']]
    pred_traj = torch.cat(outputs['pred_trajs_list'], dim=1)
    return dict(
        current_cls=cpu(outputs['cls_score']),
        current_points=cpu(outputs['refine_pts']),
        eval_current_cls=cpu(
            outputs['outs']['all_cls_scores'][-1][:, obs_mask]),
        eval_current_points=cpu(
            outputs['outs']['all_refine_pts'][-1][:, obs_mask]),
        forecast_cls=forecast_cls,
        forecast_points=forecast_points,
        pred_traj=cpu(pred_traj))


def _load_batch(cfg, split, workers_per_gpu, sample_index):
    dataset_cfg = cfg.data[split].copy()
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    if sample_index is None:
        required_slots = len(dataset.query_memory_history_target_ages) \
            if dataset.query_memory_history_selection_mode == 'target_age' \
            else dataset.query_memory_history_frames
        sample_index = next((
            index for index, info_index in enumerate(dataset.temp2nusc_map)
            if len(dataset._build_query_memory_history_infos(info_index))
            == required_slots
        ), None)
        if sample_index is None:
            raise RuntimeError(
                'dataset has no sample with every requested query-memory '
                'history slot populated')
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f'sample-index {sample_index} outside dataset length {len(dataset)}')
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False)
    iterator = iter(data_loader)
    batch = None
    for _ in range(sample_index + 1):
        batch = next(iterator)
    return dataset, batch, sample_index


def _build_and_run(cfg, checkpoint, enabled, batch, gpu_id):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    if enabled:
        query_memory_cfg = dict(model_cfg.get('query_memory_cfg', {}))
        query_memory_cfg.update(dict(
            enabled=True,
            source='cache',
            memory_finetune_mode=False,
            log_diagnostics=True))
        model_cfg.query_memory_cfg = query_memory_cfg
    else:
        model_cfg.query_memory_cfg = dict(enabled=False)

    model = build_model(model_cfg, test_cfg=cfg.get('test_cfg'))
    revise_keys = cfg.get('revise_keys', [(r'^module.', '')])
    load_checkpoint(
        model, checkpoint, map_location='cpu', revise_keys=revise_keys)
    before_digest = _tensor_digest(model)
    model.eval()

    holder = {}
    original_forward_backbone = model.forward_backbone

    def capture_forward_backbone(*args, **kwargs):
        outputs = original_forward_backbone(*args, **kwargs)
        holder['snapshot'] = _snapshot_outputs(model, outputs)
        return outputs

    model.forward_backbone = capture_forward_backbone
    parallel_model = MMDataParallel(
        model.cuda(gpu_id), device_ids=[gpu_id])
    with torch.no_grad():
        parallel_model(return_loss=False, rescale=True, **batch)
    if 'snapshot' not in holder:
        raise RuntimeError('forward_backbone capture did not execute')

    after_digest = _tensor_digest(model)
    if before_digest != after_digest:
        raise RuntimeError('base model parameters or buffers changed in eval')
    diagnostics = list(getattr(model, 'query_memory_diagnostics', []))
    if enabled:
        out_proj_abs_max = float(
            model.query_memory.fusion.out_proj.weight.detach().abs().max().item())
        motion_last = model.query_memory.motion_compensator.mlp[-1]
        motion_abs_max = max(
            float(motion_last.weight.detach().abs().max().item()),
            float(motion_last.bias.detach().abs().max().item()))
    else:
        out_proj_abs_max = None
        motion_abs_max = None
    stats = dict(
        read_count=len(diagnostics),
        fused_query_count=sum(
            int(item.get('query_count', 0)) for item in diagnostics),
        valid_history_slots=float(
            getattr(model, '_last_query_memory_valid_slots', 0.0)),
        has_candidate=any(
            bool(item.get('has_candidate').any().item())
            for item in diagnostics
            if isinstance(item.get('has_candidate'), torch.Tensor)),
        candidate_count_max=max([
            int(item['candidate_count'].max().item())
            for item in diagnostics
            if isinstance(item.get('candidate_count'), torch.Tensor)
            and item['candidate_count'].numel()
        ] or [0]),
        residual_norm_max=max([
            float(item['residual_norm'].max().item())
            for item in diagnostics
            if isinstance(item.get('residual_norm'), torch.Tensor)
            and item['residual_norm'].numel()
        ] or [0.0]),
        out_proj_abs_max=out_proj_abs_max,
        motion_last_abs_max=motion_abs_max)
    snapshot = holder['snapshot']
    del parallel_model
    del model
    torch.cuda.empty_cache()
    return snapshot, stats, before_digest


def _max_abs_diff(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f'tensor shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}')
    if not left.numel():
        return 0.0
    return float((left - right).abs().max().item())


def _compare_snapshots(off, on):
    diffs = {}
    for key in ('current_cls', 'current_points', 'eval_current_cls',
                'eval_current_points', 'pred_traj'):
        diffs[key] = _max_abs_diff(off[key], on[key])
    if len(off['forecast_cls']) != len(on['forecast_cls']):
        raise ValueError('forecast_cls horizon count mismatch')
    if len(off['forecast_points']) != len(on['forecast_points']):
        raise ValueError('forecast_points horizon count mismatch')
    for index, (left, right) in enumerate(zip(
            off['forecast_cls'], on['forecast_cls'])):
        diffs[f'forecast_cls_{index + 1}'] = _max_abs_diff(left, right)
    for index, (left, right) in enumerate(zip(
            off['forecast_points'], on['forecast_points'])):
        diffs[f'forecast_points_{index + 1}'] = _max_abs_diff(left, right)
    return diffs


def main():
    args = parse_args()
    if args.mode == 'trained' and args.memory_checkpoint is None:
        raise ValueError('--memory-checkpoint is required in trained mode')
    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    dataset, batch, sample_index = _load_batch(
        cfg, args.split, args.workers_per_gpu, args.sample_index)
    active_checkpoint = (
        args.memory_checkpoint if args.mode == 'trained' else args.checkpoint)
    off, off_stats, off_digest = _build_and_run(
        cfg, active_checkpoint, False, batch, args.gpu_id)
    on, on_stats, on_digest = _build_and_run(
        cfg, active_checkpoint, True, batch, args.gpu_id)
    if off_digest != on_digest:
        raise RuntimeError(
            'C0/C1 base model state differs before inference; checkpoint '
            'comparison is invalid')

    diffs = _compare_snapshots(off, on)
    current_keys = [
        'current_cls', 'current_points', 'eval_current_cls',
        'eval_current_points']
    future_keys = [
        key for key in diffs
        if key.startswith('forecast_') or key == 'pred_traj']
    failures = []
    if on_stats['valid_history_slots'] <= 0:
        failures.append('no valid history slots were loaded')
    if on_stats['read_count'] != 7:
        failures.append(
            f'read_count={on_stats["read_count"]}, expected 7')
    if on_stats['fused_query_count'] != 1040:
        failures.append(
            'fused_query_count='
            f'{on_stats["fused_query_count"]}, expected 1040')
    if not on_stats['has_candidate']:
        failures.append('no spatial/causal memory candidate reached attention')

    if args.mode == 'zero':
        bad = {key: value for key, value in diffs.items()
               if value > args.atol}
        if bad:
            failures.append(f'C0/C1 identity exceeded atol: {bad}')
        if on_stats['out_proj_abs_max'] != 0.0:
            failures.append('fusion out projection is not zero-initialized')
        if on_stats['motion_last_abs_max'] != 0.0:
            failures.append('motion final layer is not zero-initialized')
        if on_stats['residual_norm_max'] > args.atol:
            failures.append(
                'zero-initialized fusion applied a nonzero residual: '
                f'{on_stats["residual_norm_max"]}')
    else:
        bad_current = {
            key: diffs[key] for key in current_keys
            if diffs[key] > args.atol}
        if bad_current:
            failures.append(
                f'trained Memory changed current-frame outputs: {bad_current}')
        max_future_diff = max(diffs[key] for key in future_keys)
        if max_future_diff <= args.min_future_diff:
            failures.append(
                'trained Memory produced no required future difference: '
                f'max={max_future_diff}')
        if on_stats['out_proj_abs_max'] == 0.0:
            failures.append('trained fusion out projection is still exactly zero')
        if on_stats['residual_norm_max'] <= 0.0:
            failures.append('trained Memory applied no nonzero fusion residual')

    info_index = dataset.temp2nusc_map[sample_index]
    sample_token = str(dataset.data_infos[info_index]['token'])
    report = dict(
        mode=args.mode,
        split=args.split,
        sample_index=sample_index,
        sample_token=sample_token,
        checkpoint=str(active_checkpoint),
        atol=args.atol,
        min_future_diff=args.min_future_diff,
        max_abs_diffs=diffs,
        memory_on=on_stats,
        memory_off=off_stats,
        passed=not failures,
        failures=failures)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
