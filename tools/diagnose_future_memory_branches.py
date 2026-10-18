"""Paired A+B output interventions from ONE forward per input.

No training, no cache writes, no model/source modification. Memory retrieval
and Baseline recurrence are unchanged. Restoring one output does not rerun
the Adapter or modify either other output. Saves small confusion matrices,
not full validation predictions. The snapshot reference is not a separately
evaluated checkpoint or a cache-off experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from diagnose_future_memory_semantics import CLASSES, confusion, summarize, transitions


VARIANTS = ('ab', 'restore_position', 'restore_occupancy',
            'restore_semantic', 'baseline_snapshot')
SCAN_VARIANTS = ('ab', 'position_12_base',
                 'protected_occ_alpha_025', 'protected_occ_alpha_050',
                 'protected_occ_alpha_075')
POSITION_VARIANTS = ('ab', 'position_12_base')
PROTECTED_CLASSES = (6, 7, 8, 15, 16)  # motorcycle, pedestrian, cone, manmade, vegetation


def compose(semantic, occupancy):
    return semantic + (occupancy - semantic.max(-1, keepdim=True).values)


def intervention_inputs(base_sem, base_pos, sem, occ, pos, decoded):
    """Outputs only; no gate changes, selection changes or recurrent writes."""
    if not torch.equal(compose(sem, occ), decoded):
        raise AssertionError('A+B decoded logits do not match the captured decomposition')
    return dict(
        ab=(decoded, pos),
        restore_position=(decoded, base_pos),
        restore_occupancy=(compose(sem, base_sem.max(-1, keepdim=True).values), pos),
        restore_semantic=(compose(base_sem, occ), pos),
        baseline_snapshot=(base_sem, base_pos))


def scan_inputs(base_sem, base_pos, sem, occ, pos, decoded, horizon):
    """Point-level soft class protection on negative O residuals at 2/3 s.

    A point is eligible if either the Baseline or corrected semantic
    distribution assigns probability to protected classes. This is only a
    diagnostic output intervention; no thresholds or training weights change.
    """
    if not torch.equal(compose(sem, occ), decoded):
        raise AssertionError('A+B decoded logits do not match the captured decomposition')
    variants = dict(ab=(decoded, pos),
                    position_12_base=(decoded, base_pos if horizon in (1, 2) else pos))
    base_occ = base_sem.max(-1, keepdim=True).values
    delta = occ - base_occ
    if horizon in (2, 3):
        p_base = base_sem.softmax(-1)[..., PROTECTED_CLASSES].sum(-1, keepdim=True)
        p_sem = sem.softmax(-1)[..., PROTECTED_CLASSES].sum(-1, keepdim=True)
        mass = 1. - (1. - p_base) * (1. - p_sem)
        for alpha, label in ((.25, '025'), (.5, '050'), (.75, '075')):
            scale = torch.where(delta < 0., 1. - (1. - alpha) * mass,
                                torch.ones_like(delta))
            variants[f'protected_occ_alpha_{label}'] = (
                compose(sem, base_occ + scale * delta), pos)
    else:
        for label in ('025', '050', '075'):
            variants[f'protected_occ_alpha_{label}'] = (decoded, pos)
    return variants


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def filter_statistics(head, logits, points, gt):
    """Mirror get_occ's pre-voxel filters; final decoding uses get_occ itself.

    Truth rows are GT labels at the point's voxel (18=out of bounds), NOT
    nearest-GT matches. Columns: input, raw threshold, post-class-distance
    threshold, also Query-distance pass. Unmasked evaluator convention.
    """
    probs = logits.sigmoid()
    confidence, cls = probs.max(-1)
    normalized_distance = (points - points.mean(2, keepdim=True)).norm(dim=-1)
    scale = torch.ones_like(confidence)
    special = (cls == 15) | (cls == 16)
    # get_occ defaults thre1=thre2=0.1. Only used on the same classes.
    scale[special] = (.1 / normalized_distance[special]).clamp(max=1)
    thresholds = confidence.new_tensor(head.test_cfg.get('score_thr', .1))
    if thresholds.ndim == 0:
        thresholds = thresholds.expand(head.num_classes)
    raw_pass = confidence > thresholds[cls]
    score_pass = confidence * scale > thresholds[cls]
    bounds = torch.as_tensor(head.pc_range, device=points.device, dtype=points.dtype)
    origin = bounds[:3]
    extent = bounds[3:] - origin
    metric = points * extent + origin
    center_dist = (metric - metric.mean(2, keepdim=True)).norm(dim=-1)
    dist_pass = center_dist < head.test_cfg.get('ctr_dist_thr', 3.)
    voxel_size = torch.as_tensor(head.voxel_size, device=points.device, dtype=points.dtype)
    index = ((metric - origin) // voxel_size).long()
    grid = torch.as_tensor(gt.shape, device=points.device)
    inside = ((index >= 0) & (index < grid)).all(-1)
    labels = torch.full_like(cls, 18)
    valid_index = index[inside]
    truth = torch.as_tensor(gt, device=points.device, dtype=torch.long)
    labels[inside] = truth[valid_index[:, 0], valid_index[:, 1], valid_index[:, 2]]
    stages = [torch.ones_like(inside), raw_pass, score_pass, score_pass & dist_pass]
    by_truth = torch.stack([torch.bincount(labels[m], minlength=19) for m in stages], -1)
    return dict(by_point_voxel_gt=by_truth.cpu().numpy(),
                by_predicted_class=torch.bincount(cls.reshape(-1), minlength=17).cpu().numpy(),
                scalar=np.array([cls.numel(), int(raw_pass.sum()), int(score_pass.sum()),
                                 int((score_pass & dist_pass).sum()),
                                 int((score_pass & dist_pass & inside).sum())], dtype=np.int64))


def residual_statistics(base_sem, base_pos, sem, occ, pos, pc_range):
    """Sums by Baseline top class; divide by column 0 for means."""
    cls = base_sem.argmax(-1).reshape(-1)
    if pos.device != base_pos.device:
        raise AssertionError(f'Position snapshot devices differ: corrected={pos.device}, base={base_pos.device}')
    bounds = torch.as_tensor(pc_range, device=pos.device, dtype=pos.dtype)
    extent = bounds[3:] - bounds[:3]
    shift = ((pos - base_pos) * extent).reshape(-1, 3).float()
    delta_o = (occ - base_sem.max(-1, keepdim=True).values).reshape(-1).float()
    changed = (sem.argmax(-1).reshape(-1) != cls).float()
    rows = torch.stack([torch.ones_like(delta_o), delta_o, delta_o.abs(),
                        shift.norm(dim=-1), shift[:, 0], shift[:, 1], shift[:, 2], changed], -1)
    sums = rows.new_zeros(17, 8)
    sums.index_add_(0, cls, rows)
    return sums.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab.py')
    parser.add_argument('--checkpoint', default='work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/epoch_1.pth')
    parser.add_argument('--manifest', default='work_dirs/diagnostics-ab-miou-20261015/paired_subset256.json')
    parser.add_argument('--limit', type=int, default=256)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiment', choices=('original', 'scan', 'position_full'),
                        default='original')
    parser.add_argument('--full', action='store_true',
                        help='run all test frames; only available for position_full')
    args = parser.parse_args()
    if args.full and args.experiment != 'position_full':
        parser.error('--full is only available for position_full')
    if not args.full and not 1 <= args.limit <= 512:
        parser.error('--limit must be 1..512; full validation is not supported')
    variants = (VARIANTS if args.experiment == 'original' else
                SCAN_VARIANTS if args.experiment == 'scan' else POSITION_VARIANTS)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config
    torch.set_num_threads(1)
    torch.cuda.set_device(args.gpu_id)
    set_random_seed(0, deterministic=True)
    cfg = patch_config(compat_cfg(Config.fromfile(args.config)))
    setup_multi_processes(cfg)
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True  # test_temporal default
    qcfg = cfg.model.query_memory_cfg
    assert qcfg.source == 'cache' and qcfg.future_memory_adapter_version == 'v2'
    assert qcfg.future_memory_adapter_v2.gate_mode == 'task_split'
    dataset = build_dataset(cfg.data.test)
    if args.full:
        manifest = [dict(index=index,
                         scene=dataset.data_infos[ni]['scene_name'],
                         token=dataset.data_infos[ni]['token'])
                    for index, ni in enumerate(dataset.temp2nusc_map)]
    else:
        manifest = json.loads(Path(args.manifest).read_text())['manifest'][:args.limit]
        assert len(manifest) == args.limit and len(manifest) < len(dataset)
    for entry in manifest:
        info = dataset.data_infos[dataset.temp2nusc_map[entry['index']]]
        assert entry['token'] == info['token'] and entry['scene'] == info['scene_name']
    loader = build_dataloader(torch.utils.data.Subset(dataset, [m['index'] for m in manifest]),
                              samples_per_gpu=1, workers_per_gpu=args.workers,
                              num_gpus=1, dist=False, shuffle=False, seed=0)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'config.py').write_text(cfg.pretty_text)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    # Check frozen shared tensors against the original Baseline checkpoint.
    baseline_file = Path('ckpts/epoch_56.pth')
    baseline = torch.load(baseline_file, map_location='cpu')
    baseline_state = {k.removeprefix('module.'): v for k, v in baseline['state_dict'].items()}
    shared = sorted(set(baseline_state) & set(state))
    differing = [k for k in shared if not torch.equal(state[k], baseline_state[k])]
    baseline_check = dict(shared_tensors=len(shared), different_tensors=differing,
                          missing_baseline_keys=sorted(set(baseline_state) - set(state)))
    if differing or baseline_check['missing_baseline_keys']:
        raise AssertionError(f'Frozen Baseline tensors changed: {baseline_check}')
    del checkpoint, baseline, state, baseline_state
    model.CLASSES = dataset.CLASSES
    model = model.cuda(args.gpu_id).eval()
    parallel = MMDataParallel(model, device_ids=[args.gpu_id]).eval()
    captured = []
    original = model.forward_backbone

    def capture(self, *a, **kw):
        out = original(*a, **kw)
        captured.append(out)
        return out

    model.forward_backbone = types.MethodType(capture, model)
    audit = dict(checkpoint=args.checkpoint, checkpoint_sha256=sha256(args.checkpoint),
                 baseline_checkpoint_sha256=sha256(baseline_file), baseline_tensor_check=baseline_check,
                 manifest=manifest, variants=variants, seed=0,
                 experiment=args.experiment,
                 protected_classes=PROTECTED_CLASSES if args.experiment == 'scan' else None,
                 source_sha256={p: sha256(p) for p in [__file__,
                     'mmdet3d/models/sparsedetectors/query_memory.py',
                     'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py',
                     'mmdet3d/models/sparsedetectors/opus_head.py']},
                 semantics='One normal A+B forward; intervention changes only specified output; no training, no recurrence changes',
                 checks=dict(zero_s_equal='pending', ab_decode_equal='pending', frozen_baseline_equal=True))
    (root / 'provenance.json').write_text(json.dumps(audit, indent=2))
    histories = np.zeros((len(manifest), len(variants), 4, 18, 18), dtype=np.int64)
    filters = {v: [dict(by_point_voxel_gt=np.zeros((19, 4), dtype=np.int64),
                        by_predicted_class=np.zeros(17, dtype=np.int64),
                        scalar=np.zeros(5, dtype=np.int64)) for _ in range(3)] for v in variants}
    residuals = np.zeros((3, 17, 8), dtype=np.float64)
    changes = {v: [{} for _ in range(3)] for v in variants if v != 'ab'}
    start = time.monotonic()
    print(f'Start paired diagnostic: {len(manifest)} samples, one forward each, GPU {args.gpu_id}', flush=True)
    try:
        with torch.no_grad(), (root / 'progress.jsonl').open('x') as progress:
            for sample, data in enumerate(loader):
                captured.clear()
                results = parallel(return_loss=False, rescale=True, **data)
                assert len(captured) == 1
                out = captured.pop()
                ni = dataset.temp2nusc_map[manifest[sample]['index']]
                current = model.pts_bbox_head.get_occ(dict(cls_scores=out['base_cls_score'],
                                                         refine_pts=out['base_refine_pts']))[0].cpu().numpy()
                assert np.array_equal(current, results['semantic_occ_0s'][0])
                info = dataset.data_infos[ni]
                with np.load(Path(info['occ_path']) / 'labels.npz') as g:
                    h0 = confusion(g['semantics'], current)
                histories[sample, :, 0] = h0
                for h, step in enumerate((1, 3, 5)):
                    future_info = dataset.data_infos[ni + step + 1]
                    assert future_info['scene_name'] == info['scene_name']
                    with np.load(Path(future_info['occ_path']) / 'labels.npz') as g:
                        gt = g['semantics']
                    base_sem = out['baseline_forecast_semantics_list'][step]
                    base_pos = out['baseline_forecast_points_list'][step]
                    sem = out['forecast_semantic_logits_list'][step]
                    occ = out['forecast_occupancy_logits_list'][step]
                    pos = out['forecast_points_list'][step]
                    decoded = out['forecast_semantics_list'][step]
                    inputs = (intervention_inputs(base_sem, base_pos, sem, occ, pos, decoded)
                              if args.experiment == 'original' else
                              scan_inputs(base_sem, base_pos, sem, occ, pos, decoded, h + 1))
                    if args.experiment == 'position_full':
                        inputs = {key: inputs[key] for key in POSITION_VARIANTS}
                    residuals[h] += residual_statistics(base_sem, base_pos, sem, occ, pos, model.pc_range)
                    reference = results[f'semantic_occ_{step+1}s'][0]
                    for v, (logits, points) in inputs.items():
                        decoded_voxels = model.pts_bbox_head.get_occ(dict(cls_scores=logits, refine_pts=points))[0].cpu().numpy()
                        if v == 'ab':
                            assert np.array_equal(decoded_voxels, reference), 'Replayed A+B decoder changed'
                        histories[sample, variants.index(v), h+1] = confusion(gt, decoded_voxels)
                        stats = filter_statistics(model.pts_bbox_head, logits, points, gt)
                        for key, value in stats.items():
                            filters[v][h][key] += value
                        if v != 'ab':
                            for key, value in transitions(gt, reference, decoded_voxels).items():
                                changes[v][h][key] = changes[v][h].get(key, 0) + value
                assert not model.training and all(p.grad is None for p in model.parameters())
                progress.write(json.dumps(dict(sample=sample, **manifest[sample],
                                               elapsed=time.monotonic()-start)) + '\n')
                progress.flush()
                # Drop snapshots before the next full forward, no unbounded graph/cache retention.
                del out, results, inputs, base_sem, base_pos, sem, occ, pos, decoded
                if (sample+1) % 8 == 0 or sample == 0:
                    print(f'{sample+1}/{len(manifest)} elapsed={time.monotonic()-start:.1f}s '
                          f'peak_allocated={torch.cuda.max_memory_allocated()/2**30:.2f}GiB', flush=True)
    finally:
        model.forward_backbone = original
    audit['checks'].update(zero_s_equal=True, ab_decode_equal=True)
    audit['completed_samples'] = len(manifest)
    (root / 'provenance.json').write_text(json.dumps(audit, indent=2))
    np.savez_compressed(root / 'per_sample_confusions.npz', hist=histories)
    totals = histories.sum(0)
    report = dict(samples=len(manifest), scenes=len({m['scene'] for m in manifest}),
                  elapsed_seconds=time.monotonic()-start,
                  metrics={v: [summarize(hist) for hist in totals[i]] for i, v in enumerate(variants)},
                  changes_from_ab=changes,
                  filters={v: [{k: a.tolist() for k, a in row.items()} for row in rows]
                           for v, rows in filters.items()},
                  filter_columns=['input', 'raw_score_pass', 'post_class_distance_score_pass', 'also_query_distance_pass'],
                  filter_scalar_columns=['input', 'raw_score_pass', 'post_class_distance_score_pass',
                                         'also_query_distance_pass', 'also_in_grid'],
                  residual_by_baseline_class=residuals.tolist(),
                  residual_columns=['count', 'occupancy_shift_sum', 'abs_occupancy_shift_sum',
                                    'position_norm_sum', 'position_dx_sum', 'position_dy_sum',
                                    'position_dz_sum', 'semantic_top_class_changed_count'],
                  classes=CLASSES, checks=audit['checks'],
                  limitations=('Full test split with matched inputs; intervention probes this checkpoint, not retraining causality.'
                               if args.full else
                               'Paired subset inference, not full validation; branch interventions probe this checkpoint locally, not training causality.'))
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    for v, rows in report['metrics'].items():
        print(v, 'IoU', [round(r['iou'], 4) for r in rows],
              'mIoU', [round(r['miou'], 4) for r in rows], flush=True)
    print(f'COMPLETE: {root}', flush=True)


if __name__ == '__main__':
    main()
