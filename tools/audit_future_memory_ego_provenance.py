"""Read-only CPU audit of the external ego pickle; never runs a GPU model.

Fits *one scalar per record* to a specified three-coordinate future-displacement
vector. A small residual is an algebraic signature, not a learned predictor.
The fit alone does not recover the unpublished generator or its random seed.
Only trust pickle inputs from the existing local experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def proportional_signature(values, reference, atol=1e-10):
    """Report direction-preserving scalar fits; exclude degenerate references."""
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if values.shape != reference.shape or values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('Expected matching [N,3] values and reference vectors')
    norm2 = np.sum(reference * reference, axis=-1)
    nondegenerate = norm2 > 1e-16
    scales = np.sum(values * reference, axis=-1) / np.maximum(norm2, 1e-30)
    error = np.max(np.abs(values - scales[:, None] * reference), axis=-1)
    finite = np.isfinite(error) & np.isfinite(scales)
    matched = nondegenerate & finite & (error <= atol)
    selected = scales[nondegenerate & finite]
    return dict(records=len(values), nondegenerate=int(nondegenerate.sum()),
                matching=int(matched.sum()), atol=atol,
                max_abs_residual=float(error.max()) if len(error) else None,
                scale_range=[float(selected.min()), float(selected.max())]
                if len(selected) else None)


def summarize_records(records):
    complete = [r for r in records if np.asarray(r['gt']).shape == (7, 3)]
    if not complete:
        return dict(records=len(records), complete_future=0)
    gt = np.asarray([r['gt'] for r in complete], dtype=np.float64)
    cmd = np.asarray([r['cmd'] for r in complete])
    expected = np.eye(3)[np.where(gt[:, -1, 0] <= -2, 0,
                                 np.where(gt[:, -1, 0] >= 2, 2, 1))]
    return dict(
        records=len(records), complete_future=len(complete),
        cmd_future_rule_matches=int(np.all(cmd == expected, axis=-1).sum()),
        cmd_counts=dict(Counter(map(str, cmd.argmax(-1)))),
        v_future_signature=proportional_signature(
            [r['v'] for r in complete], gt[:, 3] - gt[:, 1]),
        a_future_signature=proportional_signature(
            [r['a'] for r in complete], (gt[:, 6] - gt[:, 1]) / 2.5),
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def lidar_global_pose(info):
    from pyquaternion import Quaternion
    def transform(rotation, translation):
        matrix = np.eye(4)
        matrix[:3, :3] = Quaternion(rotation).rotation_matrix
        matrix[:3, 3] = translation
        return matrix
    return (transform(info['ego2global_rotation'], info['ego2global_translation'])
            @ transform(info['lidar2ego_rotation'], info['lidar2ego_translation']))


def past_increment_signature(poses, index, steps=4):
    """Reconstruct the pickle's cumulative negative one-step past increments.

    Each increment is expressed in its own starting lidar frame. This checks
    provenance, not equivalence to a trajectory in a single current frame.
    No pose after ``index`` is read.
    """
    increments = []
    for k in range(1, steps+1):
        relative = np.linalg.inv(poses[index-k]) @ poses[index-k+1]
        increments.append([relative[0, 3], relative[1, 3],
                           np.arctan2(relative[1, 0], relative[0, 0])])
    return -np.cumsum(increments, axis=0)


def checkpoint_ego_sensitivity(checkpoint, records):
    """Actual checkpoint's two CPU subnets, not a full occupancy replay.

    The Sequential topology is the plan_head/points_scale_branch definition in
    SparseWorld4DTraj.__init__. Strict state loading checks every tensor shape.
    Zeroing one field is an OOD sensitivity probe, not a causal repair/eval.
    """
    import torch
    from torch import nn

    torch.set_num_threads(1)
    state = torch.load(checkpoint, map_location='cpu')['state_dict']
    plan = nn.Sequential(nn.Linear(21, 256), nn.ReLU(inplace=True),
                         nn.Linear(256, 256), nn.ReLU(inplace=True),
                         nn.Linear(256, 256)).eval()
    scale = nn.Sequential(nn.Linear(256, 64), nn.ReLU(),
                          nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)).eval()
    for prefix, module in [('plan_head.', plan), ('points_scale_branch.', scale)]:
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)
        module.requires_grad_(False)
    del state
    inputs = torch.tensor(np.asarray([
        np.concatenate([np.asarray(r[k]).reshape(-1) for k in sorted(r) if k != 'gt'])
        for r in records]), dtype=torch.float32)
    result = dict(samples=len(records), checkpoint_sha256=sha256(checkpoint),
                  scope='checkpoint CPU ego/point-scale subnets only; NOT IoU/mIoU')
    with torch.no_grad():
        original = plan(inputs)
        original_scale = (torch.tanh(scale(original)) + 1) / 2 * .7 + .8
        for field, section in [('a', slice(0, 3)), ('cmd', slice(3, 6)),
                               ('v', slice(6, 9))]:
            probe = inputs.clone()
            probe[:, section] = 0
            output = plan(probe)
            output_scale = (torch.tanh(scale(output)) + 1) / 2 * .7 + .8
            result['zero_' + field] = dict(
                changed_ego_feature_rows=int((output != original).any(-1).sum()),
                changed_point_scale_rows=int((output_scale != original_scale).any(-1).sum()),
                max_abs_ego_feature_change=float((output-original).abs().max()),
                max_abs_point_scale_change=float((output_scale-original_scale).abs().max()))
    return result


def audit(config, checkpoint=None):
    import gc
    import pickle
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    cfg = Config.fromfile(config)
    path = ROOT / 'admlp/stp3_val/data_nuscene.pkl'
    with path.open('rb') as stream:
        ego = pickle.load(stream)
    report = dict(
        scope='CPU metadata/algebra and optional ego-subnet forward; '
              'no full-model inference, cache writes or training',
        ego_file=str(path), sha256=sha256(path),
        identical_top_level_copy=sha256(ROOT / 'stp3_val/data_nuscene.pkl') == sha256(path),
        all_records=summarize_records(list(ego.values())),
        signatures={
            'v': 'v = s_v * (GT[3] - GT[1]), fitted scalar per record',
            'a': 'a = s_a * (GT[6] - GT[1]) / 2.5, fitted scalar per record',
            'cmd': 'LEFT if GT[-1,x] <= -2, RIGHT if >= 2, else FORWARD',
            'index_times': 'GT[0] is current; GT[1/3/6] nominal +0.5/+1.5/+3 seconds',
            'caution': '2.5 is a nominal interval, NOT measured sample time; '
                       'fitted scale is not proof of the generator randomization algorithm',
        },
    )
    for split in ('train', 'test'):
        dataset = build_dataset(cfg.data[split])
        infos = dataset.data_infos
        lookup = {r['token']: i for i, r in enumerate(infos)}
        records = [ego[infos[i]['token']] for i in dataset.temp2nusc_map]
        result = summarize_records(records)
        # Independently reconstruct GT from nuScenes metadata poses. Do not
        # infer time from an input field's label such as "v" or "a".
        poses = np.asarray([lidar_global_pose(r) for r in infos])
        gt_max_error = 0.
        gt_matching = 0
        x_max_error = 0.
        x_matching = 0
        offsets = []
        for i in dataset.temp2nusc_map:
            if any(infos[j]['scene_token'] != infos[i]['scene_token'] for j in range(i, i+7)):
                raise ValueError('Cross-scene future GT in metadata')
            relative = np.linalg.inv(poses[i])[None] @ poses[i:i+7]
            rebuilt = np.stack([relative[:, 0, 3], relative[:, 1, 3],
                                np.arctan2(relative[:, 1, 0], relative[:, 0, 0])], -1)
            error = float(np.max(np.abs(rebuilt - np.asarray(ego[infos[i]['token']]['gt']))))
            gt_max_error = max(gt_max_error, error)
            gt_matching += int(error <= 1e-9)
            if i < 4 or any(infos[j]['scene_token'] != infos[i]['scene_token']
                            for j in range(i-4, i)):
                raise ValueError('Insufficient same-scene past for x reconstruction')
            x_error = float(np.max(np.abs(past_increment_signature(poses, i)
                               - np.asarray(ego[infos[i]['token']]['x']))))
            x_matching += int(x_error <= 1e-9)
            x_max_error = max(x_max_error, x_error)
            offsets.append([(infos[i+k]['timestamp'] - infos[i]['timestamp']) / 1e6
                            for k in (1, 3, 6)])
        result['independent_pose_gt_check'] = dict(
            matching=gt_matching, total=len(records), atol=1e-9,
            max_abs_error=gt_max_error,
            future_time_min=np.min(offsets, axis=0).tolist(),
            future_time_max=np.max(offsets, axis=0).tolist())
        result['x_past_only_pose_reconstruction'] = dict(
            matching=x_matching, total=len(records), atol=1e-9,
            max_abs_error=x_max_error,
            note='Negative cumulative one-step increments, each in its own past frame; '
                 'not the same as expressing all positions in the current frame.')
        histories = Counter()
        unique_history = set()
        overruns = []
        for i in dataset.temp2nusc_map:
            current = infos[i]
            for hist in dataset._build_query_memory_history_infos(i):
                source = lookup[hist['sample_idx']]
                unique_history.add(hist['sample_idx'])
                slot = str(hist['slot_index'])
                histories['total'] += 1
                histories['slot_' + slot] += 1
                if infos[source+6]['scene_token'] != current['scene_token']:
                    raise ValueError('Unexpected source future pose crossing scene')
                overrun = (infos[source+6]['timestamp'] - current['timestamp']) / 1e6
                if overrun > 0:
                    histories['source_gt6_after_current'] += 1
                    histories['slot_' + slot + '_source_gt6_after_current'] += 1
                    overruns.append(overrun)
        result['selected_history_source_metadata'] = dict(
            counts=dict(histories), unique_sources=len(unique_history),
            overrun_range_seconds=[min(overruns), max(overruns)] if overruns else None,
            note='Source input future dependency, NOT a future cache record timestamp; '
                 'not a replay of historical cache tensors.')
        result['selected_history_ego_signatures'] = summarize_records(
            [ego[token] for token in sorted(unique_history)])
        if split == 'test' and checkpoint:
            selected = np.linspace(0, len(records)-1, min(256, len(records)), dtype=int)
            result['checkpoint_input_sensitivity'] = checkpoint_ego_sensitivity(
                checkpoint, [records[i] for i in selected])
        report[split] = result
        print(json.dumps({split: result}, indent=2), flush=True)
        del dataset, infos, poses, records
        gc.collect()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Optional CPU ego-subnet sensitivity probe')
    parser.add_argument('--output', help='New audit JSON path; refuses to overwrite')
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        parser.error('--output already exists; refusing to overwrite')
    result = audit(args.config, args.checkpoint)
    if args.output:
        with open(args.output, 'x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))
