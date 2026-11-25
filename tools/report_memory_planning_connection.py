"""Verify paired inputs/native outputs and report same-protocol planning scores."""

import argparse
import json
from pathlib import Path

import numpy as np

import audit_planning_pair as audit


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def report(run, reference, allow_partial=False, position_reference=None):
    run, reference = Path(run), Path(reference)
    manifest = json.loads((run / 'manifest.json').read_text())
    ref_manifest = json.loads((reference / 't1/manifest.json').read_text())
    if manifest != (ref_manifest[:len(manifest)] if allow_partial else ref_manifest):
        raise ValueError('Validation manifest mismatch')
    progress, ref_progress = rows(run / 'progress.jsonl'), rows(reference / 't1/progress.jsonl')
    if progress != (ref_progress[:len(progress)] if allow_partial else ref_progress):
        raise ValueError('Image/ego input digest or token order mismatch')
    if len(progress) != len(manifest):
        raise ValueError('Incomplete prediction')
    provenance = json.loads((run / 'provenance.json').read_text())
    ref_provenance = json.loads((reference / 't1/provenance.json').read_text())
    for key in ('checkpoint_sha256', 'source_sha256', 'exporter_sha256'):
        if provenance[key] != ref_provenance[key]:
            raise ValueError(f'Archived native model/exporter mismatch: {key}')
    if not np.array_equal(np.load(run / 'query_timestamps.npy'),
                          np.load(reference / 't1/query_timestamps.npy')):
        raise ValueError('TASS schedule mismatch')
    connected = audit.read_pickle(run / 'trajectories.pkl')
    native = audit.read_pickle(run / 'native_trajectories.pkl')
    ref_native = audit.read_pickle(reference / 't1/trajectories.pkl')
    baseline = audit.read_pickle(reference / 'baseline/trajectories.pkl')
    tokens = [r['token'] for r in manifest]
    if set(connected) != set(tokens) or set(native) != set(tokens):
        raise ValueError('Prediction token mismatch')
    if not all(np.array_equal(native[t], ref_native[t]) and
               np.array_equal(native[t], baseline[t]) for t in tokens):
        raise ValueError('Native trajectories changed independently of the connection')
    difference = np.concatenate([connected[t] - native[t] for t in tokens], axis=0)
    result = dict(
        samples=len(tokens), partial=allow_partial,
        checkpoint_sha256=provenance['checkpoint_sha256'],
        variant=json.loads((run / 'planning_connection.json').read_text())['variant'],
        matched_image_ego_inputs=True, matched_tass=True,
        native_exactly_equals_previous_t1_and_baseline=True,
        changed_trajectories=sum(not np.array_equal(connected[t], native[t]) for t in tokens),
        max_abs_cumulative_xy_difference=float(np.abs(difference).max()),
        mean_abs_cumulative_xy_difference=float(np.abs(difference).mean()),
        mean_position_change_m_per_step=np.linalg.norm(difference, axis=-1).mean(0).tolist())
    if not allow_partial:
        new = json.loads((run / 'metrics.json').read_text())
        base = json.loads((reference / 'baseline/metrics.json').read_text())
        if new['samples'] != len(tokens) or base['samples'] != len(tokens):
            raise ValueError('Metric sample count mismatch')
        for key in ('metric_source_sha256', 'metric_entry_sha256', 'gt_sha256'):
            if new[key] != base[key]:
                raise ValueError(f'Metric protocol mismatch: {key}')
        result['prefix'] = {}
        for horizon in ('1s', '2s', '3s'):
            result['prefix'][horizon] = {}
            for metric in ('L2', 'box_collision_percent', 'obj_col'):
                a, b = base['prefix'][horizon][metric], new['prefix'][horizon][metric]
                result['prefix'][horizon][metric] = dict(baseline=a, connected=b, delta=b-a)
        result['mean_L2'] = dict(
            baseline=base['prefix']['mean_L2'], connected=new['prefix']['mean_L2'],
            delta=new['prefix']['mean_L2']-base['prefix']['mean_L2'])
        result['interpretation'] = (
            'Inference-only Memory output coupling, not native T1 planning or a trained planner. '
            'Negative L2/collision deltas are better; no tuning on these results.')
        if position_reference:
            position = json.loads((Path(position_reference) / 'metrics.json').read_text())
            for key in ('samples', 'metric_source_sha256', 'metric_entry_sha256', 'gt_sha256'):
                if new[key] != position[key]:
                    raise ValueError(f'Position-only reference protocol mismatch: {key}')
            result['versus_position_connection'] = {
                h: {metric: dict(position=position['prefix'][h][metric],
                                 full=new['prefix'][h][metric],
                                 delta=new['prefix'][h][metric]-position['prefix'][h][metric])
                    for metric in ('L2', 'box_collision_percent')}
                for h in ('1s', '2s', '3s')}
            result['versus_position_connection']['mean_L2_delta'] = (
                new['prefix']['mean_L2'] - position['prefix']['mean_L2'])
    audit.write_json(run / 'paired_report.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--position-reference')
    args = parser.parse_args()
    report(args.run, args.reference, args.allow_partial, args.position_reference)
