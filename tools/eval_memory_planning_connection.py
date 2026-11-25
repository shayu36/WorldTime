"""Export T1 trajectories with an explicit, untrained Memory planning connection.

Uses the exact audit_planning_pair exporter, data pipeline and metric protocol.
Only model construction installs the opt-in planning wrapper. Score the output
with the unchanged audit_planning_pair.py metrics command.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
import sys

import audit_planning_pair as audit
from memory_planning_connection import (VARIANT, TARGET_STEPS,
                                        install_position_planning_connection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--connection', choices=('position', 'full'), default='position')
    parser.add_argument('--verify-branches', type=int, default=0,
                        help='First N samples: audit each full-output branch using cached snapshots')
    args = parser.parse_args()
    args.baseline = False
    source = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    import numpy as np
    import torch
    import mmdet3d.models as models

    build_native = models.build_model
    native_trajectories = []
    statistics = []
    log_streams = []
    isolated_max = {}
    if args.connection == 'full':
        from full_memory_planning_connection import (
            VARIANT as selected_variant, PROBABILITY_FLOOR,
            install_full_planning_connection as installer)
        connection_file = 'full_memory_planning_connection.py'
        connection_details = dict(
            connected_branch='learned semantic, occupancy and position outputs',
            disconnected_branches=[],
            raw_point_context='used upstream by all residual heads, not added again',
            semantic_projection='checkpoint semantic_query_proj.weight; bias cancels',
            semantic_projection_training='trained for Memory retrieval, not planner fusion',
            occupancy_prior='log(mean_R(sigmoid(O_memory)))-log(mean_R(sigmoid(O_base)))',
            probability_floor=PROBABILITY_FLOOR,
            formula='dXY_native + R(F_base+W_sem*dP_sem,P_memory,b_occ) - R(F_base,P_base,0)')
    else:
        if args.verify_branches:
            raise ValueError('--verify-branches requires --connection full')
        selected_variant, installer = VARIANT, install_position_planning_connection
        connection_file = 'memory_planning_connection.py'
        connection_details = dict(
            connected_branch='learned Memory position residuals only',
            disconnected_branches=['semantic', 'occupancy', 'raw point context'],
            formula='dXY_native + R(F_base,P_memory) - R(F_base,P_base)')

    def build_connected(*build_args, **build_kwargs):
        model = build_native(*build_args, **build_kwargs)
        keys_before = tuple(model.state_dict())
        audit.write_json(output / 'planning_connection.json', dict(
            variant=selected_variant, trained_for_planning=False,
            **connection_details,
            target_seconds=[1., 2., 3.], gain=1.,
            scene_recurrence='unchanged native T1',
            connection_sha256=audit.digest(
                Path(__file__).with_name(connection_file)),
            shared_connection_sha256=audit.digest(
                Path(__file__).with_name('memory_planning_connection.py')),
            runner_sha256=audit.digest(__file__),
            metric_entry='unchanged tools/audit_planning_pair.py metrics'))
        stream = (output / 'planning_connection_progress.jsonl').open('x')
        log_streams.append(stream)

        def observe(outputs, planned, delta):
            native = torch.cat(outputs['pred_trajs_list'], 1).float().cpu().numpy()
            native_trajectories.append(np.cumsum(native, axis=1))
            d = delta.float().cpu().numpy()
            record = dict(index=len(native_trajectories) - 1,
                          max_abs_increment_delta=float(np.abs(d).max()),
                          target_increment_delta=d[:, TARGET_STEPS, :].tolist(),
                          finite=bool(np.isfinite(d).all()))
            if not record['finite']:
                raise ValueError('Nonfinite Memory planning increment')
            if args.connection == 'full':
                record.update(model._full_memory_planning_audit)
                for name, value in record.get('isolated_branch_increment_max', {}).items():
                    isolated_max[name] = max(value, isolated_max.get(name, 0.))
            statistics.append(record['max_abs_increment_delta'])
            stream.write(json.dumps(record) + '\n')
            stream.flush()

        if args.connection == 'full':
            installer(model, observe, verify_branches=args.verify_branches)
        else:
            installer(model, observe)
        if tuple(model.state_dict()) != keys_before:
            raise RuntimeError('The inference connection changed checkpoint keys')
        return model

    models.build_model = build_connected
    try:
        audit.predict(args)
    finally:
        models.build_model = build_native
        for stream in log_streams:
            stream.close()
    manifest = json.loads((output / 'manifest.json').read_text())
    if len(manifest) != len(native_trajectories):
        raise RuntimeError('Native/connected trajectory counts differ')
    native = {row['token']: value for row, value in zip(manifest, native_trajectories)}
    with (output / 'native_trajectories.pkl').open('xb') as stream:
        pickle.dump(native, stream)
    summary = dict(variant=selected_variant, samples=len(manifest),
                   changed_samples=sum(s > 0 for s in statistics),
                   max_abs_increment_delta=max(statistics, default=0.))
    if args.verify_branches:
        summary['isolated_branch_increment_max'] = isolated_max
        if set(isolated_max) != {'semantic', 'occupancy', 'position'} or not all(
                value > 0 for value in isolated_max.values()):
            raise RuntimeError(f'Not every branch changed a trajectory in the probe: {isolated_max}')
    audit.write_json(output / 'connection_complete.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
