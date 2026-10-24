"""Versioned, past-only ego inputs. No GT, future states, or legacy pickle read.

21 network channels: a_xyz (m/s^2), unknown-command (three zeros), v_xyz
(m/s), four past [dx,dy,dyaw] increments cumulatively summed as in the legacy
x representation. a/v are backward estimates in the CURRENT EGO frame.
"""
import hashlib
import pickle

import numpy as np
from pyquaternion import Quaternion

SCHEMA = 1
PROTOCOL = 'past_pose_backward_unknown_command_v1'
FIELDS = ('a', 'cmd', 'v', 'x')
SHAPES = ((3,), (3,), (3,), (4, 3))


def observation_available_timestamp_us(current, adjacent=()):
    """Earliest time the complete image/LiDAR packet can be used.

    The LiDAR timestamp is a frame identifier, not a guarantee that every
    synchronized camera exposure preceded it.  Include each camera actually
    consumed by the model, including temporal image sweeps.  Adjacent frames
    must have an earlier LiDAR frame timestamp and the same scene.
    """
    current_us = int(current['timestamp'])
    scene = str(current['scene_token'])
    stamps = [current_us]
    for frame in (current,) + tuple(adjacent):
        if (str(frame['scene_token']) != scene or
                int(frame['timestamp']) > current_us):
            raise ValueError('Future or cross-scene image frame in causal observation')
        stamps.extend(int(cam['timestamp']) for cam in frame['cams'].values())
    return max(stamps)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = Quaternion(rotation).rotation_matrix
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite causal pose')
    return result


def _ego_pose(info):
    return _transform(info['ego2global_rotation'], info['ego2global_translation'])


def _lidar_pose(info):
    return _ego_pose(info) @ _transform(
        info['lidar2ego_rotation'], info['lidar2ego_translation'])


def build_record(current, candidates):
    """Read only same-scene measurements at/before current; at most 5 poses.

    Filtering happens BEFORE touching poses, labels, or any future fields.
    candidates may include future records; deleting or corrupting their poses
    cannot affect this output.
    """
    timestamp = int(current['timestamp'])
    scene = str(current['scene_token'])
    past = sorted((r for r in candidates
                   if str(r['scene_token']) == scene and int(r['timestamp']) < timestamp),
                  key=lambda r: int(r['timestamp']))[-4:]
    rows = past + [current]
    times = [int(r['timestamp']) for r in rows]
    if len(set(times)) != len(times):
        raise ValueError('Duplicate past timestamps in causal ego history')
    ego = [_ego_pose(r) for r in rows]
    v = np.zeros(3)
    a = np.zeros(3)
    if len(rows) >= 2:
        dt = (times[-1] - times[-2]) / 1e6
        world_v = (ego[-1][:3, 3] - ego[-2][:3, 3]) / dt
        v = ego[-1][:3, :3].T @ world_v
        if len(rows) >= 3:
            dt_prev = (times[-2] - times[-3]) / 1e6
            previous_v = (ego[-2][:3, 3] - ego[-3][:3, 3]) / dt_prev
            # Backward velocities describe interval midpoints, hence this dt.
            a = ego[-1][:3, :3].T @ ((world_v - previous_v) / ((dt + dt_prev) / 2))
    x = np.zeros((4, 3))
    cumulative = np.zeros(3)
    lidar = [_lidar_pose(r) for r in rows]
    for k in range(1, len(rows)):
        relative = np.linalg.inv(lidar[-k-1]) @ lidar[-k]
        cumulative -= [relative[0, 3], relative[1, 3],
                       np.arctan2(relative[1, 0], relative[0, 0])]
        x[k-1] = cumulative
    result = dict(
        sample_idx=str(current['token']), scene_id=scene, timestamp_us=timestamp,
        features=dict(a=a.tolist(), cmd=[0., 0., 0.], v=v.tolist(), x=x.tolist()),
        provenance=dict(
            protocol=PROTOCOL, max_input_timestamp_us=max(times),
            source_tokens=[str(r['token']) for r in rows], source_timestamps_us=times,
            velocity_source='past_pose_backward' if len(rows) >= 2 else 'missing_zero',
            acceleration_source='past_pose_backward' if len(rows) >= 3 else 'missing_zero',
            command_source='unknown_no_navigation',
            past_steps=len(rows)-1,
            field_max_timestamp_us={name: timestamp if name != 'cmd' else None for name in FIELDS}))
    validate_record(result)
    return result


def validate_record(record, info=None):
    if set(record['features']) != set(FIELDS):
        raise ValueError('Causal ego fields must be exactly a/cmd/v/x; no GT or implicit extras')
    ts = int(record['timestamp_us'])
    prov = record['provenance']
    source_times = prov['source_timestamps_us']
    if (prov['protocol'] != PROTOCOL or not source_times
            or len(source_times) != len(prov['source_tokens'])
            or source_times != sorted(set(source_times))
            or max(source_times) != prov['max_input_timestamp_us']
            or max(source_times) != ts):
        raise ValueError('Invalid/noncausal ego source timestamps or protocol')
    for field in FIELDS:
        ft = prov['field_max_timestamp_us'][field]
        if ft is not None and (not np.isfinite(ft) or ft > ts):
            raise ValueError('Future timestamp in causal ego field ' + field)
    for key, shape in zip(FIELDS, SHAPES):
        array = np.asarray(record['features'][key])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Invalid causal ego shape/values: ' + key)
    if (prov['command_source'] != 'unknown_no_navigation'
            or np.any(np.asarray(record['features']['cmd']) != 0)):
        raise ValueError('Unknown-command protocol forbids GT-derived navigation')
    if info is not None and (
            str(info['token']) != record['sample_idx']
            or str(info['scene_token']) != record['scene_id']
            or int(info['timestamp']) != ts):
        raise ValueError('Causal ego sample/scene/time identity mismatch')


def feature_vector(record, info=None):
    validate_record(record, info)
    return np.concatenate([np.asarray(record['features'][k], dtype=np.float32).reshape(-1)
                           for k in FIELDS])


def load_artifact(path):
    # Local trusted generated data, never an arbitrary remote pickle.
    with open(path, 'rb') as stream:
        artifact = pickle.load(stream)
    if artifact.get('schema_version') != SCHEMA or artifact.get('protocol') != PROTOCOL:
        raise ValueError('Not a versioned causal ego artifact; legacy data is forbidden')
    for token, record in artifact['records'].items():
        validate_record(record)
        if token != record['sample_idx']:
            raise ValueError('Causal ego token identity mismatch')
    return artifact, file_sha256(path)
