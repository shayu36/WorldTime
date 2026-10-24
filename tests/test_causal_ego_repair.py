"""Pure functions and CPU forwards; no gradient/optimizer/training loop."""
import copy
import importlib.util
import inspect
from pathlib import Path
import textwrap
import types

import numpy as np
import pytest
import torch

from test_future_memory_adapter_v2 import _adapter, _memory
from test_query_memory_integration import make_cache, loader_mod

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('causal_ego_test', ROOT / 'mmdet3d/datasets/causal_ego.py')
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def pose(i, scene='scene', t=None):
    return dict(token=str(i), scene_token=scene, timestamp=int(i*500000 if t is None else t),
                ego2global_rotation=[1, 0, 0, 0], ego2global_translation=[i*i*.25, 0., 0.],
                lidar2ego_rotation=[1, 0, 0, 0], lidar2ego_translation=[0., 0., 0.])


def test_packet_availability_uses_last_actual_camera_exposure():
    current, previous = pose(4), pose(3)
    current['cams'] = {'front': {'timestamp': current['timestamp'] + 481},
                       'back': {'timestamp': current['timestamp'] - 102}}
    previous['cams'] = {'front': {'timestamp': previous['timestamp'] + 200}}
    assert ce.observation_available_timestamp_us(current, [previous]) == 2000481
    wrong_scene = copy.deepcopy(previous)
    wrong_scene['scene_token'] = 'other'
    with pytest.raises(ValueError, match='cross-scene'):
        ce.observation_available_timestamp_us(current, [wrong_scene])
    future = copy.deepcopy(previous)
    future['timestamp'] = current['timestamp'] + 1
    with pytest.raises(ValueError, match='Future'):
        ce.observation_available_timestamp_us(current, [future])


def test_backward_physics_and_schema():
    rows = [pose(i) for i in range(7)]
    rec = ce.build_record(rows[4], rows)
    assert np.allclose(rec['features']['v'], [3.5, 0, 0])
    assert np.allclose(rec['features']['a'], [2, 0, 0])
    assert rec['features']['cmd'] == [0., 0., 0.]
    assert ce.feature_vector(rec).shape == (21,)
    assert rec['provenance']['source_tokens'] == ['0', '1', '2', '3', '4']
    assert rec['provenance']['max_input_timestamp_us'] == 2000000


def test_future_pose_delete_corrupt_and_labels_are_irrelevant():
    rows = [pose(i) for i in range(7)]
    original = ce.build_record(rows[4], rows)
    rows[5]['ego2global_translation'] = object()
    rows[6]['ego2global_rotation'] = object()
    rows[4]['gt_trajectory'] = object()
    rows[4]['cmd'] = object()
    assert ce.build_record(rows[4], rows) == original
    assert ce.build_record(rows[4], rows[:5]) == original
    rows[3]['ego2global_translation'][0] += .2
    assert ce.build_record(rows[4], rows) != original


def test_no_future_or_other_scene_fallback_and_short_history():
    current = pose(0)
    rec = ce.build_record(current, [pose(2), pose(-1, 'other')])
    assert rec['features']['v'] == rec['features']['a'] == [0., 0., 0.]
    assert np.array_equal(rec['features']['x'], np.zeros((4, 3)))
    assert rec['provenance']['velocity_source'] == 'missing_zero'


def test_nonuniform_time_and_current_ego_rotation():
    rows = [pose(i, t=t) for i, t in enumerate((0, 400000, 1100000))]
    for r, t in zip(rows, (0, .4, 1.1)):
        r['ego2global_translation'] = [t*t, 0, 0]
    rows[-1]['ego2global_rotation'] = [2**-.5, 0, 0, 2**-.5]
    rec = ce.build_record(rows[-1], rows)
    assert np.allclose(rec['features']['v'], [0, -1.5, 0], atol=1e-12)
    assert np.allclose(rec['features']['a'], [0, -2, 0], atol=1e-12)


@pytest.mark.parametrize('bad', ['future', 'extra', 'cmd', 'identity'])
def test_record_rejects_bad_provenance_or_fields(bad):
    row = pose(4)
    rec = ce.build_record(row, [pose(i) for i in range(4)])
    if bad == 'future': rec['provenance']['field_max_timestamp_us']['v'] += 1
    if bad == 'extra': rec['features']['gt'] = [1, 2, 3]
    if bad == 'cmd': rec['features']['cmd'] = [0, 1, 0]
    if bad == 'identity': rec['sample_idx'] = 'wrong'
    with pytest.raises(ValueError): ce.validate_record(rec, row)


@pytest.mark.parametrize('age', [-.5, 0., float('nan'), float('inf')])
@pytest.mark.parametrize('horizon', [0, 1, 2])
def test_v2_rejects_invalid_base_age_even_if_effective_age_passes(age, horizon):
    torch.manual_seed(10)
    adapter = _adapter().eval()
    memory = _memory()
    memory['memory_age'].fill_(age)
    with torch.no_grad():
        out = adapter(torch.randn(1, 2, 8), torch.zeros(1, 2, 2, 3),
                      torch.randn(1, 2, 2, 3), memory, horizon)
    for key in ('context', 'gate', 'delta_s', 'delta_o', 'delta_p'):
        assert torch.equal(out[key], torch.zeros_like(out[key])), key
    assert not out['diagnostics']['has_candidate'].any()


def test_valid_input_guard_is_bitwise_equal_to_pre_guard_full_v2_forward():
    torch.manual_seed(42)
    new = _adapter().eval()
    old = copy.deepcopy(new)
    source = textwrap.dedent(inspect.getsource(type(new)._reshape_memory))
    source = source.replace(' & (age > 0)', '')
    namespace = dict(torch=torch)
    exec(source, namespace)
    old._reshape_memory = types.MethodType(namespace['_reshape_memory'], old)
    args = (torch.randn(2, 3, 8), torch.randn(2, 3, 2, 3),
            torch.randn(2, 3, 2, 3), _memory(batch=2))
    with torch.no_grad():
        for horizon in range(3):
            before, after = old(*args, horizon), new(*args, horizon)
            for key in ('context', 'gate', 'delta_s', 'delta_o', 'delta_p'):
                assert torch.equal(before[key], after[key]), key


@pytest.mark.parametrize('bad', ['legacy', 'future', 'hash', 'identity', 'valid'])
def test_strict_causal_cache_rejects_old_or_mismatched_sources(tmp_path, bad):
    path = tmp_path / 'cache.pt'
    make_cache(path, 'sample', 'scene', 5, 5.)
    cache = torch.load(path)
    cache['input_provenance'] = dict(protocol=ce.PROTOCOL, ego_input_sha256='ego',
        source_checkpoint_sha256='checkpoint', generator_sha256='code',
        sample_idx='sample', scene_id='scene', source_timestamp_us=5000000,
        source_observation_available_timestamp_us=5000200,
        max_input_timestamp_us=5000200)
    if bad == 'legacy': del cache['input_provenance']
    if bad == 'future': cache['input_provenance']['max_input_timestamp_us'] += 1
    if bad == 'hash': cache['input_provenance']['ego_input_sha256'] = 'wrong'
    if bad == 'identity': cache['input_provenance']['sample_idx'] = 'wrong'
    loader = loader_mod.LoadQueryMemoryFromFiles(require_causal_inputs=True,
        expected_ego_input_sha256='ego', expected_source_checkpoint_sha256='checkpoint')
    kwargs = dict(hist=dict(available_timestamp_us=5000200),
                  current_available_us=6000100)
    if bad == 'valid': loader._validate_cache(cache, path, **kwargs)
    else:
        with pytest.raises(ValueError, match='provenance'):
            loader._validate_cache(cache, path, **kwargs)


def test_causal_cache_rejects_late_or_wrong_availability(tmp_path):
    path = tmp_path / 'sample.pt'
    make_cache(path, 'sample', 'scene', 5, 5.)
    cache = torch.load(path)
    cache['input_provenance'] = dict(protocol=ce.PROTOCOL, ego_input_sha256='ego',
        source_checkpoint_sha256='checkpoint', generator_sha256='code',
        sample_idx='sample', scene_id='scene', source_timestamp_us=5000000,
        source_observation_available_timestamp_us=5000500,
        max_input_timestamp_us=5000500)
    loader = loader_mod.LoadQueryMemoryFromFiles(require_causal_inputs=True,
        expected_ego_input_sha256='ego', expected_source_checkpoint_sha256='checkpoint')
    hist = dict(available_timestamp_us=5000500)
    loader._validate_cache(cache, path, hist=hist, current_available_us=5000600)
    with pytest.raises(ValueError, match='not earlier'):
        loader._validate_cache(cache, path, hist=hist, current_available_us=5000500)
    with pytest.raises(ValueError, match='availability mismatch'):
        loader._validate_cache(cache, path,
                               hist=dict(available_timestamp_us=5000499),
                               current_available_us=5000600)


def test_legacy_cache_is_still_loadable_without_causal_mode(tmp_path):
    path = tmp_path / 'legacy.pt'
    make_cache(path, 'sample', 'scene', 5, 5.)
    loader_mod.LoadQueryMemoryFromFiles()._validate_cache(torch.load(path), path)
