"""Forward-free checks for the combined occupancy/planning exporter."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('joint_eval', Path(__file__).parents[1]/'tools/eval_memory_planning_joint.py')
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)


def test_uint8_export_is_lossless_and_uses_integer_horizons():
    rng = np.random.RandomState(0)
    result = {f'semantic_occ_{i}s': [rng.randint(0, 18, (3, 2, 2)).astype(np.int64)]
              for i in range(7)}
    expected = np.stack([result[f'semantic_occ_{i}s'][0] for i in (0, 2, 4, 6)])[None]
    actual = joint.pack_occupancy(result)
    assert actual.dtype == np.uint8
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize('bad', [-1, 18, .5])
def test_export_rejects_invalid_labels(bad):
    result = {f'semantic_occ_{i}s': [np.full((2, 2, 2), bad)] for i in (0, 2, 4, 6)}
    with pytest.raises(ValueError, match='Invalid native occupancy'):
        joint.pack_occupancy(result)


def make_shards(tmp_path, duplicate=False):
    manifest = [dict(index=i, token=str(i), scene='x') for i in range(5)]
    joint.write_json(tmp_path/'manifest.json', manifest)
    for rank in (0, 1):
        source = dict(world_size=2, checkpoint_sha256='checkpoint', checkpoint_epoch=2,
                      config_sha256='config', source_sha256={'model':'model'}, exporter_sha256='exporter')
        joint.write_json(tmp_path/f'provenance_rank_{rank}.json', source)
        subset = manifest[rank::2]
        if rank == 1 and duplicate:
            subset = subset + manifest[:1]
        part = {r['token']: np.full((1, 6, 2), r['index']) for r in subset}
        path = tmp_path/f'trajectories_rank_{rank}.pkl'
        path.write_bytes(pickle.dumps(part))
        joint.write_json(tmp_path/f'complete_rank_{rank}.json', dict(
            trajectory_sha256=joint.digest(path), samples=len(part)))
    return manifest


def test_merge_preserves_original_order_without_sampler_padding(tmp_path):
    manifest = make_shards(tmp_path)
    merged, _ = joint.merge_shards(tmp_path)
    assert merged == manifest
    with (tmp_path/'trajectories.pkl').open('rb') as f:
        trajectories = pickle.load(f)
    assert list(trajectories) == [str(i) for i in range(5)]
    for i in range(5):
        assert np.all(trajectories[str(i)] == i)


def test_merge_rejects_duplicate_tokens(tmp_path):
    make_shards(tmp_path, duplicate=True)
    with pytest.raises(RuntimeError, match='Duplicate/missing'):
        joint.merge_shards(tmp_path)


def test_merge_rejects_mixed_checkpoints(tmp_path):
    make_shards(tmp_path)
    p = tmp_path/'provenance_rank_1.json'
    source = json.loads(p.read_text())
    source['checkpoint_sha256'] = 'wrong'
    joint.write_json(p, source)
    with pytest.raises(RuntimeError, match='Inconsistent prediction workers'):
        joint.merge_shards(tmp_path)
