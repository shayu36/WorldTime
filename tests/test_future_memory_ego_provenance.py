"""CPU algebra tests only; no model training or parameter updates."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    'ego_provenance', Path(__file__).parents[1] / 'tools/audit_future_memory_ego_provenance.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_scalar_signature_detects_future_vector_and_rejects_distinct_direction():
    reference = np.array([[1., 2., .3], [-2., 1., -.1]])
    values = reference * np.array([[.96], [1.04]])
    good = audit.proportional_signature(values, reference)
    assert good['matching'] == 2
    values[0, 2] += .02
    assert audit.proportional_signature(values, reference)['matching'] == 1


def test_zero_reference_does_not_count_as_evidence():
    stats = audit.proportional_signature(np.zeros((2, 3)), np.zeros((2, 3)))
    assert stats['nondegenerate'] == stats['matching'] == 0
    assert stats['scale_range'] is None


def test_future_signature_fields_and_command_and_input_immutability():
    gt = np.array([[i*i/8., i, i/10.] for i in range(7)])
    record = dict(gt=gt, v=1.01*(gt[3]-gt[1]),
                  a=.97*(gt[6]-gt[1])/2.5, cmd=[0, 0, 1])
    original = gt.copy()
    result = audit.summarize_records([record])
    assert result['complete_future'] == result['cmd_future_rule_matches'] == 1
    assert result['v_future_signature']['matching'] == 1
    assert result['a_future_signature']['matching'] == 1
    assert np.array_equal(gt, original)


def test_incomplete_future_records_reported_separately():
    assert audit.summarize_records([{'gt': np.zeros((3, 3))}]) == dict(
        records=1, complete_future=0)


def test_past_increment_reconstruction_ignores_future_poses():
    poses = np.tile(np.eye(4), (8, 1, 1))
    poses[:, 0, 3] = np.arange(8) ** 2
    original = audit.past_increment_signature(poses, 4)
    assert np.array_equal(original[:, 0], [-7, -12, -15, -16])
    poses[5:, :3, 3] += 123
    assert np.array_equal(original, audit.past_increment_signature(poses, 4))
    poses[4, 0, 3] += 1
    assert not np.array_equal(original, audit.past_increment_signature(poses, 4))
