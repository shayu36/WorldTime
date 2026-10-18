"""CPU forward checks for output-only interventions (no optimization)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from diagnose_future_memory_branches import (
    compose, intervention_inputs, filter_statistics, residual_statistics,
    scan_inputs)


def test_each_intervention_preserves_other_outputs_and_inputs():
    with torch.no_grad():
        base_sem = torch.tensor([[[[1., 2., 3.], [3., 2., 1.]]]])
        base_pos = torch.rand(1, 1, 2, 3)
        sem = base_sem.flip(-1) + .25
        occ = torch.tensor([[[[.5], [1.]]]])
        pos = base_pos + .03
        decoded = compose(sem, occ)
        snapshots = [x.clone() for x in (base_sem, base_pos, sem, occ, pos, decoded)]
        variants = intervention_inputs(base_sem, base_pos, sem, occ, pos, decoded)
        assert variants['restore_position'][0] is decoded
        assert variants['restore_position'][1] is base_pos
        for name in ('restore_occupancy', 'restore_semantic'):
            assert variants[name][1] is pos
        logits = variants['restore_occupancy'][0]
        assert torch.equal(logits.argmax(-1), sem.argmax(-1))
        assert torch.equal(logits.max(-1).values, base_sem.max(-1).values)
        logits = variants['restore_semantic'][0]
        assert torch.equal(logits.argmax(-1), base_sem.argmax(-1))
        assert torch.equal(logits.max(-1, keepdim=True).values, occ)
        assert variants['baseline_snapshot'][0] is base_sem
        assert variants['baseline_snapshot'][1] is base_pos
        for actual, expected in zip((base_sem, base_pos, sem, occ, pos, decoded), snapshots):
            assert torch.equal(actual, expected)


def test_reject_inconsistent_captured_decomposition():
    import pytest
    with torch.no_grad(), pytest.raises(AssertionError, match='decomposition'):
        sem = torch.ones(1, 2, 3, 4)
        occ = torch.zeros(1, 2, 3, 1)
        pos = torch.zeros(1, 2, 3, 3)
        intervention_inputs(sem, pos, sem, occ, pos, sem)


def test_filter_statistics_distinguish_raw_and_class_distance_thresholds():
    import numpy as np
    from types import SimpleNamespace
    head = SimpleNamespace(pc_range=torch.tensor([0., 0., 0., 1.6, 1.6, 1.6]),
                           voxel_size=torch.tensor([.4, .4, .4]), num_classes=17,
                           test_cfg=dict(score_thr=[.35]*15+[.25, .30]))
    with torch.no_grad():
        points = torch.tensor([[[[.25, .25, .25], [.75, .75, .75]]]])
        logits = torch.full((1, 1, 2, 17), -10.)
        logits[..., 15] = torch.logit(torch.tensor(.4))
        result = filter_statistics(head, logits, points, np.full((4, 4, 4), 7))
        assert result['scalar'].tolist() == [2, 2, 0, 0, 0]
        assert result['by_point_voxel_gt'][7].tolist() == [2, 2, 0, 0]


def test_residual_statistics_report_metric_displacement_and_occupancy():
    with torch.no_grad():
        sem = torch.zeros(1, 1, 2, 17)
        sem[..., 7] = 1.
        pos = torch.zeros(1, 1, 2, 3)
        result = residual_statistics(sem, pos, sem, sem.max(-1, keepdim=True).values-.5,
                                     pos + .1, torch.tensor([0., 0., 0., 2., 2., 2.]))
        assert result[7, 0] == 2
        assert result[7, 1] == -1.
        assert abs(result[7, 4] - .4) < 1e-6
        assert result[7, 7] == 0


def test_bootstrap_score_axes_match_confusion_summarizer():
    import numpy as np
    from diagnose_future_memory_semantics import confusion, summarize
    from summarize_future_memory_branches import scores
    hist = confusion(np.array([4, 4, 7, 7, 17, 17]), np.array([4, 17, 4, 7, 4, 17]))
    expected = summarize(hist)
    batched = np.broadcast_to(hist, (2, 3, 4, 18, 18))
    got = scores(batched)
    assert got['iou'].shape == (2, 3, 4)
    assert np.allclose(got['iou'], expected['iou'])
    assert np.allclose(got['miou'], expected['miou'])


def test_scan_protects_negative_occupancy_for_either_semantic_source():
    with torch.no_grad():
        base = torch.full((1, 1, 3, 17), -12.)
        sem = base.clone()
        base[..., 0, 7] = 12.   # Baseline says pedestrian
        sem[..., 0, 4] = 12.    # corrected semantics disagree
        base[..., 1, 4] = 12.   # Baseline says car
        sem[..., 1, 15] = 12.   # corrected semantics says manmade
        base[..., 2, 4] = 12.
        sem[..., 2, 4] = 12.
        base_occ = base.max(-1, keepdim=True).values
        occ = base_occ - 2.
        pos = torch.zeros(1, 1, 3, 3)
        base_pos = torch.ones_like(pos)
        decoded = compose(sem, occ)
        captured = [v.clone() for v in (base, base_pos, sem, occ, pos, decoded)]
        got = scan_inputs(base, base_pos, sem, occ, pos, decoded, 2)
        assert got['position_12_base'][1] is base_pos
        protected = got['protected_occ_alpha_025'][0].max(-1, keepdim=True).values
        assert torch.allclose(protected[..., :2, :], base_occ[..., :2, :] - .5, atol=1e-5)
        assert torch.allclose(protected[..., 2:, :], occ[..., 2:, :], atol=1e-5)
        assert torch.equal(got['protected_occ_alpha_025'][0].argmax(-1), sem.argmax(-1))
        got_1s = scan_inputs(base, base_pos, sem, occ, pos, decoded, 1)
        assert got_1s['protected_occ_alpha_025'][0] is decoded
        got_3s = scan_inputs(base, base_pos, sem, occ, pos, decoded, 3)
        assert got_3s['position_12_base'][1] is pos
        for actual, snapshot in zip((base, base_pos, sem, occ, pos, decoded), captured):
            assert torch.equal(actual, snapshot)


def test_scan_scene_partition_is_deterministic_and_summary_axes():
    import numpy as np
    from summarize_future_memory_scan import scene_partition, evaluate_partition
    assert scene_partition('scene-1') == scene_partition('scene-1')
    names = ['ab', 'protected_occ_alpha_025']
    hist = np.zeros((2, 2, 4, 18, 18), dtype=np.int64)
    hist[..., 7, 7] = 2
    hist[..., 7, 17] = 1
    result = evaluate_partition(hist, names)
    assert result['ab']['iou']['future_delta'] == 0.
    assert result[names[1]]['miou']['targeted_2s_3s_delta'] == 0.
