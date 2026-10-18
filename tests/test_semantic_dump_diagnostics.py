"""Pure CPU/numpy tests for the observational diagnostic, not training tests."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    'semantic_dump_diag', Path(__file__).parents[1] / 'tools/diagnose_future_memory_semantics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_confusion_and_miss_vs_confusion():
    gt = np.array([4, 4, 7, 7, 17, 17])
    pred = np.array([4, 17, 4, 7, 4, 17])
    out = diag.summarize(diag.confusion(gt, pred))
    assert out['iou'] == 60
    assert out['precision'] == 75
    assert out['recall'] == 75
    assert out['classes']['car']['missed_as_free'] == 1
    assert out['classes']['pedestrian']['confused_as_other_occupied'] == 1
    assert out['classes']['car']['predicted_on_free'] == 1


def test_disjoint_transition_counts():
    gt = np.array([4, 4, 4, 4, 4, 4, 17, 17])
    before = np.array([17, 17, 4, 7, 4, 7, 17, 4])
    after = np.array([4, 7, 17, 17, 7, 4, 4, 17])
    assert all(v == 1 for v in diag.transitions(gt, before, after).values())


def test_binary_iou_can_rise_while_semantic_miou_falls():
    gt = np.array([4]*90 + [7]*10 + [17]*100)
    before = np.array([4]*70 + [17]*20 + [7]*10 + [4]*30 + [17]*70)
    after = np.array([4]*85 + [17]*5 + [4]*10 + [4]*10 + [17]*90)
    a = diag.summarize(diag.confusion(gt, before))
    b = diag.summarize(diag.confusion(gt, after))
    assert b['iou'] > a['iou']
    assert b['miou'] < a['miou']


def test_invalid_prediction_rejected():
    import pytest
    with pytest.raises(ValueError, match='Prediction label'):
        diag.confusion(np.array([4]), np.array([18]))
