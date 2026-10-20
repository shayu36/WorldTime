"""CPU pure-function/forward tests for the opt-in T1 supervision experiment.

No optimization, data iteration or gradient evaluation is performed here.
"""
import ast
import copy
from types import SimpleNamespace

import pytest
import torch
from mmcv import Config

from test_future_memory_adapter_v2 import QM, ROOT, _load_sparseworld_method, _adapter


BASE_CONFIG = ROOT / ('configs/sparseworld/nuscenes-temporal/'
                      'sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center.py')
T1_CONFIG = BASE_CONFIG.with_name(BASE_CONFIG.stem + '-t1.py')


def _reduction_inputs():
    values = torch.tensor([1., 1., 1., 5., 3., 3.]).view(1, 1, 6)
    positive = torch.tensor([True] * 4 + [False] * 2).view_as(values)
    valid = torch.ones_like(positive)
    # Invalid negative labels must never be used for class means.
    labels = torch.tensor([0, 0, 0, 1, 99, -1]).view_as(values).long()
    exact = torch.tensor([True, False, True, False, False, False]).view_as(positive)
    return values, positive, valid, labels, exact


def _reduce(beta=.25, **kwargs):
    values, positive, valid, labels, exact = _reduction_inputs()
    return QM.class_balanced_occupancy_bce(
        values, positive, valid, labels, 3, beta,
        voxel_positive=exact, return_diagnostics=True, **kwargs)


def test_positive_mixture_has_expected_value_not_implicit_extra_class_weight():
    value, stats = _reduce()
    # Point mean=2, class mean=(1+5)/2=3, mixture=2.25;
    # original negative mean=3, equal group weights -> 2.625.
    assert value.item() == 2.625
    assert stats['occ_positive_bce_point_mean'].item() == 2.
    assert stats['occ_positive_bce_class_mean'].item() == 3.
    assert stats['occ_positive_bce_mixed_mean'].item() == 2.25
    assert stats['occ_negative_bce_mean'].item() == 3.
    # The absent third class must not dilute the class mean.
    assert stats['occ_positive_present_classes'].item() == 2.


# torch 2.0 CPU does not implement clamp_min for Half in the legacy reducer.
# Test exact legacy equality in its supported CPU dtypes; the new FP16->FP32
# reduction is separately covered by the 80,000-point overflow case below.
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('diagnostics', [False, True])
def test_beta_zero_is_bitwise_legacy(dtype, diagnostics):
    values, positive, valid, labels, exact = _reduction_inputs()
    values = values.to(dtype) / 7.
    expected = QM._balanced_binary_group_mean(values, positive, valid, 2., 3.)
    got, _ = QM.class_balanced_occupancy_bce(
        values, positive, valid, labels, 3, 0., 2., 3.,
        voxel_positive=exact, return_diagnostics=diagnostics)
    assert got.dtype == expected.dtype
    assert torch.equal(got, expected)


def test_class_mean_not_changed_by_duplicates_within_one_class():
    values = torch.tensor([1., 5., 3., 3.])
    positive = torch.tensor([True, True, False, False])
    valid = torch.ones_like(positive)
    labels = torch.tensor([0, 1, 99, 99])
    got, _ = QM.class_balanced_occupancy_bce(values, positive, valid, labels, 3, 1.)
    repeated, _ = _reduce(beta=1.)
    assert torch.equal(got, repeated)


@pytest.mark.parametrize('positive_weight,negative_weight', [(1., 1.), (2., 3.), (0., 1.), (1., 0.)])
def test_group_weights_and_negative_reduction_unchanged(positive_weight, negative_weight):
    loss, stats = _reduce(positive_weight=positive_weight, negative_weight=negative_weight)
    expected = (positive_weight * 2.25 + negative_weight * 3.) / (positive_weight + negative_weight)
    assert loss.item() == pytest.approx(expected)
    assert stats['occ_negative_bce_mean'].item() == 3.


def test_diagnostics_exact_and_proximity_are_exclusive_with_fixed_keys():
    _, stats = _reduce()
    assert stats['occ_positive_count'].item() == 4
    assert stats['occ_negative_count'].item() == 2
    assert stats['occ_positive_voxel_count'].item() == 2
    assert stats['occ_positive_proximity_only_count'].item() == 2
    for c, count in enumerate((3, 1, 0)):
        assert stats[f'occ_positive_count_c{c}'].item() == count
        assert torch.equal(stats[f'occ_positive_count_c{c}'],
                           stats[f'occ_positive_voxel_count_c{c}'] +
                           stats[f'occ_positive_proximity_only_count_c{c}'])
    assert all('loss' not in k and v.ndim == 0 and not v.requires_grad
               for k, v in stats.items())
    values, positive, valid, labels, exact = _reduction_inputs()
    _, empty_stats = QM.class_balanced_occupancy_bce(
        values, positive, torch.zeros_like(valid), labels, 3, .25,
        voxel_positive=exact, return_diagnostics=True)
    assert stats.keys() == empty_stats.keys()
    assert empty_stats['occ_positive_count'].item() == 0
    assert empty_stats['occ_positive_present_classes'].item() == 0


def test_real_mmdet_parser_does_not_add_diagnostics_to_objective():
    from mmdet.models.detectors.base import BaseDetector
    values, positive, valid, labels, exact = _reduction_inputs()
    # Build a graph only; no derivatives or optimizer updates are evaluated.
    values.requires_grad_(True)
    occ, stats = QM.class_balanced_occupancy_bce(
        values, positive, valid, labels, 3, .25,
        voxel_positive=exact, return_diagnostics=True)
    assert occ.requires_grad and all(not v.requires_grad for v in stats.values())
    loss_only = {'mem_3s.loss_occ': occ * .25}
    with_stats = dict(loss_only, **{f'mem_3s.{k}': v for k, v in stats.items()})
    expected, _ = BaseDetector._parse_losses(None, loss_only)
    actual, logs = BaseDetector._parse_losses(None, with_stats)
    assert torch.equal(actual, expected)
    assert logs['mem_3s.occ_positive_count'] == 4.


@pytest.mark.parametrize('case', ['no_positive', 'no_negative', 'no_valid', 'empty'])
def test_empty_groups_are_safe(case):
    values, positive, valid, labels, exact = _reduction_inputs()
    if case == 'no_positive':
        positive.zero_()
        expected = values.mean()
    elif case == 'no_negative':
        valid &= positive
        expected = torch.tensor(2.25)
    elif case == 'no_valid':
        valid.zero_()
        expected = torch.tensor(0.)
    else:
        values, positive, valid, labels, exact = (t[..., :0] for t in (values, positive, valid, labels, exact))
        expected = torch.tensor(0.)
    got, stats = QM.class_balanced_occupancy_bce(
        values, positive, valid, labels, 3, .25,
        voxel_positive=exact, return_diagnostics=True)
    assert torch.isfinite(got) and torch.allclose(got, expected)
    assert all(torch.isfinite(v) for v in stats.values())


def test_fp16_accumulation_safe_beyond_65504_points():
    values = torch.full((2, 1, 40000), 10., dtype=torch.float16)
    positive = torch.ones_like(values, dtype=torch.bool)
    labels = torch.zeros_like(values, dtype=torch.long)
    got, stats = QM.class_balanced_occupancy_bce(
        values, positive, positive, labels, 3, .25,
        voxel_positive=positive, return_diagnostics=True)
    assert got.dtype == torch.float32 and got.item() == 10.
    assert stats['occ_positive_count'].item() == 80000


@pytest.mark.parametrize('beta', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_beta_fails_clearly(beta):
    with pytest.raises(ValueError, match=r'finite.*\[0, 1\]'):
        _reduce(beta)
    with pytest.raises(ValueError, match=r'finite.*\[0, 1\]'):
        QM.validate_occupancy_class_balance_beta({'3s': beta})


def test_invalid_positive_label_raises_but_invalid_negative_labels_are_ignored():
    values, positive, valid, labels, exact = _reduction_inputs()
    labels[..., 0] = 3
    with pytest.raises(ValueError, match='Positive occupancy GT labels'):
        QM.class_balanced_occupancy_bce(values, positive, valid, labels, 3, .25)


def test_horizon_config_validation_and_no_inplace_mutation():
    original = {'1s': 0., '2s': 0., '3s': .25}
    before = original.copy()
    converted = QM.validate_occupancy_class_balance_beta(original)
    assert converted == before and converted is not original and original == before
    with pytest.raises(ValueError, match='mapping'):
        QM.validate_occupancy_class_balance_beta([0., 0., .25])
    with pytest.raises(ValueError, match='Unknown.*horizon'):
        QM.validate_occupancy_class_balance_beta({'0s': .25})


def _forward_inputs(batch=1):
    semantic = torch.zeros(batch, 1, 5, 3)
    occupancy = torch.tensor([2., 2., 2., -2., -1.]).view(1, 1, 5, 1).expand(batch, -1, -1, -1).clone()
    metric = torch.tensor([[.5, .5, .5]] * 3 + [[1.5, .5, .5], [2.75, 1.5, 1.5]])
    points = (metric / torch.tensor([3., 2., 2.])).view(1, 1, 5, 3).expand(batch, -1, -1, -1).clone()
    gt = torch.full((batch, 3, 2, 2), 3, dtype=torch.long)
    gt[:, 0, 0, 0] = 0
    gt[:, 1, 0, 0] = 1
    cfg = dict(pc_range=(0., 0., 0., 3., 2., 2.), voxel_size=(1., 1., 1.),
               empty_label=3, class_weights=[1., 2., 3.], score_thresholds=[.35, .25, .3],
               positive_radius=.1, soft_voxel_center_offset=.5,
               soft_voxel_renormalize_valid_corners=True,
               baseline_occupancy_logits=occupancy.clone())
    return semantic, occupancy, points, gt, cfg


@pytest.mark.parametrize('batch', [1, 2])
def test_loss_forward_changes_only_occupancy_and_beta_zero_is_exact(batch):
    semantic, occupancy, points, gt, cfg = _forward_inputs(batch)
    original_cfg = copy.deepcopy(cfg)
    with torch.no_grad():
        original = QM.future_memory_v2_point_losses(semantic, occupancy, points, None, gt, **cfg)
        disabled = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, None, gt, **cfg,
            occupancy_positive_class_balance_beta=0.)
        t1 = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, None, gt, **cfg,
            occupancy_positive_class_balance_beta=.25, occupancy_class_balance_diagnostics=True)
    assert original.keys() == disabled.keys()
    for key in original:
        assert torch.equal(original[key], disabled[key]), key
        if key != 'loss_occ':
            assert torch.equal(original[key], t1[key]), key
    assert t1['loss_occ'] > original['loss_occ']
    assert t1['occupancy_diagnostics']['occ_positive_count'].item() == 4 * batch
    assert cfg['class_weights'] == original_cfg['class_weights']
    assert torch.equal(cfg['baseline_occupancy_logits'], original_cfg['baseline_occupancy_logits'])


@pytest.mark.parametrize('case', ['empty_gt', 'outside', 'masked', 'empty_prediction', 'all_positive'])
def test_full_t1_loss_forward_edge_cases(case):
    semantic, occupancy, points, gt, cfg = _forward_inputs()
    mask = torch.ones(points.shape[:-1], dtype=torch.bool)
    if case == 'empty_gt':
        gt.fill_(3)
    elif case == 'outside':
        points.fill_(-1.)
    elif case == 'masked':
        mask.zero_()
    elif case == 'empty_prediction':
        semantic, occupancy, points, mask = [x[:, :0] for x in (semantic, occupancy, points, mask)]
        cfg['baseline_occupancy_logits'] = occupancy.clone()
    else:
        points[..., -1, :] = points[..., 0, :]
    with torch.no_grad():
        losses = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, mask, gt, **cfg,
            occupancy_positive_class_balance_beta=.25, occupancy_class_balance_diagnostics=True)
    assert all(torch.isfinite(v) for k, v in losses.items() if k.startswith('loss_'))
    stats = losses['occupancy_diagnostics']
    assert all(torch.isfinite(v) for v in stats.values())
    if case != 'all_positive':
        assert stats['occ_positive_count'].item() == 0
    else:
        assert stats['occ_negative_count'].item() == 0


def test_t1_uses_masks_and_gt_classes_not_semantic_weights_or_prediction():
    semantic, occupancy, points, gt, cfg = _forward_inputs()
    mask = torch.tensor([[[True, True, False, True, True]]])
    with torch.no_grad():
        first = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, mask, gt, **cfg,
            occupancy_positive_class_balance_beta=.25, occupancy_class_balance_diagnostics=True)
        occupancy[..., 2, :] = -100.
        semantic[..., 2] = 100.  # Predict class 2 everywhere, but GT positives are 0/1.
        cfg['class_weights'] = [100., 1., 7.]
        second = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, mask, gt, **cfg,
            occupancy_positive_class_balance_beta=.25, occupancy_class_balance_diagnostics=True)
    assert torch.equal(first['loss_occ'], second['loss_occ'])
    assert second['occupancy_diagnostics']['occ_positive_count_c2'].item() == 0
    assert second['occupancy_diagnostics']['occ_positive_count_c0'].item() == 2


def test_actual_3s_method_routes_beta_and_logging_only_to_3s():
    method = _load_sparseworld_method('_future_memory_v2_losses', {
        'future_memory_v2_point_losses': QM.future_memory_v2_point_losses})
    semantic, occupancy, points, gt, cfg = _forward_inputs()
    model = SimpleNamespace(
        pc_range=cfg['pc_range'], empty_idx=3,
        pts_bbox_head=SimpleNamespace(voxel_size=cfg['voxel_size'],
                                     train_cfg={'cls_weights': cfg['class_weights']},
                                     test_cfg={'score_thr': cfg['score_thresholds']}),
        future_memory_v2_loss_cfg={'positive_radius': .1, 'soft_voxel_center_offset': .5,
                                  'soft_voxel_renormalize_valid_corners': True})
    original = method(model, semantic, occupancy, points, None, gt)
    model.future_memory_v2_loss_cfg.update(
        occupancy_positive_class_balance_beta={'1s': 0., '2s': 0., '3s': .25},
        occupancy_class_balance_diagnostics=True)
    with torch.no_grad():
        for h in ('1s', '2s', '3s'):
            result = method(model, semantic, occupancy, points, None, gt, horizon_name=h)
            for key in original:
                if h != '3s' or key != 'loss_occ':
                    assert torch.equal(result[key], original[key]), (h, key)
            assert ('occupancy_diagnostics' in result) == (h == '3s')
    with pytest.raises(ValueError, match='explicit 1s/2s/3s'):
        method(model, semantic, occupancy, points, None, gt)
    with pytest.raises(ValueError, match='explicit 1s/2s/3s'):
        method(model, semantic, occupancy, points, None, gt, horizon_name='0s')


def test_forward_train_passes_the_real_horizon_name_and_detached_stats():
    tree = ast.parse((ROOT / 'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseWorld4DTraj')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_train')
    call = next(n for n in ast.walk(forward) if isinstance(n, ast.Call) and
                isinstance(n.func, ast.Attribute) and n.func.attr == '_future_memory_v2_losses')
    assert any(k.arg == 'horizon_name' and isinstance(k.value, ast.Name) and k.value.id == 'horizon_name'
               for k in call.keywords)
    source = ast.unparse(forward)
    assert "v2.get('occupancy_diagnostics', {})" in source
    assert "((1, '1s'), (3, '2s'), (5, '3s'))" in source


def test_config_matches_t0_except_opt_in_and_run_metadata():
    base, t1 = Config.fromfile(str(BASE_CONFIG)), Config.fromfile(str(T1_CONFIG))
    stripped = copy.deepcopy(t1.model)
    options = stripped.query_memory_cfg.future_memory_adapter_v2
    assert options.pop('occupancy_positive_class_balance_beta') == {'1s': 0., '2s': 0., '3s': .25}
    assert options.pop('occupancy_class_balance_diagnostics') is True
    assert stripped == base.model
    for k in ('optimizer', 'optimizer_config', 'lr_config', 'data', 'fp16', 'workflow'):
        assert t1.get(k) == base.get(k), k
    assert t1.runner.max_epochs == 3
    assert t1.resume_from.endswith('-center-from-ab1/epoch_2.pth')
    assert t1.load_from is None and t1.auto_resume is False
    assert t1.work_dir.endswith('-center-t1-from-center2')


def test_loss_options_add_no_adapter_state_or_rng_changes():
    torch.manual_seed(17)
    old = _adapter(gate_mode='task_split', horizon_gate_bias=True)
    torch.manual_seed(17)
    t1 = _adapter(gate_mode='task_split', horizon_gate_bias=True,
                  occupancy_positive_class_balance_beta={'3s': .25},
                  occupancy_class_balance_diagnostics=True)
    assert old.state_dict().keys() == t1.state_dict().keys()
    assert all(torch.equal(t, t1.state_dict()[k]) for k, t in old.state_dict().items())
