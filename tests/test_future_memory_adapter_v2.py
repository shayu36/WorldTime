"""Pure-function and forward-only invariants for FutureMemoryAdapterV2.

This file intentionally contains no backward call, optimizer step, data
iteration, or training loop.
"""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'future_memory_adapter_v2_qm', ROOT / 'mmdet3d/models/sparsedetectors/query_memory.py')
QM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QM)


def _memory(C=8, classes=3, points=2, count=3, batch=1):
    reliability = torch.linspace(.9, .5, count).reshape(1, 1, count)
    age = torch.linspace(2.5, 4.5, count).reshape(1, 1, count)
    labels = torch.arange(count).reshape(1, 1, count) % classes
    return dict(
        memory_query_feat=torch.randn(
            batch, 1, count, C),
        memory_points_metric=torch.zeros(
            batch, 1, count, points, 3),
        memory_valid=torch.ones(
            batch, 1, count, dtype=torch.bool),
        memory_reliability=reliability.expand(batch, -1, -1).clone(),
        memory_age=age.expand(batch, -1, -1).clone(),
        memory_semantic_distribution=torch.eye(classes)[
            labels].expand(batch, -1, -1, -1).clone(),
        memory_label=labels.expand(batch, -1, -1).clone())


def _adapter(**kwargs):
    return QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        coarse_topk_geometry=2, coarse_topk_semantic=2, point_topk=2,
        coarse_radius=10., point_radius=10., **kwargs)


def _load_sparseworld_method(name, extra_globals):
    """Compile one real class method without importing optional CUDA ops."""
    source = (
        ROOT / 'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py'
    ).read_text()
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'SparseWorld4DTraj')
    method = copy.deepcopy(next(
        node for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
        node.name == name))
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(extra_globals)
    exec(compile(module, str(ROOT / 'sparseworld_method.py'), 'exec'),
         namespace)
    return namespace[name]


def test_weighted_statistics_use_weight_mass_not_candidate_count():
    adapter = _adapter()
    # Exercise the exact formula independently of retrieval ordering.
    weights = torch.tensor([[[[.75, .25]]]])
    valid = torch.tensor([[[[True, True]]]])
    rel = torch.tensor([[[[.8, .4]]]])
    valid_weights = weights * valid.to(weights.dtype)
    got = (valid_weights * rel).sum(-1) / valid_weights.sum(-1).clamp_min(1e-6)
    assert torch.allclose(got, torch.tensor([[[.7]]]))
    assert not torch.allclose((valid_weights * rel).sum(-1) / 2., got)
    rel_h, _, _ = adapter.aggregate_support_statistics(
        weights.unsqueeze(3), valid.unsqueeze(3), rel.unsqueeze(3),
        rel.unsqueeze(3), rel.unsqueeze(3))
    assert torch.allclose(rel_h, torch.tensor([[[.7]]]))


def test_multihead_statistics_ignore_invalid_heads_without_nan():
    adapter = _adapter()
    w = torch.tensor([[[[[.75, .25], [1., 0.]]]]])
    valid = torch.tensor([[[[[True, True], [False, False]]]]])
    rel = torch.tensor([[[[[.8, .4], [.2, .2]]]]])
    out = adapter.aggregate_support_statistics(w, valid, rel, rel * 4, rel * 2)
    assert torch.allclose(out[0], torch.tensor([[[.7]]]))
    assert torch.isfinite(torch.stack(out)).all()


def test_effective_age_is_horizon_aware_and_cache_is_not_mutated():
    adapter = _adapter(max_effective_age=20.)
    q = torch.randn(1, 1, 8)
    p = torch.zeros(1, 1, 2, 3)
    logits = torch.randn(1, 1, 2, 3)
    memory = _memory()
    base = memory['memory_age'].clone()
    ages = []
    penalties = []
    time_penalties = []
    for horizon in range(3):
        result = adapter(q, p, logits, memory, horizon)
        ages.append(result['diagnostics']['effective_age'])
        penalties.append(result['diagnostics']['support_age'])
        time_penalties.append(result['diagnostics']['time_penalty'])
    assert torch.equal(memory['memory_age'], base)
    assert torch.allclose(ages[0], torch.tensor([[3.5, 4.5, 5.5]]))
    assert torch.allclose(ages[1], torch.tensor([[4.5, 5.5, 6.5]]))
    assert torch.allclose(ages[2], torch.tensor([[5.5, 6.5, 7.5]]))
    assert float(penalties[2].mean()) > float(penalties[1].mean()) > float(penalties[0].mean())
    assert (time_penalties[2] > time_penalties[1]).all()
    assert (time_penalties[1] > time_penalties[0]).all()


def test_point_context_gate_shapes_and_zero_candidate_identity():
    adapter = _adapter()
    q = torch.randn(1, 2, 8)
    p = torch.zeros(1, 2, 2, 3)
    logits = torch.randn(1, 2, 2, 3)
    result = adapter(q, p, logits, _memory(), 0)
    assert result['context'].shape == (1, 2, 2, 8)
    assert result['gate'].shape == (1, 2, 2, 1)
    assert result['diagnostics']['coarse_candidate_count'].shape == (1, 2)
    assert result['diagnostics']['coarse_context'].shape == (1, 2, 8)
    assert result['diagnostics']['eligible_point_count'].shape == (1, 2, 2)
    assert result['diagnostics']['selected_point_count'].shape == (1, 2, 2)
    invalid = _memory()
    invalid['memory_valid'].zero_()
    empty = adapter(q, p, logits, invalid, 0)
    assert torch.equal(empty['context'], torch.zeros_like(empty['context']))
    assert torch.equal(empty['gate'], torch.zeros_like(empty['gate']))
    assert torch.equal(empty['delta_s'], torch.zeros_like(empty['delta_s']))
    assert torch.equal(
        empty['diagnostics']['eligible_point_count'],
        torch.zeros_like(empty['diagnostics']['eligible_point_count']))
    assert torch.equal(
        empty['diagnostics']['selected_point_count'],
        torch.zeros_like(empty['diagnostics']['selected_point_count']))


def test_zero_initialized_adapter_is_strict_baseline_identity():
    adapter = _adapter()
    adapter.eval()
    query = torch.randn(1, 2, 8)
    metric_points = torch.randn(1, 2, 2, 3)
    encoded_points = torch.rand(1, 2, 2, 3)
    baseline = torch.randn(1, 2, 2, 3)
    result = adapter(query, metric_points, baseline, _memory(), 0)
    semantic, occupancy, decoded = QM.decode_semantic_occupancy_logits(
        baseline, result['delta_s'], result['delta_o'], result['gate'])
    output_points = QM.apply_metric_position_residual(
        encoded_points, result['gate'] * result['delta_p'],
        [-40., -40., -1., 40., 40., 5.4],
        result['diagnostics']['has_candidate'])
    assert torch.equal(semantic, baseline)
    assert torch.equal(decoded, baseline)
    assert torch.equal(
        occupancy, baseline.max(-1, keepdim=True).values)
    assert torch.equal(output_points, encoded_points)


def test_no_memory_candidate_composes_to_strict_baseline_identity():
    adapter = _adapter()
    adapter.eval()
    query = torch.randn(1, 1, 8)
    points = torch.rand(1, 1, 2, 3)
    baseline = torch.randn(1, 1, 2, 3)
    result = adapter(query, points, baseline, None, 0)
    _, _, decoded = QM.decode_semantic_occupancy_logits(
        baseline, result['delta_s'], result['delta_o'], result['gate'])
    output_points = QM.apply_metric_position_residual(
        points, result['gate'] * result['delta_p'],
        [-40., -40., -1., 40., 40., 5.4],
        result['diagnostics']['has_candidate'])
    assert torch.equal(decoded, baseline)
    assert torch.equal(output_points, points)


def test_chunked_and_single_chunk_point_attention_agree():
    torch.manual_seed(3)
    kwargs = dict(embed_dims=8, num_classes=3, num_points=2, num_heads=2,
                  coarse_topk_geometry=2, coarse_topk_semantic=2,
                  point_topk=2, coarse_radius=10., point_radius=10.)
    chunked = QM.FutureMemoryAdapterV2(query_chunk_size=1, **kwargs)
    single = QM.FutureMemoryAdapterV2(query_chunk_size=16, **kwargs)
    single.load_state_dict(chunked.state_dict())
    q = torch.randn(1, 4, 8); p = torch.randn(1, 4, 2, 3); l = torch.randn(1, 4, 2, 3)
    memory = _memory(count=3)
    a, b = chunked(q, p, l, memory, 0), single(q, p, l, memory, 0)
    assert torch.allclose(a['context'], b['context'], atol=1e-5)
    assert torch.allclose(a['gate'], b['gate'], atol=1e-5)


def test_single_current_point_change_is_point_local_with_fixed_coarse_set():
    torch.manual_seed(31)
    adapter = QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=3, num_points=4, num_heads=2,
        coarse_topk_geometry=1, coarse_topk_semantic=1, point_topk=2,
        coarse_radius=20., point_radius=20.)
    adapter.eval()
    query = torch.randn(1, 1, 8)
    points = torch.zeros(1, 1, 4, 3)
    logits = torch.randn(1, 1, 4, 3)
    memory = _memory(points=4, count=1)
    first = adapter(query, points, logits, memory, 0)

    changed_points = points.clone()
    changed_points[0, 0, 2, 0] = .25
    position_changed = adapter(
        query, changed_points, logits, memory, 0)
    other = torch.tensor([0, 1, 3])
    assert torch.equal(
        first['context'][0, 0, other],
        position_changed['context'][0, 0, other])

    changed_logits = logits.clone()
    changed_logits[0, 0, 2] += torch.tensor([5., -5., 0.])
    semantic_changed = adapter(
        query, points, changed_logits, memory, 0)
    assert torch.equal(
        first['diagnostics']['point_semantic_current'][0, 0, other],
        semantic_changed['diagnostics']['point_semantic_current'][
            0, 0, other])
    assert torch.equal(
        first['context'][0, 0, other],
        semantic_changed['context'][0, 0, other])
    assert torch.equal(
        first['gate'][0, 0, other],
        semantic_changed['gate'][0, 0, other])


def test_semantic_uncertainty_dropout_and_disagreement_are_explicit():
    adapter = _adapter(semantic_uncertainty_gamma=2., semantic_weight_floor=.05,
                       semantic_dropout_probability=1.)
    q = torch.randn(1, 1, 8); p = torch.zeros(1, 1, 2, 3)
    confident = torch.tensor([[[[8., -4., -4.], [8., -4., -4.]]]])
    uncertain = torch.zeros_like(confident)
    memory = _memory()
    adapter.eval()
    confident_eval = adapter(q, p, confident, memory, 0)
    uncertain_eval = adapter(q, p, uncertain, memory, 0)
    assert confident_eval['diagnostics']['semantic_weight'].mean() > uncertain_eval['diagnostics']['semantic_weight'].mean()
    repeat_eval = adapter(q, p, confident, memory, 0)
    assert not confident_eval['diagnostics']['semantic_dropout_mask'].any()
    assert torch.equal(confident_eval['context'], repeat_eval['context'])
    assert torch.equal(confident_eval['gate'], repeat_eval['gate'])
    adapter.train()
    dropped = adapter(q, p, confident, memory, 0)
    assert dropped['diagnostics']['semantic_dropout_mask'].all()
    disagreement = dropped['diagnostics']['semantic_disagreement']
    assert disagreement.shape == (1, 1, 2)
    expected_current = torch.softmax(confident.float(), -1)
    assert torch.equal(
        dropped['diagnostics']['point_semantic_current'],
        expected_current)


def test_geometry_candidate_path_survives_wrong_semantics():
    adapter = _adapter()
    q = torch.randn(1, 1, 8); p = torch.zeros(1, 1, 2, 3)
    logits = torch.tensor([[[[8., -4., -4.], [8., -4., -4.]]]])
    memory = _memory(count=3)
    memory['memory_points_metric'][0, 0, 2] = 100.
    original = adapter(q, p, logits, memory, 0)
    wrong = dict(memory)
    wrong['memory_semantic_distribution'] = torch.flip(
        memory['memory_semantic_distribution'], dims=(-1,))
    changed = adapter(q, p, logits, wrong, 0)
    assert original['diagnostics']['geometry_candidate_count'].item() == 2
    assert changed['diagnostics']['geometry_candidate_count'].item() == 2
    assert original['diagnostics']['coarse_candidate_count'].item() >= 1


def _isolation_case(kind):
    torch.manual_seed(11)
    adapter = QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        coarse_topk_geometry=1, coarse_topk_semantic=1, point_topk=2,
        coarse_radius=10., point_radius=10., max_effective_age=8.,
        query_chunk_size=2)
    with torch.no_grad():
        adapter.key_proj.weight.zero_()
        adapter.key_proj.bias.zero_()
    adapter.eval()
    query = torch.randn(1, 1, 8)
    points = torch.zeros(1, 1, 2, 3)
    logits = torch.tensor([[[[5., 0., -2.], [4., 0., -2.]]]])
    memory = _memory(count=2)
    memory['memory_query_feat'][0, 0, 0] = torch.tensor(
        [2., -1., .5, 0., 1., -2., .2, .7])
    memory['memory_query_feat'][0, 0, 1] = torch.tensor(
        [-1., .2, -.5, 2., 0., 1., -.7, .1])
    if kind == 'unselected':
        memory['memory_points_metric'][0, 0, 1, :, 0] = 8.
    elif kind == 'invalid':
        memory['memory_valid'][0, 0, 1] = False
    elif kind == 'expired':
        memory['memory_age'][0, 0, 1] = 8.
    elif kind == 'outside':
        memory['memory_points_metric'][0, 0, 1, :, 0] = 20.
    else:
        raise AssertionError(kind)
    first = adapter(query, points, logits, memory, 0)
    changed = {key: value.clone() for key, value in memory.items()}
    changed['memory_query_feat'][0, 0, 1] += torch.tensor(
        [0., .01, 0., 0., 0., 0., 0., 0.])
    second = adapter(query, points, logits, changed, 0)
    return first, second, adapter, query, points, logits, memory


@pytest.mark.parametrize(
    'kind', ['unselected', 'invalid', 'expired', 'outside'])
def test_memory_outside_effective_coarse_candidates_cannot_leak(kind):
    first, second, _, _, _, _, _ = _isolation_case(kind)
    assert torch.equal(first['context'], second['context'])
    assert torch.equal(first['gate'], second['gate'])


def test_selected_memory_changes_point_context():
    first, _, adapter, query, points, logits, memory = _isolation_case(
        'unselected')
    changed = {key: value.clone() for key, value in memory.items()}
    changed['memory_query_feat'][0, 0, 0, 0] += 1.
    second = adapter(query, points, logits, changed, 0)
    assert not torch.equal(first['context'], second['context'])


def test_coarse_union_is_stable_unique_and_weights_are_normalized():
    adapter = QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        coarse_topk_geometry=1, coarse_topk_semantic=1, point_topk=2,
        coarse_radius=10., point_radius=10.)
    adapter.eval()
    result = adapter(
        torch.randn(1, 2, 8), torch.zeros(1, 2, 2, 3),
        torch.zeros(1, 2, 2, 3), _memory(count=1), 0)
    diag = result['diagnostics']
    assert torch.equal(
        diag['coarse_candidate_count'], torch.ones(1, 2, dtype=torch.long))
    assert torch.equal(diag['coarse_valid'][..., 1],
                       torch.zeros_like(diag['coarse_valid'][..., 1]))
    assert torch.allclose(
        diag['coarse_attention_weights'].sum(-1), torch.ones(1, 2))

    invalid = _memory(count=1)
    invalid['memory_valid'].zero_()
    empty = adapter(
        torch.randn(1, 2, 8), torch.zeros(1, 2, 2, 3),
        torch.zeros(1, 2, 2, 3), invalid, 0)
    empty_weights = empty['diagnostics']['coarse_attention_weights']
    assert torch.equal(empty_weights, torch.zeros_like(empty_weights))
    assert torch.isfinite(empty['context']).all()


def test_full_semantic_disagreement_features_and_shapes():
    adapter = _adapter()
    current = torch.tensor(
        [[[[.8, .1, .1], [.1, .8, .1]]]], dtype=torch.float32)
    same_memory = current.clone()
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    difference, cosine, js = adapter.semantic_disagreement_features(
        current, same_memory, valid)
    assert difference.shape == current.shape == (1, 1, 2, 3)
    assert torch.equal(difference, torch.zeros_like(difference))
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6)
    assert torch.allclose(js, torch.zeros_like(js), atol=1e-6)

    different_memory = torch.tensor(
        [[[[.1, .1, .8], [.8, .1, .1]]]], dtype=torch.float32)
    difference2, _, js2 = adapter.semantic_disagreement_features(
        current, different_memory, valid)
    assert torch.equal(difference2, current - different_memory)
    assert (js2 > js).all()

    forward = adapter(
        torch.randn(1, 1, 8), torch.zeros(1, 1, 2, 3),
        torch.log(current), _memory(), 0)
    diag = forward['diagnostics']
    assert diag['point_semantic_current'].shape == (1, 1, 2, 3)
    assert diag['point_semantic_memory'].shape == (1, 1, 2, 3)
    assert torch.allclose(
        diag['semantic_difference'],
        diag['point_semantic_current'] -
        diag['point_semantic_memory'])


def test_semantic_diagnostics_respect_configured_17_classes():
    adapter = QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=17, num_points=2, num_heads=2,
        coarse_topk_geometry=1, coarse_topk_semantic=1, point_topk=2,
        coarse_radius=10., point_radius=10.)
    adapter.eval()
    result = adapter(
        torch.randn(1, 2, 8), torch.zeros(1, 2, 2, 3),
        torch.randn(1, 2, 2, 17),
        _memory(classes=17, count=1), 0)
    diag = result['diagnostics']
    for name in (
            'point_semantic_current', 'point_semantic_memory',
            'semantic_difference'):
        assert diag[name].shape == (1, 2, 2, 17)


def test_semantic_features_are_point_local_before_query_retrieval():
    adapter = _adapter()
    adapter.eval()
    query = torch.randn(1, 1, 8)
    points = torch.zeros(1, 1, 2, 3)
    logits = torch.tensor([[[[6., 0., -2.], [0., 6., -2.]]]])
    changed = logits.clone()
    changed[0, 0, 0] = torch.tensor([-2., 0., 6.])
    first = adapter(query, points, logits, _memory(), 0)
    second = adapter(query, points, changed, _memory(), 0)
    current_first = first['diagnostics']['point_semantic_current']
    current_second = second['diagnostics']['point_semantic_current']
    assert not torch.equal(current_first[:, :, 0], current_second[:, :, 0])
    assert torch.equal(current_first[:, :, 1], current_second[:, :, 1])


def test_eligible_and_selected_point_counts_have_distinct_meanings():
    adapter = QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        coarse_topk_geometry=1, coarse_topk_semantic=1, point_topk=2,
        coarse_radius=10., point_radius=10.)
    adapter.eval()
    query = torch.randn(1, 1, 8)
    points = torch.zeros(1, 1, 2, 3)
    logits = torch.zeros(1, 1, 2, 3)
    many = adapter(query, points, logits, _memory(points=4, count=1), 0)
    assert torch.equal(
        many['diagnostics']['eligible_point_count'],
        torch.full((1, 1, 2), 4, dtype=torch.long))
    assert torch.equal(
        many['diagnostics']['selected_point_count'],
        torch.full((1, 1, 2), 2, dtype=torch.long))

    few_adapter = QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        coarse_topk_geometry=1, coarse_topk_semantic=1, point_topk=4,
        coarse_radius=10., point_radius=10.)
    few_adapter.eval()
    few = few_adapter(
        query, points, logits, _memory(points=2, count=1), 0)
    assert torch.equal(
        few['diagnostics']['eligible_point_count'],
        few['diagnostics']['selected_point_count'])
    assert (few['diagnostics']['selected_point_count'] <= 4).all()


@pytest.mark.parametrize('batch,queries,memory_count', [
    (1, 1, 1), (1, 3, 4), (2, 2, 1), (2, 4, 5)])
def test_batch_query_and_memory_topk_dimensions(
        batch, queries, memory_count):
    adapter = _adapter()
    adapter.eval()
    result = adapter(
        torch.randn(batch, queries, 8),
        torch.randn(batch, queries, 2, 3),
        torch.randn(batch, queries, 2, 3),
        _memory(count=memory_count, batch=batch), 0)
    assert result['context'].shape == (batch, queries, 2, 8)
    assert result['gate'].shape == (batch, queries, 2, 1)
    assert result['diagnostics']['coarse_candidate_count'].shape == (
        batch, queries)


def test_point_gates_are_not_forced_to_share_across_48_points():
    torch.manual_seed(23)
    adapter = QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=3, num_points=48, num_heads=2,
        coarse_topk_geometry=2, coarse_topk_semantic=2, point_topk=4,
        coarse_radius=10., point_radius=10.)
    adapter.eval()
    points = torch.randn(1, 1, 48, 3)
    result = adapter(
        torch.randn(1, 1, 8), points,
        torch.randn(1, 1, 48, 3),
        _memory(points=48, count=2), 0)
    gate = result['gate'][0, 0, :, 0]
    assert gate.unique().numel() > 1


def test_decode_separates_semantic_order_from_occupancy_max():
    base = torch.tensor([[[[1., .5, -.2]]]])
    ds = torch.tensor([[[[2., -1., 0.]]]])
    do = torch.tensor([[[[3.]]]])
    gate = torch.ones(1, 1, 1, 1)
    sem, occ, dec = QM.decode_semantic_occupancy_logits(base, ds, do, gate)
    assert torch.equal(dec.argmax(-1), sem.argmax(-1))
    assert torch.allclose(dec.max(-1, keepdim=True).values, occ)
    zero = QM.decode_semantic_occupancy_logits(base, torch.zeros_like(ds),
                                               torch.zeros_like(do), gate)
    assert torch.equal(zero[-1], base)

    occupancy_only = QM.decode_semantic_occupancy_logits(
        base, torch.zeros_like(ds), do, gate)
    assert torch.equal(occupancy_only[0].argmax(-1), base.argmax(-1))
    assert torch.equal(
        occupancy_only[2].argmax(-1), base.argmax(-1))

    semantic_delta = torch.tensor([[[[-2., 3., 0.]]]])
    semantic_only = QM.decode_semantic_occupancy_logits(
        base, semantic_delta, torch.zeros_like(do), gate)
    assert not torch.equal(semantic_only[0].argmax(-1), base.argmax(-1))
    assert torch.allclose(
        semantic_only[2].max(-1, keepdim=True).values,
        semantic_only[1])


def test_sparse_soft_voxel_forward_is_finite():
    sem = torch.randn(1, 1, 2, 3)
    occ = torch.randn(1, 1, 2, 1)
    pts = torch.tensor([[[[.25, .25, .25], [.75, .75, .75]]]])
    gt = torch.full((1, 2, 2, 2), 3, dtype=torch.long)
    gt[0, 0, 0, 0] = 1
    with torch.no_grad():
        loss = QM.sparse_soft_voxel_iou_loss(
            sem, occ, pts, gt, pc_range=[0., 0., 0., 2., 2., 2.],
            voxel_size=[1., 1., 1.], num_classes=3)
    assert torch.isfinite(loss)


def _loss_inputs(classes=17):
    semantic = torch.randn(1, 1, 2, classes)
    occupancy = torch.tensor([[[[1.], [-1.]]]])
    points = torch.tensor(
        [[[[.25, .25, .25], [.75, .75, .75]]]])
    gt = torch.full((1, 2, 2, 2), classes, dtype=torch.long)
    gt[0, 0, 0, 0] = 1
    return semantic, occupancy, points, gt


@pytest.mark.parametrize('weights_factory', [
    lambda: [1.] * 17,
    lambda: tuple([1.] * 17),
    lambda: torch.ones(1, 17),
])
def test_class_weight_sources_complete_loss_forward(weights_factory):
    semantic, occupancy, points, gt = _loss_inputs()
    weights = weights_factory()
    original = copy.deepcopy(weights)
    with torch.no_grad():
        losses = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, None, gt,
            pc_range=[0., 0., 0., 2., 2., 2.],
            voxel_size=[1., 1., 1.], empty_label=17,
            class_weights=weights,
            score_thresholds=[.35] * 15 + [.25, .30],
            positive_radius=.5)
    assert set(losses) == {
        'loss_sem', 'loss_occ', 'loss_threshold', 'loss_soft_voxel'}
    assert all(torch.isfinite(loss) for loss in losses.values())
    if isinstance(weights, list):
        assert weights == original
    elif isinstance(weights, tuple):
        assert weights == original
    else:
        assert torch.equal(weights, original)
    converted = QM.prepare_class_weights(
        weights, 17, semantic.device, semantic.dtype)
    assert converted.shape == (17,)
    assert converted.device == semantic.device
    assert converted.dtype == semantic.dtype


def test_class_weight_wrong_length_has_clear_error():
    semantic, occupancy, points, gt = _loss_inputs()
    with pytest.raises(
            ValueError,
            match=r'class_weights has 16 entries, expected 17'):
        QM.future_memory_v2_point_losses(
            semantic, occupancy, points, None, gt,
            pc_range=[0., 0., 0., 2., 2., 2.],
            voxel_size=[1., 1., 1.], empty_label=17,
            class_weights=[1.] * 16)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is not available')
def test_cuda_tensor_class_weights_are_supported():
    source = torch.ones(17, device='cuda')
    converted = QM.prepare_class_weights(
        source, 17, torch.device('cuda'), torch.float16)
    assert converted.device.type == 'cuda'
    assert converted.dtype == torch.float16
    assert converted.shape == (17,)
    semantic, occupancy, points, gt = _loss_inputs()
    with torch.no_grad():
        losses = QM.future_memory_v2_point_losses(
            semantic.cuda(), occupancy.cuda(), points.cuda(), None,
            gt.cuda(), pc_range=[0., 0., 0., 2., 2., 2.],
            voxel_size=[1., 1., 1.], empty_label=17,
            class_weights=source,
            score_thresholds=[.35] * 15 + [.25, .30],
            positive_radius=.5)
    assert all(
        loss.device.type == 'cuda' and torch.isfinite(loss)
        for loss in losses.values())


@pytest.mark.parametrize('case', [
    'empty_gt', 'empty_prediction', 'all_outside',
    'no_positive', 'no_negative'])
def test_v2_loss_forward_edge_cases_are_finite(case):
    semantic, occupancy, points, gt = _loss_inputs()
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    radius = .5
    if case == 'empty_gt':
        gt.fill_(17)
    elif case == 'empty_prediction':
        semantic = semantic[:, :0]
        occupancy = occupancy[:, :0]
        points = points[:, :0]
        mask = mask[:, :0]
    elif case == 'all_outside':
        points.fill_(4.)
    elif case == 'no_positive':
        gt.fill_(17)
    elif case == 'no_negative':
        points[:, :, 1] = points[:, :, 0]
    with torch.no_grad():
        losses = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, mask, gt,
            pc_range=[0., 0., 0., 2., 2., 2.],
            voxel_size=[1., 1., 1.], empty_label=17,
            class_weights=[1.] * 17,
            score_thresholds=[.35] * 15 + [.25, .30],
            positive_radius=radius)
    assert all(torch.isfinite(loss) for loss in losses.values())


def test_future_points_mask_excludes_masked_point_from_losses():
    semantic, occupancy, points, gt = _loss_inputs()
    mask = torch.tensor([[[True, False]]])
    changed_semantic = semantic.clone()
    changed_occupancy = occupancy.clone()
    changed_points = points.clone()
    changed_semantic[:, :, 1] = 100.
    changed_occupancy[:, :, 1] = 100.
    changed_points[:, :, 1] = 10.
    kwargs = dict(
        pc_range=[0., 0., 0., 2., 2., 2.],
        voxel_size=[1., 1., 1.], empty_label=17,
        class_weights=[1.] * 17,
        score_thresholds=[.35] * 15 + [.25, .30],
        positive_radius=.5)
    with torch.no_grad():
        first = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, mask, gt, **kwargs)
        second = QM.future_memory_v2_point_losses(
            changed_semantic, changed_occupancy, changed_points,
            mask, gt, **kwargs)
    for name in first:
        assert torch.equal(first[name], second[name])


def test_semantic_loss_uses_semantic_not_occupancy_shifted_logits():
    semantic, occupancy, points, gt = _loss_inputs()
    kwargs = dict(
        pc_range=[0., 0., 0., 2., 2., 2.],
        voxel_size=[1., 1., 1.], empty_label=17,
        class_weights=[1.] * 17,
        score_thresholds=[.35] * 15 + [.25, .30],
        positive_radius=.5)
    with torch.no_grad():
        first = QM.future_memory_v2_point_losses(
            semantic, occupancy, points, None, gt, **kwargs)
        second = QM.future_memory_v2_point_losses(
            semantic, occupancy + 20., points, None, gt, **kwargs)
    assert torch.equal(first['loss_sem'], second['loss_sem'])


def test_position_residual_is_exact_identity_and_metric_direction():
    base = torch.tensor(
        [[[[.25, .50, .75], [.10, .20, .30]],
          [[.90, .80, .70], [.40, .60, .20]]]])
    original = base.clone()
    zero = torch.zeros_like(base)
    all_active = torch.ones(1, 2, 2, dtype=torch.bool)
    no_active = torch.zeros_like(all_active)
    pc_range = [-40., -40., -1., 40., 40., 5.4]

    zero_output = QM.apply_metric_position_residual(
        base, zero, pc_range, all_active)
    empty_output = QM.apply_metric_position_residual(
        base, torch.ones_like(base), pc_range, no_active)
    gate_zero_output = QM.apply_metric_position_residual(
        base, torch.ones_like(base), pc_range,
        torch.zeros_like(all_active))
    assert torch.equal(zero_output, base)
    assert torch.equal(empty_output, base)
    assert torch.equal(gate_zero_output, base)
    assert torch.equal(base, original)

    delta = torch.zeros_like(base)
    delta[0, 0, 1, 0] = 4.
    active = torch.zeros_like(all_active)
    active[0, 0, 1] = True
    shifted = QM.apply_metric_position_residual(
        base, delta, pc_range, active)
    unchanged = ~active
    assert torch.equal(shifted[unchanged], base[unchanged])
    shifted_metric = QM.decode_points_metric(shifted, pc_range)
    base_metric = QM.decode_points_metric(base, pc_range)
    assert torch.allclose(
        shifted_metric[0, 0, 1] - base_metric[0, 0, 1],
        torch.tensor([4., 0., 0.]), atol=1e-5)


def test_point_occupancy_threshold_uses_head_score_thresholds():
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        zero = args[0].new_zeros(())
        return dict(
            loss_sem=zero, loss_occ=zero,
            loss_threshold=zero, loss_soft_voxel=zero)

    method = _load_sparseworld_method(
        '_future_memory_v2_losses',
        {'future_memory_v2_point_losses': capture})
    model = SimpleNamespace(
        pc_range=torch.tensor([0., 0., 0., 2., 2., 2.]),
        pts_bbox_head=SimpleNamespace(
            voxel_size=torch.tensor([1., 1., 1.]),
            train_cfg={'cls_weights': [1., 2., 3.]},
            test_cfg={'score_thr': [0.6, 0.2, 0.8]}),
        future_memory_v2_loss_cfg={
            'positive_radius': 0.5, 'threshold_margin': 0.1},
        class_weights=torch.ones(3),
        empty_idx=3)
    method(
        model, torch.randn(1, 1, 2, 3),
        torch.zeros(1, 1, 2, 1),
        torch.full((1, 1, 2, 3), 0.25), None,
        torch.full((1, 2, 2, 2), 3, dtype=torch.long))
    assert (
        captured['score_thresholds'] is
        model.pts_bbox_head.test_cfg['score_thr'])
    assert (
        captured['class_weights'] is
        model.pts_bbox_head.train_cfg['cls_weights'])


def test_sparseworld_v2_loss_method_accepts_config_list_forward():
    method = _load_sparseworld_method(
        '_future_memory_v2_losses',
        {'future_memory_v2_point_losses':
             QM.future_memory_v2_point_losses})
    configured = [1.] * 17
    original = configured.copy()
    model = SimpleNamespace(
        pc_range=torch.tensor([0., 0., 0., 2., 2., 2.]),
        pts_bbox_head=SimpleNamespace(
            voxel_size=torch.tensor([1., 1., 1.]),
            train_cfg={'cls_weights': configured},
            test_cfg={'score_thr': [.35] * 15 + [.25, .30]}),
        future_memory_v2_loss_cfg={
            'positive_radius': .5, 'threshold_margin': .1},
        class_weights=None,
        empty_idx=17)
    semantic, occupancy, points, gt = _loss_inputs()
    with torch.no_grad():
        losses = method(
            model, semantic, occupancy, points, None, gt)
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert configured == original


def test_model_v2_branch_keeps_baseline_recurrence_snapshots_isolated():
    forward_method = _load_sparseworld_method(
        '_forward_backbone_future_memory_adapter',
        {
            'torch': torch,
            'np': __import__('numpy'),
            'decode_points_metric': QM.decode_points_metric,
            'decode_semantic_occupancy_logits':
                QM.decode_semantic_occupancy_logits,
        })

    class ResponsiveProjection:
        def __init__(self, output, scale=0.01):
            self.output = output
            self.scale = scale

        def __call__(self, value):
            signal = value.mean(-1, keepdim=True) * self.scale
            return signal.expand(*value.shape[:-1], self.output)

    class ResponsiveEgo:
        def forward(self, query_pos, query_feat, memory_pos, memory_feat):
            del query_pos, memory_pos
            signal = memory_feat.mean(
                dim=1, keepdim=True).mean(-1, keepdim=True)
            return query_feat + signal, None

        __call__ = forward

    class Correction:
        def __init__(self, corrections):
            self.corrections = corrections

        def __call__(
                self, query, points, logits, memory, horizon_id, **kwargs):
            del query, points, memory, kwargs
            B, Q, R, _ = logits.shape
            amount = float(self.corrections[horizon_id])
            return dict(
                delta_s=torch.ones_like(logits) * amount,
                delta_o=logits.new_zeros(B, Q, R, 1),
                delta_p=logits.new_zeros(B, Q, R, 3),
                gate=logits.new_ones(B, Q, R, 1),
                diagnostics={
                    'horizon_id': horizon_id,
                    'has_candidate': torch.ones(
                        B, Q, R, dtype=torch.bool)})

    counts = [2, 1, 1, 1, 1, 1, 1]
    stamps = torch.cat([
        torch.full((count,), index, dtype=torch.long)
        for index, count in enumerate(counts)])
    B, C, R = 1, 8, 2
    model = SimpleNamespace(
        num_refines=R,
        num_fu_frames=6,
        pc_range=torch.tensor([-40., -40., -1., 40., 40., 5.4]),
        ego_cross_attn=ResponsiveEgo(),
        traj_head=ResponsiveProjection(2),
        position_encoder=ResponsiveProjection(C),
        cls_branch=ResponsiveProjection(R * 17),
        reg_branch=ResponsiveProjection(R * 3),
        vel_branch=ResponsiveProjection(R * 2),
        future_memory_adapter_version='v2',
        training=False)
    model.refine_points = lambda points, delta: points + delta.reshape_as(
        points) * 0.01
    model._refine_future_memory_points = (
        lambda points, delta, active_mask=None:
        QM.apply_metric_position_residual(
            points, delta.reshape_as(points), model.pc_range, active_mask))

    query_feat = torch.linspace(
        -1., 1., stamps.numel() * C).reshape(B, stamps.numel(), C)
    query_pos = torch.linspace(
        .2, .8, stamps.numel() * R * 3).reshape(
            B, stamps.numel(), R, 3)
    query_cls = torch.linspace(
        -2., 2., stamps.numel() * R * 17).reshape(
            B, stamps.numel(), R, 17)

    def run(corrections):
        model.future_memory_adapter_v2 = Correction(corrections)
        return forward_method(
            model, [{'ego2lidar': torch.eye(4).numpy()}], {}, B,
            torch.ones(B, 1, C), {}, query_feat, query_pos, query_cls,
            stamps, {})

    reference = run([1., 2., 3.])
    changed_1s = run([100., 2., 3.])
    changed_2s = run([1., 200., 3.])
    assert not torch.equal(
        reference['forecast_semantics_list'][1],
        changed_1s['forecast_semantics_list'][1])
    assert torch.equal(
        reference['baseline_forecast_semantics_list'][3],
        changed_1s['baseline_forecast_semantics_list'][3])
    assert not torch.equal(
        reference['forecast_semantics_list'][3],
        changed_2s['forecast_semantics_list'][3])
    assert torch.equal(
        reference['baseline_forecast_semantics_list'][5],
        changed_2s['baseline_forecast_semantics_list'][5])
    enabled_steps = [
        diag['internal_step']
        for diag in reference['memory_adapter_diagnostics']
        if diag['enabled']]
    assert enabled_steps == [2, 4, 6]
    assert torch.equal(reference['cls_score'], query_cls[:, stamps == 0])
    for step in (0, 2, 4):
        assert torch.equal(
            reference['forecast_semantics_list'][step],
            reference['baseline_forecast_semantics_list'][step])


def test_v2_disabled_runs_baseline_forward_without_adapter_call():
    method = _load_sparseworld_method(
        'forward_backbone', {'torch': torch, 'np': __import__('numpy')})

    class Projection:
        def __init__(self, output):
            self.output = output

        def __call__(self, value):
            signal = value.mean(-1, keepdim=True)
            return signal.expand(*value.shape[:-1], self.output)

    class Ego:
        def __call__(self, query_pos, ego_feat, memory_pos, memory_feat):
            del query_pos, memory_pos
            signal = memory_feat.mean(
                dim=1, keepdim=True).mean(-1, keepdim=True)
            return ego_feat + signal, None

    class ForbiddenAdapter:
        def __call__(self, *args, **kwargs):
            raise AssertionError('disabled V2 adapter was called')

    stamps = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6])
    B, C, R = 1, 8, 2
    query_feat = torch.linspace(
        -1., 1., stamps.numel() * C).reshape(B, -1, C)
    query_pos = torch.linspace(
        .2, .8, stamps.numel() * R * 3).reshape(B, -1, R, 3)
    query_cls = torch.linspace(
        -2., 2., stamps.numel() * R * 17).reshape(
            B, -1, R, 17)
    outs = dict(
        query_feat=query_feat,
        all_refine_pts=[query_pos],
        all_cls_scores=[query_cls])
    model = SimpleNamespace(
        query_memory_log_diagnostics=False,
        query_memory_diagnostics=[],
        memory_refiner_diagnostics=[],
        plan_head=Projection(C),
        points_scale_branch=Projection(1),
        pts_bbox_head=SimpleNamespace(ind_stamps_all=stamps),
        training=False,
        future_memory_adapter_enabled=False,
        future_memory_adapter_v2=ForbiddenAdapter(),
        query_memory_enabled=False,
        memory_phase3_future_only=True,
        num_refines=R,
        num_fu_frames=6,
        query_memory_frame_interval=.5,
        ego_cross_attn=Ego(),
        traj_head=Projection(2),
        position_encoder=Projection(C),
        reg_branch=Projection(R * 3),
        cls_branch=Projection(R * 17),
        vel_branch=Projection(R * 2),
        pretrain=True)
    model.simple_test_online = lambda img_metas, img: outs
    model._forward_backbone_future_memory_adapter = ForbiddenAdapter()
    model._apply_query_memory_once = (
        lambda feat, *args, **kwargs: feat)
    model._apply_memory_conditioned_refiner = (
        lambda feat, pos, cls, horizon: (
            feat, cls, pos,
            feat.new_zeros(B, feat.shape[1], R, 2),
            {'enabled': False, 'horizon': horizon}))
    model._memory_refiner_active = lambda: False
    model.refine_points = (
        lambda points, delta: points + delta.reshape_as(points) * .01)

    output = method(
        model, torch.zeros(B, 1, 1),
        [{}], temporal_ego_states=[torch.ones(B, 1, 3)])
    assert torch.equal(output['cls_score'], query_cls[:, stamps == 0])
    assert torch.equal(
        output['base_cls_score'], query_cls[:, stamps == 0])
    assert len(output['forecast_semantics_list']) == 6


def test_v2_loss_names_are_only_wired_for_1s_2s_3s():
    source = (
        ROOT / 'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py'
    ).read_text()
    assert "((1, '1s'), (3, '2s')," in source
    assert "(5, '3s'))" in source
    for suffix in (
            'loss_base_cls', 'loss_sem', 'loss_occ', 'loss_threshold',
            'loss_soft_voxel', 'loss_pts'):
        assert f"mem_{{horizon_name}}.{suffix}" in source
    assert 'mem_0s.loss_sem' not in source
