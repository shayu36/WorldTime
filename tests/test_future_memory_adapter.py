"""Fast CPU-only acceptance tests for the independent future adapter."""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qm = _load('future_memory_adapter_qm',
           ROOT / 'mmdet3d/models/sparsedetectors/query_memory.py')
sw = importlib.import_module(
    'mmdet3d.models.sparsedetectors.sparseworld_4d_traj')


def _inputs(embed_dims=8, classes=3, points=2, queries=3, batch=1):
    torch.manual_seed(7)
    query = torch.randn(batch, queries, embed_dims)
    points_metric = torch.zeros(batch, queries, points, 3)
    logits = torch.randn(batch, queries, points, classes)
    return query, points_metric, logits


def _memory(embed_dims=8, classes=3, points=2, count=2, batch=1):
    # Use genuinely different directions (not affine shifts).  LayerNorm in
    # the adapter removes per-token affine components, so a simple arange
    # fixture would make all memory keys/values identical after normalization.
    feat = torch.zeros(batch, 1, count, embed_dims, dtype=torch.float32)
    for batch_index in range(batch):
        for index in range(count):
            feat[batch_index, 0, index, index % embed_dims] = 1.0
            feat[batch_index, 0, index, (index + 3) % embed_dims] = -0.35
    sem = torch.zeros(batch, 1, count, classes)
    for batch_index in range(batch):
        for index in range(count):
            sem[batch_index, 0, index, index % classes] = 1.0
    return dict(
        memory_query_feat=feat,
        memory_points_metric=torch.zeros(batch, 1, count, points, 3),
        memory_valid=torch.ones(batch, 1, count, dtype=torch.bool),
        memory_reliability=torch.ones(batch, 1, count),
        memory_age=torch.ones(batch, 1, count),
        memory_label=(torch.arange(count, dtype=torch.long).reshape(
            1, 1, count) % classes).expand(batch, -1, -1).clone(),
        memory_semantic_distribution=sem,
    )


def _corrected(adapter, query, points, logits, memory, horizon=0):
    result = adapter(query, points, logits, memory, horizon)
    gate = result['gate'][..., None]
    return logits + gate * (result['delta_s'] + result['delta_o']), result


def test_zero_init_and_no_candidate_are_strict_baseline_identity():
    adapter = qm.FutureMemoryAdapter(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2)
    query, points, logits = _inputs()
    memory = _memory()
    corrected, result = _corrected(adapter, query, points, logits, memory)
    assert torch.equal(corrected, logits)
    assert torch.equal(result['delta_s'], torch.zeros_like(result['delta_s']))
    assert torch.equal(result['delta_o'], torch.zeros_like(result['delta_o']))
    assert torch.equal(result['delta_p'], torch.zeros_like(result['delta_p']))

    no_memory = dict(memory)
    no_memory['memory_valid'] = torch.zeros_like(memory['memory_valid'])
    corrected, result = _corrected(adapter, query, points, logits, no_memory)
    assert torch.equal(corrected, logits)
    assert torch.equal(result['gate'], torch.zeros_like(result['gate']))


def test_initialization_and_horizon_conditioning():
    adapter = qm.FutureMemoryAdapter(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2)
    assert torch.all(adapter.delta_s_head[-1].weight == 0)
    assert torch.all(adapter.delta_o_head[-1].weight == 0)
    assert torch.all(adapter.delta_p_head[-1].weight == 0)
    assert adapter.gate_head.bias.item() == pytest.approx(-1.0)
    for layer in (adapter.query_proj, adapter.key_proj, adapter.value_proj,
                  adapter.semantic_memory_proj):
        assert float(layer.weight.abs().sum()) > 0
    query, points, logits = _inputs()
    result0 = adapter(query, points, logits, _memory(), 0)
    result1 = adapter(query, points, logits, _memory(), 1)
    result2 = adapter(query, points, logits, _memory(), 2)
    assert result0['diagnostics']['horizon_id'] == 0
    assert result1['diagnostics']['horizon_id'] == 1
    assert result2['diagnostics']['horizon_id'] == 2
    assert not torch.allclose(result0['attention_query'],
                              result1['attention_query'])
    assert not torch.allclose(result1['attention_query'],
                              result2['attention_query'])


def test_batch_size_greater_than_one_uses_explicit_batch_broadcasts():
    adapter = qm.FutureMemoryAdapter(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        spatial_radius=10.0)
    query, points, logits = _inputs(queries=5, batch=4)
    result = adapter(query, points, logits, _memory(count=3, batch=4), 0)
    assert result['context'].shape == (4, 5, 8)
    assert result['diagnostics']['attention_scores'].shape == (4, 2, 5, 3)
    assert result['diagnostics']['has_candidate'].shape == (4, 5)


def test_memory_semantics_feature_geometry_age_and_reliability_are_causal():
    adapter = qm.FutureMemoryAdapter(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        spatial_radius=10.0)
    query, points, logits = _inputs(queries=1)
    memory = _memory(count=2)

    score_a = adapter(query, points, logits, memory, 0)['diagnostics'][
        'attention_scores']
    changed_sem = dict(memory)
    changed_sem['memory_semantic_distribution'] = memory[
        'memory_semantic_distribution'].flip(-1)
    score_b = adapter(query, points, logits, changed_sem, 0)['diagnostics'][
        'attention_scores']
    assert not torch.allclose(score_a, score_b)

    changed_feature = dict(memory)
    changed_feature['memory_query_feat'] = memory[
        'memory_query_feat'].flip(2)
    _, feature_result = _corrected(adapter, query, points, logits,
                                   changed_feature)
    _, original_result = _corrected(adapter, query, points, logits, memory)
    assert not torch.allclose(feature_result['context'],
                              original_result['context'])

    changed_rel = dict(memory)
    changed_rel['memory_reliability'] = torch.tensor([[[1.0, 0.2]]])
    rel_result = adapter(query, points, logits, changed_rel, 0)
    assert not torch.allclose(rel_result['diagnostics']['attention_scores'],
                              score_a)
    changed_age = dict(memory)
    changed_age['memory_age'] = torch.tensor([[[1.0, 4.0]]])
    age_result = adapter(query, points, logits, changed_age, 0)
    assert not torch.allclose(age_result['diagnostics']['attention_scores'],
                              score_a)
    changed_pos = dict(memory)
    changed_pos['memory_points_metric'] = memory[
        'memory_points_metric'].clone()
    changed_pos['memory_points_metric'][..., 1, :, 0] = 2.0
    pos_result = adapter(query, points, logits, changed_pos, 0)
    assert not torch.allclose(pos_result['diagnostics']['attention_scores'],
                              score_a)

    # Labels are a functional prior even when a soft distribution is present.
    changed_label = dict(memory)
    changed_label['memory_label'] = torch.ones_like(memory['memory_label'])
    label_result = adapter(query, points, logits, changed_label, 0)
    assert not torch.allclose(label_result['diagnostics']['attention_scores'],
                              score_a)


def test_label_fallback_and_shuffled_memory_change_adapter_context():
    adapter = qm.FutureMemoryAdapter(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        spatial_radius=10.0)
    query, points, logits = _inputs(queries=1)
    memory = _memory(count=2)
    fallback = dict(memory)
    fallback.pop('memory_semantic_distribution')
    fallback['memory_label'] = torch.tensor([[[0, 1]]])
    score = adapter(query, points, logits, fallback, 0)['diagnostics'][
        'attention_scores']
    fallback['memory_label'] = torch.tensor([[[1, 1]]])
    score_changed = adapter(query, points, logits, fallback, 0)['diagnostics'][
        'attention_scores']
    assert not torch.allclose(score, score_changed)

    for head in (adapter.delta_s_head, adapter.delta_o_head,
                 adapter.delta_p_head):
        nn.init.constant_(head[-1].weight, 0.05)
    _, original = _corrected(adapter, query, points, logits, memory)
    # Swap feature content without swapping the other fields.  This models a
    # shuffled/corrupted cache record and verifies that the context is
    # functionally dependent on memory content (a consistent permutation of
    # every field would correctly be permutation invariant).
    shuffled = dict(memory)
    shuffled['memory_query_feat'] = memory['memory_query_feat'].flip(2)
    _, shuffled_result = _corrected(adapter, query, points, logits, shuffled)
    assert not torch.allclose(original['context'], shuffled_result['context'])


def test_two_step_backward_reaches_residual_heads_then_attention_and_gate():
    adapter = qm.FutureMemoryAdapter(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        spatial_radius=10.0)
    frozen_baseline = nn.Linear(4, 8)
    for parameter in frozen_baseline.parameters():
        parameter.requires_grad = False
    baseline_input = torch.randn(1, 2, 4)
    query, points, logits = _inputs(queries=2)
    # Route the synthetic query through a frozen Baseline component so the
    # backward assertion covers the model-level freeze boundary as well.
    query = frozen_baseline(baseline_input)
    memory = _memory(count=2)
    target_s = torch.randn_like(logits)
    target_o = torch.randn(1, 2, 2, 1)
    target_p = torch.randn(1, 2, 2, 3)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    first_grads = None
    for step in range(2):
        optimizer.zero_grad()
        result = adapter(query, points, logits, memory, 0)
        gate = result['gate'][..., None]
        corrected = logits + gate * (result['delta_s'] + result['delta_o'])
        loss = ((corrected - target_s)**2).mean()
        loss = loss + ((result['delta_o'] - target_o)**2).mean()
        loss = loss + ((result['delta_p'] - target_p)**2).mean()
        loss.backward()
        assert all(parameter.grad is None
                   for parameter in frozen_baseline.parameters())
        if step == 0:
            first_grads = [adapter.delta_s_head[-1].weight.grad,
                           adapter.delta_o_head[-1].weight.grad,
                           adapter.delta_p_head[-1].weight.grad]
            assert all(g is not None and float(g.abs().sum()) > 0
                       for g in first_grads)
        else:
            for parameter in (adapter.query_proj.weight,
                              adapter.key_proj.weight,
                              adapter.value_proj.weight,
                              adapter.adapter[0].weight,
                              adapter.gate_head.weight):
                assert parameter.grad is not None
                assert float(parameter.grad.abs().sum()) > 0
        optimizer.step()


class _Shape(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = int(output)

    def forward(self, x):
        return x.new_zeros(*x.shape[:-1], self.output)


class _Ego(nn.Module):
    def forward(self, query_pos, query_feat, memory_pos, memory_feat):
        del query_pos, memory_pos, memory_feat
        return query_feat.new_zeros(query_feat.shape), None


class _Correction(nn.Module):
    def forward(self, query, points, logits, memory, horizon_id, **kwargs):
        del points, memory, kwargs
        B, Q, R, C = logits.shape
        return dict(
            delta_s=logits.new_ones(B, Q, R, C),
            delta_o=logits.new_zeros(B, Q, R, 1),
            delta_p=logits.new_zeros(B, Q, R, 3),
            gate=logits.new_ones(B, Q, 1),
            diagnostics={'horizon_id': horizon_id})


def test_active_queries_and_corrected_logits_never_feed_baseline_state():
    model = sw.SparseWorld4DTraj.__new__(sw.SparseWorld4DTraj)
    nn.Module.__init__(model)
    counts = [720, 60, 60, 60, 60, 40, 40]
    stamps = torch.cat([
        torch.full((count,), index, dtype=torch.long)
        for index, count in enumerate(counts)])
    B, C, R = 1, 8, 2
    model.num_refines = R
    model.num_fu_frames = 6
    model.pc_range = torch.tensor([-40., -40., -1., 40., 40., 5.4])
    model.pts_bbox_head = nn.Module()
    model.pts_bbox_head.ind_stamps_all = stamps
    model.ego_cross_attn = _Ego()
    model.traj_head = _Shape(2)
    model.position_encoder = _Shape(C)
    model.cls_branch = _Shape(R * 17)
    model.reg_branch = _Shape(R * 3)
    model.vel_branch = _Shape(R * 2)
    model.refine_points = lambda points, delta: points
    model.future_memory_adapter = _Correction()
    query = torch.zeros(B, stamps.numel(), C)
    points = torch.zeros(B, stamps.numel(), R, 3)
    logits = torch.zeros(B, stamps.numel(), R, 17)
    outputs = model._forward_backbone_future_memory_adapter(
        [{'ego2lidar': torch.eye(4).numpy()}],
        {'temporal_trajs': torch.zeros(B, 6, 2)}, B, torch.zeros(B, 1, C),
        {}, query, points, logits, stamps, {})
    target_counts = [d['query_count'] for d in
                     outputs['memory_adapter_diagnostics'] if d['enabled']]
    assert target_counts == [840, 960, 1040]
    assert outputs['memory_adapter_diagnostics'][0]['enabled'] is False
    assert outputs['memory_adapter_diagnostics'][2]['enabled'] is False
    assert outputs['memory_adapter_diagnostics'][4]['enabled'] is False
    # Correction is visible at 1/2/3 s, but baseline state snapshots remain
    # zero and therefore cannot have received a previous corrected logit.
    assert float(outputs['forecast_semantics_list'][1].abs().sum()) > 0
    assert float(outputs['forecast_semantics_list'][3].abs().sum()) > 0
    assert float(outputs['forecast_semantics_list'][5].abs().sum()) > 0
    assert torch.equal(outputs['baseline_forecast_semantics_list'][1],
                       torch.zeros_like(outputs[
                           'baseline_forecast_semantics_list'][1]))
    assert torch.equal(outputs['baseline_forecast_semantics_list'][3],
                       torch.zeros_like(outputs[
                           'baseline_forecast_semantics_list'][3]))
    for corrected_points, baseline_points in zip(
            outputs['forecast_points_list'],
            outputs['baseline_forecast_points_list']):
        assert torch.allclose(corrected_points, baseline_points)
