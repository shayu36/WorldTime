"""CPU-level invariants for FutureMemoryAdapterV2."""

import importlib
import importlib.util
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'future_memory_adapter_v2_qm', ROOT / 'mmdet3d/models/sparsedetectors/query_memory.py')
QM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QM)


def _memory(C=8, classes=3, points=2, count=3):
    return dict(
        memory_query_feat=torch.randn(1, 1, count, C),
        memory_points_metric=torch.zeros(1, 1, count, points, 3),
        memory_valid=torch.ones(1, 1, count, dtype=torch.bool),
        memory_reliability=torch.tensor([[[.9, .7, .5]]]),
        memory_age=torch.tensor([[[2.5, 3.5, 4.5]]]),
        memory_semantic_distribution=torch.eye(classes)[
            torch.arange(count) % classes].reshape(1, 1, count, classes),
        memory_label=torch.arange(count).reshape(1, 1, count) % classes)


def _adapter(**kwargs):
    return QM.FutureMemoryAdapterV2(
        embed_dims=8, num_classes=3, num_points=2, num_heads=2,
        coarse_topk_geometry=2, coarse_topk_semantic=2, point_topk=2,
        coarse_radius=10., point_radius=10., **kwargs)


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
    for horizon in range(3):
        result = adapter(q, p, logits, memory, horizon)
        ages.append(result['diagnostics']['effective_age'])
        penalties.append(result['diagnostics']['support_age'])
    assert torch.equal(memory['memory_age'], base)
    assert torch.allclose(ages[0], torch.tensor([[3.5, 4.5, 5.5]]))
    assert torch.allclose(ages[1], torch.tensor([[4.5, 5.5, 6.5]]))
    assert torch.allclose(ages[2], torch.tensor([[5.5, 6.5, 7.5]]))
    assert float(penalties[2].mean()) > float(penalties[1].mean()) > float(penalties[0].mean())


def test_point_context_gate_shapes_and_zero_candidate_identity():
    adapter = _adapter()
    q = torch.randn(1, 2, 8)
    p = torch.zeros(1, 2, 2, 3)
    logits = torch.randn(1, 2, 2, 3)
    result = adapter(q, p, logits, _memory(), 0)
    assert result['context'].shape == (1, 2, 2, 8)
    assert result['gate'].shape == (1, 2, 2, 1)
    assert result['diagnostics']['coarse_candidate_count'].shape == (1, 2)
    invalid = _memory()
    invalid['memory_valid'].zero_()
    empty = adapter(q, p, logits, invalid, 0)
    assert torch.equal(empty['gate'], torch.zeros_like(empty['gate']))
    assert torch.equal(empty['delta_s'], torch.zeros_like(empty['delta_s']))


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


def test_semantic_uncertainty_dropout_and_disagreement_are_explicit():
    adapter = _adapter(semantic_uncertainty_gamma=2., semantic_weight_floor=.05,
                       semantic_dropout_probability=1.)
    q = torch.randn(1, 1, 8); p = torch.zeros(1, 1, 2, 3)
    confident = torch.tensor([[[[8., -4., -4.], [8., -4., -4.]]]])
    uncertain = torch.zeros_like(confident)
    adapter.eval()
    confident_eval = adapter(q, p, confident, _memory(), 0)
    uncertain_eval = adapter(q, p, uncertain, _memory(), 0)
    assert confident_eval['diagnostics']['semantic_weight'].mean() > uncertain_eval['diagnostics']['semantic_weight'].mean()
    adapter.train()
    dropped = adapter(q, p, confident, _memory(), 0)
    assert dropped['diagnostics']['semantic_dropout_mask'].all()
    adapter.semantic_dropout_probability = 0.
    disagreement = dropped['diagnostics']['semantic_disagreement']
    assert disagreement.shape == (1, 1, 2)


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


def test_sparse_soft_voxel_has_semantic_occupancy_and_position_gradients():
    sem = torch.randn(1, 1, 2, 3, requires_grad=True)
    occ = torch.randn(1, 1, 2, 1, requires_grad=True)
    pts = torch.tensor([[[[.25, .25, .25], [.75, .75, .75]]]], requires_grad=True)
    gt = torch.full((1, 2, 2, 2), 3, dtype=torch.long)
    gt[0, 0, 0, 0] = 1
    loss = QM.sparse_soft_voxel_iou_loss(
        sem, occ, pts, gt, pc_range=[0., 0., 0., 2., 2., 2.],
        voxel_size=[1., 1., 1.], num_classes=3)
    loss.backward()
    assert sem.grad is not None and float(sem.grad.abs().sum()) > 0
    assert occ.grad is not None and float(occ.grad.abs().sum()) > 0
    assert pts.grad is not None and torch.isfinite(pts.grad).all()


def test_point_occupancy_threshold_uses_configured_score_thresholds():
    sw = importlib.import_module('mmdet3d.models.sparsedetectors.sparseworld_4d_traj')
    class Head:
        voxel_size = torch.tensor([1., 1., 1.])
        test_cfg = {'score_thr': [0.6, 0.2, 0.8]}
    model = object.__new__(sw.SparseWorld4DTraj)
    model.pc_range = torch.tensor([0., 0., 0., 2., 2., 2.])
    model.pts_bbox_head = Head()
    model.future_memory_v2_loss_cfg = {'positive_radius': 0.5, 'threshold_margin': 0.1}
    model.class_weights = torch.ones(3)
    model.empty_idx = 3
    sem = torch.randn(1, 1, 2, 3, requires_grad=True)
    occ = torch.zeros(1, 1, 2, 1, requires_grad=True)
    pts = torch.tensor([[[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]]], requires_grad=True)
    gt = torch.full((1, 2, 2, 2), 3, dtype=torch.long)
    gt[0, 0, 0, 0] = 1
    losses = model._future_memory_v2_losses(sem, occ, pts, None, gt)
    assert all(name in losses for name in ('loss_sem', 'loss_occ', 'loss_threshold', 'loss_soft_voxel'))
    sum(losses.values()).backward()
    assert occ.grad is not None and torch.isfinite(occ.grad).all()


def test_model_v2_branch_keeps_baseline_recurrence_snapshots_isolated():
    sw = importlib.import_module('mmdet3d.models.sparsedetectors.sparseworld_4d_traj')

    class Shape(nn.Module):
        def __init__(self, output): super().__init__(); self.output = output
        def forward(self, x): return x.new_zeros(*x.shape[:-1], self.output)

    class Ego(nn.Module):
        def forward(self, query_pos, query_feat, memory_pos, memory_feat):
            del query_pos, memory_pos, memory_feat
            return query_feat.new_zeros(query_feat.shape), None

    class Correction(nn.Module):
        def forward(self, query, points, logits, memory, horizon_id, **kwargs):
            del query, points, memory, kwargs
            B, Q, R, C = logits.shape
            return dict(delta_s=torch.ones_like(logits), delta_o=logits.new_zeros(B, Q, R, 1),
                        delta_p=logits.new_zeros(B, Q, R, 3), gate=logits.new_ones(B, Q, R, 1),
                        diagnostics={'horizon_id': horizon_id})

    model = sw.SparseWorld4DTraj.__new__(sw.SparseWorld4DTraj); nn.Module.__init__(model)
    counts = [720, 60, 60, 60, 60, 40, 40]
    stamps = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)])
    B, C, R = 1, 8, 2
    model.num_refines = R; model.num_fu_frames = 6
    model.pc_range = torch.tensor([-40., -40., -1., 40., 40., 5.4])
    model.pts_bbox_head = nn.Module(); model.pts_bbox_head.ind_stamps_all = stamps
    model.ego_cross_attn = Ego(); model.traj_head = Shape(2); model.position_encoder = Shape(C)
    model.cls_branch = Shape(R * 17); model.reg_branch = Shape(R * 3); model.vel_branch = Shape(R * 2)
    model.refine_points = lambda points, delta: points
    model.future_memory_adapter_v2 = Correction(); model.future_memory_adapter_version = 'v2'
    out = model._forward_backbone_future_memory_adapter(
        [{'ego2lidar': torch.eye(4).numpy()}], {'temporal_trajs': torch.zeros(B, 6, 2)}, B,
        torch.zeros(B, 1, C), {}, torch.zeros(B, stamps.numel(), C),
        torch.zeros(B, stamps.numel(), R, 3), torch.zeros(B, stamps.numel(), R, 17), stamps, {})
    assert [d['query_count'] for d in out['memory_adapter_diagnostics'] if d['enabled']] == [840, 960, 1040]
    assert torch.equal(out['baseline_forecast_semantics_list'][3], torch.zeros_like(out['baseline_forecast_semantics_list'][3]))
