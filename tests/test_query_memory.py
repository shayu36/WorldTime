import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qm = load_module(
    'query_memory_under_test',
    ROOT / 'mmdet3d/models/sparsedetectors/query_memory.py')
loader_mod = load_module(
    'loading_query_memory_under_test',
    ROOT / 'mmdet3d/datasets/pipelines/loading_query_memory.py')


def z_rotation(theta):
    c = torch.cos(torch.tensor(theta))
    s = torch.sin(torch.tensor(theta))
    mat = torch.eye(4)
    mat[0, 0] = c
    mat[0, 1] = -s
    mat[1, 0] = s
    mat[1, 1] = c
    return mat


def make_memory(B=1, K=1, M=4, C=8, R=2):
    return dict(
        memory_query_feat=torch.randn(B, K, M, C),
        memory_points_metric=torch.zeros(B, K, M, R, 3),
        memory_conf=torch.full((B, K, M), 0.9),
        memory_valid=torch.ones(B, K, M, dtype=torch.bool),
        memory_source_ego2global=torch.eye(4).repeat(B, K, 1, 1),
        memory_age=torch.ones(B, K, M))


def test_ego_pose_aligner_identity_translation_rotation_and_batch_isolation():
    aligner = qm.EgoPoseAligner()
    points = torch.tensor([[[[1.0, 2.0, 0.5], [3.0, 4.0, 0.5]]]])
    eye = torch.eye(4).unsqueeze(0)
    assert torch.allclose(aligner(points, eye, eye), points)

    source = torch.eye(4).unsqueeze(0)
    source[0, 0, 3] = 10.0
    translated = aligner(points, source, eye)
    assert torch.allclose(translated[..., 0], points[..., 0] + 10.0)

    rotated = aligner(
        torch.tensor([[[[1.0, 0.0, 0.0]]]]),
        z_rotation(torch.pi / 2).unsqueeze(0),
        eye)
    assert torch.allclose(
        rotated[0, 0, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)

    batch_points = torch.zeros(2, 1, 1, 3)
    batch_source = torch.eye(4).repeat(2, 1, 1)
    batch_source[0, 0, 3] = 1.0
    batch_source[1, 1, 3] = 2.0
    aligned = aligner(batch_points, batch_source, torch.eye(4).repeat(2, 1, 1))
    assert torch.allclose(aligned[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(aligned[1, 0, 0], torch.tensor([0.0, 2.0, 0.0]))


def test_logits_to_query_confidence_is_sigmoid_max_mean():
    logits = torch.tensor([[[[0.0, 2.0], [-1.0, 1.0]]]])
    expected = logits.float().sigmoid().amax(dim=-1).mean(dim=-1)
    got = qm.logits_to_query_confidence(logits)
    assert got.shape == (1, 1)
    assert torch.allclose(got, expected)


def test_attention_uses_true_multihead_age_radius_and_topk():
    torch.manual_seed(0)
    attn = qm.CausalQueryMemoryAttention(
        embed_dims=8, num_heads=2, spatial_radius=2.0, topk=2,
        max_age=3.0, dropout=0.0)
    query_feat = torch.randn(1, 1, 8)
    query_points = torch.zeros(1, 1, 2, 3)
    memory = make_memory(C=8, R=2)
    centers = torch.tensor([0.5, 1.0, 1.5, 10.0])
    memory['memory_points_metric'][0, 0, :, :, 0] = centers[:, None]
    memory['memory_age'][0, 0] = torch.tensor([1.0, 2.0, -0.5, 1.0])

    readout, diag = attn(
        query_feat, query_points, memory['memory_query_feat'],
        memory['memory_points_metric'], memory['memory_conf'],
        memory['memory_age'], memory['memory_valid'])

    assert readout.shape == query_feat.shape
    assert diag['attention_shape'] == (1, 2, 1, 2)
    assert diag['candidate_count'].item() == 2
    assert diag['topk_candidate_count'].item() == 2
    assert diag['has_candidate'].item() is True


def test_age_and_radius_can_mask_every_candidate_without_nan():
    attn = qm.CausalQueryMemoryAttention(
        embed_dims=8, num_heads=2, spatial_radius=1.0, topk=2,
        max_age=3.0, dropout=0.0)
    query_feat = torch.randn(1, 2, 8)
    query_points = torch.zeros(1, 2, 2, 3)
    memory = make_memory(C=8, R=2)
    memory['memory_points_metric'].fill_(10.0)
    memory['memory_age'].fill_(0.0)

    readout, diag = attn(
        query_feat, query_points, memory['memory_query_feat'],
        memory['memory_points_metric'], memory['memory_conf'],
        memory['memory_age'], memory['memory_valid'])

    assert torch.isfinite(readout).all()
    assert torch.equal(readout, torch.zeros_like(readout))
    assert not diag['has_candidate'].any()


def test_stac_all_invalid_and_empty_memory_are_exact_identity():
    stac = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=1.0,
        topk=2, max_age=3.0, dropout=0.0)
    query_feat = torch.randn(1, 3, 8)
    query_points = torch.zeros(1, 3, 2, 3)
    query_conf = torch.ones(1, 3)
    memory = make_memory(C=8, R=2)
    memory['memory_points_metric'].fill_(100.0)

    fused, diag = stac(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=torch.eye(4).unsqueeze(0))
    assert torch.equal(fused, query_feat)
    assert not diag['has_candidate'].any()

    empty_fused, _ = stac(
        query_feat, query_points, query_conf, memory=None,
        target_ego2global=torch.eye(4).unsqueeze(0))
    assert torch.equal(empty_fused, query_feat)


def test_layerscale_fusion_has_live_first_step_gradients():
    torch.manual_seed(7)
    stac = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=4, max_age=8.0, dropout=0.0,
        fusion_gate_bias=-1.0, fusion_alpha_init=1e-3,
        fusion_out_proj_gain=0.1)
    query_feat = torch.randn(1, 2, 8)
    memory = make_memory(B=1, K=1, M=2, C=8, R=2)
    fused, diagnostics = stac(
        query_feat, torch.zeros(1, 2, 2, 3), torch.ones(1, 2),
        memory=memory, target_ego2global=torch.eye(4).unsqueeze(0))
    fused.square().mean().backward()

    for prefix in (
            'attention.q_proj.', 'attention.k_proj.', 'attention.v_proj.',
            'fusion.out_proj.', 'fusion.gate_mlp.'):
        grads = [param.grad for name, param in stac.named_parameters()
                 if name.startswith(prefix)]
        assert grads and any(
            grad is not None and torch.count_nonzero(grad).item() > 0
            for grad in grads)
    assert stac.fusion.alpha.grad is not None
    assert torch.count_nonzero(stac.fusion.alpha.grad).item() > 0
    assert float(diagnostics['conditional_gate']) > 0.0
    assert float(diagnostics['candidate_ratio']) == 1.0


def test_zero_layerscale_is_strict_identity_but_alpha_is_the_only_path():
    stac = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=4, max_age=8.0, dropout=0.0,
        fusion_gate_bias=-1.0, fusion_alpha_init=0.0,
        fusion_out_proj_gain=0.1)
    query_feat = torch.randn(1, 1, 8)
    memory = make_memory(B=1, K=1, M=1, C=8, R=2)
    fused, _ = stac(
        query_feat, torch.zeros(1, 1, 2, 3), torch.ones(1, 1),
        memory=memory, target_ego2global=torch.eye(4).unsqueeze(0))
    assert torch.equal(fused, query_feat)
    fused.square().mean().backward()
    assert stac.fusion.alpha.grad is not None
    for prefix in ('attention.q_proj.', 'attention.k_proj.',
                   'attention.v_proj.', 'fusion.gate_mlp.',
                   'fusion.out_proj.'):
        grads = [param.grad for name, param in stac.named_parameters()
                 if name.startswith(prefix)]
        assert all(grad is None or torch.count_nonzero(grad).item() == 0
                   for grad in grads)


def test_enabled_false_is_baseline_identity_without_memory_requirements():
    stac = qm.STACQueryMemory(enabled=False, embed_dims=8, num_heads=2)
    query_feat = torch.randn(1, 2, 8)
    fused, diag = stac(
        query_feat, torch.zeros(1, 2, 1, 3), torch.ones(1, 2),
        memory={'memory_valid': torch.ones(1, 1, 1, dtype=torch.bool)})
    assert torch.equal(fused, query_feat)
    assert diag['enabled'] is False


def test_online_bank_read_after_write_causality_and_scene_isolation():
    bank = qm.QueryMemoryBank(
        history_frames=2, max_queries_per_frame=3, write_threshold=0.0)
    feat = torch.randn(1, 4, 8)
    points = torch.zeros(1, 4, 2, 3)
    logits = torch.zeros(1, 4, 2, 3)
    ego = torch.eye(4).unsqueeze(0)

    assert bank.read('scene-a', 'sample-0', 0, 0.0) is None
    wrote = bank.write(
        feat, points, cls_scores=logits, ego2global=ego, timestamp=0.0,
        scene_id='scene-a', sample_idx='sample-0', frame_idx=0)
    assert wrote is True
    assert bank.read('scene-a', 'sample-0', 0, 0.0) is None
    assert bank.read('scene-b', 'sample-1', 1, 1.0) is None

    mem = bank.read(
        'scene-a', 'sample-1', 1, 1.0, device=torch.device('cpu'))
    assert mem is not None
    assert mem['memory_valid'].sum().item() == 3
    assert torch.all(mem['memory_age'][mem['memory_valid']] > 0)

    duplicate = bank.write(
        feat, points, cls_scores=logits, ego2global=ego, timestamp=0.0,
        scene_id='scene-a', sample_idx='sample-0', frame_idx=0)
    assert duplicate is False


def test_online_bank_rejects_batch_size_greater_than_one():
    bank = qm.QueryMemoryBank()
    with pytest.raises(RuntimeError, match='batch_size=1'):
        bank.write(
            torch.zeros(2, 1, 8), torch.zeros(2, 1, 1, 3),
            cls_scores=torch.zeros(2, 1, 1, 2),
            ego2global=torch.eye(4).repeat(2, 1, 1), timestamp=0.0)


def make_cache(path, sample_idx, scene_id, frame_idx, timestamp, M=2, C=4, R=2):
    cache = dict(
        schema_version=1,
        sample_idx=sample_idx,
        scene_id=scene_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        ego2global=torch.eye(4),
        query_feat=torch.arange(M * C, dtype=torch.float32).reshape(M, C),
        query_points_metric=torch.zeros(M, R, 3),
        query_conf=torch.tensor([0.9, 0.8])[:M],
        valid_mask=torch.ones(M, dtype=torch.bool),
        pc_range=[-1, -1, -1, 1, 1, 1],
        embed_dims=C,
        num_points=R,
        num_classes=3,
        source_config='cfg.py',
        source_checkpoint='model.pth')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def test_cache_loader_padding_and_non_strict_missing(tmp_path):
    root = tmp_path / 'cache'
    make_cache(root / 'scene-a' / 'sample-0.pt', 'sample-0', 'scene-a', 0, 0.0)
    loader = loader_mod.LoadQueryMemoryFromFiles(
        cache_root=str(root), history_frames=2, max_queries_per_frame=3,
        strict=False, embed_dims=4, num_points=2)
    results = dict(
        sample_idx='sample-2', scene_token='scene-a', frame_idx=2,
        timestamp=2.0,
        query_memory_history_infos=[
            dict(sample_idx='missing', scene_id='scene-a', frame_idx=0,
                 timestamp=0.0),
            dict(sample_idx='sample-0', scene_id='scene-a', frame_idx=0,
                 timestamp=0.0),
        ])

    out = loader(results)
    assert out['memory_query_feat'].shape == (2, 3, 4)
    assert out['memory_points_metric'].shape == (2, 3, 2, 3)
    assert out['memory_valid'][0].sum().item() == 0
    assert out['memory_valid'][1].sum().item() == 2
    assert torch.equal(out['memory_age'][1, :2], torch.tensor([2.0, 2.0]))


def test_cache_loader_strict_missing_reports_path(tmp_path):
    loader = loader_mod.LoadQueryMemoryFromFiles(
        cache_root=str(tmp_path), history_frames=1, strict=True,
        embed_dims=4, num_points=2)
    results = dict(
        sample_idx='sample-1', scene_name='scene-a', frame_idx=1,
        timestamp=1.0,
        query_memory_history_infos=[
            dict(sample_idx='missing', scene_id='scene-a', frame_idx=0,
                 timestamp=0.0)
        ])
    with pytest.raises(FileNotFoundError, match='missing'):
        loader(results)


def test_configuration_errors_are_clear():
    with pytest.raises(ValueError, match='embed_dims % num_heads'):
        qm.CausalQueryMemoryAttention(embed_dims=7, num_heads=2)
    with pytest.raises(ValueError, match='spatial_radius'):
        qm.CausalQueryMemoryAttention(
            embed_dims=8, num_heads=2, spatial_radius=0.0)


# ---------------------------------------------------------------------------
# Problem 4 - per-query semantic reliability (schema v2)
# ---------------------------------------------------------------------------
def test_compute_query_reliability_keys_range_and_empty_label():
    logits = torch.tensor([[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])  # [1, R, C]
    rel = qm.compute_query_reliability(logits)
    for key in ('query_semantic_distribution', 'query_label', 'query_margin',
                'query_entropy', 'query_reliability'):
        assert key in rel
    assert rel['query_reliability'].shape == (1,)
    assert 0.0 <= float(rel['query_reliability'][0]) <= 1.0
    assert int(rel['query_label'][0]) == 0

    # empty query (R == 0) must produce label -1 and zero reliability, no NaN.
    empty = qm.compute_query_reliability(torch.zeros(1, 0, 3))
    assert int(empty['query_label'][0]) == -1
    assert float(empty['query_reliability'][0]) == 0.0
    assert torch.isfinite(empty['query_reliability']).all()


def test_compute_query_reliability_peaked_beats_uniform():
    peaked = torch.full((1, 2, 4), 0.0)
    peaked[..., 0] = 10.0                       # sharply concentrated
    uniform = torch.zeros(1, 2, 4)              # maximally ambiguous
    logits = torch.cat([peaked, uniform], dim=0)  # [2, R, C]
    rel = qm.compute_query_reliability(logits)
    assert float(rel['query_reliability'][0]) > float(rel['query_reliability'][1])
    assert float(rel['query_margin'][0]) > float(rel['query_margin'][1])
    assert (rel['query_reliability'] >= 0).all()
    assert (rel['query_reliability'] <= 1).all()


# ---------------------------------------------------------------------------
# Problem 2 - future-aware effective age
# ---------------------------------------------------------------------------
def test_compute_effective_age_obs_and_scheduled_offsets():
    base = torch.tensor([1.0, 2.0, 3.0])
    # observation query keeps its base age (offset 0).
    assert torch.equal(qm.compute_effective_age(base, 0.0), base)
    # scheduled future group k uses (k+1) * frame_interval.
    frame_interval = 0.5
    for k in range(3):
        offset = (k + 1) * frame_interval
        got = qm.compute_effective_age(base, offset)
        assert torch.allclose(got, base + offset)
    # scalar base is promoted to a tensor.
    assert torch.allclose(
        qm.compute_effective_age(2.0, 0.5), torch.tensor(2.5))


# ---------------------------------------------------------------------------
# Problem 5 - deterministic reliability + class + spatial diversity selection
# ---------------------------------------------------------------------------
def _spread_points(x_centers, R=2):
    pts = torch.zeros(len(x_centers), R, 3)
    for i, x in enumerate(x_centers):
        pts[i, :, 0] = float(x)
    return pts


def test_select_diverse_deterministic_and_reliability_ordered():
    reliability = torch.tensor([0.1, 0.9, 0.5, 0.7])
    points = _spread_points([0.0, 100.0, 200.0, 300.0])  # all distinct cells
    idx_a = qm.select_diverse_memory_queries(
        points, reliability, valid=None, labels=None, max_queries=4,
        spatial_cell_size=4.0, max_per_spatial_cell=16, max_per_class=64)
    idx_b = qm.select_diverse_memory_queries(
        points, reliability, valid=None, labels=None, max_queries=4,
        spatial_cell_size=4.0, max_per_spatial_cell=16, max_per_class=64)
    # deterministic + reliability-descending when every query is a novel cell.
    assert torch.equal(idx_a, idx_b)
    assert idx_a.tolist() == [1, 3, 2, 0]


def test_select_diverse_spatial_and_class_caps():
    reliability = torch.tensor([0.9, 0.8, 0.7, 0.6])
    # (a) spatial cap: all queries in one cell, cap of 2 -> only 2 survive.
    same_cell = _spread_points([0.0, 0.0, 0.0, 0.0])
    sel = qm.select_diverse_memory_queries(
        same_cell, reliability, labels=None, max_queries=10,
        spatial_cell_size=4.0, max_per_spatial_cell=2, max_per_class=64)
    assert sel.numel() == 2
    assert sel.tolist() == [0, 1]  # top-2 reliability
    # (b) class cap: distinct cells but one class, cap of 2 -> only 2 survive.
    distinct = _spread_points([0.0, 100.0, 200.0, 300.0])
    labels = torch.zeros(4, dtype=torch.long)
    sel_c = qm.select_diverse_memory_queries(
        distinct, reliability, labels=labels, max_queries=10,
        spatial_cell_size=4.0, max_per_spatial_cell=16, max_per_class=2)
    assert sel_c.numel() == 2
    assert sel_c.tolist() == [0, 1]


def test_select_diverse_v1_unknown_labels_degrades():
    # schema-v1 fallback: all labels == -1 must DISABLE the class constraint.
    reliability = torch.tensor([0.9, 0.8, 0.7])
    distinct = _spread_points([0.0, 100.0, 200.0])
    labels = torch.full((3,), -1, dtype=torch.long)
    sel = qm.select_diverse_memory_queries(
        distinct, reliability, labels=labels, max_queries=10,
        spatial_cell_size=4.0, max_per_spatial_cell=16, max_per_class=1)
    assert sel.numel() == 3  # class cap ignored because labels are unknown
    # min_reliability gate drops low-reliability queries.
    gated = qm.select_diverse_memory_queries(
        distinct, reliability, labels=labels, max_queries=10,
        min_reliability=0.75, spatial_cell_size=4.0,
        max_per_spatial_cell=16, max_per_class=64)
    assert gated.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Problem 3 - zero-initialized motion compensator
# ---------------------------------------------------------------------------
def test_motion_compensator_zero_init_is_noop():
    comp = qm.QueryMotionCompensator(embed_dims=8, max_velocity=20.0, max_age=8.0)
    feat = torch.randn(1, 1, 3, 8)
    age = torch.full((1, 1, 3), 2.0)
    velocity = comp(feat, age)
    assert velocity.shape == (1, 1, 3, 3)
    # last layer is zero-initialized -> exact no-op until trained.
    assert torch.equal(velocity, torch.zeros_like(velocity))


# ---------------------------------------------------------------------------
# Problem 4 - reliability drives the attention score
# ---------------------------------------------------------------------------
def test_attention_prefers_higher_reliability_memory():
    attn = qm.CausalQueryMemoryAttention(
        embed_dims=8, num_heads=2, spatial_radius=50.0, topk=8,
        max_age=10.0, dropout=0.0)
    query_feat = torch.randn(1, 1, 8)
    query_points = torch.zeros(1, 1, 2, 3)
    shared = torch.randn(8)
    mem_feat = shared.view(1, 1, 1, 8).repeat(1, 1, 2, 1)  # identical k/v
    mem_points = torch.zeros(1, 1, 2, 2, 3)                # co-located, in range
    mem_conf = torch.ones(1, 1, 2)
    mem_age = torch.ones(1, 1, 2)
    mem_valid = torch.ones(1, 1, 2, dtype=torch.bool)
    mem_rel = torch.tensor([[[0.99, 0.01]]])
    _, diag = attn(
        query_feat, query_points, mem_feat, mem_points, mem_conf, mem_age,
        mem_valid, memory_reliability=mem_rel)
    assert diag['has_candidate'].any()
    # identical features => only the reliability term differentiates the two;
    # weight collapses onto the high-reliability candidate.
    assert float(diag['support_reliability'][0, 0]) > 0.5
