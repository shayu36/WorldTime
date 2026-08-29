"""STAC-QM modeling-repair integration tests (synthetic, CPU only).

These exercise the end-to-end wiring of the modeling-structure repairs
(Problems 1-6) across the online bank, the cache loader, and the STAC-QM
module -- WITHOUT any dataset, checkpoint, GPU, or training. Everything runs on
hand-built tensors so the tests are deterministic and fast.

Documented as NOT run here (require the full dataset / GPU / checkpoints and are
therefore out of scope for this modeling-structure-only phase):
  * tools/query_memory/precompute_query_memory.py end-to-end cache generation
    (needs nuScenes + a trained checkpoint).
  * A real nuScenes forward pass through SparseWorld4DTraj.forward_backbone
    (needs the dataset and the base checkpoint).
  * Any metric / mIoU evaluation.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qm = load_module(
    'query_memory_under_test_integration',
    ROOT / 'mmdet3d/models/sparsedetectors/query_memory.py')
loader_mod = load_module(
    'loading_query_memory_under_test_integration',
    ROOT / 'mmdet3d/datasets/pipelines/loading_query_memory.py')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_cache(path, sample_idx, scene_id, frame_idx, timestamp,
               conf=(0.9, 0.8), reliability=None, label=None,
               schema_version=1, C=4, R=2):
    M = len(conf)
    num_classes = max(3, max(label) + 1) if label is not None else 3
    cache = dict(
        schema_version=schema_version,
        sample_idx=sample_idx,
        scene_id=scene_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        ego2global=torch.eye(4),
        query_feat=torch.arange(M * C, dtype=torch.float32).reshape(M, C),
        query_points_metric=torch.zeros(M, R, 3),
        query_conf=torch.tensor(list(conf), dtype=torch.float32),
        valid_mask=torch.ones(M, dtype=torch.bool),
        pc_range=[-1, -1, -1, 1, 1, 1],
        embed_dims=C,
        num_points=R,
        num_classes=num_classes,
        source_config='cfg.py',
        source_checkpoint='model.pth')
    if schema_version >= 2:
        rel = torch.tensor(list(reliability), dtype=torch.float32)
        labels = torch.tensor(list(label), dtype=torch.long)
        distribution = torch.zeros(M, cache['num_classes'])
        distribution[torch.arange(M), labels] = 1.0
        cache['query_semantic_distribution'] = distribution
        cache['query_label'] = labels
        cache['query_margin'] = rel.clone()
        cache['query_entropy'] = torch.zeros(M)
        cache['query_reliability'] = rel
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def _write_bank_frame(bank, scene, sample, frame, ts, Q=3, C=8, R=2):
    feat = torch.randn(1, Q, C)
    points = torch.zeros(1, Q, R, 3)
    logits = torch.zeros(1, Q, R, 4)
    ego = torch.eye(4).unsqueeze(0)
    return bank.write(
        feat, points, cls_scores=logits, ego2global=ego, timestamp=ts,
        scene_id=scene, sample_idx=sample, frame_idx=frame)


# ---------------------------------------------------------------------------
# Problem 6 - online bank target-age slot assignment
# ---------------------------------------------------------------------------
def test_online_bank_target_age_slot_assignment_and_tolerance():
    # targets 1s / 2s match written frames; 5s has no frame within tolerance.
    bank = qm.QueryMemoryBank(
        history_selection_mode='target_age',
        history_target_ages=[1.0, 2.0, 5.0],
        history_age_tolerance=0.35,
        retention_seconds=20.0,
        max_bank_entries=16,
        write_threshold=0.0,
        max_queries_per_frame=4)
    for frame, ts in enumerate([0.0, 1.0, 2.0]):
        assert _write_bank_frame(bank, 'scene-a', f'sample-{frame}', frame, ts)

    mem = bank.read('scene-a', 'sample-cur', 3, 3.0, device=torch.device('cpu'))
    assert mem is not None
    # 3 slots (one per target age); slot0<-age1 (frame2), slot1<-age2 (frame1),
    # slot2 (target 5s) unmatched -> stays fully invalid.
    assert mem['memory_valid'].shape[1] == 3
    assert mem['memory_valid'][0, 0].sum().item() == 3
    assert mem['memory_valid'][0, 1].sum().item() == 3
    assert mem['memory_valid'][0, 2].sum().item() == 0
    # ages land on their target slots.
    assert abs(float(mem['memory_age'][0, 0].max()) - 1.0) < 1e-5
    assert abs(float(mem['memory_age'][0, 1].max()) - 2.0) < 1e-5
    # schema-v2 outputs are present.
    assert 'memory_reliability' in mem and 'memory_label' in mem


def test_online_bank_target_age_all_out_of_tolerance_returns_none():
    bank = qm.QueryMemoryBank(
        history_selection_mode='target_age',
        history_target_ages=[1.0, 2.0, 3.0],
        history_age_tolerance=0.2,
        retention_seconds=20.0,
        write_threshold=0.0,
        max_queries_per_frame=4)
    for frame, ts in enumerate([0.0, 1.0, 2.0]):
        _write_bank_frame(bank, 'scene-a', f'sample-{frame}', frame, ts)
    # read at 3.4s -> ages 3.4/2.4/1.4, none within 0.2 of any target.
    mem = bank.read('scene-a', 'sample-cur', 3, 3.4, device=torch.device('cpu'))
    assert mem is None


def test_dataset_target_age_uses_only_cacheable_split_frames():
    from mmdet3d.datasets.nuscenes_dataset_occ_trajectory import (
        NuScenesDatasetOccpancy4DTraj)

    dataset = NuScenesDatasetOccpancy4DTraj.__new__(
        NuScenesDatasetOccpancy4DTraj)
    dataset.data_infos = [
        dict(token=f'sample-{index}', scene_token='scene-a',
             scene_name='scene-a', frame_idx=index,
             timestamp=index * 1_000_000)
        for index in range(5)
    ]
    dataset.query_memory_history_frames = 2
    dataset.query_memory_history_selection_mode = 'target_age'
    dataset.query_memory_history_target_ages = [1.0, 3.0]
    dataset.query_memory_history_age_tolerance = 0.1
    dataset._query_memory_cacheable_indices = {1, 3, 4}

    histories = dataset._build_query_memory_history_infos(4)
    assert [item['sample_idx'] for item in histories] == [
        'sample-3', 'sample-1']
    assert [item['slot_index'] for item in histories] == [0, 1]
    assert len({item['sample_idx'] for item in histories}) == 2


# ---------------------------------------------------------------------------
# Problem 5/6 - cache loader target-age slots + schema versions
# ---------------------------------------------------------------------------
def test_loader_target_age_slot_placement_schema_v2(tmp_path):
    root = tmp_path / 'cache'
    make_cache(root / 'scene-a' / 'sample-0.pt', 'sample-0', 'scene-a', 0, 4.0,
               conf=(0.9, 0.8), reliability=(0.9, 0.8), label=(0, 1),
               schema_version=2)
    make_cache(root / 'scene-a' / 'sample-1.pt', 'sample-1', 'scene-a', 1, 2.0,
               conf=(0.7, 0.6), reliability=(0.7, 0.6), label=(2, 3),
               schema_version=2)
    loader = loader_mod.LoadQueryMemoryFromFiles(
        cache_root=str(root), history_selection_mode='target_age',
        history_target_ages=[1.0, 2.0, 3.0], max_queries_per_frame=3,
        strict=True, embed_dims=4, num_points=2)
    results = dict(
        sample_idx='sample-cur', scene_token='scene-a', frame_idx=5,
        timestamp=5.0,
        query_memory_history_infos=[
            dict(sample_idx='sample-0', scene_id='scene-a', frame_idx=0,
                 timestamp=4.0, slot_index=0, target_age=1.0),
            dict(sample_idx='sample-1', scene_id='scene-a', frame_idx=1,
                 timestamp=2.0, slot_index=2, target_age=3.0),
        ])

    out = loader(results)
    assert out['memory_query_feat'].shape == (3, 3, 4)
    assert out['memory_valid'][0].sum().item() == 2   # slot 0 filled
    assert out['memory_valid'][1].sum().item() == 0   # slot 1 empty
    assert out['memory_valid'][2].sum().item() == 2   # slot 2 filled
    # schema-v2 labels survive selection (reliability-descending order).
    assert out['memory_label'][0, :2].tolist() == [0, 1]
    assert out['memory_label'][2, :2].tolist() == [2, 3]
    # base_age = current_ts - history_ts.
    assert abs(float(out['memory_age'][0, 0]) - 1.0) < 1e-5
    assert abs(float(out['memory_age'][2, 0]) - 3.0) < 1e-5


def test_loader_schema_v1_reliability_label_fallback(tmp_path):
    root = tmp_path / 'cache'
    make_cache(root / 'scene-a' / 'sample-0.pt', 'sample-0', 'scene-a', 0, 4.0,
               conf=(0.9, 0.8), schema_version=1)
    loader = loader_mod.LoadQueryMemoryFromFiles(
        cache_root=str(root), history_frames=3, max_queries_per_frame=3,
        strict=True, embed_dims=4, num_points=2)
    results = dict(
        sample_idx='sample-cur', scene_token='scene-a', frame_idx=5,
        timestamp=5.0,
        query_memory_history_infos=[
            dict(sample_idx='sample-0', scene_id='scene-a', frame_idx=0,
                 timestamp=4.0),
        ])
    out = loader(results)
    # recent mode right-aligns the single frame into the last slot.
    slot = out['memory_valid'].shape[0] - 1
    assert out['memory_valid'][slot].sum().item() == 2
    # v1 has no reliability/label -> reliability degrades to conf, label to -1.
    assert torch.allclose(
        out['memory_reliability'][slot, :2], torch.tensor([0.9, 0.8]))
    assert out['memory_label'][slot, :2].tolist() == [-1, -1]


def test_loader_rejects_incomplete_schema_v2(tmp_path):
    root = tmp_path / 'cache'
    path = root / 'scene-a' / 'sample-0.pt'
    make_cache(
        path, 'sample-0', 'scene-a', 0, 4.0, conf=(0.9, 0.8),
        reliability=(0.9, 0.8), label=(0, 1), schema_version=2)
    cache = torch.load(path, map_location='cpu')
    cache.pop('query_entropy')
    torch.save(cache, path)
    loader = loader_mod.LoadQueryMemoryFromFiles(
        cache_root=str(root), history_frames=1, max_queries_per_frame=3,
        strict=True, embed_dims=4, num_points=2)
    results = dict(
        sample_idx='sample-cur', scene_token='scene-a', frame_idx=5,
        timestamp=5.0,
        query_memory_history_infos=[
            dict(sample_idx='sample-0', scene_id='scene-a', frame_idx=0,
                 timestamp=4.0),
        ])
    with pytest.raises(KeyError, match='query_entropy'):
        loader(results)


# ---------------------------------------------------------------------------
# Problems 1/2/3 - STAC forward: future-aware age + causal filter + motion
# ---------------------------------------------------------------------------
def _single_candidate_memory(base_age, C=8, R=2):
    return dict(
        memory_query_feat=torch.randn(1, 1, 1, C),
        memory_points_metric=torch.zeros(1, 1, 1, R, 3),
        memory_conf=torch.full((1, 1, 1), 0.9),
        memory_reliability=torch.full((1, 1, 1), 0.9),
        memory_valid=torch.ones(1, 1, 1, dtype=torch.bool),
        memory_source_ego2global=torch.eye(4).repeat(1, 1, 1, 1),
        memory_age=torch.full((1, 1, 1), float(base_age)))


def test_future_offset_effective_age_and_causal_filter():
    stac = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=8, max_age=3.0, dropout=0.0)
    query_feat = torch.randn(1, 1, 8)
    query_points = torch.zeros(1, 1, 2, 3)
    query_conf = torch.ones(1, 1)
    target = torch.eye(4).unsqueeze(0)
    memory = _single_candidate_memory(base_age=2.9)

    fused0, diag0 = stac(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target, future_offset=0.0)
    fused1, diag1 = stac(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target, future_offset=0.5)

    # base age unchanged; effective age shifts by exactly the future offset.
    assert torch.allclose(diag0['base_age'], diag1['base_age'])
    assert abs(float(diag1['effective_age_mean'] -
                     diag0['effective_age_mean']) - 0.5) < 1e-5
    # 2.9s is within max_age=3s; 2.9+0.5=3.4s exceeds it -> candidate filtered.
    assert bool(diag0['has_candidate'].any())
    assert not bool(diag1['has_candidate'].any())
    # zero-init fusion => exact identity at initialization either way.
    assert torch.equal(fused0, query_feat)
    assert torch.equal(fused1, query_feat)


def test_empty_memory_training_identity_has_zero_gradient_graph():
    stac = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=8, max_age=8.0, dropout=0.0)
    stac.train()
    query_feat = torch.randn(1, 2, 8)
    memory = _single_candidate_memory(base_age=2.0)
    memory['memory_valid'].zero_()

    fused, diagnostics = stac(
        query_feat, torch.zeros(1, 2, 2, 3), torch.ones(1, 2),
        memory=memory, target_ego2global=torch.eye(4).unsqueeze(0))

    assert torch.equal(fused, query_feat)
    assert not bool(diagnostics['has_candidate'].any())
    loss = fused.square().mean()
    assert loss.requires_grad
    loss.backward()
    grads = [param.grad for param in stac.parameters() if param.requires_grad]
    assert grads and all(grad is not None for grad in grads)
    assert all(torch.count_nonzero(grad).item() == 0 for grad in grads)


def test_motion_compensation_toggle_and_trained_shift():
    query_feat = torch.randn(1, 1, 8)
    query_points = torch.zeros(1, 1, 2, 3)
    query_conf = torch.ones(1, 1)
    target = torch.eye(4).unsqueeze(0)
    memory = _single_candidate_memory(base_age=2.0)

    # motion ON but zero-init -> no velocity, no residual.
    stac_on = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=8, max_age=8.0, dropout=0.0, motion_compensation=True)
    _, diag_init = stac_on(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target)
    assert float(diag_init['motion_velocity_norm']) == 0.0
    assert float(diag_init['motion_residual_mean']) == 0.0

    # after "training" the last layer, velocity + residual become non-zero.
    with torch.no_grad():
        stac_on.motion_compensator.mlp[-1].weight.fill_(0.05)
        stac_on.motion_compensator.mlp[-1].bias.fill_(0.05)
    _, diag_trained = stac_on(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target)
    assert float(diag_trained['motion_velocity_norm']) > 0.0
    assert float(diag_trained['motion_residual_mean']) > 0.0

    # motion OFF -> module emits no motion diagnostics at all.
    stac_off = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=8, max_age=8.0, dropout=0.0, motion_compensation=False)
    _, diag_off = stac_off(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target)
    assert 'motion_velocity_norm' not in diag_off


# ---------------------------------------------------------------------------
# Baseline compatibility - disabled STAC is a tensor-level identity
# ---------------------------------------------------------------------------
def test_disabled_stac_is_tensor_identity_regardless_of_memory():
    stac = qm.STACQueryMemory(enabled=False, embed_dims=8, num_heads=2)
    query_feat = torch.randn(1, 5, 8)
    memory = _single_candidate_memory(base_age=1.0)
    fused, diag = stac(
        query_feat, torch.zeros(1, 5, 2, 3), torch.ones(1, 5),
        memory=memory, target_ego2global=torch.eye(4).unsqueeze(0),
        future_offset=0.5)
    assert torch.equal(fused, query_feat)  # byte-for-byte identity
    assert diag['enabled'] is False


# ---------------------------------------------------------------------------
# Real SparseWorld4DTraj control-flow integration (synthetic CPU fakes)
# ---------------------------------------------------------------------------
class _ShapeModule(nn.Module):
    def __init__(self, out_dims):
        super().__init__()
        self.out_dims = int(out_dims)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, tensor):
        return tensor.new_zeros(*tensor.shape[:-1], self.out_dims) + \
            self.anchor * 0


class _FakeEgoCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, query_pos, query_feat, memory_pos, memory_feat):
        del query_pos, memory_pos, memory_feat
        return query_feat.new_zeros(query_feat.shape) + self.anchor * 0, None


class _ForwardHead(nn.Module):
    def __init__(self, ind_stamps_all):
        super().__init__()
        self.register_buffer('ind_stamps_all', ind_stamps_all)
        self.points_scale = None


class _FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ind_mask = None


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeSelfAttention()


class _FakeTemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.register_buffer('num_stamps_all', torch.tensor([
            [10, 1], [9, 2], [1, 10]], dtype=torch.long))
        self.pretrain = True
        self.freeze_tass_state = False
        decoder = nn.Module()
        decoder.decoder_layers = nn.ModuleList([_FakeDecoderLayer()])
        self.transformer = nn.Module()
        self.transformer.decoder = decoder

    def reset_mask(self):
        mask = torch.zeros(3, 3)
        for stamp in range(2):
            rows = (self.ind_stamps_all == stamp).nonzero(as_tuple=True)[0]
            cols = (self.ind_stamps_all > stamp).nonzero(as_tuple=True)[0]
            if rows.numel() and cols.numel():
                grid_row, grid_col = torch.meshgrid(rows, cols, indexing='ij')
                mask[grid_row, grid_col] = -1e5
        self.transformer.decoder.decoder_layers[0].self_attn.ind_mask = mask


def _make_sparseworld_shell(memory_finetune_mode=True,
                            memory_joint_finetune_mode=False,
                            memory_phase2_finetune_mode=False,
                            memory_phase3_finetune_mode=False):
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import \
        SparseWorld4DTraj

    model = SparseWorld4DTraj.__new__(SparseWorld4DTraj)
    nn.Module.__init__(model)
    model.query_memory_enabled = True
    model.memory_finetune_mode = memory_finetune_mode
    model.memory_joint_finetune_mode = memory_joint_finetune_mode
    model.memory_phase2_finetune_mode = memory_phase2_finetune_mode
    model.memory_phase3_finetune_mode = memory_phase3_finetune_mode
    model.memory_conditioned_refiner_enabled = memory_phase3_finetune_mode
    model.query_memory_source = 'cache'
    model.query_memory_cfg = dict(
        freeze_base_model=True,
        memory_finetune_mode=memory_finetune_mode,
        memory_joint_finetune_mode=memory_joint_finetune_mode,
        memory_phase2_finetune_mode=memory_phase2_finetune_mode,
        memory_phase3_finetune_mode=memory_phase3_finetune_mode)
    model.query_memory = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1))
    model.img_backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1))
    model.img_neck = nn.Linear(4, 4)
    model.pts_bbox_head = _FakeTemporalHead()
    model.plan_head = nn.Linear(4, 4)
    model.points_scale_branch = nn.Linear(4, 3)
    model.traj_head = nn.Linear(4, 2)
    model.position_encoder = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1))
    model.reg_branch = nn.Linear(4, 3)
    model.vel_branch = nn.Linear(4, 2)
    model.cls_branch = nn.Linear(4, 17)
    model.ego_cross_attn = _FakeEgoCrossAttention()
    model.memory_refiner = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    model.memory_cls_branch = nn.Linear(4, 17)
    model.memory_reg_branch = nn.Linear(4, 3)
    model.memory_vel_branch = nn.Linear(4, 2)
    model.memory_horizon_embedding = nn.Embedding(2, 4)
    model.num_query = 2
    model.num_fu_query = [1]
    model.num_fu_frames = 1
    model.pretrain = True
    model.frozen_num_stamps_all = None
    model.frozen_ind_stamps_all = None
    model._frozen_rap_masks = None
    model._configure_query_memory_trainability()
    return model


def test_memory_finetune_trainability_optimizer_and_module_modes():
    model = _make_sparseworld_shell()
    model._freeze_memory_finetune_temporal_state()
    optimizer = torch.optim.Adam(model.query_memory.parameters(), lr=1e-4)
    summary = model.validate_query_memory_training_setup(optimizer=optimizer)
    bad_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with pytest.raises(RuntimeError, match='frozen or unexpected'):
        model.validate_query_memory_training_setup(optimizer=bad_optimizer)
    assert summary['trainable_names']
    assert all(name.startswith('query_memory.')
               for name in summary['trainable_names'])
    assert all(not param.requires_grad for param in model.img_backbone.parameters())

    model.train()
    assert model.training is True
    assert model.query_memory.training is True
    assert model.img_backbone.training is False
    assert model.img_neck.training is False
    assert model.pts_bbox_head.training is False
    model.eval()
    assert model.training is False
    assert model.query_memory.training is False


@pytest.mark.parametrize(
    'memory_finetune_mode,memory_joint_finetune_mode,'
    'memory_phase2_finetune_mode',
    [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ])
def test_memory_tuning_modes_are_mutually_exclusive(
        memory_finetune_mode, memory_joint_finetune_mode,
        memory_phase2_finetune_mode):
    with pytest.raises(ValueError, match='mutually exclusive'):
        _make_sparseworld_shell(
            memory_finetune_mode=memory_finetune_mode,
            memory_joint_finetune_mode=memory_joint_finetune_mode,
            memory_phase2_finetune_mode=memory_phase2_finetune_mode)


def test_memory_joint_trainability_optimizer_lrs_and_module_modes():
    from mmcv.runner import build_optimizer
    from mmdet3d.core.optimizer import TrainableOnlyOptimizerConstructor  # noqa

    model = _make_sparseworld_shell(
        memory_finetune_mode=False,
        memory_joint_finetune_mode=True)
    model._freeze_memory_finetune_temporal_state()
    optimizer_cfg = dict(
        type='AdamW',
        constructor='TrainableOnlyOptimizerConstructor',
        lr=1e-5,
        weight_decay=1e-2,
        paramwise_cfg=dict(
            custom_keys={
                'query_memory': dict(lr_mult=5.0),
                'ego_cross_attn': dict(lr_mult=0.5),
            },
            bypass_duplicate=True))
    optimizer = build_optimizer(model, optimizer_cfg)
    summary = model.validate_query_memory_training_setup(
        optimizer=optimizer, optimizer_cfg=optimizer_cfg)
    bad_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    with pytest.raises(RuntimeError, match='frozen or unexpected'):
        model.validate_query_memory_training_setup(optimizer=bad_optimizer)

    expected_prefixes = model._MEMORY_JOINT_TRAINABLE_PREFIXES
    assert summary['trainable_names']
    assert all(any(name.startswith(prefix) for prefix in expected_prefixes)
               for name in summary['trainable_names'])
    for prefix in expected_prefixes:
        assert any(name.startswith(prefix)
                   for name in summary['trainable_names'])

    named_by_id = {id(param): name for name, param in model.named_parameters()}
    lrs = {}
    optimizer_ids = []
    for group in optimizer.param_groups:
        for param in group['params']:
            optimizer_ids.append(id(param))
            lrs[named_by_id[id(param)]] = group['lr']
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(lrs) == set(summary['trainable_names'])
    assert all(lr == pytest.approx(5e-5)
               for name, lr in lrs.items()
               if name.startswith('query_memory.'))
    assert all(lr == pytest.approx(5e-6)
               for name, lr in lrs.items()
               if name.startswith('ego_cross_attn.'))
    assert all(lr == pytest.approx(1e-5)
               for name, lr in lrs.items()
               if not name.startswith(('query_memory.', 'ego_cross_attn.')))

    # A resumed MMCV optimizer carries scheduler-adjusted current LRs.  Their
    # absolute values differ from the config while the initial LRs and the
    # relative paramwise multipliers remain valid.
    scheduler_scale = 0.9829799502313896
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
        group['lr'] *= scheduler_scale
    model.validate_query_memory_training_setup(
        optimizer=optimizer, optimizer_cfg=optimizer_cfg)

    optimizer.param_groups[0]['lr'] *= 0.5
    with pytest.raises(RuntimeError, match='inconsistent lr scheduler scales'):
        model.validate_query_memory_training_setup(
            optimizer=optimizer, optimizer_cfg=optimizer_cfg)

    for module_name in ('img_backbone', 'img_neck', 'pts_bbox_head',
                        'plan_head', 'points_scale_branch', 'traj_head'):
        assert all(not param.requires_grad
                   for param in getattr(model, module_name).parameters())

    model.train()
    assert model.training is True
    for module_name in model._MEMORY_JOINT_TRAIN_MODULES:
        assert getattr(model, module_name).training is True
    for module_name in ('img_backbone', 'img_neck', 'pts_bbox_head',
                        'plan_head', 'points_scale_branch', 'traj_head'):
        assert getattr(model, module_name).training is False

    invalid_cfg = dict(
        type='AdamW',
        constructor='TrainableOnlyOptimizerConstructor',
        lr=1e-5,
        weight_decay=1e-2,
        paramwise_cfg=dict(norm_decay_mult=0.0))
    with pytest.raises(ValueError, match='does not support'):
        build_optimizer(model, invalid_cfg)


def test_memory_phase2_trainability_optimizer_lrs_and_module_modes():
    from mmcv.runner import build_optimizer
    from mmdet3d.core.optimizer import TrainableOnlyOptimizerConstructor  # noqa

    model = _make_sparseworld_shell(
        memory_finetune_mode=False,
        memory_phase2_finetune_mode=True)
    model._freeze_memory_finetune_temporal_state()
    optimizer_cfg = dict(
        type='AdamW',
        constructor='TrainableOnlyOptimizerConstructor',
        lr=1e-5,
        weight_decay=1e-2,
        paramwise_cfg=dict(
            custom_keys={
                'query_memory': dict(lr_mult=5.0),
                'ego_cross_attn': dict(lr_mult=0.5),
                'pts_bbox_head': dict(lr_mult=0.5),
            },
            bypass_duplicate=True))
    optimizer = build_optimizer(model, optimizer_cfg)
    summary = model.validate_query_memory_training_setup(
        optimizer=optimizer, optimizer_cfg=optimizer_cfg)

    expected_prefixes = model._MEMORY_PHASE2_TRAINABLE_PREFIXES
    assert summary['trainable_names']
    assert all(any(name.startswith(prefix) for prefix in expected_prefixes)
               for name in summary['trainable_names'])
    for prefix in expected_prefixes:
        assert any(name.startswith(prefix)
                   for name in summary['trainable_names'])

    named_by_id = {id(param): name for name, param in model.named_parameters()}
    lrs = {}
    optimizer_ids = []
    for group in optimizer.param_groups:
        for param in group['params']:
            optimizer_ids.append(id(param))
            lrs[named_by_id[id(param)]] = group['lr']
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(lrs) == set(summary['trainable_names'])
    assert all(lr == pytest.approx(5e-5)
               for name, lr in lrs.items()
               if name.startswith('query_memory.'))
    assert all(lr == pytest.approx(5e-6)
               for name, lr in lrs.items()
               if name.startswith(('ego_cross_attn.', 'pts_bbox_head.')))
    assert all(lr == pytest.approx(1e-5)
               for name, lr in lrs.items()
               if not name.startswith((
                   'query_memory.', 'ego_cross_attn.', 'pts_bbox_head.')))

    for module_name in ('img_backbone', 'img_neck', 'plan_head',
                        'points_scale_branch', 'traj_head'):
        assert all(not param.requires_grad
                   for param in getattr(model, module_name).parameters())
    assert all(param.requires_grad
               for param in model.pts_bbox_head.parameters())

    model.train()
    assert model.training is True
    for module_name in model._MEMORY_PHASE2_TRAIN_MODULES:
        assert getattr(model, module_name).training is True
    for module_name in ('img_backbone', 'img_neck', 'plan_head',
                        'points_scale_branch', 'traj_head'):
        assert getattr(model, module_name).training is False

    bad_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    with pytest.raises(RuntimeError, match='frozen or unexpected'):
        model.validate_query_memory_training_setup(optimizer=bad_optimizer)


def test_memory_phase3_trainability_and_frozen_boundary():
    from mmcv.runner import build_optimizer

    model = _make_sparseworld_shell(
        memory_finetune_mode=False,
        memory_phase3_finetune_mode=True)
    model._freeze_memory_finetune_temporal_state()
    optimizer_cfg = dict(
        type='AdamW',
        constructor='TrainableOnlyOptimizerConstructor',
        lr=1e-5,
        weight_decay=1e-2,
        paramwise_cfg=dict(
            custom_keys={'query_memory': dict(lr_mult=5.0)},
            bypass_duplicate=True))
    optimizer = build_optimizer(model, optimizer_cfg)
    summary = model.validate_query_memory_training_setup(
        optimizer=optimizer, optimizer_cfg=optimizer_cfg)

    expected_prefixes = model._MEMORY_PHASE3_TRAINABLE_PREFIXES
    assert summary['trainable_names']
    assert all(any(name.startswith(prefix) for prefix in expected_prefixes)
               for name in summary['trainable_names'])
    for prefix in expected_prefixes:
        assert any(name.startswith(prefix)
                   for name in summary['trainable_names'])
    for module_name in model._MEMORY_PHASE3_TRAIN_MODULES:
        assert any(param.requires_grad
                   for param in getattr(model, module_name).parameters())
    for module_name in ('img_backbone', 'img_neck', 'pts_bbox_head',
                        'plan_head', 'points_scale_branch', 'traj_head',
                        'position_encoder', 'reg_branch', 'vel_branch',
                        'cls_branch', 'ego_cross_attn'):
        assert all(not param.requires_grad
                   for param in getattr(model, module_name).parameters())

    model.train()
    assert model.training is True
    for module_name in model._MEMORY_PHASE3_TRAIN_MODULES:
        assert getattr(model, module_name).training is True
    for module_name in ('img_backbone', 'img_neck', 'pts_bbox_head',
                        'plan_head', 'points_scale_branch', 'traj_head',
                        'position_encoder', 'reg_branch', 'vel_branch',
                        'cls_branch', 'ego_cross_attn'):
        assert getattr(model, module_name).training is False


@pytest.mark.parametrize(
    'memory_enabled,refiner_enabled,expected_prefixes',
    [
        (False, True, ('memory_refiner.', 'memory_cls_branch.',
                       'memory_reg_branch.', 'memory_vel_branch.',
                       'memory_horizon_embedding.')),
        (True, False, ('query_memory.',)),
        (True, True, ('query_memory.', 'memory_refiner.',
                      'memory_cls_branch.', 'memory_reg_branch.',
                      'memory_vel_branch.', 'memory_horizon_embedding.')),
    ])
def test_phase3_2x2_ablation_trainability_is_disjoint(
        memory_enabled, refiner_enabled, expected_prefixes):
    """Each non-baseline 2x2 cell optimizes exactly its enabled side(s)."""
    model = _make_sparseworld_shell(
        memory_finetune_mode=False, memory_phase3_finetune_mode=True)
    model.query_memory_enabled = memory_enabled
    if not memory_enabled:
        model.query_memory = None
    model.memory_conditioned_refiner_enabled = refiner_enabled
    model._configure_query_memory_trainability()
    trainable = [
        name for name, param in model.named_parameters()
        if param.requires_grad]
    assert trainable
    assert all(any(name.startswith(prefix) for prefix in expected_prefixes)
               for name in trainable)
    for prefix in expected_prefixes:
        assert any(name.startswith(prefix) for name in trainable)
    if not memory_enabled:
        assert all(not name.startswith('query_memory.') for name in trainable)
    if not refiner_enabled:
        assert all(not name.startswith('memory_refiner.') and
                   not name.startswith('memory_cls_branch.') and
                   not name.startswith('memory_reg_branch.') and
                   not name.startswith('memory_vel_branch.') and
                   not name.startswith('memory_horizon_embedding.')
                   for name in trainable)


def test_memory_conditioned_refiner_uses_encoded_point_refinement():
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import (
        MemoryConditionedRefiner, SparseWorld4DTraj)

    model = SparseWorld4DTraj.__new__(SparseWorld4DTraj)
    nn.Module.__init__(model)
    model.memory_conditioned_refiner_enabled = True
    model.num_refines = 2
    model.pc_range = torch.tensor([-4., -4., -1., 4., 4., 5.])
    model.memory_refiner = MemoryConditionedRefiner(8)
    model.memory_horizon_embedding = nn.Embedding(3, 8)
    model.memory_cls_branch = nn.Linear(8, 2 * 17)
    model.memory_reg_branch = nn.Linear(8, 2 * 3)
    model.memory_vel_branch = nn.Linear(8, 2 * 2)

    calls = []
    original_refine = SparseWorld4DTraj.refine_points.__get__(
        model, SparseWorld4DTraj)

    def record_refine(self, points, delta):
        calls.append((tuple(points.shape), tuple(delta.shape)))
        return original_refine(points, delta)

    model.refine_points = types.MethodType(record_refine, model)
    query_feat = torch.randn(1, 3, 8, requires_grad=True)
    query_pos = torch.rand(1, 3, 2, 3)
    base_cls = torch.zeros(1, 3, 2, 17)
    refined_feat, refined_cls, refined_pts, refined_vel, diagnostics = \
        model._apply_memory_conditioned_refiner(
            query_feat, query_pos, base_cls, horizon_index=2)

    assert refined_feat.shape == query_feat.shape
    assert refined_cls.shape == base_cls.shape
    assert refined_pts.shape == query_pos.shape
    assert refined_vel.shape == (1, 3, 2, 2)
    assert calls == [((1, 3, 2, 3), (1, 3, 6))]
    assert diagnostics['enabled'] is True
    (refined_feat.square().mean() + refined_cls.square().mean() +
     refined_pts.square().mean() + refined_vel.square().mean()).backward()
    assert any(param.grad is not None and
               torch.count_nonzero(param.grad).item() > 0
               for param in model.memory_refiner.parameters())
    assert any(param.grad is not None and
               torch.count_nonzero(param.grad).item() > 0
               for param in model.memory_cls_branch.parameters())
    assert any(param.grad is not None and
               torch.count_nonzero(param.grad).item() > 0
               for param in model.memory_reg_branch.parameters())
    assert any(param.grad is not None and
               torch.count_nonzero(param.grad).item() > 0
               for param in model.memory_vel_branch.parameters())


def test_simple_test_uses_refined_current_outputs_for_0s():
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import \
        SparseWorld4DTraj

    model = SparseWorld4DTraj.__new__(SparseWorld4DTraj)
    nn.Module.__init__(model)
    model.query_memory_enabled = False
    model.num_fu_frames = 1
    model.pts_bbox_head = nn.Module()
    model.pts_bbox_head.get_occ = types.MethodType(
        lambda self, pred_dict: torch.zeros(
            pred_dict['refine_pts'].shape[0], 2, 2, 2, dtype=torch.long),
        model.pts_bbox_head)
    raw_cls = torch.full((1, 2, 2, 17), -3.0)
    raw_pts = torch.zeros(1, 2, 2, 3)
    refined_cls = torch.full((1, 2, 2, 17), 4.0)
    refined_pts = torch.full((1, 2, 2, 3), 0.5)
    future_cls = torch.zeros(1, 2, 2, 17)
    future_pts = torch.zeros(1, 2, 2, 3)
    outputs = dict(
        cls_score=refined_cls,
        refine_pts=refined_pts,
        outs=dict(all_cls_scores=[raw_cls], all_refine_pts=[raw_pts]),
        forecast_semantics_list=[future_cls],
        forecast_points_list=[future_pts],
        pred_trajs_list=[torch.zeros(1, 1, 2)])
    model.forward_backbone = types.MethodType(
        lambda self, img, img_metas, **kwargs: outputs, model)

    seen = []
    original_get_occ = model.pts_bbox_head.get_occ

    def record_get_occ(self, pred_dict):
        seen.append(pred_dict)
        return original_get_occ(pred_dict)

    model.pts_bbox_head.get_occ = types.MethodType(
        record_get_occ, model.pts_bbox_head)
    model.eval()
    result = model.simple_test([{}], torch.zeros(1, 1), **{})
    assert result['semantic_occ_0s']
    assert seen[0]['cls_scores'] is refined_cls
    assert seen[0]['refine_pts'] is refined_pts
    assert seen[0]['cls_scores'] is not raw_cls
    assert seen[0]['refine_pts'] is not raw_pts


@pytest.mark.parametrize(
    'invalid_setup,error',
    [
        ('disabled', 'requires enabled STAC-QM'),
        ('online', 'requires source="cache"'),
        ('unfrozen', 'requires freeze_base_model=True'),
    ])
def test_memory_phase2_requires_safe_cache_finetune_setup(
        invalid_setup, error):
    model = _make_sparseworld_shell(
        memory_finetune_mode=False,
        memory_phase2_finetune_mode=True)
    if invalid_setup == 'disabled':
        model.query_memory_enabled = False
        model.query_memory = None
    elif invalid_setup == 'online':
        model.query_memory_source = 'online'
    else:
        model.query_memory_cfg['freeze_base_model'] = False

    with pytest.raises(ValueError, match=error):
        model._configure_query_memory_trainability()


def test_memory_joint_requires_strict_loader_tensors():
    model = _make_sparseworld_shell(
        memory_finetune_mode=False,
        memory_joint_finetune_mode=True)
    with pytest.raises(KeyError, match='strict loader'):
        model._query_memory_context(
            {}, [], torch.device('cpu'), torch.float32)


def test_memory_phase2_requires_strict_loader_tensors():
    model = _make_sparseworld_shell(
        memory_finetune_mode=False,
        memory_phase2_finetune_mode=True)
    with pytest.raises(KeyError, match='strict loader'):
        model._query_memory_context(
            {}, [], torch.device('cpu'), torch.float32)


def test_memory_finetune_requires_strict_loader_tensors():
    model = _make_sparseworld_shell()
    with pytest.raises(KeyError, match='strict loader'):
        model._query_memory_context(
            {}, [], torch.device('cpu'), torch.float32)


@pytest.mark.parametrize(
    'memory_finetune_mode,memory_joint_finetune_mode,'
    'memory_phase2_finetune_mode',
    [(True, False, False), (False, True, False), (False, False, True)])
def test_memory_tuning_tass_state_is_frozen_across_epochs(
        memory_finetune_mode, memory_joint_finetune_mode,
        memory_phase2_finetune_mode):
    model = _make_sparseworld_shell(
        memory_finetune_mode=memory_finetune_mode,
        memory_joint_finetune_mode=memory_joint_finetune_mode,
        memory_phase2_finetune_mode=memory_phase2_finetune_mode)
    model._freeze_memory_finetune_temporal_state()
    frozen_num = model.frozen_num_stamps_all.clone()
    frozen_ind = model.frozen_ind_stamps_all.clone()
    frozen_mask = model._frozen_rap_masks[0].clone()

    model.set_epoch(0)
    model.set_epoch(11)
    assert torch.equal(model.pts_bbox_head.num_stamps_all, frozen_num)
    assert torch.equal(model.pts_bbox_head.ind_stamps_all, frozen_ind)
    assert torch.equal(
        model.pts_bbox_head.transformer.decoder.decoder_layers[
            0].self_attn.ind_mask,
        frozen_mask)
    assert model.pretrain is False
    assert model.pts_bbox_head.pretrain is False
    assert model.pts_bbox_head.freeze_tass_state is True

    model.pts_bbox_head.num_stamps_all[0, 0] += 1
    with pytest.raises(RuntimeError, match='num_stamps_all changed'):
        model.set_epoch(12)


@pytest.mark.parametrize(
    'memory_finetune_mode,memory_joint_finetune_mode,'
    'memory_phase2_finetune_mode',
    [(True, False, False), (False, True, False), (False, False, True)])
def test_forward_backbone_runtime_reads_seven_groups_and_1040_queries(
        monkeypatch, memory_finetune_mode, memory_joint_finetune_mode,
        memory_phase2_finetune_mode):
    import mmdet3d.models.sparsedetectors.sparseworld_4d_traj as sw_mod
    SparseWorld4DTraj = sw_mod.SparseWorld4DTraj
    monkeypatch.setattr(sw_mod, 'device', torch.device('cpu'))

    model = SparseWorld4DTraj.__new__(SparseWorld4DTraj)
    nn.Module.__init__(model)
    counts = [720, 60, 60, 60, 60, 40, 40]
    ind_stamps = torch.cat([
        torch.full((count,), stamp, dtype=torch.long)
        for stamp, count in enumerate(counts)
    ])
    B, C, R = 1, 8, 2
    Q = int(ind_stamps.numel())
    outs = dict(
        query_feat=torch.zeros(B, Q, C),
        all_refine_pts=[torch.zeros(B, Q, R, 3)],
        all_cls_scores=[torch.zeros(B, Q, R, 17)])

    model.query_memory_enabled = True
    model.memory_finetune_mode = memory_finetune_mode
    model.memory_joint_finetune_mode = memory_joint_finetune_mode
    model.memory_phase2_finetune_mode = memory_phase2_finetune_mode
    model.query_memory_log_diagnostics = False
    model.query_memory_frame_interval = 0.5
    model.num_refines = R
    model.num_fu_frames = 6
    model.finetune_epoch = 5
    model.curr_epoch = 0
    model.pretrain = False
    model.pc_range = torch.tensor([-40., -40., -1., 40., 40., 5.4])
    model.pts_bbox_head = _ForwardHead(ind_stamps)
    model.plan_head = _ShapeModule(C)
    model.points_scale_branch = _ShapeModule(3)
    model.ego_cross_attn = _FakeEgoCrossAttention()
    model.traj_head = _ShapeModule(2)
    model.position_encoder = _ShapeModule(C)
    model.reg_branch = _ShapeModule(R * 3)
    model.cls_branch = _ShapeModule(R * 17)
    model.vel_branch = _ShapeModule(R * 2)

    model.extract_feat = types.MethodType(
        lambda self, img, img_metas: None, model)
    model.pts_bbox_head.forward = types.MethodType(
        lambda self, img_feats, img_metas: outs, model.pts_bbox_head)
    model._query_memory_context = types.MethodType(
        lambda self, kwargs, img_metas, device, dtype: object(), model)
    model.refine_points = types.MethodType(
        lambda self, points, delta: points, model)
    model.trans_points = types.MethodType(
        lambda self, points, delta, transform: points, model)

    calls = []

    def _record(self, feat, pos, cls, img_metas, memory, future_offset=0.0):
        del pos, cls, img_metas, memory
        calls.append((int(feat.shape[1]), float(future_offset)))
        return feat

    model._apply_query_memory_once = types.MethodType(_record, model)
    model.training = True
    img = torch.zeros(B, 1)
    img_metas = [dict(ego2lidar=torch.eye(4).numpy())]
    outputs = model.forward_backbone(
        img, img_metas,
        temporal_ego_states=[torch.zeros(B, 1, 4)],
        temporal_trajs=torch.zeros(B, 6, 2))

    assert [count for count, _ in calls] == counts
    assert [offset for _, offset in calls] == [
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert len(calls) == 7
    assert sum(count for count, _ in calls) == 1040
    assert len(outputs['forecast_semantics_list']) == 6
    # The first read is only the 720 observation queries; later reads are only
    # newly scheduled groups, never the growing active set.
    assert max(count for count, _ in calls[1:]) == 60
