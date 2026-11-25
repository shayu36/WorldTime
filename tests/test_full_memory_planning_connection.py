"""Pure forward checks, including the real OPUS planner attention implementation."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import BaseModule

from test_memory_planning_connection import fixture_model, CONNECTION as POSITION
from test_future_memory_adapter_v2 import QM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
SPEC = importlib.util.spec_from_file_location(
    'full_memory_planning_connection', ROOT / 'tools/full_memory_planning_connection.py')
FULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL)


def real_attention_class():
    """Compile unchanged real class bodies without importing optional CUDA ops."""
    namespace = dict(torch=torch, nn=nn, F=F, BaseModule=BaseModule,
                     warnings=warnings, DUMP=SimpleNamespace(enabled=False),
                     decode_points=QM.decode_points_metric)
    for filename, classname in (
            ('bbox/utils.py', 'MultiheadAttention'),
            ('opus_transformer.py', 'OPUSCrossAttention')):
        path = ROOT / 'mmdet3d/models/sparsedetectors' / filename
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, str(path), 'exec'), namespace)
    return namespace['OPUSCrossAttention']


class TriTaskMemory(nn.Module):
    def __init__(self, branch=None, disabled=False, target=None):
        super().__init__()
        self.semantic_query_proj = nn.Linear(17, 8)
        self.branch = branch
        self.disabled = disabled
        self.target = target

    def forward(self, query, points, logits, memory, horizon_id, **kwargs):
        gate = torch.ones_like(points[..., :1])
        if self.disabled or (self.target is not None and self.target != horizon_id):
            gate.zero_()
        if gate.shape[0] == 2:
            gate[1] = 0
        ds, do, dp = torch.zeros_like(logits), torch.zeros_like(gate), torch.zeros_like(points)
        if self.branch in (None, 'semantic'):
            ds[:, 0, :, 0] = 3.
        if self.branch in (None, 'occupancy'):
            do[:, 1, :, 0] = -3.
        if self.branch in (None, 'position'):
            dp[:, 0, :, 0] = 2.
        return dict(delta_s=ds, delta_o=do, delta_p=dp, gate=gate,
                    gate_sem=gate, gate_occ=gate, gate_pos=gate,
                    diagnostics={'has_candidate': gate.squeeze(-1).bool()})


def full_fixture(batch=1, points=2, branch=None, disabled=False, target=None):
    torch.manual_seed(17)
    model, args = fixture_model(batch=batch, points=points)
    model.ego_cross_attn = real_attention_class()(
        embed_dims=8, num_heads=2, dropout=0., pc_range=model.pc_range)
    model.traj_head = nn.Linear(8, 2)
    model.future_memory_adapter_v2 = TriTaskMemory(branch, disabled, target)
    model.eval()
    return model, args


@pytest.mark.parametrize('branch', FULL.BRANCHES)
@pytest.mark.parametrize('batch', [1, 2])
def test_each_branch_changes_real_planner_and_preserves_native_scene(branch, batch):
    model, args = full_fixture(batch=batch, points=3, branch=branch)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        native = model._forward_backbone_future_memory_adapter(*args)
        restore = FULL.install_full_planning_connection(model)
        connected = model._forward_backbone_future_memory_adapter(*args)
        restore()
        restored = model._forward_backbone_future_memory_adapter(*args)
    for name in ('baseline_forecast_points_list', 'baseline_forecast_semantics_list',
                 'forecast_points_list', 'forecast_semantics_list'):
        assert all(torch.equal(a, b) for a, b in zip(native[name], connected[name]))
    for step, (a, b) in enumerate(zip(native['pred_trajs_list'], connected['pred_trajs_list'])):
        assert torch.equal(a, restored['pred_trajs_list'][step])
        if step in FULL.TARGET_STEPS:
            assert not torch.equal(a[0], b[0]), branch
            if batch == 2:
                assert torch.equal(a[1], b[1])
        else:
            assert torch.equal(a, b)
    assert weights.keys() == model.state_dict().keys()
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    assert not model.cls_branch._forward_pre_hooks
    assert not model.ego_cross_attn.attention._forward_pre_hooks


@pytest.mark.parametrize('target', [0, 1, 2])
def test_full_connection_targets_the_same_horizon_including_3s(target):
    model, args = full_fixture(target=target)
    with torch.no_grad():
        native = model._forward_backbone_future_memory_adapter(*args)
        FULL.install_full_planning_connection(model, verify_branches=1)
        connected = model._forward_backbone_future_memory_adapter(*args)
    a, b = (torch.cat(x['pred_trajs_list'], 1) for x in (native, connected))
    assert (a != b).any(-1)[0].nonzero().flatten().tolist() == [2 * target + 1]
    assert all(v > 0 for v in model._full_memory_planning_audit['isolated_branch_increment_max'].values())


def test_no_memory_all_three_gates_zero_returns_native_exactly():
    model, args = full_fixture(batch=2, disabled=True)
    with torch.no_grad():
        native = model._forward_backbone_future_memory_adapter(*args)
        FULL.install_full_planning_connection(model)
        connected = model._forward_backbone_future_memory_adapter(*args)
    assert all(torch.equal(a, b) for a, b in zip(native['pred_trajs_list'], connected['pred_trajs_list']))


def test_zero_semantic_occupancy_is_exact_previous_position_connection():
    model, args = full_fixture(branch='position')
    with torch.no_grad():
        restore = POSITION.install_position_planning_connection(model)
        position = model._forward_backbone_future_memory_adapter(*args)
        restore()
        FULL.install_full_planning_connection(model)
        full = model._forward_backbone_future_memory_adapter(*args)
    assert all(torch.equal(a, b) for a, b in zip(position['pred_trajs_list'], full['pred_trajs_list']))


def scene_tensors(dtype=torch.float32):
    features = torch.ones(2, 3, 8, dtype=dtype)
    points = torch.full((2, 3, 2, 3), .5, dtype=dtype)
    logits = torch.zeros(2, 3, 2, 17, dtype=dtype)
    occupancy = logits.max(-1, keepdim=True).values
    projection = nn.Linear(17, 8)
    return features, points, logits, occupancy, projection


def test_full_scene_zero_residual_identity_and_semantic_bias_cancels():
    features, points, logits, occupancy, projection = scene_tensors()
    original = [v.clone() for v in (features, points, logits, occupancy)]
    with torch.no_grad():
        f, p, prior, changed, _ = FULL.full_scene_inputs(
            features, points, points, logits, logits, occupancy, projection)
        assert torch.equal(f, features) and torch.equal(p, points)
        assert torch.count_nonzero(prior) == 0 and not changed.any()
        semantic = logits.clone()
        semantic[:, 0, :, 2] = 4.
        a = FULL.full_scene_inputs(features, points, points, logits, semantic, occupancy, projection)
        projection.bias.fill_(10000.)
        b = FULL.full_scene_inputs(features, points, points, logits, semantic, occupancy, projection)
        expected = F.linear((semantic.softmax(-1)-logits.softmax(-1)).mean(-2), projection.weight)
    assert torch.equal(a[0], features + expected)
    assert torch.equal(a[0], b[0])
    assert all(torch.equal(a, b) for a, b in zip(original, (features, points, logits, occupancy)))


def test_occupancy_prior_ratio_is_correct_not_hard_mask():
    features, points, logits, occupancy, projection = scene_tensors()
    occupancy = occupancy.clone()
    occupancy[:, 1] = torch.log(torch.tensor(3.))  # sigmoid(log(3)) = .75, baseline .5
    with torch.no_grad():
        f, _, prior, _, _ = FULL.full_scene_inputs(
            features, points, points, logits, logits, occupancy, projection)
    assert torch.equal(f, features)
    assert torch.allclose(prior[:, 1], torch.full((2,), 1.5).log())
    assert torch.equal(prior[:, (0, 2)], torch.zeros(2, 2))
    assert torch.isfinite(prior).all()


def test_half_precision_extreme_logits_are_finite():
    features, points, logits, occupancy, projection = scene_tensors(torch.float16)
    logits.fill_(-1000.)
    semantic = logits.clone()
    semantic[..., 3] = 1000.
    occupancy.fill_(-1000.)
    occupancy[:, 1] = 1000.
    with torch.no_grad():
        f, _, prior, _, _ = FULL.full_scene_inputs(
            features, points, points, logits, semantic, occupancy, projection)
    assert torch.isfinite(f).all() and torch.isfinite(prior).all()


def test_attention_prior_batch_head_layout_matches_individual_samples():
    model, _ = full_fixture(batch=2)
    torch.manual_seed(10)
    ego, points, feat = torch.randn(2, 1, 8), torch.rand(2, 5, 3, 3), torch.randn(2, 5, 8)
    prior = torch.tensor([[0., 1., 2., -1., .5], [2., 0., -1., .3, -2.]])
    captured = []
    handle = model.ego_cross_attn.attention.attn.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs['attn_mask'].clone()), with_kwargs=True)
    with torch.no_grad():
        FULL.read_with_occupancy_prior(model, ego, points, feat, torch.zeros_like(prior))
        together = FULL.read_with_occupancy_prior(model, ego, points, feat, prior)
        separate = torch.cat([FULL.read_with_occupancy_prior(
            model, ego[i:i+1], points[i:i+1], feat[i:i+1], prior[i:i+1]) for i in range(2)])
    handle.remove()
    expected = captured[0] + prior[:, None, None, :].expand(2, 2, 1, 5).reshape(4, 1, 5)
    assert torch.equal(captured[1], expected)
    assert torch.allclose(together, separate, atol=2e-6, rtol=1e-6)
    assert not model.ego_cross_attn.attention._forward_pre_hooks


def test_attention_hook_is_removed_after_exception():
    model, _ = full_fixture()

    class Fail(nn.Module):
        def forward(self, value):
            raise ValueError('deliberate read failure')

    model.traj_head = Fail()
    with torch.no_grad(), pytest.raises(ValueError, match='deliberate'):
        FULL.read_with_occupancy_prior(model, torch.ones(1, 1, 8), torch.rand(1, 2, 3, 3),
                                       torch.ones(1, 2, 8), torch.ones(1, 2))
    assert not model.ego_cross_attn.attention._forward_pre_hooks


def test_labels_unused_training_rejected_and_duplicate_installation_rejected():
    model, args = full_fixture()
    FULL.install_full_planning_connection(model)
    with pytest.raises(RuntimeError, match='already installed'):
        FULL.install_full_planning_connection(model)
    with pytest.raises(RuntimeError, match='no_grad'):
        model._forward_backbone_future_memory_adapter(*args)
    with torch.no_grad():
        first = model._forward_backbone_future_memory_adapter(*args)
        other = list(args)
        other[1] = {'temporal_trajs': torch.full((1, 6, 2), 9999.)}
        second = model._forward_backbone_future_memory_adapter(*other)
    assert all(torch.equal(a, b) for a, b in zip(first['pred_trajs_list'], second['pred_trajs_list']))
    model.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match='eval'):
        model._forward_backbone_future_memory_adapter(*args)


def test_invalid_shapes_and_probability_floor_fail_clearly():
    features, points, logits, occupancy, projection = scene_tensors()
    with pytest.raises(ValueError, match='probability_floor'):
        FULL.full_scene_inputs(features, points, points, logits, logits, occupancy, projection,
                               probability_floor=0.)
    with pytest.raises(ValueError, match='Misaligned'):
        FULL.full_scene_inputs(features, points, points, logits, logits[..., :-1], occupancy, projection)
