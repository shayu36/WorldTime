"""Forward-only checks for the opt-in Memory-position planning experiment."""

import importlib.util
from pathlib import Path
import types

import pytest
import torch
from torch import nn

from test_future_memory_adapter_v2 import QM, _load_sparseworld_method

SPEC = importlib.util.spec_from_file_location(
    'memory_planning_connection',
    Path(__file__).resolve().parents[1] / 'tools/memory_planning_connection.py')
CONNECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONNECTION)


class Projection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, value):
        signal = value.mean(-1, keepdim=True) * .01
        return signal.expand(*value.shape[:-1], self.channels)


class ResponsivePlanner(nn.Module):
    def forward(self, query_pos, query_feat, points, features):
        # Nonconstant, spatially sensitive read: interventions must change it.
        geometry = points.square().mean((1, 2, 3))[:, None, None]
        content = features.mean((1, 2))[:, None, None]
        return query_feat + geometry + content, None


class Memory(nn.Module):
    def __init__(self, corrections, disabled=False, semantic=0.):
        super().__init__()
        self.corrections = corrections
        self.disabled = disabled
        self.semantic = semantic

    def forward(self, query, points, logits, memory, horizon_id, **kwargs):
        gate = torch.ones_like(points[..., :1]) * (not self.disabled)
        # B=2 exercises an unchanged sample alongside a changed sample.
        if gate.shape[0] == 2:
            gate[1] = 0
        return dict(
            delta_s=torch.full_like(logits, self.semantic),
            delta_o=torch.zeros_like(gate),
            delta_p=torch.ones_like(points) * self.corrections[horizon_id],
            gate=gate, gate_occ=gate, gate_pos=gate,
            diagnostics={'has_candidate': gate.squeeze(-1).bool()})


def fixture_model(batch=1, points=2, corrections=(1., 2., 3.), disabled=False,
                  semantic=0.):
    forward = _load_sparseworld_method('_forward_backbone_future_memory_adapter', {
        'torch': torch, 'np': __import__('numpy'),
        'decode_points_metric': QM.decode_points_metric,
        'decode_semantic_occupancy_logits': QM.decode_semantic_occupancy_logits})
    model = nn.Module()
    model.num_refines = points
    model.num_fu_frames = 6
    model.pc_range = torch.tensor([-40., -40., -1., 40., 40., 5.4])
    model.ego_cross_attn = ResponsivePlanner()
    model.traj_head = Projection(2)
    model.position_encoder = Projection(8)
    model.cls_branch = Projection(points * 17)
    model.reg_branch = Projection(points * 3)
    model.vel_branch = Projection(points * 2)
    model.future_memory_adapter_enabled = True
    model.future_memory_adapter_version = 'v2'
    model.future_memory_adapter_v2 = Memory(corrections, disabled, semantic)
    model.refine_points = lambda pos, delta: pos + .01 * delta.reshape_as(pos)
    model._refine_future_memory_points_v2 = lambda pos, delta, active_mask=None: (
        QM.apply_metric_position_residual(
            pos, delta.reshape_as(pos), model.pc_range, active_mask))
    model._forward_backbone_future_memory_adapter = types.MethodType(forward, model)
    model.eval()
    counts = [2, 1, 1, 1, 1, 1, 1]
    stamps = torch.repeat_interleave(torch.arange(7), torch.tensor(counts))
    queries = len(stamps)
    feat = torch.linspace(-1, 1, batch * queries * 8).reshape(batch, queries, 8)
    pos = torch.linspace(.2, .8, batch * queries * points * 3).reshape(
        batch, queries, points, 3)
    logits = torch.zeros(batch, queries, points, 17)
    args = ([], {}, batch, torch.ones(batch, 1, 8), {}, feat, pos, logits, stamps, None)
    return model, args


@pytest.mark.parametrize('batch,points', [(1, 2), (2, 3)])
def test_real_forward_memory_changes_plans_without_changing_scene_recurrence(batch, points):
    model, args = fixture_model(batch, points)
    original_pos = args[6].clone()
    observed = []
    keys = tuple(model.state_dict())
    with torch.no_grad():
        native = model._forward_backbone_future_memory_adapter(*args)
        restore = CONNECTION.install_position_planning_connection(
            model, lambda output, planned, delta: observed.append(delta))
        connected = model._forward_backbone_future_memory_adapter(*args)
        restore()
        restored = model._forward_backbone_future_memory_adapter(*args)
    assert tuple(model.state_dict()) == keys
    assert torch.equal(args[6], original_pos)
    assert not model.cls_branch._forward_pre_hooks
    for name in ('baseline_forecast_points_list', 'baseline_forecast_semantics_list',
                 'forecast_points_list', 'forecast_semantics_list'):
        assert all(torch.equal(a, b) for a, b in zip(native[name], connected[name]))
    for step in range(6):
        a, b = native['pred_trajs_list'][step], connected['pred_trajs_list'][step]
        assert torch.equal(a, restored['pred_trajs_list'][step])
        if step in CONNECTION.TARGET_STEPS:
            assert not torch.equal(a[0], b[0])
            if batch == 2:
                assert torch.equal(a[1], b[1])
        else:
            assert torch.equal(a, b)
    assert observed[0].shape == (batch, 6, 2)


@pytest.mark.parametrize('disabled,corrections', [(False, (0., 0., 0.)),
                                                (True, (1., 2., 3.))])
def test_no_memory_or_zero_positions_is_exact_identity(disabled, corrections):
    model, args = fixture_model(disabled=disabled, corrections=corrections)
    with torch.no_grad():
        native = model._forward_backbone_future_memory_adapter(*args)
        CONNECTION.install_position_planning_connection(model)
        connected = model._forward_backbone_future_memory_adapter(*args)
    assert all(torch.equal(a, b) for a, b in
               zip(native['pred_trajs_list'], connected['pred_trajs_list']))


@pytest.mark.parametrize('target', [0, 1, 2])
def test_same_horizon_read_includes_3s_not_a_one_step_delayed_read(target):
    corrections = [0., 0., 0.]
    corrections[target] = 1.
    model, args = fixture_model(corrections=corrections)
    with torch.no_grad():
        native = model._forward_backbone_future_memory_adapter(*args)
        CONNECTION.install_position_planning_connection(model)
        connected = model._forward_backbone_future_memory_adapter(*args)
    a = torch.cat(native['pred_trajs_list'], 1)
    b = torch.cat(connected['pred_trajs_list'], 1)
    changed = (a != b).any(-1)[0].nonzero().flatten().tolist()
    assert changed == [2 * target + 1]
    # Increment addition is followed by exactly one cumulative coordinate sum.
    assert torch.equal(a.cumsum(1)[:, :2*target+1], b.cumsum(1)[:, :2*target+1])
    assert not torch.equal(a.cumsum(1)[:, 2*target+1:], b.cumsum(1)[:, 2*target+1:])


def test_semantic_only_is_explicitly_not_connected_and_labels_are_unused():
    model, args = fixture_model(corrections=(0., 0., 0.), semantic=100.)
    with torch.no_grad():
        native = model._forward_backbone_future_memory_adapter(*args)
        CONNECTION.install_position_planning_connection(model)
        connected = model._forward_backbone_future_memory_adapter(*args)
        changed_args = list(args)
        changed_args[1] = {'temporal_trajs': torch.full((1, 6, 2), 10000.)}
        changed = model._forward_backbone_future_memory_adapter(*changed_args)
    for a, b, c in zip(native['pred_trajs_list'], connected['pred_trajs_list'],
                       changed['pred_trajs_list']):
        assert torch.equal(a, b) and torch.equal(b, c)


def test_reject_training_grad_mode_and_double_installation():
    model, args = fixture_model()
    restore = CONNECTION.install_position_planning_connection(model)
    with pytest.raises(RuntimeError, match='already installed'):
        CONNECTION.install_position_planning_connection(model)
    with pytest.raises(RuntimeError, match='no_grad'):
        model._forward_backbone_future_memory_adapter(*args)
    model.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match='eval'):
        model._forward_backbone_future_memory_adapter(*args)
    restore()


def test_failed_native_forward_does_not_leave_hook_registered():
    model, args = fixture_model()

    def fail(*args):
        raise ValueError('deliberate test failure')

    model._forward_backbone_future_memory_adapter = fail
    CONNECTION.install_position_planning_connection(model)
    with torch.no_grad(), pytest.raises(ValueError, match='deliberate'):
        model._forward_backbone_future_memory_adapter(*args)
    assert not model.cls_branch._forward_pre_hooks
