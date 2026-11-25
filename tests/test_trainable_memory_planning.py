"""Small CPU gradient/forward checks; no optimizer steps or dataset iteration."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_memory_planning_connection import fixture_model, Memory
from test_future_memory_adapter_v2 import _load_sparseworld_method

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('planning_fusion', ROOT /
    'mmdet3d/models/sparsedetectors/memory_planning_fusion.py')
FUSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FUSION)


def inputs(batch=2, queries=4, points=3, classes=5, dims=8, empty=False):
    torch.manual_seed(17)
    module = FUSION.MemoryPlanningFusion(dims, classes, hidden_dims=8, num_heads=2)
    valid = torch.ones(batch, queries, points, dtype=torch.bool)
    valid[:, -1] = False
    if empty:
        valid.zero_()
    base = torch.rand(batch, queries, points, 3)
    logits = torch.randn(batch, queries, points, classes)
    sem = logits + torch.randn_like(logits) * .3
    occ = logits.max(-1, keepdim=True).values - .2
    gate = torch.full_like(occ, .3)
    adapter = dict(context=torch.randn(batch, queries, points, dims), gate=gate,
                   diagnostics=dict(has_candidate=valid))
    tokens, qvalid = module.pack_scene(torch.randn(batch, queries, dims), base,
        base + .01, logits, sem, occ, adapter, (-40, -40, -1, 40, 40, 5.4))
    ego, native = torch.randn(batch, 1, dims), torch.randn(batch, 1, 2)
    return module, (ego, tokens, qvalid, 1, native)


@pytest.mark.parametrize('batch,queries,points,classes', [(1, 3, 2, 3), (2, 5, 4, 17)])
def test_zero_initialization_and_upstream_gradient_after_head_activation(batch, queries, points, classes):
    module, args = inputs(batch, queries, points, classes)
    delta = module(*args)
    assert delta.shape == (batch, 1, 2) and torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(args[-1] + delta, args[-1])
    loss = ((args[-1] + delta - .7) ** 2).mean()
    loss.backward()
    assert module.residual_head[-1].weight.grad.abs().sum() > 0
    assert module.token_encoder[0].weight.grad.abs().sum() == 0
    # A nonzero head tests differentiability without an optimizer update here.
    with torch.no_grad():
        module.residual_head[-1].weight.normal_(std=.02)
    module.zero_grad(set_to_none=True)
    loss = ((args[-1] + module(*args) - .7) ** 2).mean()
    loss.backward()
    for group in ('token_encoder', 'ego_encoder', 'horizon_embedding', 'attention', 'residual_head'):
        grads = [p.grad for n, p in module.named_parameters() if n.startswith(group)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0, group
    # All three semantic vector blocks, occupancy and geometry columns receive
    # loss gradients, not just the ego-only path.
    grad = module.token_encoder[0].weight.grad
    for section in (slice(8, 16), slice(16, 16+3*classes),
                    slice(16+3*classes, 19+3*classes), slice(19+3*classes, 28+3*classes)):
        assert grad[:, section].abs().sum() > 0


def test_empty_candidates_are_finite_zero_with_live_ddp_graph():
    module, args = inputs(empty=True)
    with torch.no_grad():
        module.residual_head[-1].weight.fill_(.1)
        module.residual_head[-1].bias.fill_(1.)
    delta = module(*args)
    assert torch.equal(delta, torch.zeros_like(delta))
    ((delta + args[-1]) ** 2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert all(torch.count_nonzero(p.grad) == 0 for p in module.parameters())


def test_invalid_tokens_cannot_influence_read_and_batch_does_not_mix():
    module, args = inputs()
    with torch.no_grad():
        module.residual_head[-1].weight.normal_(std=.1)
        original = module(*args)
        modified = list(args)
        modified[1] = args[1].clone()
        modified[1][:, -1] = 100.
        assert torch.equal(original, module(*modified))
        individual = torch.cat([module(args[0][i:i+1], args[1][i:i+1], args[2][i:i+1],
                                      args[3], args[4][i:i+1]) for i in range(2)])
        assert torch.allclose(original, individual, atol=1e-6)


class SnapshotMemory(Memory):
    num_classes = 17
    voxel_size = (.4, .4, .4)
    ablation_mode = 'full'

    def forward(self, query, points, logits, memory, horizon_id, **kwargs):
        result = super().forward(query, points, logits, memory, horizon_id, **kwargs)
        result['context'] = query[:, :, None, :].expand(-1, -1, points.shape[2], -1) * .5
        return result


class FrozenFixture(nn.Module):
    """Responsive frozen backbone; real recurrence and loss methods are reused."""
    def __init__(self, **kwargs):
        super().__init__()
        base, self.args = fixture_model()
        for name, value in base.__dict__.items():
            if not name.startswith('_'):
                setattr(self, name, value)
        for name, value in base.named_children():
            setattr(self, name, value)
        self._refine_future_memory_points_v2 = base._refine_future_memory_points_v2
        self.future_memory_adapter_v2 = SnapshotMemory((1., 2., 3.))
        self.out_dim = 8
        self.query_memory_source = 'cache'
        self.query_memory_cfg = dict(freeze_base_model=True)
        self.l2_loss = lambda pred, target: ((pred-target)**2).mean(0).sum()

    def memory_tuning_trainable_prefixes(self):
        return ('future_memory_adapter_v2.',)

    def memory_tuning_train_modules(self):
        return ('future_memory_adapter_v2',)

    def _configure_query_memory_trainability(self):
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(self.memory_tuning_trainable_prefixes()))

    def train(self, mode=True):
        super().train(mode)
        for name, child in self.named_children():
            child.train(mode and name in self.memory_tuning_train_modules())
        return self

    def forward_backbone(self, img, img_metas, **kwargs):
        arguments = list(self.args)
        arguments[0], arguments[1] = img_metas, kwargs
        # Compile/use the real current recurrence method, bound to this model.
        return self.args_method(self, *arguments)

    args_method = staticmethod(_load_sparseworld_method('_forward_backbone_future_memory_adapter',
        {'torch': torch, 'decode_points_metric': __import__('test_future_memory_adapter_v2').QM.decode_points_metric,
         'decode_semantic_occupancy_logits': __import__('test_future_memory_adapter_v2').QM.decode_semantic_occupancy_logits}))
    loss_traj = _load_sparseworld_method('loss_traj', {})


def subclass():
    source = ROOT / 'mmdet3d/models/sparsedetectors/sparseworld_memory_planning.py'
    node = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef))
    node.decorator_list = []
    namespace = dict(torch=torch, SparseWorld4DTraj=FrozenFixture,
                     MemoryPlanningFusion=FUSION.MemoryPlanningFusion)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[node.name]


def test_real_forward_train_uses_existing_loss_and_no_gt_enters_fusion():
    model = subclass()(memory_planning_cfg=dict(hidden_dims=8, num_heads=2)).train()
    target = torch.ones(1, 6, 2)
    outputs = model.forward_backbone(None, [])
    assert all(torch.equal(a, b) for a, b in zip(outputs['pred_trajs_list'], outputs['baseline_pred_trajs_list']))
    losses = model.forward_train(img=None, img_metas=[], temporal_trajs=target)
    assert set(losses) == {f'loss_traj_{i}s' for i in range(1, 7)}
    expected = {f'loss_traj_{i+1}s': ((p.squeeze(1)-target[:, i])**2).mean(0).sum()
                for i, p in enumerate(outputs['pred_trajs_list'])}
    assert all(torch.equal(losses[k], expected[k]) for k in losses)
    sum(losses.values()).backward()
    assert model.memory_planning_fusion.residual_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith('memory_planning_fusion.'))
    with torch.no_grad():
        other = model.forward_backbone(None, [], temporal_trajs=target * 100.)
    assert all(torch.equal(a, b) for a, b in zip(outputs['pred_trajs_list'], other['pred_trajs_list']))


def test_source_loading_rejects_missing_frozen_keys():
    model = subclass()(memory_planning_cfg=dict(hidden_dims=8, num_heads=2))
    # Add a frozen parameter so the rejection is meaningful for the fixture.
    model.frozen_probe = nn.Linear(2, 2)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('memory_planning_fusion.')}
    info = model.load_frozen_t1_state(state)
    assert info['missing_new_fusion']
    state.pop('frozen_probe.weight')
    with pytest.raises(RuntimeError, match='frozen_probe.weight'):
        model.load_frozen_t1_state(state)
