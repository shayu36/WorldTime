"""Task/horizon gate migration, isolation and training graph invariants."""
import importlib.util

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from test_future_memory_adapter_v2 import QM, ROOT, _adapter, _memory

spec = importlib.util.spec_from_file_location(
    'prepare_ab', ROOT / 'tools/prepare_future_memory_ab.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def pair():
    torch.manual_seed(17)
    old = _adapter().eval()
    # Exercise a trained, nonzero residual, not merely zero-init equivalence.
    with torch.no_grad():
        for head in (old.delta_s_head, old.delta_o_head, old.delta_p_head):
            head[-1].weight.normal_(0, .1)
            head[-1].bias.fill_(.2)
    new = _adapter(gate_mode='task_split', horizon_gate_bias=True).eval()
    new.load_state_dict(migration.migrate_shared_state(old.state_dict(), ''), strict=True)
    return old, new


@pytest.mark.parametrize('batch', [1, 2])
def test_migration_preserves_nonzero_outputs_all_horizons(batch):
    old, new = pair()
    args = (torch.randn(batch, 3, 8), torch.zeros(batch, 3, 2, 3),
            torch.randn(batch, 3, 2, 3), _memory(batch=batch))
    with torch.no_grad():
        for horizon in range(3):
            a, b = old(*args, horizon), new(*args, horizon)
            for key in ('delta_s', 'delta_o', 'delta_p', 'gate', 'point_context'):
                assert torch.equal(a[key], b[key]), key
            assert torch.equal(a['gate'], b['gate_occ'])
            assert torch.equal(a['gate'], b['gate_pos'])
    assert 'gate_occ_head.weight' not in old.state_dict()
    assert not new.task_horizon_bias.any()


def test_horizon_and_task_gate_isolation():
    _, model = pair()
    features = torch.randn(2, 3, 2, 8)
    before = [model.task_gates(features, h) for h in range(3)]
    with torch.no_grad():
        model.task_horizon_bias[2, 1] = 2.
    for h in range(3):
        after = model.task_gates(features, h)
        for task in range(3):
            if (h, task) == (2, 1):
                assert (after[task] > before[h][task]).all()
            else:
                assert torch.equal(after[task], before[h][task])


@pytest.mark.parametrize('missing', [True, False])
def test_no_candidate_gates_and_outputs_are_exact_baseline(missing):
    _, model = pair()
    memory = _memory()
    memory['memory_valid'].zero_()
    base = torch.randn(1, 2, 2, 3)
    pos = torch.rand(1, 2, 2, 3)
    out = model(torch.randn(1, 2, 8), pos, base,
                None if missing else memory, 0)
    for key in ('gate_sem', 'gate_occ', 'gate_pos'):
        assert torch.equal(out[key], torch.zeros_like(out[key]))
    sem, occ, decoded = QM.decode_semantic_occupancy_logits(
        base, out['delta_s'], out['delta_o'], out['gate_sem'], out['gate_occ'])
    assert torch.equal(sem, base) and torch.equal(decoded, base)
    final = QM.apply_metric_position_residual(
        pos, out['gate_pos'] * out['delta_p'], model.pc_range,
        active_mask=out['gate_pos'].squeeze(-1) != 0)
    assert torch.equal(final, pos)


def test_decode_independent_occupancy_gate():
    base = torch.tensor([[[[2., 0., -2.]]]])
    delta_s = torch.tensor([[[[-4., 4., 0.]]]])
    delta_o = torch.ones(1, 1, 1, 1)
    zero, one = torch.zeros_like(delta_o), torch.ones_like(delta_o)
    sem, occ, decoded = QM.decode_semantic_occupancy_logits(base, delta_s, delta_o, one, zero)
    assert sem.argmax(-1).item() == 1
    assert torch.equal(occ, base.max(-1, keepdim=True).values)
    assert torch.equal(decoded.max(-1, keepdim=True).values, occ)
    other = QM.decode_semantic_occupancy_logits(base, delta_s, delta_o, one, one)
    assert torch.equal(other[0], sem)
    assert torch.equal(other[2].argmax(-1), decoded.argmax(-1))


def test_checkpointed_backward_reaches_all_new_heads():
    _, model = pair()
    model.train()
    q, p, s = torch.randn(2, 2, 8), torch.zeros(2, 2, 2, 3), torch.randn(2, 2, 2, 3)
    memory = _memory(batch=2)
    total = 0.
    for h in range(3):
        def run(q, p, s, horizon=h):
            result = model(q, p, s, memory, horizon, return_diagnostics=False)
            return tuple(result[k] for k in ('delta_s', 'delta_o', 'delta_p',
                                            'gate_sem', 'gate_occ', 'gate_pos'))
        ds, do, dp, gs, go, gp = checkpoint(run, q, p, s, use_reentrant=False)
        total = total + (gs * ds).square().mean() + (go * do).square().mean() + (gp * dp).square().mean()
    total.backward()
    for name in ('gate_head.weight', 'gate_occ_head.weight',
                 'gate_pos_head.weight', 'task_horizon_bias'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_empty_memory_graph_reaches_every_parameter():
    _, model = pair()
    out = model(torch.zeros(1, 1, 8), torch.zeros(1, 1, 2, 3),
                torch.zeros(1, 1, 2, 3), None, 0)
    sum(out[key].sum() for key in ('gate_sem', 'gate_occ', 'gate_pos')).backward()
    assert all(p.grad is not None and not p.grad.any() for p in model.parameters())
