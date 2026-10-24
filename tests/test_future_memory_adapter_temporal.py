"""Round-1 temporal Adapter: CPU forward/structure tests only, no optimization."""
import copy
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmcv import Config
from torch.utils.checkpoint import checkpoint

from test_future_memory_adapter_v2 import QM, ROOT, _adapter, _memory, _load_sparseworld_method


CONFIG = ROOT / 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-t1-temporal.py'
CONTROL = CONFIG.with_name(CONFIG.stem + '-control.py')


def _pair(**kw):
    torch.manual_seed(23)
    old = _adapter(gate_mode='task_split', horizon_gate_bias=True, **kw).eval()
    with torch.no_grad():
        for head in (old.delta_s_head, old.delta_o_head, old.delta_p_head):
            head[-1].weight.normal_(0, .15)
            head[-1].bias.normal_(0, .1)
    new = _adapter(gate_mode='task_split', horizon_gate_bias=True,
                   temporal_adapter_enabled=True, **kw).eval()
    incompatible = new.load_state_dict(old.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        'temporal_read.' + k for k in new.temporal_read.state_dict()}
    return old, new


def _open_read(adapter):
    with torch.no_grad():
        adapter.temporal_read.output.weight.copy_(torch.eye(adapter.embed_dims) * .5)


def test_opt_in_preserves_old_keys_initialization_and_rng():
    torch.manual_seed(17)
    old = _adapter()
    rng = torch.get_rng_state()
    torch.manual_seed(17)
    new = _adapter(temporal_adapter_enabled=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(k.startswith('temporal_read.') for k in old.state_dict())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    assert not new.temporal_read.output.weight.any()


@pytest.mark.parametrize('batch', [1, 2])
def test_trained_t1_zero_contribution_identity_all_horizons(batch):
    old, new = _pair()
    memory = _memory(batch=batch)
    state = None
    for h, ids in enumerate(([9, 2], [9, 2, 6], [9, 2, 6, 1])):
        q, p = torch.randn(batch, len(ids), 8), torch.rand(batch, len(ids), 2, 3)
        s = torch.randn(batch, len(ids), 2, 3)
        with torch.no_grad():
            a = old(q, p, s, memory, h)
            b = new(q, p, s, memory, h, temporal_state=state, query_ids=torch.tensor(ids))
        for key in ('delta_s', 'delta_o', 'delta_p', 'gate_sem', 'gate_occ', 'gate_pos', 'point_context'):
            assert torch.equal(a[key], b[key]), key
        da = QM.decode_semantic_occupancy_logits(s, a['delta_s'], a['delta_o'], a['gate'], a['gate_occ'])
        db = QM.decode_semantic_occupancy_logits(s, b['delta_s'], b['delta_o'], b['gate'], b['gate_occ'])
        assert all(torch.equal(x, y) for x, y in zip(da, db))
        assert not b['temporal_stats'][2].any()
        state = b['temporal_state']
        assert state['features'].shape == (batch, len(ids), 8)
        assert not state['features'].requires_grad
        assert torch.equal(state['query_ids'], torch.tensor(ids))


def test_alignment_uses_ids_and_batch_not_prefix_or_point_slots():
    features = torch.arange(2 * 2 * 8).reshape(2, 2, 8).float()
    state = dict(features=features, valid=torch.tensor([[True, False], [True, True]]),
                 query_ids=torch.tensor([7, 3]), horizon_id=0)
    out, valid = QM.AdapterTemporalRead.align(state, torch.tensor([3, 9, 7]), torch.zeros(2, 3, 8), 1)
    assert torch.equal(out[:, 2], features[:, 0])
    assert torch.equal(out[1, 0], features[1, 1])
    assert not out[0, 0].any() and not out[:, 1].any()
    assert torch.equal(valid, torch.tensor([[False, False, True], [True, False, True]]))
    bad = dict(state, features=features.clone())
    bad['features'][1, 0, 0] = float('nan')
    out, valid = QM.AdapterTemporalRead.align(bad, torch.tensor([7]), torch.zeros(2, 1, 8), 1)
    assert not valid[1, 0] and torch.isfinite(out).all()


def test_state_pool_is_point_permutation_invariant_masked_and_fp16_safe():
    features = torch.tensor([[[[1., 2.], [3., 4.], [99., 99.]]]])
    valid = torch.tensor([[[True, True, False]]])
    expected = torch.tensor([[[2., 3.]]])
    assert torch.equal(QM.AdapterTemporalRead.pool(features, valid), expected)
    assert torch.equal(QM.AdapterTemporalRead.pool(features.flip(2), valid.flip(2)), expected)
    assert not QM.AdapterTemporalRead.pool(features, torch.zeros_like(valid)).any()
    large = torch.full((2, 2, 48, 8), 3000., dtype=torch.float16)
    got = QM.AdapterTemporalRead.pool(large, torch.ones(2, 2, 48, dtype=torch.bool))
    assert torch.isfinite(got).all() and (got == 3000.).all()


@pytest.mark.parametrize('case', ['missing', 'invalid', 'expired', 'radius'])
def test_no_current_memory_blocks_temporal_read_and_clears_state(case):
    _, model = _pair()
    _open_read(model)
    q, p, s = torch.randn(2, 2, 8), torch.zeros(2, 2, 2, 3), torch.randn(2, 2, 2, 3)
    ids, memory = torch.tensor([3, 9]), _memory(batch=2)
    with torch.no_grad():
        state = model(q, p, s, memory, 0, query_ids=ids)['temporal_state']
        if case == 'missing':
            memory = None
        elif case == 'invalid':
            memory['memory_valid'].zero_()
        elif case == 'expired':
            memory['memory_age'].fill_(20.)
        else:
            memory['memory_points_metric'].fill_(1000.)
        out = model(q, p, s, memory, 1, temporal_state=state, query_ids=ids)
    for key in ('delta_s', 'delta_o', 'delta_p', 'gate_sem', 'gate_occ', 'gate_pos'):
        assert not out[key].any(), key
    assert not out['temporal_state']['features'].any()
    assert not out['temporal_state']['valid'].any()
    assert not out['temporal_stats'].any()
    decoded = QM.decode_semantic_occupancy_logits(s, out['delta_s'], out['delta_o'], out['gate'], out['gate_occ'])[2]
    assert torch.equal(decoded, s)
    normalized = torch.rand_like(p)
    corrected = QM.apply_metric_position_residual(normalized, out['gate_pos'] * out['delta_p'], model.pc_range,
                                                  active_mask=out['gate_pos'].squeeze(-1) != 0)
    assert torch.equal(corrected, normalized)


def test_open_read_is_pointwise_bounded_and_cannot_reach_new_query():
    old, model = _pair()
    _open_read(model)
    q, p, s = torch.randn(2, 3, 8), torch.rand(2, 3, 2, 3), torch.randn(2, 3, 2, 3)
    memory = _memory(batch=2)
    with torch.no_grad():
        state = model(q[:, :2], p[:, :2], s[:, :2], memory, 0, query_ids=torch.tensor([7, 3]))['temporal_state']
        a = old(q, p, s, memory, 1)
        b = model(q, p, s, memory, 1, temporal_state=state, query_ids=torch.tensor([7, 3, 5]))
    assert not torch.equal(a['delta_s'][:, :2], b['delta_s'][:, :2])
    for key in ('delta_s', 'delta_o', 'delta_p', 'gate_sem', 'gate_occ', 'gate_pos'):
        assert torch.equal(a[key][:, 2], b[key][:, 2]), key
    assert torch.equal(a['point_context'], b['point_context'])
    gates = b['diagnostics']['temporal_gate']
    assert not torch.equal(gates[:, :2, 0], gates[:, :2, 1])
    x = torch.randn(2, 3, 2, 8)
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    with torch.no_grad():
        _, g, update = model.temporal_read(x, torch.randn(2, 3, 8), mask.any(-1), mask)
    assert update.abs().max() <= model.temporal_read.max_delta
    assert ((g > 0) & (g < 1)).all()
    mask[..., 1] = False
    with torch.no_grad():
        y, g, update = model.temporal_read(x, torch.randn(2, 3, 8), mask.any(-1), mask)
    assert not g[..., 1, :].any() and not update[..., 1, :].any()
    assert torch.equal(y[..., 1, :], x[..., 1, :])


def test_chunked_and_unchunked_temporal_results_agree():
    _, model = _pair(query_chunk_size=1)
    _open_read(model)
    large = copy.deepcopy(model)
    large.query_chunk_size = 16
    q, p, s = torch.randn(2, 3, 8), torch.rand(2, 3, 2, 3), torch.randn(2, 3, 2, 3)
    ids, memory = torch.tensor([5, 2, 8]), _memory(batch=2)
    state = None
    with torch.no_grad():
        for h in range(3):
            a = model(q, p, s, memory, h, temporal_state=state, query_ids=ids)
            b = large(q, p, s, memory, h, temporal_state=state, query_ids=ids)
            for key in ('delta_s', 'delta_o', 'delta_p', 'gate_sem', 'gate_occ', 'gate_pos'):
                assert torch.allclose(a[key], b[key], atol=1e-6, rtol=1e-5), key
            state = a['temporal_state']
            assert torch.allclose(state['features'], b['temporal_state']['features'], atol=1e-6, rtol=1e-5)


def test_state_validation_and_no_implicit_cross_sequence_storage():
    _, model = _pair()
    q, p, s = torch.randn(1, 2, 8), torch.zeros(1, 2, 2, 3), torch.randn(1, 2, 2, 3)
    memory, ids = _memory(), torch.tensor([1, 4])
    with pytest.raises(ValueError, match='query_ids'):
        model(q, p, s, memory, 0)
    with pytest.raises(ValueError, match='unique'):
        model(q, p, s, memory, 0, query_ids=torch.tensor([1, 1]))
    with torch.no_grad():
        a = model(q, p, s, memory, 0, query_ids=ids)
        state = a['temporal_state']
        for h in (0, 2):
            with pytest.raises(ValueError, match='preceding horizon'):
                model(q, p, s, memory, h, temporal_state=state, query_ids=ids)
        model(q, p, s, memory, 1, temporal_state=state, query_ids=ids)
        b = model(q, p, s, memory, 0, query_ids=ids)
    assert torch.equal(a['delta_s'], b['delta_s'])
    assert not hasattr(model, 'temporal_state')


class _Projection:
    def __init__(self, count):
        self.count = count

    def __call__(self, x):
        return x.mean(-1, keepdim=True) * torch.linspace(.001, .01, self.count)


class _Ego:
    def __call__(self, query_pos, query_feat, memory_pos, memory_feat):
        return (query_feat + memory_feat.mean((1, 2), keepdim=True) * .01 +
                memory_pos.mean((1, 2, 3)).reshape(-1, 1, 1) * .01), None


def _recurrence_fixture(training=False, use_checkpoint=False, checkpoint_impl=checkpoint):
    method = _load_sparseworld_method('_forward_backbone_future_memory_adapter', {
        'torch': torch, 'np': np, 'checkpoint': checkpoint_impl,
        'decode_points_metric': QM.decode_points_metric,
        'decode_semantic_occupancy_logits': QM.decode_semantic_occupancy_logits})
    torch.manual_seed(31)
    adapter = QM.FutureMemoryAdapterV2(embed_dims=8, num_classes=17, num_points=2,
        num_heads=2, coarse_topk_geometry=2, coarse_topk_semantic=2, point_topk=2,
        coarse_radius=10., point_radius=10., query_chunk_size=2,
        temporal_adapter_enabled=True, gate_mode='task_split', horizon_gate_bias=True)
    with torch.no_grad():
        for head in (adapter.delta_s_head, adapter.delta_o_head, adapter.delta_p_head):
            head[-1].weight.normal_(0, .1)
    _open_read(adapter)
    model = SimpleNamespace(num_refines=2, num_fu_frames=6,
        pc_range=torch.tensor([-4., -4., -2., 4., 4., 2.]),
        ego_cross_attn=_Ego(), traj_head=_Projection(2), position_encoder=_Projection(8),
        cls_branch=_Projection(34), vel_branch=_Projection(4), reg_branch=_Projection(6),
        future_memory_adapter_version='v2', future_memory_adapter_v2=adapter,
        future_memory_adapter_v2_activation_checkpoint=use_checkpoint,
        training=training)
    model.refine_points = lambda p, d: p + d.reshape_as(p) * .01
    model._refine_future_memory_points_v2 = lambda p, d, active_mask=None: QM.apply_metric_position_residual(
        p, d.reshape_as(p), model.pc_range, active_mask)
    model.trans_points = lambda p, t, e: p
    stamps = torch.tensor([1, 0, 2, 0, 3, 4, 5, 6])
    q, p, s = torch.randn(2, 8, 8), torch.full((2, 8, 2, 3), .5), torch.randn(2, 8, 2, 17)
    memory = _memory(batch=2, classes=17)

    def run(offset=0.):
        return method(model, [{'ego2lidar': np.eye(4)}] * 2,
                      {'temporal_trajs': torch.zeros(2, 6, 2)}, 2, torch.ones(2, 1, 8), {},
                      q + offset, p, s, stamps, memory)
    return model, run


@pytest.mark.parametrize('horizon', [0, 1])
@pytest.mark.parametrize('change', ['hidden', 'outputs'])
def test_only_hidden_state_propagates_baseline_and_planning_stay_isolated(horizon, change):
    model, run = _recurrence_fixture()
    with torch.no_grad():
        original = run()
        responsive = run(.3)
        assert not torch.equal(original['baseline_forecast_points_list'][3], responsive['baseline_forecast_points_list'][3])
        ids_seen = []

        def perturb(module, args, kwargs, out):
            h = args[4]
            ids_seen.append(kwargs['query_ids'].tolist())
            if h == horizon:
                if change == 'hidden':
                    state = dict(out['temporal_state'])
                    state['features'] = state['features'].flip(-1) * -3.
                    out['temporal_state'] = state
                else:
                    out['delta_s'] = out['delta_s'] + torch.arange(17) * .1
                    out['delta_o'] = out['delta_o'] + 2.
                    out['delta_p'] = out['delta_p'] + .5
            return out

        hook = model.future_memory_adapter_v2.register_forward_hook(perturb, with_kwargs=True)
        changed = run()
        hook.remove()
        again = run()
    assert ids_seen == [[1, 3, 0, 2], [1, 3, 0, 2, 4, 5], [1, 3, 0, 2, 4, 5, 6, 7]]
    for key in ('baseline_forecast_points_list', 'baseline_forecast_semantics_list', 'pred_trajs_list'):
        assert all(torch.equal(a, b) for a, b in zip(original[key], changed[key])), key
    for key in ('forecast_points_list', 'forecast_semantics_list'):
        for j in range(6):
            should_change = (j in (3, 5) and j > 2 * horizon + 1) if change == 'hidden' else j == 2 * horizon + 1
            assert torch.equal(original[key][j], changed[key][j]) != should_change, (key, j)
            assert torch.equal(original[key][j], again[key][j]), 'state leaked between calls'
    assert torch.equal(original['cls_score'], changed['cls_score'])


def test_checkpoint_training_forward_preserves_state_and_fixed_log_keys():
    model, run = _recurrence_fixture(training=True, use_checkpoint=False)
    # Build only a forward graph. No derivatives or optimizer operations.
    direct = run()
    model.future_memory_adapter_v2_activation_checkpoint = True
    checked = run()
    for key in ('forecast_points_list', 'forecast_semantics_list', 'pred_trajs_list'):
        assert all(torch.equal(a, b) for a, b in zip(direct[key], checked[key])), key
    for j in (1, 3, 5):
        a = direct['memory_adapter_diagnostics'][j]['temporal_stats']
        b = checked['memory_adapter_diagnostics'][j]['temporal_stats']
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) and not b[k].requires_grad for k in a)
        assert all('loss' not in k for k in a)
    assert all(p.grad is None for p in model.future_memory_adapter_v2.parameters())


def test_checkpoint_callbacks_bind_each_horizon_state_for_later_replay():
    captured = []

    def capture(function, *args, **kwargs):
        values = checkpoint(function, *args, **kwargs)
        assert values[0].requires_grad
        assert not values[6].requires_grad  # detached inter-horizon state
        captured.append((function, args, tuple(t.detach().clone() for t in values)))
        return values

    model, run = _recurrence_fixture(training=True, use_checkpoint=True,
                                    checkpoint_impl=capture)
    run()
    assert len(captured) == 3
    # Explicit forward-only replay AFTER the full loop has advanced to 3s.
    # This catches late-bound horizon/state closures without differentiation.
    with torch.no_grad():
        for function, args, expected in reversed(captured):
            replay = function(*args)
            assert all(torch.equal(a, b) for a, b in zip(expected, replay))
    assert all(p.grad is None for p in model.future_memory_adapter_v2.parameters())


def test_matched_configs_keep_losses_data_optimizer_freeze_and_start_identical():
    a, b = Config.fromfile(str(CONTROL)), Config.fromfile(str(CONFIG))
    assert a.load_from == b.load_from and a.load_from.endswith('t1-from-center2/epoch_3.pth')
    assert a.resume_from is None and b.resume_from is None
    assert not a.auto_resume and not b.auto_resume
    assert a.work_dir != b.work_dir and a.runner.max_epochs == b.runner.max_epochs == 3
    for key in ('data', 'optimizer', 'optimizer_config', 'lr_config', 'fp16', 'seed', 'workflow'):
        assert a.get(key) == b.get(key), key
    assert a.model.query_memory_cfg.freeze_base_model and b.model.query_memory_cfg.freeze_base_model
    assert b.optimizer.constructor == 'TrainableOnlyOptimizerConstructor'
    reference, treated = copy.deepcopy(a.model), copy.deepcopy(b.model)
    opts = treated.query_memory_cfg.future_memory_adapter_v2
    assert opts.temporal_adapter_enabled and opts.temporal_gate_bias == -2. and opts.temporal_max_delta == 1.
    assert opts.occupancy_positive_class_balance_beta == {'1s': 0., '2s': 0., '3s': .25}
    assert opts.soft_voxel_center_offset == .5
    for k in ('temporal_adapter_enabled', 'temporal_gate_bias', 'temporal_max_delta'):
        opts.pop(k)
    reference.query_memory_cfg.future_memory_adapter_v2.pop('temporal_adapter_enabled')
    assert reference == treated


def test_real_freezing_policy_and_optimizer_include_temporal_but_no_baseline():
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj
    from mmdet3d.core.optimizer import TrainableOnlyOptimizerConstructor

    # Exercise real policies without instantiating the CUDA-only OPUS head.
    model = SparseWorld4DTraj.__new__(SparseWorld4DTraj)
    torch.nn.Module.__init__(model)
    model.baseline = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(.5))
    model.future_memory_adapter_v2 = _pair()[1]
    model.query_memory_cfg = dict(freeze_base_model=True)
    model.query_memory_source = 'cache'
    model.future_memory_adapter_version = 'v2'
    model.future_memory_adapter_finetune_mode = True
    model.future_memory_adapter_enabled = True
    model._configure_query_memory_trainability()
    model.train()
    assert model.training and not model.baseline.training
    assert model.future_memory_adapter_v2.training
    trainable = {id(p) for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {id(p) for p in model.future_memory_adapter_v2.parameters()}
    constructor = TrainableOnlyOptimizerConstructor(dict(type='AdamW', lr=2e-5, weight_decay=.01))
    optimizer = constructor(model)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == trainable
    assert all(p.grad is None and not p.requires_grad for p in model.baseline.parameters())


def test_checkpoint_check_rejects_missing_old_keys_and_never_mutates_source():
    spec = importlib.util.spec_from_file_location('check_temporal', ROOT / 'tools/check_future_memory_temporal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old, new = _pair()
    weights = {'module.future_memory_adapter_v2.' + k: v.clone() for k, v in old.state_dict().items()}
    original = copy.deepcopy(weights)
    module.load_t1_adapter(new, weights)
    assert all(torch.equal(v, weights[k]) for k, v in original.items())
    weights.pop('module.future_memory_adapter_v2.query_proj.weight')
    with pytest.raises(ValueError, match='not a compatible T1 source'):
        module.load_t1_adapter(new, weights)
    with pytest.raises(ValueError, match='no future_memory_adapter_v2'):
        module.load_t1_adapter(new, {})


def test_cpu_autocast_temporal_read_forward_is_finite():
    read = QM.AdapterTemporalRead(8)
    with torch.no_grad():
        read.output.weight.copy_(torch.eye(8))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out, gate, update = read(torch.randn(2, 3, 2, 8), torch.randn(2, 3, 8),
                                     torch.ones(2, 3, dtype=torch.bool),
                                     torch.ones(2, 3, 2, dtype=torch.bool))
    assert all(torch.isfinite(t).all() for t in (out, gate, update))
