"""Internal ablation gradient checks (kept out of the forward-only suite)."""
import torch

from test_future_memory_adapter_v2 import _adapter, _memory


def test_d5_backward_excludes_hidden_slots_and_reaches_selected_memory():
    torch.manual_seed(19)
    adapter = _adapter(history_slot_limit=1, gate_mode='task_split',
                       horizon_gate_bias=True)
    # A nonzero trained residual is needed to probe the upstream read path.
    with torch.no_grad():
        for head in (adapter.delta_s_head, adapter.delta_o_head, adapter.delta_p_head):
            head[-1].weight.normal_(0., .1)
    memory = _memory(count=6, batch=2)
    memory['memory_points_metric'].normal_(0., .2)
    for key, value in memory.items():
        memory[key] = value.reshape(2, 3, 2, *value.shape[3:])
    memory['memory_query_feat'].requires_grad_()
    out = adapter(torch.randn(2, 2, 8), torch.zeros(2, 2, 2, 3),
                  torch.randn(2, 2, 2, 3), memory, 1)
    loss = sum((out[g] * out[d]).square().mean() for g, d in (
        ('gate_sem', 'delta_s'), ('gate_occ', 'delta_o'), ('gate_pos', 'delta_p')))
    loss.backward()
    grad = memory['memory_query_feat'].grad
    assert torch.isfinite(grad).all()
    assert grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(grad[:, 1:]) == 0
    for name in ('query_proj.weight', 'key_proj.weight', 'value_proj.weight',
                 'point_query_proj.weight', 'point_key_proj.weight',
                 'point_value_proj.weight', 'gate_head.weight',
                 'gate_occ_head.weight', 'gate_pos_head.weight',
                 'delta_s_head.2.weight', 'delta_o_head.2.weight',
                 'delta_p_head.2.weight', 'horizon_embedding.weight'):
        grad = dict(adapter.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
