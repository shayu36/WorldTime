"""Checkpoint the trainable adapter while retaining detached planning evidence."""
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_adapter_with_context(adapter, args, target_ego2global=None):
    query, points, logits, memory, horizon = args[:5]

    def evaluate(q, p, s):
        result = adapter(q, p, s, memory, horizon,
                         target_ego2global=target_ego2global, return_diagnostics=True)
        gate = result['gate']
        return (result['delta_s'], result['delta_o'], result['delta_p'], gate,
                result.get('gate_occ', gate), result.get('gate_pos', gate),
                result['context'].detach(), result['diagnostics']['has_candidate'].detach())

    ds, do, dp, gs, go, gp, context, valid = checkpoint(
        evaluate, query, points, logits, use_reentrant=False, preserve_rng_state=True)
    return dict(delta_s=ds, delta_o=do, delta_p=dp, gate=gs, point_gate=gs,
                gate_sem=gs, gate_occ=go, gate_pos=gp, context=context.detach(),
                diagnostics=dict(has_candidate=valid.detach()))
