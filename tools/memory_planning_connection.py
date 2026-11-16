"""Opt-in, inference-only Memory position -> frozen planner experiment.

This is NOT the archived T1 model's native planning behavior. It has no new
parameters or training and connects only its learned POSITION corrections.
Semantic/occupancy corrections and raw Memory features are not planner inputs.
"""

import types

import torch


VARIANT = 'memory_position_read_delta_v1'
TARGET_STEPS = (1, 3, 5)  # zero-based 0.5 s increments: exactly 1/2/3 s


def corrected_planning_steps(model, ego_feat, features, outputs):
    """Return six increments; preserve the frozen scene recurrence entirely.

    For each target t, F_t and P_t are the SAME post-update Baseline snapshot
    that the adapter consumed. Let R(F,P) = traj_head(ego_cross_attn(ego,F,P)).
    New increment = native_increment + R(F_t, P_memory_t) - R(F_t, P_base_t).

    The matched subtraction avoids accidentally shifting the native planner
    from its pre-update snapshot to a post-update snapshot even with no Memory.
    All points retain the model's encoded coordinates; OPUSCrossAttention owns
    their conversion to metric coordinates. No coordinate re-encoding occurs.
    The caller converts increments to cumulative XY exactly once, as before.
    A changed increment at 1s affects subsequent cumulative positions as well.
    """
    if model.training or torch.is_grad_enabled():
        raise RuntimeError('Memory planning connection requires eval() and no_grad()')
    native = outputs['pred_trajs_list']
    base_points = outputs['baseline_forecast_points_list']
    memory_points = outputs['forecast_points_list']
    if not (len(native) == len(features) == len(base_points) ==
            len(memory_points) == 6):
        raise ValueError('Expected six aligned 0.5s scene/planning snapshots')
    planned = list(native)
    deltas = [torch.zeros_like(step) for step in native]
    for step in TARGET_STEPS:
        base, corrected, feat = base_points[step], memory_points[step], features[step]
        if (base.ndim != 4 or base.shape[-1] != 3 or
                corrected.shape != base.shape or feat.shape[:2] != base.shape[:2]
                or feat.shape[0] != ego_feat.shape[0]):
            raise ValueError(f'Misaligned scene tensors at step {step + 1}')
        changed = (corrected != base).flatten(1).any(-1)
        # Inference only: bypass both reads when the position branch is zero.
        if not bool(changed.any()):
            continue
        query_points = ego_feat.new_ones(ego_feat.shape[0], 1, 3) * 0.5
        base_read, _ = model.ego_cross_attn(query_points, ego_feat, base, feat)
        memory_read, _ = model.ego_cross_attn(query_points, ego_feat, corrected, feat)
        delta = model.traj_head(memory_read) - model.traj_head(base_read)
        if delta.shape != native[step].shape:
            raise ValueError('Planner read must produce one XY increment per batch')
        active = changed[:, None, None]
        deltas[step] = torch.where(active, delta, torch.zeros_like(delta))
        planned[step] = torch.where(active, native[step] + delta, native[step])
    return planned, torch.cat(deltas, dim=1)


def install_position_planning_connection(model, observer=None):
    """Wrap only the evaluation method, without registering modules/parameters.

    The original full forward runs once and builds unchanged occupancy outputs
    and recurrent states. A temporary cls_branch pre-hook records F_t [B,Q_t,C]
    at all six updates. It is removed even when the original forward fails.
    The wrapper reads P_base/P_memory [B,Q_t,R,3] only after recurrence finishes.
    No GT field is read by this connection. Default/native evaluation never
    calls this installer. The returned function restores the original method.
    """
    if (not model.future_memory_adapter_enabled or
            model.future_memory_adapter_version != 'v2'):
        raise ValueError('This connection requires a V2 adapter model')
    if getattr(model, '_memory_planning_connection_installed', False):
        raise RuntimeError('Memory planning connection already installed')
    original = model._forward_backbone_future_memory_adapter

    def wrapped(self, img_metas, kwargs, B, ego_feat, outs, query_feat,
                query_pos, query_cls, ind_stamps_all, memory_context):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Memory planning connection requires eval() and no_grad()')
        features = []

        def capture_snapshot(module, inputs):
            features.append(inputs[0].detach())

        handle = self.cls_branch.register_forward_pre_hook(capture_snapshot)
        try:
            outputs = original(img_metas, kwargs, B, ego_feat, outs, query_feat,
                               query_pos, query_cls, ind_stamps_all, memory_context)
        finally:
            handle.remove()
        planned, delta = corrected_planning_steps(self, ego_feat, features, outputs)
        if observer is not None:
            observer(outputs, planned, delta)
        # Replace ONLY the returned planning list. No source Tensor is mutated.
        return dict(outputs, pred_trajs_list=planned)

    model._forward_backbone_future_memory_adapter = types.MethodType(wrapped, model)
    model._memory_planning_connection_installed = True

    def restore():
        model._forward_backbone_future_memory_adapter = original
        model._memory_planning_connection_installed = False

    return restore
