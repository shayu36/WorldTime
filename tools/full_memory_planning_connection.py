"""Inference-only connection of all three learned Memory OUTPUT branches.

Uses T1's semantic projection, occupancy log-priors and corrected positions.
No new parameters, no raw-context addition, no scene recurrence modification.
The existing projection is trained for Memory retrieval, NOT for this planner:
this new connection must be evaluated as an untrained inference experiment.
"""

import types

import torch
from torch.nn import functional as F

from memory_planning_connection import TARGET_STEPS


VARIANT = 'memory_full_output_read_delta_v1'
BRANCHES = ('semantic', 'occupancy', 'position')
PROBABILITY_FLOOR = 1e-6  # Numerical log floor, not a detection threshold.


def full_scene_inputs(features, base_points, memory_points, base_logits,
                      semantic_logits, occupancy_logits, semantic_projection,
                      branches=BRANCHES, probability_floor=PROBABILITY_FLOOR):
    """Map learned, already-gated outputs into frozen planner inputs.

    features [B,Q,C], points [B,Q,R,3], logits [B,Q,R,K], occupancy [B,Q,R,1].
    dF = W_sem * mean_R(softmax(S_sem)-softmax(S_base)), shape [B,Q,C].
    b_occ = log(mean_R(sigmoid(O))) - log(mean_R(sigmoid(max(S_base)))),
    shape [B,Q]. Adding b_occ to attention scores changes relative key weights.
    A uniform b_occ across every Query cancels under softmax, as expected for
    an attention prior. This is not an absolute occupancy/collision optimizer.

    W_sem is the checkpoint's existing semantic_query_proj.weight. Its bias
    cancels in a residual projection and is not added. Task gates are already
    included in S_sem, O and P_memory; do NOT multiply them a second time.
    """
    if not set(branches).issubset(BRANCHES):
        raise ValueError(f'Unknown planning branches: {branches}')
    if not 0 < probability_floor < 1:
        raise ValueError('probability_floor must lie in (0,1)')
    if (features.ndim != 3 or base_points.ndim != 4 or
            base_points.shape[-1] != 3 or memory_points.shape != base_points.shape or
            features.shape[:2] != base_points.shape[:2] or
            base_logits.shape[:3] != base_points.shape[:3] or
            semantic_logits.shape != base_logits.shape or
            occupancy_logits.shape != base_logits.shape[:-1] + (1,) or
            semantic_projection.weight.shape != (features.shape[-1], base_logits.shape[-1])):
        raise ValueError('Misaligned full-Memory planning inputs or semantic projection')

    semantic_delta = torch.zeros_like(features)
    if 'semantic' in branches:
        probability_delta = (semantic_logits.float().softmax(-1) -
                             base_logits.float().softmax(-1)).mean(-2)
        semantic_delta = F.linear(probability_delta,
                                  semantic_projection.weight.float(), None).to(features.dtype)
    log_prior = features.new_zeros(features.shape[:2], dtype=torch.float32)
    if 'occupancy' in branches:
        base_occupancy = base_logits.max(-1, keepdim=True).values
        base_probability = base_occupancy.float().sigmoid().mean(-2).squeeze(-1)
        memory_probability = occupancy_logits.float().sigmoid().mean(-2).squeeze(-1)
        log_prior = (memory_probability.clamp_min(probability_floor).log() -
                     base_probability.clamp_min(probability_floor).log())
    connected_features = features + semantic_delta
    connected_points = memory_points if 'position' in branches else base_points
    changed = ((connected_features != features).flatten(1).any(-1) |
               (connected_points != base_points).flatten(1).any(-1) |
               (log_prior != 0).any(-1))
    diagnostics = dict(
        semantic_feature_delta_max=float(semantic_delta.abs().max()),
        occupancy_log_prior_max=float(log_prior.abs().max()),
        encoded_position_delta_max=float((connected_points-base_points).abs().max()))
    return connected_features, connected_points, log_prior, changed, diagnostics


def read_with_occupancy_prior(model, ego_feat, points, features, log_prior):
    """Use the real OPUSCrossAttention; add only a scoped soft attention prior.

    Native attn_mask [B*H,N_ego,Q]. prior [B,Q] expands in B,H order, without
    merging examples. No mask is modified in-place; the temporary hook is
    always removed, including on exceptions. Zero priors bypass the hook.
    """
    query_points = ego_feat.new_ones(ego_feat.shape[0], 1, 3) * 0.5
    if not bool((log_prior != 0).any()):
        fused, _ = model.ego_cross_attn(query_points, ego_feat, points, features)
        return model.traj_head(fused)
    attention = model.ego_cross_attn.attention
    batch, queries = log_prior.shape
    calls = []

    def add_prior(module, args, kwargs):
        mask = kwargs.get('attn_mask')
        if (mask is None or mask.ndim != 3 or mask.shape[-1] != queries or
                mask.shape[0] % batch != 0 or not mask.is_floating_point()):
            raise ValueError('Expected OPUS floating attention mask [B*H,N_ego,Q]')
        heads = mask.shape[0] // batch
        bias = log_prior[:, None, None, :].expand(batch, heads, mask.shape[1], queries)
        bias = bias.reshape_as(mask).to(device=mask.device, dtype=mask.dtype)
        calls.append(True)
        return args, dict(kwargs, attn_mask=mask + bias)

    handle = attention.register_forward_pre_hook(add_prior, with_kwargs=True)
    try:
        fused, _ = model.ego_cross_attn(query_points, ego_feat, points, features)
        if len(calls) != 1:
            raise RuntimeError('Occupancy prior must reach exactly one planner attention call')
        return model.traj_head(fused)
    finally:
        handle.remove()


def corrected_full_planning_steps(model, ego_feat, features, outputs,
                                  branches=BRANCHES, probability_floor=PROBABILITY_FLOOR):
    if model.training or torch.is_grad_enabled():
        raise RuntimeError('Full Memory planning connection requires eval() and no_grad()')
    names = ('pred_trajs_list', 'baseline_forecast_points_list', 'forecast_points_list',
             'baseline_forecast_semantics_list', 'forecast_semantic_logits_list',
             'forecast_occupancy_logits_list')
    if len(features) != 6 or any(len(outputs[name]) != 6 for name in names):
        raise ValueError('Expected six aligned 0.5s scene/planning snapshots')
    native = outputs['pred_trajs_list']
    planned = list(native)
    deltas = [torch.zeros_like(value) for value in native]
    diagnostics = []
    for step in TARGET_STEPS:
        base_points = outputs['baseline_forecast_points_list'][step]
        feat, points, prior, changed, stats = full_scene_inputs(
            features[step], base_points, outputs['forecast_points_list'][step],
            outputs['baseline_forecast_semantics_list'][step],
            outputs['forecast_semantic_logits_list'][step],
            outputs['forecast_occupancy_logits_list'][step],
            model.future_memory_adapter_v2.semantic_query_proj, branches, probability_floor)
        diagnostics.append(dict(horizon_seconds=(step + 1) / 2, **stats))
        if not bool(changed.any()):
            continue
        base_read = read_with_occupancy_prior(
            model, ego_feat, base_points, features[step], torch.zeros_like(prior))
        memory_read = read_with_occupancy_prior(model, ego_feat, points, feat, prior)
        delta = memory_read - base_read
        if delta.shape != native[step].shape:
            raise ValueError('Planner read must produce one XY increment per batch')
        active = changed[:, None, None]
        deltas[step] = torch.where(active, delta, torch.zeros_like(delta))
        planned[step] = torch.where(active, native[step] + delta, native[step])
    return planned, torch.cat(deltas, dim=1), diagnostics


def install_full_planning_connection(model, observer=None, verify_branches=0):
    """Opt-in full output connection, preserving all native model state and keys.

    verify_branches only performs extra planner reads on the first N forwards,
    using the SAME cached snapshots. It does not rerun or change the adapter,
    nor use labels. These branch checks are not separate validation metrics.
    """
    if (not model.future_memory_adapter_enabled or
            model.future_memory_adapter_version != 'v2'):
        raise ValueError('This connection requires a V2 adapter model')
    if getattr(model, '_memory_planning_connection_installed', False):
        raise RuntimeError('Memory planning connection already installed')
    original = model._forward_backbone_future_memory_adapter
    forwards = 0

    def wrapped(self, img_metas, kwargs, B, ego_feat, outs, query_feat,
                query_pos, query_cls, ind_stamps_all, memory_context):
        nonlocal forwards
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Full Memory planning connection requires eval() and no_grad()')
        features = []
        handle = self.cls_branch.register_forward_pre_hook(
            lambda module, inputs: features.append(inputs[0].detach()))
        try:
            outputs = original(img_metas, kwargs, B, ego_feat, outs, query_feat,
                               query_pos, query_cls, ind_stamps_all, memory_context)
        finally:
            handle.remove()
        planned, delta, diagnostics = corrected_full_planning_steps(self, ego_feat, features, outputs)
        audit = dict(input_changes=diagnostics)
        if forwards < verify_branches:
            audit['isolated_branch_increment_max'] = {}
            for branch in BRANCHES:
                _, branch_delta, _ = corrected_full_planning_steps(
                    self, ego_feat, features, outputs, branches=(branch,))
                audit['isolated_branch_increment_max'][branch] = float(branch_delta.abs().max())
        self._full_memory_planning_audit = audit
        forwards += 1
        if observer is not None:
            observer(outputs, planned, delta)
        return dict(outputs, pred_trajs_list=planned)

    model._forward_backbone_future_memory_adapter = types.MethodType(wrapped, model)
    model._memory_planning_connection_installed = True

    def restore():
        model._forward_backbone_future_memory_adapter = original
        model._memory_planning_connection_installed = False

    return restore
