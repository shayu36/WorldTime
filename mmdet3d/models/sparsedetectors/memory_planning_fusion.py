"""Trainable, zero-initialized Memory-to-planning residual fusion."""

import torch
from torch import nn


class MemoryPlanningFusion(nn.Module):
    """Read frozen per-Query Memory evidence and predict one XY increment.

    Frozen point evidence is pooled within each Query using its candidate mask.
    Only this small token encoder / ego attention / trajectory head is trained.
    No historical Query outside the upstream adapter's selected set is read.
    """

    def __init__(self, embed_dims=256, num_classes=17, hidden_dims=128,
                 num_heads=4, horizon_count=3, voxel_size=(.4, .4, .4)):
        super().__init__()
        if hidden_dims % num_heads or min(embed_dims, num_classes, hidden_dims,
                                         num_heads, horizon_count) <= 0:
            raise ValueError('Positive dimensions and divisible attention heads required')
        if len(voxel_size) != 3 or min(voxel_size) <= 0:
            raise ValueError('voxel_size must contain three positive sizes')
        self.embed_dims = int(embed_dims)
        self.num_classes = int(num_classes)
        self.voxel_size = tuple(float(x) for x in voxel_size)
        # Query / Point Context, 3 semantic vectors, 3 occupancy scalars,
        # base/Memory centers + metric displacement (9), task gates (3),
        # candidate fraction (1). No hardcoded class/point/Query counts.
        self.token_dims = 2 * embed_dims + 3 * num_classes + 16
        self.token_encoder = nn.Sequential(
            nn.Linear(self.token_dims, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.GELU(), nn.Linear(hidden_dims, hidden_dims))
        self.ego_encoder = nn.Sequential(nn.Linear(embed_dims, hidden_dims),
                                         nn.LayerNorm(hidden_dims), nn.GELU())
        self.horizon_embedding = nn.Embedding(horizon_count, hidden_dims)
        self.attention = nn.MultiheadAttention(hidden_dims, num_heads,
                                               dropout=0., batch_first=True)
        self.residual_head = nn.Sequential(
            nn.Linear(2 * hidden_dims + 2, hidden_dims), nn.GELU(),
            nn.Linear(hidden_dims, 2))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @torch.no_grad()
    def pack_scene(self, query_feat, base_points, memory_points, base_logits,
                   semantic_logits, occupancy_logits, adapter_result, pc_range):
        """Frozen inputs -> [B,Q,D] tokens and [B,Q] valid mask.

        Positions stay in the model's linear normalized encoding; displacement
        is converted to meters using the confirmed pc_range then voxel units.
        Semantic input is S_sem, never occupancy-shifted S_decode. All logits
        and gates are post-adapter outputs (do not apply task gates twice).
        """
        B, Q, R, K = base_logits.shape
        context = adapter_result.get('point_context', adapter_result.get('context'))
        valid = adapter_result.get('diagnostics', {}).get('has_candidate')
        if context is None or valid is None:
            raise RuntimeError('Frozen V2 must return Point Context and candidate diagnostics')
        if (K != self.num_classes or query_feat.shape != (B, Q, self.embed_dims) or
                context.shape != (B, Q, R, self.embed_dims) or valid.shape != (B, Q, R) or
                base_points.shape != (B, Q, R, 3) or memory_points.shape != base_points.shape or
                semantic_logits.shape != base_logits.shape or occupancy_logits.shape != (B, Q, R, 1)):
            raise ValueError('Memory planning snapshot shape mismatch')
        valid = valid.bool()
        mass = valid.float().sum(-1, keepdim=True)
        weights = valid.float().unsqueeze(-1) / mass.clamp_min(1.).unsqueeze(-1)

        def pool(value):
            # Mask before multiplication; invalid padded features do not enter
            # the summary even if their unselected values are nonfinite.
            value = torch.where(valid.unsqueeze(-1), value.float(), 0.)
            return (value * weights).sum(-2)

        base_prob = base_logits.float().softmax(-1)
        semantic_prob = semantic_logits.float().softmax(-1)
        base_occ = base_logits.float().max(-1, keepdim=True).values
        extent = torch.as_tensor(pc_range, device=query_feat.device, dtype=torch.float32)
        extent = extent[3:] - extent[:3]
        voxel = extent.new_tensor(self.voxel_size)
        gates = [adapter_result.get(name, adapter_result['gate'])
                 for name in ('gate_sem', 'gate_occ', 'gate_pos')]
        tokens = torch.cat([
            query_feat.float(), pool(context), pool(base_prob), pool(semantic_prob),
            pool(semantic_prob-base_prob), pool(base_occ.sigmoid()),
            pool(occupancy_logits.float().sigmoid()), pool(occupancy_logits-base_occ),
            pool(base_points), pool(memory_points), pool((memory_points-base_points)*extent/voxel),
            *[pool(gate.float().mean(-1, keepdim=True)) for gate in gates], mass / R,
        ], dim=-1)
        query_valid = valid.any(-1)
        tokens = torch.where(query_valid.unsqueeze(-1), tokens, 0.)
        if tokens.shape != (B, Q, self.token_dims):
            raise RuntimeError('Planning token width differs from constructed encoder')
        return tokens.detach(), query_valid.detach()

    def forward(self, ego_feat, tokens, query_valid, horizon_id, native_increment):
        # Keep the tiny learned branch in FP32 even when the frozen image model
        # runs under MMCV AMP. Inputs are frozen; the following ops are NOT
        # detached/no_grad and carry the trajectory-loss gradient.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            encoded = self.token_encoder(tokens.detach().float())
            ego = self.ego_encoder(ego_feat.detach().float())
            horizon = self.horizon_embedding(
                torch.as_tensor(horizon_id, device=tokens.device, dtype=torch.long))
            query = ego + horizon
            # Always-valid zero null key makes the all-empty case finite.
            null = encoded.new_zeros(encoded.shape[0], 1, encoded.shape[-1])
            keys = torch.cat([encoded, null], 1)
            padding = torch.cat([~query_valid.bool(), torch.zeros(
                query_valid.shape[0], 1, device=tokens.device, dtype=torch.bool)], 1)
            context, _ = self.attention(query, keys, keys, key_padding_mask=padding,
                                        need_weights=False)
            hidden = torch.cat([ego, context, native_increment.detach().float()], -1)
            delta = self.residual_head(hidden)
            # Tensor mask preserves a live (possibly zero) graph on every rank.
            delta = delta * query_valid.any(-1)[:, None, None].to(delta.dtype)
        return delta.to(native_increment.dtype)
