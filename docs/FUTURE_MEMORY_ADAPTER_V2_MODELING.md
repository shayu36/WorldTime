# Future Memory Adapter V2 modeling notes

## 1. V1 baseline and preservation policy

Official V1 results remain unchanged and reproducible: epoch 4 Future Mean mIoU is
13.2500 (+0.0267 relative to Baseline), and epoch 6 Future Mean IoU is 22.2467
(+0.0333 relative to Baseline). `FutureMemoryAdapter`, its configurations, and checkpoint paths
have not been renamed or overwritten; V2 is enabled through the separate `FutureMemoryAdapterV2` class and separate configurations.

This stage covers code-level acceptance only, with no official V2 training or full NuScenes validation, so there are no new
IoU/mIoU conclusions.

## 2. Problems addressed by V2

V2 fixes five interrelated problems: candidate-support statistics are no longer incorrectly reduced by the top-k count; prediction
times use future effective ages; Query-level coarse retrieval is followed by Point retrieval within each Query; semantic matching is constrained by
uncertainty, dual-path candidates, and Semantic Dropout to avoid confirmation bias from Baseline errors;
training jointly supervises semantics, occupancy, threshold boundaries, and a sparse differentiable soft-voxel objective, bringing training signals
closer to the final IoU/mIoU decoding process.

## 3. Query → Point two-stage data flow

For each target horizon (1s, 2s, 3s), read the Baseline snapshot `Q_base/P_base/S_base`:

1. Query-level coarse retrieval computes feature/geometry and semantic-enhanced paths, taking
   `coarse_topk_geometry=8` and `coarse_topk_semantic=8`, respectively, and forms their union after stable deduplication.
2. Read geometry candidates and semantic candidates separately using masked softmax within the union, and fuse them with
   `coarse_geometry_context_weight` (default 0.5) to obtain each Query's own
   `H_coarse:[B,Q,C]`; incorrect semantics therefore cannot shut off the geometry path during reading.
3. For each current Query, expand only historical Queries in the union and their 48 historical points; historical
   Query semantic distributions are broadcast to points, but each historical point retains its own spatial position encoding.
4. Compute Point Attention in chunks of `query_chunk_size=64`, selecting at most
   `point_topk=8`, to obtain `H_qr`: `[B,Q,R,C]` and `G_qr`: `[B,Q,R,1]`.
   By default, half the complete attention heads (`point_geometry_head_fraction=0.5`) read only
   geometry candidates and geometry/feature/time scores, receiving no explicit semantic term; only the remaining heads use
   semantic-enhanced scores. Multi-head features are concatenated rather than divided by the number of heads.
5. Point Adapter inputs include the Query, current-point position encoding, Context, Query–Context
   differences, horizon embedding, candidate count, reliability, effective age, distance, uncertainty, and
   current/historical semantic disagreement; outputs are semantic/occupancy/position residuals and a Point Gate.

At the default scale, coarse scores have a maximum shape of `[B,Q,768]` (up to 3×256 historical Queries), and Point
Attention has a maximum shape of `[B,64,48,8,768]` (64 current Queries, 48 points, 8 heads,
and up to 16 coarse candidates×48 historical points). There is no global matrix of all current points×all historical points.

## 4. Uncertainty-aware semantic matching and dual-path candidates

Each current point uses `softmax(S_base)`. Its normalized entropy is `U_qr`, and the top1-top2 margin is also
recorded; explicit semantic weight is `(1-U_qr)^gamma`, with a lower bound imposed by
`semantic_weight_floor`. High-entropy points reduce semantic matching without discarding spatially/feature-consistent Memory.

Coarse retrieval always includes two paths:

```text
A_geometry = A_feature - A_distance - A_effective_age + A_reliability
A_semantic = A_geometry + w_sem * A_semantic_compatibility
M_q = stable_unique(TopK(A_geometry) ∪ TopK(A_semantic))
```

The geometry path is always retained, allowing candidates to enter the union even when explicit semantics are wrong. During training,
`semantic_dropout_probability` can replace current semantics with a uniform distribution; it is strictly disabled during evaluation.
The Point Adapter receives complete current/historical per-point semantic probabilities, semantic-difference vectors, cosine similarity, and
JS divergence. Nearby Memory with inconsistent semantics can therefore be used for correction rather than being hard-deleted
at retrieval.

## 5. Effective age

The cache stores only base age, `base_age = t_current - t_history`. V2 creates a
new tensor for each target horizon without modifying the cache:

```text
effective_age = base_age + horizon_seconds
1s → +1.0s, 2s → +2.0s, 3s → +3.0s
```

Maximum-age filtering, temporal scores, support age, and Adapter diagnostics all use
`effective_age`; diagnostics also retain `base_age`, `effective_age`, and
`horizon_seconds`.

Candidate statistics use actual weight mass:

```python
valid_weights = weights * top_valid.to(weights.dtype)
weight_mass = valid_weights.sum(-1).clamp_min(EPS)
support = (valid_weights * selected).sum(-1) / weight_mass
```

They are no longer divided by the number of valid candidates, and remain finite for partially invalid candidates, dropout, no candidates, and multi-head aggregation.

## 6. Semantic / occupancy decoupling

V2 produces:

```text
S_sem = S_base + G * DeltaS
O     = max_c(S_base) + G * DeltaO
S_decode = S_sem + (O - max_c(S_sem))
```

Thus, `argmax(S_decode) == argmax(S_sem)` and `max(S_decode) == O`. Class ranking is determined only by the
semantic branch, and crossing a class threshold is determined only by the occupancy branch; the existing `get_occ()`
can still use `S_decode`. Zero residuals or no valid Memory reduce strictly to Baseline.

## 7. New losses

Only the following losses are added at each target horizon, with all weights in the V2 configuration:

```text
mem_1s.loss_base_cls (1.0)
mem_1s.loss_sem       (0.25)
mem_1s.loss_occ       (0.25)
mem_1s.loss_occ_anchor(0.05)
mem_1s.loss_threshold (0.10)
mem_1s.loss_soft_voxel(0.10)
mem_1s.loss_pts       (0.50)
```

`mem_2s.*` and `mem_3s.*` have the same structure. The original future classification loss is retained as the stable
`loss_base_cls` anchor, and original point matching is retained as `loss_pts`. Occupancy positives are first determined by membership
in 0.4m non-free GT voxels, consistent with final evaluation; points that fall just inside an adjacent free voxel but remain close to GT
are supplemented by proximity to a non-free voxel center. The default proximity radius comes from half the voxel's three-dimensional diagonal:
`0.5 * ||voxel_size||_2 = 0.3464m`, replacing 0.20m, which covers only the cube's inscribed sphere.
Points at corners of non-free voxels therefore will not be incorrectly supervised as free.

Occupancy BCE and threshold margin loss calculate separate positive- and negative-group means, then combine them using the configured
`occupancy_{positive,negative}_weight` and
`threshold_{positive,negative}_weight`, preventing numerically dominant negative samples from suppressing all future points.
`loss_occ_anchor` uses Smooth-L1 to conservatively anchor the independent occupancy logit `O` to the frozen Baseline's
`max_c(S_base)`; its default weight, 0.05, is below the main occupancy loss weight of 0.25, suppressing only global recall
collapse without disabling the learnable occupancy residual. Semantic loss uses existing
class weights only on reliably matched occupied points; threshold loss dynamically converts `pts_bbox_head.test_cfg.score_thr` into
logit thresholds. SoftVoxel uses trilinear weights and noisy-OR over the 8 continuous-coordinate voxels around each point,
computing Soft-IoU only on the sparse union of predicted voxels and non-free GT voxels, while retaining gradients for semantic logits,
occupancy logits, and continuous point positions.

No Memory loss is added at 0s, and 0.5s, 1.5s, and 2.5s also do not receive V2 losses.

This correction arose from reduced future occupancy recall observed in forward evaluation: the old 0.20m center-distance rule treated
valid points inside a 0.4m voxel but more than 0.20m from its center as negatives; the independent occupancy
branch then suppressed all classes below threshold through `max(S_decode)=O`. This is a mismatch between the training objective and
the voxel-evaluation definition, rather than an error in semantic argmax itself. V2
checkpoints trained with the old objective will not automatically change after the code fix; subsequent validation should start
new V2 training from the frozen Baseline checkpoint. This document does not claim IoU/mIoU improvements on that basis.

## 8. Recurrence isolation and freezing boundaries

V2 only reads the Baseline snapshot at each target horizon. Context, gates, corrected logits, occupancy
logits, corrected points, and corrected query features are never written back into OPUS/SparseWorld future recurrence;
changing the 1s output does not change the 2s Baseline state, and changing the 2s output does not change the 3s state.

The V2 optimizer allowlist contains only:

```text
future_memory_adapter_v2.*
```

Baseline, original query encoding, the OPUS head, recurrence, and original classification/regression branches are all frozen,
and frozen modules remain in eval mode. V2 q/k/v, position, and semantic projections use normal Xavier initialization; gate
bias is -1, and the last layers of the three residual heads are zero-initialized; there is no `1e-3` global scaling.

## 9. Candidate isolation and coarse Context within candidates

Early V2 combined coarse_k.mean and coarse_v.mean over all Memory Queries into
coarse_summary and injected it into every current Query. This bypassed memory_valid, effective age,
maximum age, spatial radius, and the Top-K union, and has now been completely removed.

Each Query's coarse read is now strictly restricted to:

    U_q = stable_unique(TopK(A_geometry) union TopK(A_semantic))
    alpha_qm^g = MaskedSoftmax(A_geometry[q,m]), m in U_q^g
    alpha_qm^s = MaskedSoftmax(A_semantic[q,m]), m in U_q^s
    H_g[q] = sum(m in U_q^g) alpha_qm^g * V_m
    H_s[q] = sum(m in U_q^s) alpha_qm^s * V_m
    H_coarse[q] = lambda_g H_g[q] + lambda_s H_s[q]

Here, `lambda_g` is at least the configured nonzero lower bound when both paths have candidates; when only one path is
valid, the other coefficient is zero. `H_coarse` has shape [B,Q,C]. A candidate selected more than once retains only
its first occurrence, and duplicate semantic slots are not accumulated again; each path's original continuous scores
participate in softmax within its own deduplicated candidate slots. When all candidates are invalid, both alpha groups and H_coarse
are strictly zero.
Memory that is unselected, has memory_valid=False, exceeds the effective-age limit, or lies outside the coarse radius cannot enter
coarse Context or expand into Point Attention. The current point's own qpoint serves only as the Query
condition for Point Attention and is not accumulated into historical Point Memory Context.

Point Attention likewise expands only historical points from historical Queries in `U_q`. Geometry heads'
masked softmax sees only geometry slots and adds no explicit semantic-compatibility term; semantic heads see the stable
union and add an uncertainty-weighted semantic term. Thus, incorrect high-confidence Baseline semantics cannot suppress geometry
candidate read weights to zero.

## 10. Complete per-point semantic disagreement features

Semantic Dropout modifies only retrieval conditions P_retrieval and w_retrieval, not Baseline
logits, and does not replace the P_current used by the Adapter. Historical Query-level semantics are broadcast to historical points
within candidates and produce a separate P_memory for each current point after Point Attention.

The Adapter explicitly receives the following shapes:

    P_current:             [B,Q,R,17]
    P_memory:              [B,Q,R,17]
    P_current-P_memory:    [B,Q,R,17]
    cosine_similarity:     [B,Q,R,1]
    JS_divergence:         [B,Q,R,1]
    current_entropy:       [B,Q,R,1]
    top1_top2_margin:      [B,Q,R,1]
    semantic_weight:       [B,Q,R,1]

With no valid candidates, P_memory, the semantic-difference vector, cosine, and JS are safely set to zero, and the Point Gate is zero.
The Adapter input width is derived from module parameters as
5*embed_dims + 3*num_classes + 10, which defaults to 1341; Query count, point count, and
class count are not hard-coded. Semantic disagreement still does not act as a hard mask, and the geometry branch remains available throughout.

## 11. Eligible and selected candidate counts

- eligible_point_count [B,Q,R]: the number of historical points eligible for Point Top-K selection after passing Memory validity, effective age, coarse-candidate
  restrictions, and point radius.
- selected_point_count [B,Q,R]: the number of valid historical points actually participating in Point Attention
  after Point Top-K, no greater than point_topk.

The Adapter uses both: eligible is divided by this Query's coarse-candidate expansion capacity
Ku*R_memory, and selected is divided by point_topk. Padding and invalid candidates are not counted; with no candidates,
both are zero.

## 12. Strict identity of position residuals

SparseWorld point proposals have been verified to use the linear normalized coordinates defined by
encode_points/decode_points. V2 processes only points with candidates and nonzero gates, computing relative encoding changes:

    decoded_base = decode(base_pos)
    encoded_base = encode(decoded_base)
    encoded_shifted = encode(decoded_base + delta_position_metric)
    encoded_delta = encoded_shifted - encoded_base
    output_pos = base_pos + encoded_delta

When delta_position_metric is zero, both encode calls have identical inputs, so encoded_delta is strictly
zero; with no candidates or a zero gate, the original base_pos is returned directly. Nonzero residuals are still converted using pc_range,
inputs are not modified in place, and the position-learning path is not detached.

V1 does not call this relative-encoding helper. V1's `_refine_future_memory_points` retains the original
`encode(decode(base_pos) + delta)` path; V2 separately calls
`_refine_future_memory_points_v2`, preserving the execution path of V1 checkpoints.

## 13. Unified cls_weights conversion

Weight-source priority is unchanged: first read pts_bbox_head.train_cfg['cls_weights'], and if it is absent,
fall back to model class_weights. Lists, tuples, CPU Tensors, and CUDA Tensors all ultimately use
torch.as_tensor with S_sem's device and dtype, then flatten to one dimension.

After conversion, the element count is strictly checked against num_classes; errors report both actual and expected counts.
The original configuration object is unchanged. The same conversion function is also used by sparse SoftVoxel loss.

## 14. Acceptance scope for this round

This round completed only static checks, Python compilation, configuration parsing, and small pure-function/forward
acceptance checks without backpropagation, covering B=1/B=2, Memory smaller/larger than Top-K, candidate isolation, semantic features, strict position
identity, and loss-forward boundary inputs.

This round did not run backward, torch.autograd.grad, optimizer steps, gradient acceptance checks, training
data iteration, a 200-iteration smoke test, V1 control experiments, official V2 training, or full nuScenes
evaluation. The current conclusion is only that known code issues have been corrected and static/forward acceptance passed; gradient
acceptance and the pre-training smoke test remain outstanding. This does not establish official V2 training conclusions, and there are no new
IoU/mIoU improvement conclusions.

For subsequent V2 training on 24 GB cards, the separate configuration defaults to
`activation_checkpoint=True` and `save_on_cpu=False`. Point Attention and the Adapter
MLP still process `query_chunk_size` chunks, but chunking only reduces transient workspace; it cannot shorten the lifetime of the backward graph
saved until the unified backward pass. Training therefore uses non-reentrant activation
checkpointing to recompute V2 during backward rather than saving complete point-level graphs for 1s/2s/3s simultaneously.
The checkpointed path returns only delta_s/delta_o/delta_p/gate needed for losses, without constructing large audit
diagnostics; evaluation and ordinary forward tests retain full diagnostics.

The previously attempted full `save_on_cpu=True` is now a non-default option. In an actual test with query_chunk_size=1,
this scheme avoided CUDA OOM, but thousands of chunks across three horizons accumulated host saved tensors
to approximately 128.9 GB, eventually triggering the Linux OOM killer (SIGKILL -9). It is therefore only a debugging fallback
and cannot be combined with small chunks for official training. Activation checkpointing does not change forward values, candidate
sets, Baseline recurrence, or V1 paths; its cost is extra V2 recomputation during backward.