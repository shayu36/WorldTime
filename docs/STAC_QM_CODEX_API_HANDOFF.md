# STAC-QM Codex API Continuation Handoff

Last updated: 2026-08-31 UTC (historical handoff; current results consolidated in `docs/EXPERIMENT_DEBUG_REPORT.md`)

This document is a historical self-contained handoff for a prior Codex API
session. It summarizes the repository state, user constraints, completed
implementation, and the Phase 3 design as it was proposed at that time. For
the authoritative current Phase 2/3/4 and ablation results, read
`docs/EXPERIMENT_DEBUG_REPORT.md`. Do not use the old proposal sections as a
status report.

## 1. Current Decision

Phase 2 is complete and negative under the agreed criterion.

- The epoch-56 Memory-OFF baseline future mIoU is `13.2233`.
- Phase 2 completed all 12 epochs and validated every epoch on 4,219 samples.
- Its best checkpoint is epoch 4 with future mIoU `13.2233`, exactly tied with
  the baseline. No epoch exceeded the baseline.
- Epoch 12 ended at `13.1833`, or `-0.0400` versus the baseline.
- Unfreezing `pts_bbox_head` did not create a measurable Memory benefit.
- Per the prior agreement, do not extend Phase 2, widen its trainable boundary,
  refresh caches, or start an unplanned experiment. The next work is an
  architecture-level discussion and, only after user confirmation, a new phase.

The leading hypothesis has two parts:

1. `gate_bias=-4` plus a zero-initialized fusion `out_proj` severely suppresses
   the useful gradient path into attention and motion parameters.
2. STAC-QM is applied after the six-layer OPUS occupancy decoder has already
   produced its 0s logits and points. The 0s train/test path therefore does not
   directly consume Memory, even in `memory_phase2_finetune_mode`.

The recommended next architecture combines a nonzero Memory projection with a
small LayerScale/ReZero coefficient and a Memory-conditioned refinement block
whose outputs are used by both 0s occupancy and future SCF.

## 2. Repository and Environment

```text
Repository:    /data/jxy/projects
Branch:        master
HEAD:          76b3c99
Python:        /data/jxy/projects/env/bin/python
PyTorch:       2.0.1+cu118
GPU:           2 x NVIDIA RTX 3090, 24 GB each
Dataset:       nuScenes under data/
Base weights:  ckpts/epoch_56.pth
```

Epoch-56 schema-v2 caches:

```text
train: data/query_memory/sparseworld_epoch56_schema2_train
       19,730 records, approximately 8.1 GB
val:   data/query_memory/sparseworld_epoch56_schema2_val
       4,219 records, approximately 1.8 GB
test:  routed to the fixed val cache
```

The val cache is the fixed comparison reference and must not be regenerated.

Phase 2 artifacts currently occupy approximately 8.0 GB:

```text
work_dirs/sparseworld-traj-memory-phase2/
  epoch_1.pth ... epoch_12.pth
  latest.pth -> epoch_12.pth
  train.log
```

These are generated artifacts and must remain uncommitted.

## 3. Hard User Constraints

These constraints remain active unless the user explicitly changes them.

1. Codex must not run commands expected to take more than a few minutes,
   including training, full evaluation, cache generation, or planning
   evaluation. Provide the exact command for the user to run instead.
2. Do not run `git push`, force push, destructive Git commands, or broad
   recursive deletion.
3. Do not commit datasets, cache tensors, checkpoints, `output_data.pkl`, large
   logs, predictions, or evaluation artifacts.
4. Never modify the original baseline config:
   `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py`.
5. Preserve the epoch-56 cache feature-space boundary. Do not unfreeze
   `img_backbone` or `img_neck` while using fixed epoch-56 Query caches.
6. After user-run training or evaluation, record the actual result in the
   handoff document before proceeding.
7. If a new architecture still produces no improvement, stop and discuss the
   evidence rather than autonomously multiplying experiments.
8. Correctness and causal interpretability are preferred over a minimal diff.

## 4. Dirty Worktree: Preserve User-Owned State

Current `git status --short` snapshot (historical, captured at the 2026-08-21 handoff; not the current worktree):

```text
 D DIAGNOSTIC_REPORT.md
 D README.md
 M docs/STAC_QM_Implementation.md
 D docs/install.md
 M mmdet3d/apis/train.py
 M mmdet3d/core/hook/__init__.py
 M mmdet3d/core/hook/query_memory_training_hook.py
 M mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py
 D setup.cfg
 D setup.py
 D setup_remaining.sh
 M tests/test_query_memory_integration.py
 M tests/test_sparseworld_evaluation.py
?? configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2-smoke.py
?? configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py
?? docs/PHASE2_HANDOFF.md
?? docs/archive/conversation_transcript.md
?? docs/STAC_QM_CODEX_API_HANDOFF.md
```

The deletions and unrelated untracked files in this historical snapshot were
not authorization to restore, delete, or commit them. `docs/install.md` remains
outside this organization task. The conversation transcript is now archived at
`docs/archive/conversation_transcript.md`. Treat unrelated status entries as
user-owned unless the user explicitly says otherwise.

## 5. Completed STAC-QM Foundation

The following implementation already exists and must not be reimplemented from
scratch.

### 5.1 Memory representation and cache

`mmdet3d/models/sparsedetectors/query_memory.py` contains:

- `STACQueryMemory`
- `CausalQueryMemoryAttention`
- `ConfidenceGatedFusion`
- `QueryMotionCompensator`
- `QueryMemoryBank`
- deterministic history and Query selection utilities

Schema-v2 cache records contain, after fixed-shape loading:

```text
memory_query_feat
memory_points_metric
memory_conf
memory_reliability
memory_label                 optional in model context
memory_valid
memory_source_ego2global
memory_age
```

Important invariants:

- Observation Queries read history once before SCF.
- Each of six scheduled future Query groups reads once when introduced.
- Already active Queries are never reread.
- Expected count is 7 reads and 1,040 fused Queries per forward.
- Future-aware age is `effective_age = base_age + future_offset`.
- Cache timestamps themselves are immutable.
- History target ages are `[2.5, 3.5, 4.5]` seconds with tolerance `0.35`.
- History is strictly past, same-scene, unique per slot, and deterministic.
- Motion compensation starts as ego-only alignment because the learned object
  motion residual is zero-initialized.

### 5.2 Training and evaluation safety

Implemented mechanisms include:

- exact trainable-prefix allowlists for Memory-only, joint, and Phase 2 modes;
- frozen modules held in eval mode, including frozen BN/dropout behavior;
- `TrainableOnlyOptimizerConstructor` that includes only trainable parameters;
- post-checkpoint-load validation of optimizer membership and LR tiers;
- resume-safe LR validation using `initial_lr` and a common scheduler ratio;
- TASS timestamp assignment and RAP masks frozen and hash-checked;
- differentiable, numerically exact empty-history identity fallback;
- connectivity hooks checking target gradients, frozen state, TASS, and 7/1040;
- SparseWorld-specific DDP evaluation with correct 4,219-sample trimming;
- flattened scalar IoU/mIoU logging compatible with TensorBoard.

Latest static/synthetic verification before the formal run:

```text
59 passed, 19 warnings
py_compile passed
git diff --check passed
```

The short verification commands that Codex may run are:

```bash
cd /data/jxy/projects
/data/jxy/projects/env/bin/python -m pytest -q tests/test_query_memory.py \
  tests/test_query_memory_integration.py tests/test_sparseworld_evaluation.py
/data/jxy/projects/env/bin/python -m py_compile \
  mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py \
  mmdet3d/apis/train.py \
  mmdet3d/core/evaluation/sparseworld_eval_hooks.py
git diff --check
```

## 6. Completed Training Modes

### 6.1 Memory-only

Trainable parameters: `query_memory.*` only.

Full validation result at epoch 12:

```text
Memory OFF epoch-56:
  mIoU [18.20, 14.96, 13.18, 11.53]
  future mean 13.2233

Memory-only epoch 12:
  mIoU [18.20, 14.95, 13.17, 11.51]
  future mean 13.2100
  delta -0.0133
```

Conclusion: the Memory path is active but does not improve aggregate occupancy.

### 6.2 Joint fine-tuning

Trainable modules:

```text
query_memory
position_encoder
reg_branch
vel_branch
cls_branch
ego_cross_attn
```

The best joint checkpoint was epoch 4 with future mIoU `13.2233`, tied with
the baseline. Epoch 12 was `13.2033`.

Planning results already available for the baseline and selected joint epochs:

```text
baseline L2: 0.1596 / 0.1945 / 0.2393
baseline collision: 0.0948% / 0.1007% / 0.1106%

joint epoch 4 L2: 0.16034 / 0.19570 / 0.24088
joint epoch 4 collision: 0.0830% / 0.0948% / 0.1067%
```

The collision changes were small and did not accompany an occupancy gain.

### 6.3 Phase 2 occupancy-head fine-tuning

Phase 2 added `memory_phase2_finetune_mode=True` and is already implemented.
Do not recreate it.

Trainable modules and effective base learning rates:

```text
query_memory.*       5e-5
position_encoder.*   1e-5
reg_branch.*         1e-5
vel_branch.*         1e-5
cls_branch.*         1e-5
ego_cross_attn.*     5e-6
pts_bbox_head.*      5e-6
```

Frozen and held in eval mode:

```text
img_backbone.*
img_neck.*
plan_head.*
points_scale_branch.*
traj_head.*
TASS timestamp allocation and RAP masks
```

Relevant files:

```text
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2-smoke.py
mmdet3d/core/hook/query_memory_training_hook.py
  QueryMemoryPhase2ConnectivityOptimizerHook
```

The 200-iteration smoke completed successfully with 49,812,743 trainable
scalars. It preserved 7 reads / 1,040 fused Queries, TASS, frozen parameter
hashes, and buffer hashes. `pts_bbox_head` gradients were strongly nonzero.

Key smoke diagnostics:

```text
iter 200:
  qm_grad_pts_bbox_head       55.4997
  memory_has_candidate_ratio  0.5412
  memory_gate_mean             0.0098
  memory_residual_norm         0.0009
  qm_grad_attention_q          approximately 0 at printed precision
  qm_grad_attention_k          approximately 0 at printed precision
  qm_grad_attention_v          0.0001
```

Because `memory_gate_mean` includes Queries without candidates, the approximate
gate conditional on a candidate was:

```text
0.0098 / 0.5412 = 0.0181
```

This is almost exactly `sigmoid(-4) = 0.0180`, which is evidence that the gate
remained near its initialization during smoke. It is not a formal-run trace,
because formal training disabled detailed Memory diagnostics.

## 7. Complete Phase 2 Formal Results

Command run by the user:

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 --master_port=29520 \
  tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py \
  --work-dir work_dirs/sparseworld-traj-memory-phase2 \
  --launcher pytorch --validate --deterministic \
  2>&1 | tee work_dirs/sparseworld-traj-memory-phase2/train.log
```

All values below are from the full 4,219-sample validation set.

| Epoch | mIoU [0s, 1s, 2s, 3s] | Future mean | Delta vs 13.2233 |
| ---: | --- | ---: | ---: |
| 1 | [18.19, 14.96, 13.13, 11.48] | 13.1900 | -0.0333 |
| 2 | [18.22, 14.91, 13.10, 11.45] | 13.1533 | -0.0700 |
| 3 | [18.17, 14.91, 13.12, 11.48] | 13.1700 | -0.0533 |
| 4 | [18.20, 14.97, 13.17, 11.53] | 13.2233 | +0.0000 |
| 5 | [18.23, 14.96, 13.16, 11.51] | 13.2100 | -0.0133 |
| 6 | [18.21, 14.94, 13.13, 11.51] | 13.1933 | -0.0300 |
| 7 | [18.18, 14.90, 13.13, 11.47] | 13.1667 | -0.0566 |
| 8 | [18.19, 14.92, 13.13, 11.49] | 13.1800 | -0.0433 |
| 9 | [18.22, 14.95, 13.17, 11.51] | 13.2100 | -0.0133 |
| 10 | [18.21, 14.93, 13.14, 11.51] | 13.1933 | -0.0300 |
| 11 | [18.19, 14.92, 13.15, 11.50] | 13.1900 | -0.0333 |
| 12 | [18.20, 14.92, 13.14, 11.49] | 13.1833 | -0.0400 |

There is no upward trend. The run is complete and no training process remains.

## 8. Current Forward Path and Structural Diagnosis

### 8.1 OPUS before Memory

`OPUSHead.forward()` creates learned initial points and zero Query features,
then runs six `OPUSTransformerDecoderLayer` instances. Each decoder layer does:

```text
position encoding
multi-view/multi-level image sampling
AdaptiveMixing
Query self-attention
FFN
classification branch
point refinement branch
```

The head returns final `query_feat`, `all_cls_scores`, and `all_refine_pts`.
TASS `ind_stamps_all` then divides the final Queries into the observation group
and six future groups.

Key source locations:

```text
mmdet3d/models/sparsedetectors/opus_head.py
  OPUSHead.forward
mmdet3d/models/sparsedetectors/opus_transformer.py
  OPUSTransformerDecoder.forward
  OPUSTransformerDecoderLayer.forward
```

### 8.2 STAC-QM read

For a current Query and cached history, STAC-QM:

1. decodes normalized Query points into metric coordinates;
2. aligns historical points into the current ego frame;
3. applies a learned object-motion translation `effective_age * velocity`;
4. filters by valid past age, reliability, and a 6 m spatial radius;
5. chooses at most top-16 candidates;
6. computes multi-head attention with semantic, spatial, temporal, and
   reliability terms;
7. fuses the readout through a confidence gate.

The attention score is conceptually:

```text
score = semantic dot-product
        - lambda_position * normalized squared distance
        - lambda_time * normalized age
        + lambda_reliability * log(reliability)
```

### 8.3 Why 0s does not consume Memory

In `SparseWorld4DTraj.forward_backbone`, the final OPUS values are extracted and
saved into `outputs` before `_apply_query_memory_once` is called:

```text
query_feat/query_pos/query_cls = final OPUS outputs
outputs['cls_score']            = pre-Memory current logits
outputs['refine_pts']           = pre-Memory current points
curr_query_feat                 = STAC-QM(curr_query_feat, ...)
```

During testing, 0s occupancy is explicitly decoded from
`outs['all_cls_scores'][-1]` and `outs['all_refine_pts'][-1]`, which are the
pre-Memory OPUS outputs. During training, the OPUS current occupancy losses also
use `outs` before Memory-conditioned replacement.

Memory directly affects the future recurrence because fused current and
scheduled future Query features are fed to the external `position_encoder`,
`cls_branch`, `reg_branch`, and `vel_branch`. It does not directly affect the 0s
semantic/geometry prediction.

Therefore Phase 2's unfreezing of `pts_bbox_head` allowed future losses to adapt
upstream OPUS representations, but it did not put Memory inside the six-layer
occupancy decoder or make the 0s output depend on Memory.

### 8.4 The double gradient suppression

Current `ConfidenceGatedFusion` uses:

```python
nn.init.zeros_(self.out_proj.weight)
nn.init.zeros_(self.out_proj.bias)
nn.init.constant_(self.gate_mlp[-1].bias, -4.0)

residual = self.out_proj(memory_output)
gate = torch.sigmoid(self.gate_mlp(gate_input))
fused = query_feat + has_candidate * gate * residual
```

At initialization:

- the output residual is exactly zero;
- the initial gate is approximately `0.018`;
- zero projection weights block first-step gradients from reaching the
  attention output and therefore q/k/v through this path;
- the small gate also suppresses the gradient that first updates `out_proj`.

This preserves the baseline exactly, but it creates a difficult optimization
startup. The smoke result is consistent with the model remaining close to an
identity mapping.

## 9. Historical Phase 3 Design Proposal

Status: this section records the design discussion before Phase 3 was
implemented. The actual implementation and negative/diagnostic results are in
`docs/EXPERIMENT_DEBUG_REPORT.md`.

### 9.1 Recommended first architecture

Keep the existing six-layer OPUS head and add a post-OPUS, Memory-conditioned
refinement path:

```text
image backbone + neck
        |
OPUS decoder layers 1-6
        |
q_base, p_base, s_base
        |
STAC-QM + LayerScale
        |
q_memory
        |
shared Memory-conditioned refiner + horizon embedding
        |
q_refined, delta_cls, delta_pts, delta_vel
        |
0s occupancy and future SCF both use refined state
```

Suggested fusion:

```python
memory_context = attention(...)
memory_residual = out_proj(memory_norm(memory_context))
gate = sigmoid(gate_mlp(...))
q_memory = q_base + has_candidate * alpha * gate * memory_residual
```

Recommended first ablation initialization:

```python
nn.init.xavier_uniform_(out_proj.weight, gain=0.1)
nn.init.zeros_(out_proj.bias)
nn.init.constant_(gate_mlp[-1].bias, -1.0)
alpha = nn.Parameter(torch.tensor(1e-3))
```

Rationale:

- nonzero `out_proj` provides a live gradient path into Memory attention;
- `gate_bias=-1` starts near `0.269` instead of `0.018`;
- the small nonzero `alpha` keeps the perturbation tiny while allowing q/k/v,
  gate, and projection parameters to receive gradients on the first step;
- an exact `alpha=0` ReZero start preserves strict identity, but only `alpha`
  receives the first-step gradient. Other Memory parameters begin learning only
  after `alpha` moves away from zero.

Do not blindly apply an unscaled Xavier projection followed by normalization;
it can make a noisy historical residual comparable in norm to the base Query.
Use a controlled projection scale and monitor the actual residual ratio.

### 9.2 Refiner outputs and routing

The refiner should update feature, semantics, and geometry:

```python
q_refined = memory_refiner(q_memory, p_base, horizon_embedding)
refined_cls = s_base + memory_cls_branch(q_refined)
refined_pts = refine_points(p_base, memory_reg_branch(q_refined))
```

Do not implement `refined_pts = p_base + delta_pts`. OPUS points are encoded
relative to `pc_range` and have multiple refinement samples. Reuse the existing
`refine_points()` decode/refine/encode behavior.

The implementation is incomplete unless all of these routes change:

1. The 0s training loss must use `refined_cls` and `refined_pts`.
2. The 0s test path must decode refined outputs instead of directly reading
   `outs['all_cls_scores'][-1]` and `outs['all_refine_pts'][-1]`.
3. Future SCF must start from `q_refined`, `refined_pts`, and `refined_cls`.
4. Memory diagnostics must describe the refined path actually used by losses.
5. Online Memory write semantics must be explicitly decided and tested. Do not
   accidentally switch fixed-cache training to recursive, post-Memory writes.

Use one shared refiner across 0s and future horizons, plus a horizon embedding,
before considering separate per-horizon decoders. This gives direct supervision
without multiplying parameters unnecessarily.

### 9.3 Initial trainable boundary

For the cleanest causal experiment:

```text
Train:
  query_memory.*
  memory_refiner.*
  memory_cls_branch.*
  memory_reg_branch.*
  memory_vel_branch.* if used
  horizon embedding

Freeze and hold in eval mode:
  img_backbone.*
  img_neck.*
  original OPUS decoder layers 1-6
  plan_head.*
  unrelated trajectory modules
  TASS state and RAP masks
```

This boundary forces the new branch to demonstrate a Memory contribution
instead of letting the large OPUS head absorb the optimization. If it works,
the next controlled ablation may unfreeze only original OPUS decoder layer 6 at
a low LR. Backbone and neck remain frozen with fixed caches.

### 9.4 Losses

Initial loss design:

```text
L = L_current_memory_conditioned
    + lambda_future * L_future_memory_conditioned
    + lambda_base_aux * L_original_OPUS_aux
    + existing trajectory loss where appropriate
```

Reasonable starting values for discussion, not fixed decisions:

```text
lambda_future = 1.0
lambda_base_aux = 0.25
```

Do not add an unconstrained `L_memory_consistency` in the first experiment.
Attention top-k is not an identity match. A naive consistency loss can reinforce
wrong historical associations or penalize real motion. Add consistency only
after defining a reliable pairing method such as GT/Hungarian matching, track
IDs, a motion-compensated high-confidence match, or a stop-gradient teacher.

### 9.5 Diagnostics

The next smoke hook should log and assert connectivity for:

```text
conditional_gate = memory_gate_mean / max(has_candidate_ratio, eps)
residual_ratio = norm(alpha * gate * residual) / (norm(query) + eps)

gradient norms:
  q_proj
  k_proj
  v_proj
  attention out_proj
  fusion out_proj
  gate MLP
  alpha
  memory refiner
  memory cls/reg/vel branches

behavior:
  candidate coverage
  attention entropy
  selected distance and age
  per-horizon gate and residual ratio
  empty-history identity behavior
  7 reads / 1040 fused Queries
  TASS and frozen-state hashes
```

Values such as conditional gate `0.1-0.3` and residual ratio `1%-5%` are useful
observation ranges, not success criteria. Do not force the network to hit them.
The causal success criteria are non-degenerate gradients, a measurable ON/OFF
difference, and improved validation metrics.

### 9.6 Gate warmup is deferred

A proposed alternative was:

- first 500 iterations force candidate gate to 1;
- next 500 iterations restore the learned gate;
- optionally anneal a positive gate floor to zero.

Do not include this in the first architecture experiment. Initialization,
refiner routing, and an iteration-dependent gate schedule would change several
variables simultaneously and complicate attribution and resume behavior. First
try nonzero projection, `gate_bias=-1`, and `alpha=1e-3` without warmup. Add a
DDP-safe, checkpointed gate schedule only if q/k/v gradients remain too small.

### 9.7 Deeper decoder integration

The safer deeper variant is to append a seventh Memory-conditioned OPUS decoder
layer after the current final layer:

```text
OPUS layers 1-6
  -> STAC-QM using final-layer cached Queries
  -> new OPUS refinement layer 7
  -> final occupancy outputs
```

This preserves feature-space compatibility because both current and cached
history are final-layer epoch-56 Query features. The new layer can resample
current multi-level image evidence after Memory changes the Query.

Do not directly inject final-layer cache features between OPUS layers 4/5 or
5/6 and assume they are aligned. True internal insertion requires a schema-v3
cache containing layer-specific historical Query features, or a separately
validated cross-layer adapter. That is a later, higher-cost research direction.

## 10. Minimal Next Experimental Sequence

Nothing in this sequence is authorized to run yet merely by this document.
Implement only after the user confirms the architecture.

1. Write tests for nonzero fusion gradients, alpha behavior, encoded point
   refinement, exact 0s train/test routing, and future refined-state routing.
2. Implement LayerScale fusion diagnostics without starting formal training.
3. Implement the shared Memory-conditioned 0s/future refiner.
4. Run the short unit/static suite locally.
5. Give the user a 200-iteration smoke command. Codex must not run it.
6. Require the smoke to demonstrate nonzero q/k/v and refiner gradients plus
   sensible residual ratios before giving a formal training command.
7. Run a controlled user-operated diagnostic experiment with the original six
   OPUS layers frozen.
8. Evaluate the same checkpoint with Memory ON and OFF. Add shuffled-history
   and shuffled-time controls if practical.
9. Only if the new path produces a causal benefit, consider unfreezing OPUS
   layer 6 or appending Memory-conditioned decoder layer 7.
10. If residual usage is healthy but mIoU still does not improve, stop changing
    fusion strength. Investigate cache content, matching quality, instance
    identity, motion alignment, and temporal noise instead.

The primary comparison remains:

```text
epoch-56 baseline mIoU [18.20, 14.96, 13.18, 11.53]
future mean 13.2233
```

Report 0s/1s/2s/3s separately. Small changes of `0.01-0.07` have already
appeared as epoch-level variation, so a new claim should include a same-weight
Memory ON/OFF ablation and preferably multiple seeds or a sample-level bootstrap
confidence interval.

## 11. Important Non-Directions

Do not do the following as the immediate next step:

- continue Phase 2 beyond 12 epochs;
- increase `samples_per_gpu` or LR as a substitute for architectural evidence;
- unfreeze backbone/neck against fixed epoch-56 caches;
- regenerate the fixed val cache;
- refresh the train cache mid-run without a separate, approved protocol;
- add consistency loss before defining reliable historical/current matches;
- change gate initialization, warmup, top-k, radius, cache policy, trainable
  boundary, and decoder location in one experiment;
- start with multi-scale or world-model scope before testing the direct 0s
  Memory-conditioned path.

## 12. File Map

```text
Primary handoffs:
  docs/EXPERIMENT_DEBUG_REPORT.md           current authoritative results/debug
  docs/README.md                            current documentation index
  docs/STAC_QM_CODEX_API_HANDOFF.md       this document
  docs/PHASE2_HANDOFF.md                  detailed Phase 1/2 runbook and results
  docs/STAC_QM_Implementation.md          implementation and command history

Model:
  mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py
  mmdet3d/models/sparsedetectors/query_memory.py
  mmdet3d/models/sparsedetectors/opus_head.py
  mmdet3d/models/sparsedetectors/opus_transformer.py

Training and evaluation:
  mmdet3d/apis/train.py
  mmdet3d/core/optimizer/trainable_only_optimizer_constructor.py
  mmdet3d/core/hook/query_memory_training_hook.py
  mmdet3d/core/evaluation/sparseworld_eval_hooks.py
  mmdet3d/apis/sparseworld_test.py

Cache:
  mmdet3d/datasets/pipelines/loading_query_memory.py
  tools/query_memory/precompute_query_memory.py
  tools/query_memory/audit_query_memory_cache.py

Configs:
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint.py
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py

Tests:
  tests/test_query_memory.py
  tests/test_query_memory_integration.py
  tests/test_sparseworld_evaluation.py
```

## 13. First Actions for the Next Codex API Session

1. Read this file completely.
2. Read `docs/PHASE2_HANDOFF.md` sections 3, 4.5, 7, and 9.
3. Inspect `git status` and preserve all unrelated user-owned changes.
4. Inspect the exact 0s and future routing in `SparseWorld4DTraj` before editing.
5. Restate the proposed Phase 3 trainable boundary, gradient strategy, and why
   backbone/neck remain frozen.
6. Ask for or confirm user authorization for Phase 3 implementation. This
   handoff records a design; it is not authorization to start a new experiment.
7. If authorized, implement in small verified steps and never run long jobs.

## 14. Result Recording Template for Future Sessions

Append a dated record here after every user-run smoke, training, or evaluation:

```text
Stage:
Date:
Code/config identity:
Command supplied to user:
Command actually run:
Checkpoint/source cache:
Trainable parameter count and prefixes:
Smoke invariants:
Gradient diagnostics:
Gate/residual diagnostics:
Occupancy IoU/mIoU [0s, 1s, 2s, 3s]:
Future mean and delta vs 13.2233:
Memory ON/OFF or shuffled controls:
Planning metrics if run:
Conclusion:
Next decision:
```

### 14.1 Phase 2 tmux result confirmation

```text
Stage: phase2 formal training, tmux verification
Date: 2026-08-22 (logs completed 2026-08-20 UTC)
Session: tmux 0, cwd=/data/jxy/projects
Process state: training and validation completed; pane returned to bash; no
  torchrun/tools/train.py process remains
Code/config identity: sparseworld-traj-memory-phase2.py
Checkpoint/source cache: epoch-56 source checkpoint; fixed schema-v2 val cache
Occupancy IoU: [25.67, 23.15, 22.27, 21.20]
Occupancy mIoU [0s, 1s, 2s, 3s]: [18.20, 14.92, 13.14, 11.49]
Future mean and delta vs 13.2233: 13.1833, -0.0400
Sample count: 4219 validation samples (4220 sampler outputs trimmed to 4219)
Conclusion: confirms the previously recorded negative Phase 2 result; no Phase 2
  extension or broader fine-tuning is authorized.
Next decision at the time of this Phase 2 record: use the separately
  implemented Phase 3 smoke configuration; the subsequent Phase 3 smoke result
  is recorded in §14.2.
```

### 14.2 Phase 3 connectivity smoke result

```text
Stage: phase3 connectivity smoke
Date: 2026-08-22 (UTC)
Session: tmux 0, cwd=/data/jxy/projects
Process state: 200/200 iterations completed; pane returned to bash; no
  torchrun/tools/train.py process remains
Code/config identity: configs/sparseworld/nuscenes-temporal/
  sparseworld-traj-memory-phase3-smoke.py
Command actually run:
  CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/train.py
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase3-smoke.py
  --work-dir work_dirs/sparseworld-traj-memory-phase3-smoke --gpu-id 0
  --deterministic
Checkpoint/source cache: iter_100.pth and iter_200.pth written; source
  checkpoint ckpts/epoch_56.pth; fixed epoch-56 feature-space cache boundary
Trainable parameter count: 1,725,756 scalars. Trainable prefixes are
  query_memory.*, memory_refiner.*, memory_cls_branch.*, memory_reg_branch.*,
  memory_vel_branch.*, and memory_horizon_embedding.*
Smoke invariants: Phase 3 connectivity hook passed at iteration 100;
  frozen parameter/buffer hash checks passed at after_run; TASS state and
  7 reads / 1040 fused Queries invariants passed.
Gradient diagnostics (iteration 100 -> 200):
  fusion out 0.0049 -> 0.0093; fusion gate 0.0017 -> 0.0041;
  fusion alpha 0.2853 -> 0.6673; attention q 0.0001 -> 0.0001;
  attention k 0.0001 -> 0.0001; attention v 0.0019 -> 0.0050;
  memory refiner 0.8698 -> 0.8685; cls branch 0.6291 -> 0.5836;
  reg branch 1.2599 -> 1.2592; vel branch 0.4239 -> 0.3959.
Gate/residual diagnostics (iteration 100 -> 200):
  conditional gate 0.2334 -> 0.2996; refiner residual ratio
  0.0062 -> 0.0068; direct fusion residual ratio 0.0002 -> 0.0007;
  fusion alpha 0.0030 -> 0.0042; has-candidate ratio 0.5705 -> 0.4774.
Occupancy IoU/mIoU [0s, 1s, 2s, 3s]: not evaluated by this smoke.
Future mean and delta vs 13.2233: not evaluated.
Memory ON/OFF or shuffled controls: not run.
Conclusion: the Phase 3 path is causally connected in the training graph:
  q/k/v, fusion projection, gate, alpha, and all refiner branches receive
  nonzero gradients. The refiner is active at roughly 0.6-0.7% residual
  usage; direct STAC fusion remains small but nonzero (0.02-0.07%). This is
  a connectivity smoke only and provides no mIoU claim.
Next decision: do not infer occupancy benefit from this smoke. If proceeding,
  run the separately authorized formal Phase 3 training and then evaluate the
  same checkpoint with Memory ON/OFF, reporting 0s/1s/2s/3s mIoU against the
  13.2233 baseline future mean.
```

### 14.3 Phase 3 formal training result

```text
Stage: phase3 formal 12-epoch training
Date: 2026-08-23 (UTC; final validation completed 2026-08-23 21:06)
Session: tmux 0, cwd=/data/jxy/projects
Process state: training and validation completed through Epoch 12; pane
  returned to bash; no torchrun/tools/train.py process remains
Code/config identity: configs/sparseworld/nuscenes-temporal/
  sparseworld-traj-memory-phase3.py
Command actually run: resumed the formal Phase 3 run from
  work_dirs/sparseworld-traj-memory-phase3/epoch_7.pth with two GPUs,
  data.samples_per_gpu=4, data.workers_per_gpu=2, deterministic mode, and
  OMP/MKL/OpenBLAS/NumExpr thread caps at 1
Checkpoint/source cache: epoch_8.pth through epoch_12.pth written;
  latest.pth -> epoch_12.pth; source ckpts/epoch_56.pth and fixed epoch-56
  schema-v2 train/val feature caches
Occupancy mIoU [0s, 1s, 2s, 3s]: [15.48, 14.22, 12.52, 10.95]
Occupancy IoU [0s, 1s, 2s, 3s]: [25.18, 23.01, 22.03, 20.86]
mIoU mean: 13.2925
Future mIoU mean: 12.5633
Delta vs epoch-56 baseline [18.20, 14.96, 13.18, 11.53]:
  [-2.72, -0.74, -0.66, -0.58] mIoU points; future mean delta -0.6600
Best formal epoch by reported mIoU mean/future mean: Epoch 12 (tied with
  Epoch 11 at the printed precision); the curve largely plateaued after
  Epoch 9.
Memory ON/OFF or shuffled controls: not run during training.
Conclusion: Phase 3 did not produce a causal occupancy improvement. The
  current 0s path is substantially below the frozen epoch-56 baseline and
  all future horizons remain below baseline; the final future mean is
  12.5633 versus 13.2233. Do not continue Phase 3 beyond Epoch 12 or multiply
  training variants based on this result.
Next decision: run a same-checkpoint Memory ON/OFF evaluation on epoch_12.pth
  to verify that the degradation is caused by the Memory-conditioned route.
  If OFF restores the epoch-56 baseline, stop this Phase 3 topology and
  investigate refiner output calibration, cache matching/alignment, and the
  0s/future routing before any new training experiment.
```

### 14.4 Phase 4 future-only formal training result

```text
Stage: phase4 future-only formal 12-epoch training
Date: 2026-08-25 (UTC; final validation completed 2026-08-25 19:14)
Session: tmux 0, cwd=/data/jxy/projects
Process state: training and validation completed through Epoch 12; pane
  returned to bash; no torchrun/tools/train.py process remains
Code/config identity: configs/sparseworld/nuscenes-temporal/
  sparseworld-traj-memory-phase4-future-only.py
Checkpoint/source cache: epoch_1.pth through epoch_12.pth written;
  latest.pth -> epoch_12.pth; source ckpts/epoch_56.pth and fixed epoch-56
  schema-v2 train/val feature caches
Routing: raw OPUS observation Query is preserved for 0s; STAC-QM and the
  Memory-conditioned refiner are applied only to scheduled future Query
  groups. Frozen backbone/neck/OPUS boundary and 2-GPU batch/workers were
  unchanged.
Occupancy mIoU [0s, 1s, 2s, 3s]: [18.20, 14.38, 12.75, 11.23]
Occupancy IoU [0s, 1s, 2s, 3s]: [25.68, 22.74, 21.62, 20.47]
mIoU mean: 14.1400
Future mIoU mean: 12.7867
Delta vs epoch-56 baseline [18.20, 14.96, 13.18, 11.53]:
  [0.00, -0.58, -0.43, -0.30] mIoU points; future mean delta -0.4366
Best future mean: Epoch 9, 12.7900; Epochs 10 and 12 were 12.7867.
The curve plateaued after approximately Epoch 8-9.
Memory ON/OFF or shuffled controls: not run in this training.
Conclusion: preserving the raw 0s route successfully restores 0s to the
  epoch-56 baseline exactly, confirming that Phase 3's 0s refiner route was
  responsible for the large current-horizon degradation. However, applying
  Memory only to future Query groups still lowers every future horizon and
  remains 0.4366 below the baseline future mean. This topology does not
  provide a causal occupancy gain.
Next decision: stop Phase 4 training and do not add epochs or unfreeze the
  backbone/neck/OPUS. The remaining issue is future Memory signal quality or
  routing, not 0s preservation. Before any new long run, perform a diagnostic
  evaluation of epoch_12 with future Memory disabled versus enabled while
  keeping raw 0s fixed, then inspect future Query/cache matching, motion
  compensation, scheduled-group geometry, and future loss weighting.
```
