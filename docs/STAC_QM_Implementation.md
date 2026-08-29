# STAC-QM Implementation and Runbook

STAC-QM adds Spatio-Temporally Aligned, Reliability-Gated Causal Query Memory to SparseWorld trajectory forecasting. The authoritative modeling and safety details are in [STAC_QM_Modeling_Repair.md](STAC_QM_Modeling_Repair.md).

## Implemented Data Flow

Only raw observation queries from real past frames are cached:

```text
base SparseWorld RAP observation queries
  -> schema-v2 per-sample cache
  -> target-age same-scene history selection
  -> ego-pose alignment
  -> zero-initialized motion residual
  -> reliability-aware causal attention
  -> confidence-gated residual fusion
  -> original recursive SCF future prediction
```

STAC-QM does not add predicted-query memory, instance tracking, query-to-track conversion, dynamic/static query splitting, SCF parallelization, fixed OccWorld tokens, or VQ-VAE.

## Schema-v2 Cache Record

Each selected observation-query record contains:

```text
query_feat                     [M, C]
query_points_metric            [M, R, 3]
query_conf                     [M]
query_semantic_distribution    [M, C_sem]
query_label                    [M]
query_margin                   [M]
query_entropy                  [M]
query_reliability              [M]
valid_mask                     [M]
ego2global                     [4, 4]
timestamp, frame_idx, scene_id, sample_idx
pc_range, embed_dims, num_points, num_classes
source_config, source_checkpoint, schema_version
```

Schema v1 remains readable with `query_reliability=query_conf` and `query_label=-1`. Formal Memory-only runs require complete schema-v2 caches.

## SCF Integration

Memory reads are bounded by construction:

- observation queries read once before SCF at `future_offset=0`;
- each of six scheduled groups reads once when introduced;
- already-active queries never re-read Memory.

Default runtime expectation:

```text
query counts:   [720, 60, 60, 60, 60, 40, 40]
future offsets: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
reads:          7
fused queries:  1040
```

## Configurations

### Split Cache Roots

```text
train: data/query_memory/sparseworld_epoch56_schema2_train
val:   data/query_memory/sparseworld_epoch56_schema2_val
test:  data/query_memory/sparseworld_epoch56_schema2_val
```

Train, val, and test routes are assigned separately. Formal loaders use `strict=True`.

### Formal Memory-Only Config

```text
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py
```

It sets:

- `load_from='ckpts/epoch_56.pth'`;
- `resume_from=None`;
- `memory_finetune_mode=True`;
- `freeze_base_model=True`;
- `source='cache'`;
- AdamW, LR `1e-4`, weight decay `1e-2`;
- gradient clipping at norm 5;
- two GPUs with `samples_per_gpu=2` (global batch size 4);
- 12 epochs and one checkpoint per epoch.

Only `query_memory.*` is trainable and optimized. TASS assignment is finalized once after checkpoint loading and then asserted immutable. Scene-boundary batches with no valid target-age history remain numerically exact identities while retaining a zero-valued Query Memory autograd bridge; backward therefore remains valid, and the zero gradients are not counted as connectivity evidence.

### Connectivity Smoke Config

```text
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only-smoke.py
```

This is a separate 200-iteration run. Its optimizer hook checks base gradients/state, TASS immutability, 7/1040 runtime behavior, and gradient connectivity for fusion and attention. It reports motion connectivity without claiming success when the motion final layer has no gradient.

## User-Run Commands

These commands require the repository's real nuScenes data, checkpoint, and GPU environment. Cache generation, both audits, zero-initialization identity, the 200-iteration Memory-only smoke, formal 12-epoch Memory-only training, and full Memory ON/OFF validation have been completed. The new joint-finetune smoke and formal training commands below are user-run operations and have not been executed by the assistant.

### 1. Generate Train Cache

```bash
/data/jxy/projects/env/bin/python tools/query_memory/precompute_query_memory.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --split train \
  --output-dir data/query_memory/sparseworld_epoch56_schema2_train \
  --max-queries-per-frame 256 \
  --write-threshold 0.35 \
  --min-reliability 0.0 \
  --spatial-cell-size 4.0 \
  --max-per-spatial-cell 16 \
  --max-per-class 64 \
  --workers-per-gpu 2 \
  --skip-existing
```

### 2. Generate Val Cache

```bash
/data/jxy/projects/env/bin/python tools/query_memory/precompute_query_memory.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --split val \
  --output-dir data/query_memory/sparseworld_epoch56_schema2_val \
  --max-queries-per-frame 256 \
  --write-threshold 0.35 \
  --min-reliability 0.0 \
  --spatial-cell-size 4.0 \
  --max-per-spatial-cell 16 \
  --max-per-class 64 \
  --workers-per-gpu 2 \
  --skip-existing
```

Do not pass `--overwrite` unless replacing existing caches is intentional and verified.

### 3. Audit Train and Val Caches

```bash
/data/jxy/projects/env/bin/python tools/query_memory/audit_query_memory_cache.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --split train \
  --expected-schema-version 2 \
  --expected-source-checkpoint ckpts/epoch_56.pth \
  --json-out work_dirs/stacqm_cache_audit_train.json
```

```bash
/data/jxy/projects/env/bin/python tools/query_memory/audit_query_memory_cache.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --split val \
  --expected-schema-version 2 \
  --expected-source-checkpoint ckpts/epoch_56.pth \
  --json-out work_dirs/stacqm_cache_audit_val.json
```

Do not continue to training if either audit exits nonzero.

### 4. Run Zero-Initialization C0/C1 Identity

```bash
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python \
  tools/query_memory/check_query_memory_identity.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --mode zero \
  --split val \
  --gpu-id 0 \
  --atol 1e-6
```

The tool automatically chooses the first sample with all target-age slots unless `--sample-index` is specified. It must report valid histories/candidates, 7 reads, 1040 fused queries, zero residual, and identity within `1e-6`.

### 5. Run the 200-Iteration Connectivity Smoke

```bash
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only-smoke.py \
  --work-dir work_dirs/sparseworld-traj-memory-only-smoke \
  --gpu-id 0 \
  --deterministic
```

Inspect the hook output and final run status. Do not treat a produced checkpoint alone as proof that connectivity checks passed.

### 6. Check Trained Memory ON/OFF Behavior

After a successful smoke run, point `--memory-checkpoint` to its checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python \
  tools/query_memory/check_query_memory_identity.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --memory-checkpoint work_dirs/sparseworld-traj-memory-only-smoke/iter_200.pth \
  --mode trained \
  --split val \
  --gpu-id 0 \
  --atol 1e-6 \
  --min-future-diff 1e-6
```

This mode expects current-frame identity and a nonzero difference in at least one future output.

### 7. Start Formal 12-Epoch Training

Only after cache audits, C0/C1 identity, smoke connectivity, and trained ON/OFF checks pass:

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --work-dir work_dirs/sparseworld-traj-memory-only \
  --launcher pytorch \
  --validate \
  --deterministic
```

Do not use `--auto-resume` for the initial run. The config loads `ckpts/epoch_56.pth` and sets `resume_from=None`. This command was used for the completed formal run. Because the inherited evaluation interval was 24 epochs, it did not evaluate during the 12-epoch run.

### 8. Evaluate Memory ON and OFF

Run these full-validation commands sequentially. The first evaluates trained Memory ON. The second disables Memory and evaluates the unchanged `epoch_56.pth` base under the same repaired model/data configuration.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 \
  --master_port=29501 \
  tools/test.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint work_dirs/sparseworld-traj-memory-only/epoch_12.pth \
  --launcher pytorch \
  --eval segm \
  --deterministic \
  2>&1 | tee work_dirs/sparseworld-traj-memory-only/eval_epoch12_memory_on.log
```

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 \
  --master_port=29502 \
  tools/test.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --launcher pytorch \
  --eval segm \
  --deterministic \
  --cfg-options \
    model.query_memory_cfg.enabled=False \
    model.query_memory_cfg.memory_finetune_mode=False \
    model.query_memory_cfg.freeze_base_model=False \
  2>&1 | tee work_dirs/sparseworld-traj-memory-only/eval_epoch56_memory_off.log
```

Compare the reported `IoU` and `mIoU` arrays at 0s, 1s, 2s, and 3s. Current-frame values should remain equal within evaluation rounding; effectiveness is determined primarily by positive 1s/2s/3s deltas.

## Joint Future-Occupancy Fine-Tuning

The clean-start joint experiment keeps the epoch-56 observation-query feature space fixed while adapting STAC-QM and the future occupancy path together.

Trainable from iteration one:

```text
query_memory.*       lr=5e-5
position_encoder.*   lr=1e-5
reg_branch.*         lr=1e-5
vel_branch.*         lr=1e-5
cls_branch.*         lr=1e-5
ego_cross_attn.*     lr=5e-6
```

Frozen and held in eval mode:

```text
img_backbone.*
img_neck.*
pts_bbox_head.*
plan_head.*
points_scale_branch.*
traj_head.*
```

TASS state and RAP masks remain immutable. ResNet/neck/`pts_bbox_head` must stay frozen because all schema-v2 history caches were generated by `ckpts/epoch_56.pth`; changing the observation-query generator while retaining fixed caches would place current and historical Query features in different spaces.

Configs:

```text
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint.py
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint-smoke.py
```

### 9. Run Joint Connectivity Smoke

Run this before formal joint training:

```bash
cd /data/jxy/projects
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint-smoke.py \
  --work-dir work_dirs/sparseworld-traj-memory-joint-smoke \
  --gpu-id 0 \
  --deterministic
```

The hook must finish successfully after checking optimizer scope, frozen parameters/buffers, immutable TASS, 7 reads, 1040 fused queries, and nonzero connectivity for STAC-QM plus every joint future module. Do not infer success solely from the existence of `iter_200.pth`.

### 10. Start Formal Two-GPU Joint Training

Only after the joint smoke succeeds:

```bash
cd /data/jxy/projects
mkdir -p work_dirs/sparseworld-traj-memory-joint
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 \
  --master_port=29510 \
  tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint.py \
  --work-dir work_dirs/sparseworld-traj-memory-joint \
  --launcher pytorch \
  --validate \
  --deterministic \
  2>&1 | tee work_dirs/sparseworld-traj-memory-joint/train.log
```

This starts cleanly from `ckpts/epoch_56.pth`, does not resume the Memory-only optimizer/checkpoint, trains all six future horizons immediately, saves every epoch, and validates every epoch. The primary checkpoint-selection value is mean future mIoU over 1s/2s/3s, with 0s monitored for unexpected regression.

The first formal launch completed epoch 1 and saved a valid checkpoint, then failed at validation sample 0 because MMDetection's generic distributed test function indexed the model's dictionary result as `result[0]`. Training-time validation now selects SparseWorld-specific evaluation hooks, parses the dictionary occupancy/trajectory result, interleaves distributed sampler shards, and truncates sampler padding to exactly the 4,219 validation samples. Other detector models continue to use the generic MMDetection hooks.

After resuming from epoch 1, epoch 2 training and all 4,219 validation samples completed successfully. Evaluation reported `IoU=[25.68, 23.11, 22.23, 21.18]` and `mIoU=[18.20, 14.94, 13.15, 11.49]`, but the subsequent TensorBoard logger rejected those list-valued metrics. SparseWorld evaluation now converts temporal metric lists into scalar fields (`IoU_0s` through `IoU_3s`, `mIoU_0s` through `mIoU_3s`, means, and future means) before writing the runner log buffer. This preserves the printed evaluator output and prevents scalar loggers from receiving lists.

The latest checkpoint was inspected successfully:

```text
work_dirs/sparseworld-traj-memory-joint/epoch_2.pth
meta.epoch:         2
meta.iter:          9866
state_dict tensors: 710
optimizer groups:   63
optimizer states:   63
```

Resume from epoch 2 instead of restarting or repeating completed epochs:

```bash
cd /data/jxy/projects
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 \
  --master_port=29513 \
  tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint.py \
  --work-dir work_dirs/sparseworld-traj-memory-joint \
  --resume-from work_dirs/sparseworld-traj-memory-joint/epoch_2.pth \
  --launcher pytorch \
  --validate \
  --deterministic \
  2>&1 | tee -a work_dirs/sparseworld-traj-memory-joint/train.log
```

Generated checkpoints, logs, predictions, `output_data.pkl`, and cache artifacts must remain uncommitted.

## Phase2 Occupancy-Head Fine-Tuning

Phase2 adds `memory_phase2_finetune_mode=True`. The mode is mutually
exclusive with Memory-only and joint tuning and requires enabled cache-backed
STAC-QM plus `freeze_base_model=True`.

Trainable from iteration one:

```text
query_memory.*       lr=5e-5
position_encoder.*   lr=1e-5
reg_branch.*         lr=1e-5
vel_branch.*         lr=1e-5
cls_branch.*         lr=1e-5
ego_cross_attn.*     lr=5e-6
pts_bbox_head.*      lr=5e-6
```

Frozen and held in eval mode:

```text
img_backbone.*
img_neck.*
plan_head.*
points_scale_branch.*
traj_head.*
```

The image backbone and neck remain frozen because the schema-v2 history
caches were generated in the epoch-56 feature space. Allowing the current
frame feature extractor to drift while cached history Queries remain fixed
would break their alignment. Phase2 instead adapts the final occupancy query
refinement and semantic output path at a conservative LR. TASS
`num_stamps_all`, `ind_stamps_all`, and RAP causal masks remain immutable, and
the 7-read/1040-fused-query invariant is unchanged.

Configs:

```text
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2-smoke.py
```

The formal config starts cleanly from `ckpts/epoch_56.pth`, trains for 12
epochs, and validates every epoch. It uses
`TrainableOnlyOptimizerConstructor`; the phase2 smoke hook additionally
requires a nonzero aggregate gradient for `pts_bbox_head.*` while preserving
the joint connectivity, frozen-state, TASS, and 7/1040 checks.

Run the user-operated 200-iteration smoke first:

```bash
cd /data/jxy/projects
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2-smoke.py \
  --work-dir work_dirs/sparseworld-traj-memory-phase2-smoke \
  --gpu-id 0 --deterministic
```

Only after the hook completes successfully, start formal training:

```bash
cd /data/jxy/projects
mkdir -p work_dirs/sparseworld-traj-memory-phase2
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 --master_port=29520 \
  tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py \
  --work-dir work_dirs/sparseworld-traj-memory-phase2 \
  --launcher pytorch --validate --deterministic \
  2>&1 | tee work_dirs/sparseworld-traj-memory-phase2/train.log
```

Phase2 effectiveness is measured against the 4,219-sample epoch-56 baseline.
Any epoch with future mean mIoU above `13.2233` is an improvement; 0s mIoU
must also be monitored for regression. If no epoch improves, stop and discuss
architecture-level changes instead of expanding the trainable boundary.

| Checkpoint | mIoU [0s, 1s, 2s, 3s] | Future mean | Delta vs 13.2233 | Planning | Status |
| --- | --- | ---: | ---: | --- | --- |
| epoch-56 baseline | [18.20, 14.96, 13.18, 11.53] | 13.2233 | 0.0000 | recorded above | complete |
| phase2 smoke | not evaluated | not evaluated | not applicable | not applicable | 200/200 hook passed |
| phase2 best epoch | pending | pending | pending | pending | user-run pending |

## Completed Verification

### Static and Synthetic

```text
59 passed, 19 warnings
```

The regression coverage proves that a Memory-only batch with no valid
historical slot remains an exact numerical identity but still supports
backward with explicit zero gradients. It also covers phase2's exact trainable
set and LR tiers, three-mode mutual exclusion, required cache-backed safety
settings, module train/eval states, immutable TASS state, 7/1040 behavior,
strict cache routing, trainable-only optimizer guard, and smoke-hook
registration. SparseWorld dictionary outputs are parsed without generic
sequence indexing, distributed rank shards are restored in dataset order and
trimmed after sampler padding, SparseWorld selects the custom evaluation
hooks, and ordinary detectors retain MMDetection's generic hooks. Modified
Python files compile, `git diff --check` passes, all eight STAC-QM configs load,
and the training and evaluation hooks are registered.

### Real Cache and GPU Acceptance

Observed on the configured nuScenes splits and `ckpts/epoch_56.pth`:

- train schema-v2 cache: `19,730 / 19,730` records;
- val schema-v2 cache: `4,219 / 4,219` records;
- train and val audits: `failure_count=0`, no missing, corrupt, orphan, noncausal, scene, temporal, shape, schema, or source-checkpoint failures;
- zero-initialization C0/C1: passed with exact `0.0` output differences, three valid history slots, 7 reads, 1040 fused queries, and zero residual;
- 200-iteration connectivity smoke: completed with nonzero fusion output/gate, attention Q/K/V, and motion-final-layer gradients; 7/1040 held throughout, and the base parameter/buffer plus frozen TASS checks passed;
- trained Memory ON/OFF: passed with exact current-frame identity, nonzero differences at all six future horizons, `out_proj_abs_max=0.0034669`, `motion_last_abs_max=0.0030774`, and `residual_norm_max=0.0197589`.

Formal 12-epoch Memory-only training completed on two GPUs with `samples_per_gpu=2` (global batch size 4), producing `epoch_12.pth` after 59,196 iterations. The run ended cleanly without NaN/Inf, all 679 base-model tensors remained bitwise identical to `ckpts/epoch_56.pth`, and the final optimizer contained only the 29 trainable Query Memory tensors. Epoch-12 trained ON/OFF verification passed with exact current-frame identity, all six future horizons and trajectory output changed, 7 reads, 1040 fused queries, three valid history slots, `out_proj_abs_max=0.0962831`, `motion_last_abs_max=0.0379095`, and `residual_norm_max=7.7569` on the checked validation sample.

The inherited evaluation interval is 24 epochs, so this 12-epoch run did not evaluate during training even though validation was enabled at launch. A subsequent full 4,219-sample validation compared epoch-12 Memory ON against the unchanged epoch-56 Memory OFF base. ON reported `IoU=[25.68, 23.14, 22.28, 21.21]` and `mIoU=[18.20, 14.95, 13.17, 11.51]`; OFF reported `IoU=[25.68, 23.15, 22.27, 21.21]` and `mIoU=[18.20, 14.96, 13.18, 11.53]`. The ON-minus-OFF mIoU deltas were therefore `[0.00, -0.01, -0.01, -0.02]`, with mean future mIoU changing from `13.2233` to `13.2100` (`-0.0133`). The repaired historical Query path is functionally active but epoch 12 does not improve the aggregate validation metrics.

Generated caches, checkpoints, logs, and evaluation products remain local artifacts and must not be committed.
