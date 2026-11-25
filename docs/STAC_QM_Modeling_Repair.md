# STAC-QM Modeling Fixes and Training Constraints

This document records modeling fixes, the cache schema, and training constraints. Experiment values, debugging findings, and current unresolved issues are collected in [EXPERIMENT_DEBUG_REPORT.md](EXPERIMENT_DEBUG_REPORT.md). Commands are provided in [STAC_QM_Implementation.md](STAC_QM_Implementation.md).

## Goals of the fixes

STAC-QM introduces causal query memory into SparseWorld trajectory forecasting while ensuring that:

- Memory reads only observed past frames;
- The semantics, positions, reliability, and timing recorded in the cache can be audited;
- The added Memory branch is trainable;
- Memory training does not silently change the baseline reproduction path;
- Empty history or batches without candidates reduce to a numerical identity mapping.

## Six modeling fixes

### 1. Each query reads observed history only once

Observation queries read Memory once before SCF; each scheduled future query group reads Memory once when introduced. Already active queries do not read it again.

Default schedule:

```text
observation queries: 720
future scheduled groups: [60, 60, 60, 60, 40, 40]
future offsets: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
```

The legacy STAC-QM integration expects `7` Memory calls and `1040` fused queries by default.

### 2. Future-aware effective age

The cache retains the base time difference between each historical frame and the current frame:

```text
base_age = current_timestamp - history_timestamp
```

When a future step reads Memory, it uses:

```text
effective_age = base_age + future_offset
```

Causal filtering, max-age filtering, the temporal attention penalty, motion compensation, and diagnostics all use `effective_age`. The original cached timestamps are not overwritten.

### 3. Zero-initialized motion compensation

`QueryMotionCompensator` predicts bounded velocity from memory features and time features:

```text
velocity = v_max * tanh(MLP([LN(memory_feature), time_features]))
aligned_points = ego_aligned_points + effective_age * velocity
```

The final layer is initialized to zero, making the initial behavior equivalent to ego-pose alignment alone.

### 4. Per-query semantic reliability

The schema-v2 cache stores the following for each selected query:

```text
query_semantic_distribution [M, C_sem]
query_label                 [M]
query_margin                [M]
query_entropy               [M]
query_reliability           [M]
```

Reliability combines the top-1 probability, the top-1/top-2 margin, and normalized entropy, and is clamped to `[0, 1]`. A schema-v1 fallback remains available:

```text
query_reliability = query_conf
query_label = -1
```

Formal Memory experiments should use the schema-v2 cache.

### 5. Shared deterministic diversity selector

Cache precomputation, cache loading, and the online memory bank share the same `select_diverse_memory_queries(...)`. Selection uses a stable sort by reliability, prioritizes coverage of different classes and spatial cells, and then fills the remaining capacity.

### 6. target-age history selection

The repaired configuration uses the following defaults:

```python
history_selection_mode = 'target_age'
history_target_ages = [2.5, 3.5, 4.5]
history_age_tolerance = 0.35
```

Historical frames must satisfy the following conditions:

- strictly past;
- same scene;
- At most one frame per target slot;
- A frame cannot fill multiple slots;
- Exclude samples at split boundaries for which the cache cannot be generated.

## Schema-v2 cache validation

Key fields in each record:

```text
query_feat
query_points_metric
query_conf
query_semantic_distribution
query_label
query_margin
query_entropy
query_reliability
valid_mask
ego2global
timestamp
frame_idx
scene_id
sample_idx
pc_range
embed_dims
num_points
num_classes
source_config
source_checkpoint
schema_version
```

The loader must check that:

- Required fields are present;
- Feature, point, label, semantic distribution, reliability, and time fields have consistent shapes;
- The class dimension of the semantic distribution is valid;
- Values are finite;
- Probabilities are nonnegative and their row sums are valid;
- Reliability and margin values are within valid ranges;
- Scene membership and temporal causality are valid.

Missing fields or incorrect shapes must raise explicit errors rather than trigger a silent fallback. The schema-v1 fallback exists only for compatibility with older records and is not the main path for formal experiments.

## Training constraints

### Legacy Memory-only path

The legacy Memory-only mode uses:

```python
query_memory_cfg = dict(
    enabled=True,
    source='cache',
    memory_finetune_mode=True,
    freeze_base_model=True,
)
```

Only `query_memory.*` parameters may be trained. The base model, OPUS head, TASS assignment, and frozen buffers must remain unchanged.

### Legacy joint path

Joint fine-tuning was used to test whether expanding the trainable scope allowed Memory signals to reach the future head. It is a historical experiment entry point rather than the current main approach.

### Current Future Memory Adapter path

Future Adapter mode uses a separate module:

```python
future_memory_adapter_enabled=True
future_memory_adapter_finetune_mode=True
future_memory_target_horizons=[1.0, 2.0, 3.0]
```

Constraints:

- Freeze the baseline image backbone/neck;
- Freeze the original OPUS/SparseWorld head;
- Freeze the original query encoding, future recurrence, and classification/regression branches;
- Train only `future_memory_adapter.*`;
- Do not write Memory-corrected logits, points, or features back into baseline future recurrence;
- Bypass the Future Adapter at 0s and at future steps outside the target horizons.

## Future Adapter modeling invariants

The current Future Adapter fixes cross-time logit accumulation in the legacy Phase3/Phase4 topology. Each target horizon is computed independently:

```text
S_base(t) = ClsBranch(Q_base(t))
S_final(t) = S_base(t) + G(t) * DeltaS_memory(t) + G(t) * DeltaO_memory(t)
P_final(t) = SafeRefine(P_base(t), G(t) * DeltaP_memory(t))
```

The following are prohibited:

- `S(t) = S_final(t-1) + ClsBranch(Q(t))`;
- Passing the 1s corrected logits to 2s;
- Passing the 2s corrected logits to 3s;
- Using Memory correction to overwrite the baseline recurrence state for the next step.

Target horizon mapping:

| horizon | internal future step | output key |
| ---: | ---: | --- |
| 1s | 2 | `semantic_occ_2s` |
| 2s | 4 | `semantic_occ_4s` |
| 3s | 6 | `semantic_occ_6s` |

The number of active queries is derived from the actual schedule tensors; the default expected values are `840 / 960 / 1040`.

## Initialization strategy

The added Memory path must be both trainable and initially equivalent to the baseline:

- Use standard nonzero initialization for q/k/v, the memory projection, and the semantic projection;
- Set the gate bias to approximately `-1`;
- Initialize the final-layer weights and biases of the semantic/occupancy/position residual heads to zero;
- Do not use a global alpha on the order of `1e-3`;
- Force the gates to zero when no valid Memory candidates exist, so the output reduces exactly to the baseline.

## Acceptance criteria

Code-level acceptance checks should cover:

- zero-init baseline identity;
- no-candidate baseline fallback;
- Applying the horizon embedding before the attention query;
- A causal effect of memory features, semantic distributions, labels, reliability, age, and relative position on candidate scores or outputs;
- No gradients for baseline parameters;
- Gradients for residual heads and upstream adapter parameters during a synthetic backward pass or a small optimizer step;
- Preserving all cache schema/collate fields.

Formal IoU/mIoU conclusions may be added to the experiment summary only after formal training and full validation are complete.
