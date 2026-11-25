# T0/T1: class balancing only for 3s occupancy-positive BCE

## Starting point and evidence (2026-10-18)

The starting point is the **A+B + SoftVoxel-center epoch 2 checkpoint with native inference**:

`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-from-ab1/epoch_2.pth`

Existing official results on 4219 frames are IoU `[25.68,23.85,22.84,21.68]` and mIoU
`[18.20,15.28,13.21,11.39]`, with Future Mean **22.7900 / 13.2933**.
These are not this round's T1 results; position corrections remain enabled at every horizon.

The user-run D0 paired diagnosis on 256 frames from 150 scenes is located at
`work_dirs/diagnostics-center-occ-20261018-074738/summary.json`.
When only Baseline occupancy is restored while current semantics and position are retained:

| Horizon | Binary IoU difference (percentage points) | mIoU difference | Scene-bootstrap 95% CI for the mIoU difference |
| --- | ---: | ---: | --- |
| 1s | -0.6288 | -0.2524 | [-0.4218,-0.0789] |
| 2s | -0.4870 | +0.0616 | [-0.1242,+0.2637] |
| 3s | -0.3202 | +0.1947 | [-0.0145,+0.4220] |

At 3s, pedestrian/manmade/vegetation change by +2.0268/+1.0605/+1.3963, respectively, and
driveable_surface by -1.0283. Restoring semantics changes Future mIoU by -0.3533; restoring position changes
Future mIoU by -0.0041, with -0.0639 at 3s. Therefore, retain semantics, occupancy, position, and A+B gates,
and experiment only with class aggregation of 3s supervision. D0 supports the "tradeoffs between classes" hypothesis, but **does not prove that training-positive
class imbalance is the sole cause**; the overall 3s mIoU intervention interval still crosses zero. This is not full validation,
nor does it characterize variation across training seeds.

## The only loss change

The existing positive/negative-balanced BCE already calculates separate means for positive and negative points; this round leaves positive/negative label rules unchanged.
Let P be all valid positive points in the current rank/local batch, P_c the subset whose GT target label=c,
and C_present the set of classes actually present in P:

```
L_pos_point = mean_{i in P} BCE(O_i, 1)
L_pos_class = mean_{c in C_present} mean_{i in P_c} BCE(O_i, 1)
L_pos_mix   = (1-beta) * L_pos_point + beta * L_pos_class
L_occ      = (w_pos * I_pos * L_pos_mix + w_neg * I_neg * L_neg)
             / max(w_pos * I_pos + w_neg * I_neg, EPS)
```

Configuration: `occupancy_positive_class_balance_beta={'1s':0., '2s':0., '3s':0.25}`.
0.25 is a conservative first-round mixing ratio: retain 75% of the original point-wise objective, with only 25% of the positive term equally weighting each
GT class actually present. All 17 non-free classes can participate; there is no specific small-class list,
no use of predicted classes, and no modification of existing semantic class weights; absent classes are excluded from the denominator.
Class imbalance is reduced rather than eliminated; if rare classes contain more label noise, that noise may also be amplified and must be measured.

Positives remain points that "fall inside a non-free GT voxel" or whose "nearest non-free center is within the original radius"; the former's
GT class takes priority, and the latter uses the original nearest-label. The points mask and out-of-bounds filtering are fully retained.
The negative term, positive/negative group weights, outer `loss_occ_weight=0.25`, threshold, anchor,
semantics, SoftVoxel-center, and original future classification/position losses are **unchanged**.
T1 replaces only the internal positive-sample mean in `mem_3s.loss_occ`, retaining all loss names.

With no positive samples, only the original negative term remains; with no negative samples, only the positive term remains; all-empty input returns a finite zero.
The new path accumulates in FP32 to avoid FP16 overflow in large sample counts/sums (explicit double inputs retain
double). When beta=0 and new statistics are not requested, execution strictly follows the original reducer, preserving operation order and dtype.
Only C-dimensional scatter/bincount and point-level vectors are added; no dense points×classes tensor is generated.

As in the original loss, DDP uses **local-batch aggregation per rank followed by averaging rank losses**,
not per-class count normalization over the global batch. Keep GPU count, samples_per_gpu, seed, and
sampling order fixed so that this meaning does not change between experiments. Beta acts only at 3s, but shared parameters can still affect
subsequent 1s/2s inference; bitwise equality at those two horizons after training cannot be promised.

## Diagnostic logs

T1 defaults to `occupancy_class_balance_diagnostics=True`, and records only 3s, where beta>0:

- `mem_3s.occ_positive_bce_point_mean / class_mean / mixed_mean`
- `mem_3s.occ_negative_bce_mean`
- `mem_3s.occ_class_balance_beta / occ_positive_present_classes`
- `mem_3s.occ_positive_count / occ_negative_count`
- `mem_3s.occ_positive_voxel_count / occ_positive_proximity_only_count`
- Per-class `occ_positive_count_c0` through `c16`, with corresponding `voxel_count_c*` and
  `proximity_only_count_c*`. The implementation derives the class count from logits rather than hard-coding 17.

These count GT classes matched by current predicted points, **not GT voxel volume**.
Exact and proximity-only matches are mutually exclusive, and their per-class sum equals positive_count; the original,
potentially overlapping voxel_positive and proximity_positive masks cannot simply be added.
All statistic tensors are detached and contain no `loss` substring, so they do not enter the optimization objective.
Absent classes and empty positive sets retain the same scalar keys, avoiding different numbers of logging collectives
across DDP ranks. MMDetection averages logs across ranks, and the logger then averages over windows, so printed
counts may be fractional: they represent mean counts per rank per iteration, not global totals.

## Matched continuation training and commands

Both groups resume from **center epoch 2**, including optimizer/runner state, and each adds one epoch
to reach **epoch 3**; T1 does not continue from T0's trained epoch 3.

| Item | T0 | T1 |
| --- | --- | --- |
| Model / inference | Original center | Identical |
| Starting point | center-from-ab1/epoch_2.pth | Same checkpoint |
| trainable | Original `future_memory_adapter_v2.*` | No parameters added or removed |
| 3s positive BCE | Original point-wise mean | 75% point-wise + 25% class-wise |
| seed / DDP / batch | 17 / 2 GPU / 1 per card | Same |
| LR / scheduler | Restore original AdamW, fixed 2e-5 | Same |
| q chunk / AMP | 48 / dynamic FP16 | Same |
| Total runner epochs | 3 (only 1 added) | 3 (only 1 added) |

The user has manually launched T0 in
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t0-from-center2`.
T0 uses the original center configuration with CLI overrides max_epochs=3 and a separate work-dir, and **must explicitly specify**
`--resume-from .../epoch_2.pth`; CLI defaults in this project's tools/train.py may override
resume_from in the configuration, so setting only the configuration path is insufficient. The original configuration file is unchanged.

Manual T1 pre-run check (CPU, without starting training):

```bash
bash /data/jxy/projects/tools/run_train_future_memory_occ_t1.sh --check
```

After T0 completes and both GPUs are idle, run in tmux:

```bash
bash /data/jxy/projects/tools/run_train_future_memory_occ_t1.sh
```

The script checks epoch2/optimizer/matching configuration, rejects existing output directories and busy GPUs, uses
the same environment and seed as T0, uses separate port 29555, and explicitly resumes the same epoch2.
It records checkpoint/source SHA256 and configuration, without recording environment variables/secrets or modifying old checkpoints.
Output is `work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth`.
No automatic-launch task or task that automatically runs once GPUs become idle is created.

## Acceptance checks and subsequent decisions

This round performs only CPU static checks, configuration/standalone Adapter construction, and forward/pure-function acceptance; the assistant does not launch training,
backward, optimizer steps, a GPU smoke test, or full evaluation. Tests cover beta=0 identity,
the exact mixing formula, absent classes/empty samples, valid-point masks, FP16 accumulation, B=1/2, strict preservation of other
losses, actual horizon wiring, exclusion of logs from the loss, and configuration/checkpoint-key compatibility.
This is not T1 gradient acceptance or a conclusion about training/metric improvements. Existing T0 training was launched by the user and is left untouched.

The CPU construction attempt for the full model **did not complete**: the old `opus_head.py:85` directly calls
`torch.tensor(...).cuda()` during initialization, causing `No CUDA GPUs are available` in the check environment with GPUs hidden.
Baseline is not modified, and T0's GPUs are not used to bypass this; full T1 model build/GPU smoke/DDP startup
acceptance is deferred until GPUs are idle. `--check` verifies only strict CPU Adapter loading and restoration of optimizer state
groups/shapes; it is not described as a passing full-model check.

Commands actually executed in this round:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
env/bin/python -m pytest -q \
  tests/test_future_memory_occ_class_balance.py \
  tests/test_future_memory_adapter_v2.py \
  tests/test_future_memory_adapter_ab.py::test_migration_preserves_nonzero_outputs_all_horizons \
  tests/test_future_memory_adapter_ab.py::test_horizon_and_task_gate_isolation \
  tests/test_future_memory_adapter_ab.py::test_no_candidate_gates_and_outputs_are_exact_baseline \
  tests/test_future_memory_adapter_ab.py::test_decode_independent_occupancy_gate \
  tests/test_future_memory_softvoxel_center.py::test_centre_is_optimal_and_neighbours_symmetric \
  tests/test_future_memory_softvoxel_center.py::test_boundary_stays_finite_and_out_of_bounds_safe \
  -k 'not cuda'
```

Final result: **94 passed, 1 deselected**; gradient tests in the AB/SoftVoxel files were not selected.
The initial test found that the old CPU Half reducer does not support `clamp_min`, so strict legacy equality
was checked only with supported FP32/FP64; the new reducer passed the check promoting 80,000-point FP16 input to FP32.
This is not CUDA AMP runtime acceptance. A missing instance argument in the test call to `_parse_losses`
was also corrected, after which the actual MMDetection parser verified that new statistics do not enter the total loss.

`bash -n`, the launch script's `--check`, Python compilation, and `git diff --check` all passed.
The original loss function from HEAD was also loaded as a read-only control and checked on 10 inputs in total:
normal, empty GT, out-of-bounds, fully masked, and empty predictions for B=1/B=2. All default losses were `torch.equal`.
ASTs for the V1/V2 Adapter classes, decoding, position, occupancy target, and SoftVoxel functions
are identical to HEAD; the source epoch2 SHA256 matches D0 provenance.
HEAD remains `3c057d61ab2c7d8ea68ceb4ddd319d9bf7e1bf73`; there were no commits, pushes, or changes to
existing checkpoints/cache; the user's pre-existing diagnostic-launcher and position-recheck-document changes were preserved.

Next, first compare **T1 epoch3 against T0 epoch3**, retaining center epoch2 as a starting-point reference.
The objective is to improve 3s and Future mIoU while retaining occupied IoU gains; a Future IoU
decline of no more than 0.05 percentage points from the starting point can serve as a pre-agreed exploration budget (22.7400), not a
significance boundary or permission to ignore T0. Inspect per-horizon, per-class TP/FP/FN, especially tradeoffs involving pedestrians,
vegetation, manmade objects, and roads. 1 epoch is only a screening run; do not automatically add epochs or combine C.
Formal conclusions still require matched full validation; collision rate/L2 must be evaluated separately on the finally selected checkpoint.