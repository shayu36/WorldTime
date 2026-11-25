# 2026-11-24: Joint Adaptation with Five Seeds and 1 Full Epoch per Seed

The user requested a five-seed experiment on the current server with the occupancy Adapter unfrozen.
The independent directory is `/data/jxy/WorldTime-planning-joint-5seed-20261124`;
the original directory and the preceding planning-only results are not overwritten.
The new model class is `SparseWorld4DTrajMemoryPlanningJoint`.

## Fixed protocol

- Seeds 17 / 23 / 41 / 73 / 101. Each seed uses all 19730 training samples, two GPUs at batch 1 per GPU,
  9865 iterations, and one epoch.
- Each seed starts from the same best T1 epoch_3, SHA256
  `bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`.
  The new planning-fusion module is initialized according to that seed; no previous optimizer,
  AMP scaler, or iteration state is loaded.
- Baseline parameters and runtime state remain frozen. Training covers 77 parameter tensors and 1722147 scalars:
  `future_memory_adapter_v2.*` (1498145) and `memory_planning_fusion.*` (224002).
- Retain the original T1 occupancy objectives and original six-step trajectory supervision.
  Occupancy losses update V2; trajectory losses update the fusion module.
  Scene token packing retains stop-gradient, so this experiment does not update V2 end to end through the planning loss.
- Learning rates: 2e-5 for V2 (retaining the final T1 continuation rate) and 1e-4 for the new planning module
  (retaining the planning-only rate). AdamW weight_decay=.01, fixed LR, 100 iter linear warmup, initial ratio .5.
- V2 q40, activation checkpoint=True, save_on_cpu=False, FP16 dynamic loss scale.
  Retain the actual Memory Context and candidate mask from the same forward pass for planning fusion.
  Joint training restores the future point masks skipped by the original planning-only path, for occupancy objectives.
- No cross-time correction is written back into the Baseline recurrence. The Memory cache was not rebuilt.
- The new planning residual starts at zero; T1 occupancy residuals retain their trained values from the source checkpoint.

## Checks and execution

Dedicated CPU tests verify recomputation values/gradients and RNG under checkpointing, the no-candidate boundary,
compatibility with the original planning-only path, optimizer scope/learning rates for both modules,
Baseline isolation and V2 updates in checkpoints, and preservation of results at every horizon.
The real TRAIN index=64 check includes three backward passes and two temporary updates;
it separately verifies the two gradient routes, gradients upstream of fusion, an unchanged Baseline,
and updates to both modules. This check does not save a training checkpoint.
Its log and acceptance.json are in deployment/joint_acceptance_v2.log and work_dirs/joint_acceptance_v2.
The earlier failure log for missing masks is retained for audit; the check was rerun after the fix.

The two-GPU performance check uses the first 40 real training iterations and saves no checkpoint.
The official five-seed runs restart from the fixed source.
The official queue completes training → checkpoint audit → two-GPU inference → CPU evaluation for each seed
before proceeding to the next.
Each job uses at most 90% of the PyTorch allocator budget on each GPU.
Two joint-training jobs are not run concurrently on the same GPUs.

Controller: prepare/check/run in `tools/run_memory_planning_joint_five_seeds.py`.
Before launch, registry.json records SHA256 values for Python source, all five complete configurations,
the source checkpoint, and the Baseline evaluation evidence.
Read-only inputs are shared; code and outputs are isolated.
Changes to registered code prevent the queue from mixing versions.
The official session is named `planning-joint-five-seeds` and can continue after SSH disconnects.

## Required artifacts for each seed

- epoch_1.pth, training logs, configuration, and training timings and parameter-state summaries for both ranks.
- Native occupancy predictions for 4219 unique tokens (lossless uint8), and cumulative positions from trajectory increments.
- IoU/mIoU: 0s, 1s, 2s, 3s, Future Mean over only 1/2/3s, and raw confusion matrices.
- L2 / Obj Collision / Box Collision: all six time steps from 0.5s to 3s, plus 1/2/3s prefix means and endpoint values.
  Planning evaluator and GT SHA256 values must match the existing Baseline;
  record the number of valid planning samples separately.
- Checkpoint/prediction/configuration/evaluator provenance summaries and completion markers.
  Do not fill incomplete seeds with zero or include them in statistics.
- results_by_horizon.json / .csv / statistics_by_horizon.csv in the top-level output directory retain
  each of the five runs, their means, and sample standard deviations (ddof=1).

This experiment measures seed variation in one-epoch joint adaptation from a fixed best T1 starting point.
It cannot be described as the variance of training the entire method from scratch.
Changes in training loss do not establish improvements in IoU/mIoU or planning;
the complete evaluation of each seed is the evidence.
