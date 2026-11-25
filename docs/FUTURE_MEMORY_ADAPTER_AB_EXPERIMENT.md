# A+B task- and horizon-gated continuation training

Starting point: occupancy-corrected `occfix-q48/epoch_11.pth`, corresponding to Future IoU
22.7967 and Future mIoU 13.1100. HEAD before implementation was
`852977be7f9a734b2d32cef781093c42831350a3`; the original workspace was clean.

## The only structural change

A: Retain the original `gate_head` as the semantic head and copy its trained weight/bias into
`gate_occ_head` and `gate_pos_head`. B: Add zero-initialized
`task_horizon_bias[3,3]` (rows=1/2/3s, columns=sem/occ/pos).
Each Gate is `sigmoid(head_task(F) + bias[horizon, task])`, with output
`[B,Q,R,1]`. C is not enabled, and no class-group gating is added.

Semantics, occupancy, and position each apply their own Gate; decode still preserves class ranking, with the maximum
logit equal to occupancy. Position updates retain metric relative-encoding residuals.
All three gates become zero when there are no candidates; Baseline is frozen, 0s/non-target steps are unchanged, and corrections do not feed back into recurrence.
The default is `gate_mode=shared`, compatible with old V1/V2 configurations and parameter keys.

Migration does not zero the trained residual heads or reset the original Gate bias. Initial output should equal the trained
source checkpoint, rather than the frozen Baseline. The migration tool generates a new file, refuses overwriting,
preserves all original state_dict tensors, records the source SHA256, and discards old optimizer/epoch state.
Strict loading of the full model and a synthetic B=2 forward pass at three horizons with actual Adapter weights serve as startup checks;
this is not an equivalence assessment over full validation.

## Run settings for this round

Separate configuration `sparseworld-traj-memory-future-adapter-v2-ab.py`, separate directory
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11`.
Launch only one A+B training run; decide on C later based on results, without launching C automatically.

- Two RTX3090 cards, batch/card=1, worker/card=1, chunk=48.
- activation checkpoint=True, save_on_cpu=False, dynamic FP16, seed=17.
- New AdamW optimizer, learning rate 2e-5, weight decay retained at 0.01.
- 100 iter linear warmup (0.5→1 times), then a fixed learning rate.
- 2e-5 is a conservative short-adaptation value below the source epoch11 rate of approximately 3.09e-5; it has not been optimized by tuning.
- Run 3 local epochs, saving epoch_1/2/3, corresponding to three continuation stages after source epoch11.
  This is not an exact resume to the original runner's epoch12/13/14; data-sampling counts also restart from local epoch0.
- Original occupancy labels, positive/negative balancing, anchor, and remaining loss/retrieval parameters are inherited from the corrected version.

Training logs add per-horizon valid-candidate proportions, mean/std of the three Gates over valid candidates,
mean absolute semantic and occupancy corrections, mean metric position-correction norms, and signed occupancy shifts.
These detached statistics do not enter the total loss. Activation checkpointing returns all gates; DDP
graph anchors cover all three gates, and the empty-Memory branch retains zero-gradient connections.

## Acceptance checks and attribution

First compare exact outputs before/after migration, independent branch/horizon control, no-candidate fallback, B=2,
checkpoint recomputation gradients, and all-empty-Memory parameter connections; then inspect the first actual training iterations.
Do not overwrite old checkpoints or commit/push.

After training, report IoU/mIoU and per-class results at each horizon, and inspect occupancy precision/recall.
The aim is to retain IoU gains while improving 2s/3s semantics; success is not claimed in advance.
This round runs only A+B exploration; a shared-Gate continuation control with the same starting point, optimizer reset,
learning rate/sampling/budget has not been run. Improvements over source epoch11 therefore include the effect of additional training and cannot be
attributed solely to the gating structure. Any future C run should also use a control with the same budget.

## Official evaluation results and current selection (2026-10-16)

All results below come from full validation on 4219 frames; Future Mean averages only 1s, 2s, and 3s.

| checkpoint | IoU [0s,1s,2s,3s] | Future IoU | mIoU [0s,1s,2s,3s] | Future mIoU |
| --- | --- | ---: | --- | ---: |
| A+B epoch1 | [25.68, 23.83, 22.84, 21.70] | **22.7900** | [18.20, 15.08, 13.07, 11.35] | **13.1667** |
| A+B epoch2 | [25.68, 23.80, 22.79, 21.64] | 22.7433 | [18.20, 15.05, 13.02, 11.25] | 13.1067 |
| A+B epoch3 | [25.68, 23.84, 22.82, 21.64] | 22.7667 | [18.20, 15.08, 13.05, 11.26] | 13.1300 |

A+B epoch1 is the latest best result retained for this round of the V2 approach. Relative to historical Baseline Future IoU
22.2133, it improves by 0.5767, but Future mIoU 13.1667 remains below Baseline 13.2233 by 0.0566.
"Best" therefore denotes the overall selection among current V2 experiments, primarily based on occupied IoU; it does not mean that 17-class
semantic mIoU has surpassed Baseline and does not alter historical V1 records.

Original results are saved at
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/eval_epoch_{1,2,3}/metrics.json`.

## Abandoned C scheme: class-group Semantic Gate

C starts from A+B epoch1, retaining the three task Gates for semantics/occupancy/position and the horizon bias, while further splitting only the semantic Gate
into three predefined class groups: background classes, traffic participants, and road facilities. A zero-initialized
`Linear(256,3)` predicts three group biases for each point, then expands them using a fixed class-to-group mapping into a
`[B,Q,R,17]` Semantic Gate; occupancy and position Gates remain `[B,Q,R,1]`.
At migration, zero group biases give strict equivalence with A+B; all V2 parameters subsequently continue training together.

Full-evaluation results:

| checkpoint | Future IoU | Future mIoU | Relative to best A+B IoU | Relative to best A+B mIoU |
| --- | ---: | ---: | ---: | ---: |
| C epoch1 | 22.7900 | 13.1667 | +0.0000 | +0.0000 |
| C epoch2 | 22.7600 | 13.1133 | -0.0300 | -0.0534 |
| C epoch3 | 22.7700 | 13.1400 | -0.0200 | -0.0267 |

C was abandoned and removed from production code because:

1. None of the three rounds exceeded A+B epoch1; continued training did not produce an upward trend.
2. C did not resolve the key tradeoff: occupied IoU remained relatively high, but 2s/3s semantic mIoU remained below Baseline.
3. Fixed coarse class groups contain both improving and declining classes. For example, the background group includes road classes alongside
   manmade/vegetation, and one group bias cannot express opposing needs within the group.
4. A class-group Semantic Gate can only adjust semantic residuals on existing points; it cannot restore points filtered out by the occupancy branch
   or displaced from the correct voxel by position residuals, so it does not fully match the principal observed error mechanisms.
5. It adds parameters, checkpoint migration, and class-level Gate maintenance costs without measurable metric gains.

C's training checkpoints and evaluation artifacts are temporarily retained at
`work_dirs/sparseworld-traj-memory-future-adapter-v2-c-from-ab1-5ep` as experimental evidence,
but its configuration, migration tool, training/evaluation scripts, dedicated tests, and class-group model path have all been removed.
Subsequent development starts from A+B code and checkpoints; training will no longer continue from C.