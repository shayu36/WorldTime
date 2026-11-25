# Notable Experiment Records

Updated: 2026-10-16 UTC

Experiment directory: `work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix`

Evaluation scope: `epoch_1` through `epoch_12`, for which complete occupancy + planning evaluations are available. The repository currently contains only `epoch_1.pth` through `epoch_12.pth`; `epoch_14.pth` / `epoch_16.pth` were not found, so no results are filled in for them.

Raw logs:

- Occupancy:`work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/eval_selected_epochs/epoch_{N}_occ_eval.log`
- Planning:`work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/eval_selected_epochs/planning_eval/planning_epoch{N}.log`

## Baseline

The Baseline uses `ckpts/epoch_56.pth` and comes from existing formal records. Units are those directly output by the evaluation scripts.

| Metric | 0s | 1s | 2s | 3s | Mean | Future Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IoU | 25.68 | 23.15 | 22.28 | 21.21 | 23.0800 | 22.2133 |
| mIoU | 18.20 | 14.96 | 13.18 | 11.53 | 14.4675 | 13.2233 |

No planning evaluation log was found for the epoch-56 baseline, so planning metrics in this file summarize only results within the current experiment and do not report baseline deltas.

## Occupancy summary

All `Delta` values are relative to the baseline future mean: IoU baseline future mean = `22.2133`, mIoU baseline future mean = `13.2233`.

| Epoch | IoU [0s,1s,2s,3s] | IoU Future Mean | Delta | mIoU [0s,1s,2s,3s] | mIoU Future Mean | Delta | Conclusion |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | [25.68, 23.14, 22.27, 21.21] | 22.2067 | -0.0067 | [18.20, 14.96, 13.18, 11.53] | 13.2233 | +0.0000 | mIoU matches the baseline |
| 2 | [25.68, 23.15, 22.28, 21.23] | 22.2200 | +0.0067 | [18.20, 14.97, 13.20, 11.55] | 13.2400 | +0.0167 | One of the early best results |
| 3 | [25.68, 23.14, 22.27, 21.23] | 22.2133 | +0.0000 | [18.20, 14.96, 13.19, 11.55] | 13.2333 | +0.0100 | Above the baseline |
| 4 | [25.68, 23.17, 22.29, 21.25] | 22.2367 | +0.0233 | [18.20, 14.98, 13.20, 11.57] | 13.2500 | +0.0267 | Current best mIoU |
| 5 | [25.68, 23.16, 22.28, 21.26] | 22.2333 | +0.0200 | [18.20, 14.97, 13.19, 11.56] | 13.2400 | +0.0167 | Above the baseline |
| 6 | [25.68, 23.17, 22.29, 21.28] | 22.2467 | +0.0333 | [18.20, 14.97, 13.19, 11.56] | 13.2400 | +0.0167 | Current best IoU |
| 7 | [25.68, 23.18, 22.29, 21.25] | 22.2400 | +0.0267 | [18.20, 14.98, 13.20, 11.56] | 13.2467 | +0.0233 | Close to the best mIoU |
| 8 | [25.68, 23.17, 22.29, 21.27] | 22.2433 | +0.0300 | [18.20, 14.98, 13.19, 11.57] | 13.2467 | +0.0233 | Close to the best overall result |
| 9 | [25.68, 23.17, 22.28, 21.24] | 22.2300 | +0.0167 | [18.20, 14.97, 13.19, 11.55] | 13.2367 | +0.0133 | Above the baseline |
| 10 | [25.68, 23.16, 22.28, 21.24] | 22.2267 | +0.0133 | [18.20, 14.97, 13.19, 11.56] | 13.2400 | +0.0167 | Above the baseline |
| 11 | [25.68, 23.16, 22.28, 21.26] | 22.2333 | +0.0200 | [18.20, 14.97, 13.19, 11.56] | 13.2400 | +0.0167 | Above the baseline |
| 12 | [25.68, 23.16, 22.28, 21.26] | 22.2333 | +0.0200 | [18.20, 14.97, 13.19, 11.56] | 13.2400 | +0.0167 | Above the baseline |

## Planning summary

In all complete planning logs, L2 and collision rates are exactly identical from `epoch_1` through `epoch_12`. The full record for each epoch is retained here for row-by-row comparison with the occupancy table.

| Epoch | L2@1s | L2@2s | L2@3s | Obj Col@1s | Obj Col@2s | Obj Col@3s | Box Col@1s | Box Col@2s | Box Col@3s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 2 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 3 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 4 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 5 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 6 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 7 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 8 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 9 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 10 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 11 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |
| 12 | 0.159597 | 0.194546 | 0.239346 | 0.000000 | 0.000000 | 0.000000 | 0.000948 | 0.001007 | 0.001106 |

Planning means: `L2_mean = 0.197830`, `Box Col_mean = 0.001021`, `Obj Col_mean = 0.000000`.

## Current best results

| Objective | Best epoch | Value | Relative to baseline |
| --- | ---: | ---: | ---: |
| mIoU Future Mean | 4 | 13.2500 | +0.0267 |
| IoU Future Mean | 6 | 22.2467 | +0.0333 |
| mIoU@1s | 4 / 7 / 8 | 14.98 | +0.02 |
| mIoU@2s | 4 / 7 | 13.20 | +0.02 |
| mIoU@3s | 4 / 8 | 11.57 | +0.04 |
| IoU@3s | 6 | 21.28 | +0.07 |

## Assessment

Evaluation is complete for all Future Adapter formal-batch4fix checkpoints from `epoch_1` through `epoch_12`. Apart from `epoch_1`, whose future IoU mean is slightly below the baseline, all epochs have a future IoU mean at least as high as the baseline. Future mIoU means from `epoch_2` through `epoch_12` are above or equal to the baseline, with small gains.

In the occupancy results, `epoch_4` is the current best checkpoint by mIoU and `epoch_6` is the best by IoU. If mIoU is the primary metric, prioritize retaining `epoch_4.pth`; if occupied IoU is the primary metric, prioritize `epoch_6.pth`.

Planning metrics are exactly identical from `epoch_1` through `epoch_12`, indicating that changes in this Future Adapter run primarily affect occupancy forecasting rather than trajectory outputs. Without baseline planning logs, no planning improvement over the baseline can be claimed.

## 2026-10-16: Best Retained Experiment from the V2 A+B Stage

This section appends the latest V2 experiments without rewriting the historical V1 values above.
A+B splits one shared Point Gate into three task gates for semantic, occupancy, and position,
and adds task-level horizon biases for 1s/2s/3s.
The Baseline remains frozen, and Memory corrections are not fed back into future recurrence.

| Version | Future IoU | Relative to Baseline | Future mIoU | Relative to Baseline | Selection |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 22.2133 | — | 13.2233 | — | Frozen reference |
| A+B epoch1 | **22.7900** | **+0.5767** | **13.1667** | -0.0566 | Best V2 retained at that stage |
| A+B epoch2 | 22.7433 | +0.5300 | 13.1067 | -0.1166 | Not retained as the best |
| A+B epoch3 | 22.7667 | +0.5534 | 13.1300 | -0.0933 | Not retained as the best |

A+B epoch1 results by horizon:

```text
IoU  = [25.68, 23.83, 22.84, 21.70]
mIoU = [18.20, 15.08, 13.07, 11.35]
```

The selected experiment at this stage was A+B epoch1. It substantially improves occupied IoU,
but Future mIoU remains slightly below the Baseline, so it cannot be described as improving all semantic metrics.
See `docs/FUTURE_MEMORY_ADAPTER_AB_EXPERIMENT.md` for the detailed design, limitations, and raw-result paths.

### C Class-Group Gate Experiment: Abandoned

On top of A+B, C attempted to subdivide the Semantic Gate into three fixed class groups:
background, traffic participants, and road infrastructure.
It generated group biases for each point and expanded them into gates for 17 classes.
Its complete evaluation results are:

| C checkpoint | Future IoU | Future mIoU |
| --- | ---: | ---: |
| epoch1 | 22.7900 | 13.1667 |
| epoch2 | 22.7600 | 13.1133 |
| epoch3 | 22.7700 | 13.1400 |

None of C's three rounds exceeded A+B epoch1, and further training showed no upward trend.
The fixed class groups also put classes with different performance trends under the same gate,
and cannot directly repair missed detections caused by occupancy filtering or position offsets.
C is therefore no longer part of the current code approach; its dedicated code, configurations,
and run entry points have been removed. Its experimental artifacts are retained only as historical evidence.
Subsequent experiments start from A+B epoch1.

## 2026-10-18: SoftVoxel Center Alignment, Current Best Checkpoint by mIoU

The model, optimizer, and runner were fully resumed from A+B epoch1. The only changes were
setting the SoftVoxel loss voxel-center offset to 0.5 and normalizing valid corners at boundaries,
followed by one further epoch of training. All 706 non-V2 tensors from the source checkpoint
are exactly identical in the resulting checkpoint; 58 of the 60 V2 tensors changed.
Complete validation on 4219 frames:

| checkpoint | IoU 1s/2s/3s | Future IoU | mIoU 1s/2s/3s | Future mIoU |
| --- | --- | ---: | --- | ---: |
| A+B epoch2 (same-round control) | 23.80 / 22.79 / 21.64 | 22.7433 | 15.05 / 13.02 / 11.25 | 13.1067 |
| SoftVoxel center alignment epoch2 | 23.85 / 22.84 / 21.68 | 22.7900 | 15.28 / 13.21 / 11.39 | **13.2933** |
| occupancy correction epoch11 (best IoU) | 23.84 / 22.85 / 21.70 | **22.7967** | 15.06 / 13.04 / 11.23 | 13.1100 |

At 0s, IoU remains 25.68 and mIoU remains 18.20.
The current highest Future IoU and highest Future mIoU come from different checkpoints.
When selecting one trained checkpoint, center-aligned epoch2 is only 0.0067 percentage points below
the highest IoU, while its Future mIoU is 0.1833 percentage points higher. Raw metrics are in
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-from-ab1/eval_epoch_2/metrics.json`.
This experiment uses a single starting point and a single round;
it does not establish stable gains across multiple random seeds.
