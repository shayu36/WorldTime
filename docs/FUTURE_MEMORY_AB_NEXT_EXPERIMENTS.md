# A+B follow-up experiments, step by step: diagnose first, then change the loss

Status (2026-10-16): Only code, configurations, and CPU static/small-scale tests have been completed here. The two 3090 cards are running another training job, `sparseworld-traj-prescf.py`; this round did not launch A+B diagnostics, full validation, or new training. The interventions below are output experiments with a fixed A+B epoch 1 checkpoint, not retraining results.

## Existing evidence and limitations

A+B epoch 1 full validation: Future IoU 22.7900, Future mIoU 13.1667. The earlier Baseline was approximately 22.2133 / 13.2233. Paired diagnosis on 256 frames showed that restoring Baseline position raised Future mIoU by +0.1434 (95% CI +0.0628 to +0.2240); restoring Baseline occupancy changed binary Future IoU by -0.5006; restoring Baseline semantics changed Future mIoU by -0.2894. These are output interventions and do not establish causal effects after retraining; see `FUTURE_MEMORY_AB_PAIRED_GPU_DIAGNOSIS.md` for the complete record.

## Stage 1: output-level diagnosis without changing weights

Fix the same A+B epoch 1 checkpoint and the same forward pass. Compare original A+B, restoration of Baseline position only at 1s/2s, and soft protection of occupancy **negative** residuals only at 2s/3s (retaining 25%, 50%, or 75% of the original residual). Protection mass is calculated from the probability union of the Baseline and corrected semantic distributions over motorcycle, pedestrian, traffic cone, manmade, and vegetation, without hard argmax. Positive occupancy residuals, the Semantic branch, Memory retrieval, and future recurrence remain unchanged. The three coefficients are only a diagnostic grid, not recommended training hyperparameters.

First run a 256-frame scan. `summarize_future_memory_scan.py` splits scenes into selection/holdout sets by the parity of their SHA256. Select a scheme only on the selection set, requiring better 2s/3s mIoU and a binary IoU decline of no more than 0.10 percentage points, then inspect the holdout set. The holdout set also cannot replace full validation.

```bash
bash /data/jxy/projects/tools/run_next_future_memory_diagnostics.sh scan
```

Regardless of scan results, the 1s/2s position intervention has shown significance only on the existing 256 frames; separate validation on 4219 frames is required. This script compares only original A+B against disabling 1s/2s position corrections, avoiding wasted GPU time on full evaluations of multiple schemes. Evaluation will take a while and should run after the GPUs become idle:

```bash
bash /data/jxy/projects/tools/run_next_future_memory_diagnostics.sh position-full
```

The two output directories are `work_dirs/diagnostics-ab-miou-20261016/scan256` and `position_full4219`. The script refuses to overwrite existing directories. Compare 1s/2s/3s binary IoU, mIoU, and per-class TP/FP/FN, focusing on pedestrian, vegetation, and manmade at 2s/3s. Consider actually changing model configuration only if the scan's holdout set agrees in direction and full validation does not harm key metrics; new training/full validation is then required, and output interventions cannot directly count as new model scores.

## Stage 2: single-variable SoftVoxel center-alignment experiment

Old SoftVoxel treats index `k` as `(origin+k·voxel_size)`, whereas GT index `k` actually represents voxel center `(origin+(k+0.5)·voxel_size)`. The new optional `soft_voxel_center_offset=0.5` makes the training loss's trilinear interpolation symmetric around GT centers; at boundaries, invalid corners are discarded and valid-corner weights are normalized, while points outside the scene are not projected back onto its boundary. Original V2 defaults remain 0/False, leaving inference with existing checkpoints unchanged.

The separate configuration restores optimizer/runner from A+B epoch 1 and adds only one epoch at a fixed `2e-5` learning rate, producing local epoch 2; original A+B epoch 2 is the control. Note: this experiment tests the training loss and will not directly change inference with an already-trained checkpoint; output interventions cannot replace it. After completing Stage 1 diagnosis and confirming that the GPUs are idle, the user runs locally:

```bash
bash /data/jxy/projects/tools/run_train_future_memory_softvoxel_center.sh
```

Save to the separate directory `work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center`, without overwriting A+B or V1 checkpoints. After training, fully validate the new epoch 2 using existing dump-only + offline evaluator, and compare Future IoU/mIoU and per-class recall against original A+B epoch 2 and the same Baseline. If the new loss does not improve 2s/3s mIoU, or binary IoU declines substantially, stop this branch; if the direction is favorable, consider a further round of longer training. Before training, only "the code is connected" can be claimed, not improved metrics.

## Possible follow-ups (not implemented or launched yet)

If the output scan confirms that occupancy protection helps, design trainable class/time conditioning or a Baseline anchor; avoid directly treating output-level `alpha` as a training result. If full position validation confirms damage at 1s/2s, adjust the position gate or position loss separately. Change only one factor at a time; do not simultaneously change Top-K, thresholds, and multiple loss weights.

## 2026-10-18 progress

The 256-frame scan has completed: the selection set chose `alpha=0.25`, but holdout-set 2s/3s binary IoU declined by
0.2367 percentage points, so it is not adopted directly. On the holdout set, `alpha=0.75` yielded mIoU +0.1851 and
IoU +0.0603 percentage points, still only a diagnostic result. In the full 4219-frame position output intervention (A+B epoch1),
restoring 1s/2s position to Baseline changed Future IoU from 22.7907→22.8043 and mIoU from
13.1653→13.3007; this is not a newly trained checkpoint.

SoftVoxel center alignment **correctly resumed** from A+B epoch1 and continued for only one epoch; the resulting checkpoint's
706 non-V2 tensors are identical to the source. Full evaluation on 4219 frames yielded IoU
`[25.68,23.85,22.84,21.68]` and mIoU `[18.20,15.28,13.21,11.39]`,
with Future means of 22.7900/13.2933; the old-SoftVoxel control at the same epoch was 22.7433/13.1067.
An earlier erroneous training run that omitted `--resume-from` is retained in the directory without the `-from-ab1` suffix
and must not be used for comparison. Next, prioritize rechecking the position-branch intervention on the center-aligned checkpoint,
then decide whether to change the trainable Position Gate; do not combine output-intervention and training results as if they came from one checkpoint.