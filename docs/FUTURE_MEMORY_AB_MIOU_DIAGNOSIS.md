# A+B: Diagnosing the 2s/3s mIoU decline (2026-10-15)

2026-10-16 addendum: After GPUs became available, single-snapshot branch interventions on 256 frames were completed; see
[Paired GPU diagnosis](FUTURE_MEMORY_AB_PAIRED_GPU_DIAGNOSIS.md). The evidence and limitations as of 10/15 are preserved below.
The subsequent paired experiment supports class-selective recall damage from occupancy and identifies net damage from the position branch; the semantic branch is beneficial overall.

> 2026-10-16 status update: C has completed three full evaluations, with Future IoU/mIoU of
> 22.7900/13.1667, 22.7600/13.1133, and 22.7700/13.1400; none exceeded A+B epoch1.
> The C class-group Gate scheme has been abandoned and removed from model code and run entrypoints; A+B epoch1 is currently
> the latest retained V2 result. References below to "C is running" describe the environment at the time of diagnosis.

This round involved only code reading, existing-log analysis, CPU sampling of saved predictions, and small forward-pass experiments.
No production model/loss/configuration changes, backward/optimizer steps, new training,
reruns of full validation, or interruption of the ongoing C continuation training occurred.
Starting HEAD: `852977be7f9a734b2d32cef781093c42831350a3`.
Existing uncommitted A+B/C changes were left unchanged.

## 1. Distinguish the comparison targets first

The following are existing **full evaluations on 4219 frames**, not evaluations rerun in this round. IoU/mIoU are percentages; differences are percentage points.

| Model | Future IoU | Future mIoU | mIoU 1s | mIoU 2s | mIoU 3s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (historical official record) | 22.2133 | 13.2233 | 14.96 | 13.18 | 11.53 |
| occfix V2 epoch11, the source model for A+B | 22.7967 | 13.1100 | 15.06 | 13.04 | 11.23 |
| A+B epoch1 | 22.7900 | 13.1667 | 15.08 | 13.07 | 11.35 |
| A+B epoch2 | 22.7433 | 13.1067 | 15.05 | 13.02 | 11.25 |
| A+B epoch3 | 22.7667 | 13.1300 | 15.08 | 13.05 | 11.26 |

The decline already existed in the source V2. A+B epoch1 improved mIoU over its source model (especially 3s, +0.12),
but did not recover the original Baseline. The entire degradation cannot be attributed to A+B, nor can A+B itself be credited with +0.58 IoU.
The three rounds show no sustained improvement trend; there is also no shared-Gate continuation control with the same budget, so the effect cannot be attributed solely to splitting the Gate.

Sources: `docs/EXPERIMENT_RECORDS.md`;
`work_dirs/sparseworld-traj-memory-future-adapter-v2-occfix-q48/eval_epoch_11/metrics.json`;
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/eval_epoch_{1,2,3}/metrics.json`.

The per-class Baseline comes from the existing `work_dirs/sparseworld-traj-memory-only/eval_epoch56_memory_off.log`.
Note that the old log reports 2s binary IoU as 22.27, whereas the historical official summary reports 22.28; this report does not rewrite historical values
or interpret the 0.01 recording difference as a model gain. The mIoU values agree in both places.

## 2. Full evaluation: which classes reduce mIoU

Comparing A+B epoch1, which has the best mIoU, against Baseline gives the following per-class IoU differences:

| Class | 2s delta | 3s delta |
| --- | ---: | ---: |
| pedestrian | -2.35 | -1.83 |
| motorcycle | -1.05 | -1.22 |
| traffic_cone | -0.59 | -0.82 |
| manmade | -0.93 | -1.12 |
| vegetation | -1.22 | -1.34 |
| driveable_surface | +1.32 | +1.67 |
| sidewalk | +0.75 | +0.83 |
| terrain | +0.82 | +0.88 |
| car | +0.87 | +0.28 |

This cannot be summarized as "all background classes improve and only dynamic classes suffer": manmade/vegetation also suffer.
At 2s, all improving classes contribute approximately +0.2924 to the mean, while all declining classes contribute -0.3965;
at 3s, the corresponding contributions are +0.2712 / -0.4524 (calculated from the rounded per-class log values).
The five affected classes in the table already contribute approximately -0.3612 / -0.3724, enough to offset the gains from the remaining classes.

Binary occupied IoU distinguishes only free/occupied. Predicting a pedestrian as road can still count as a binary TP,
but semantically it increases both pedestrian FN and road FP; predicting a pedestrian as free damages both metrics.
mIoU weights the 17 classes equally, while binary IoU does not weight classes equally.
Thus, "improvement in some large classes/regions + missed detections/misclassification of small classes" can raise binary IoU while lowering mIoU.
This is the current output-level tradeoff; it does not imply that the two metrics are necessarily mutually exclusive, nor does it prove that a particular loss actively sacrifices semantics.

## 3. CPU diagnosis of saved predictions: 256 frames covering 150 scenes

The script `tools/diagnose_future_memory_semantics.py` selects 256 evenly spaced indices using the actual dataset.temp2nusc_map;
it compares native11, occfix11, AB1, and AB3 on the same tokens, the same GT, and 0/1/2/3s.
Following the existing evaluator, it uses no camera/lidar mask, computes an 18-class confusion matrix, and excludes free from mIoU.
It reads only existing dumps and performs no inference for new predictions. Results and the token list are saved at:

`work_dirs/diagnostics-ab-miou-20261015/paired_subset256.json`

| Model/time | binary precision % | binary recall % | binary IoU % | mIoU % |
| --- | ---: | ---: | ---: | ---: |
| native11 2s | 37.6673 | 28.4023 | 19.3212 | 11.0913 |
| occfix11 2s | 35.1434 | 39.8871 | 22.9750 | 13.0562 |
| AB1 2s | 35.2326 | 39.7569 | 22.9697 | 13.0919 |
| native11 3s | 36.6076 | 26.0472 | 17.9505 | 9.4292 |
| occfix11 3s | 33.9689 | 37.8342 | 21.8008 | 11.4442 |
| AB1 3s | 34.0823 | 37.6861 | 21.7980 | 11.5453 |

This contradicts **directly treating the old 0.20m positive-sample error as the cause of current A+B behavior**:
occfix has recovered substantial recall, and the saved A+B configuration indeed uses positive_radius=None,
separate means for positive and negative groups, and occupancy anchor=0.05. Losses remain, but this is not the unrepaired original configuration.
The native→occfix comparison includes retraining and other changes, so it is not a strict ablation of label correction alone.

Where AB1 sends each true GT class (each row is normalized by the number of GT voxels of that class):

| Class/time | Correct class % | Predicted free % | Predicted as another non-free class % |
| --- | ---: | ---: | ---: |
| pedestrian 2s | 3.12 | 76.96 | 19.92 |
| pedestrian 3s | 1.13 | 78.69 | 20.18 |
| motorcycle 2s | 6.64 | 66.17 | 27.19 |
| motorcycle 3s | 5.63 | 68.69 | 25.68 |
| traffic_cone 2s | 5.45 | 59.95 | 34.59 |
| traffic_cone 3s | 3.44 | 58.95 | 37.61 |
| manmade 2s | 17.56 | 78.57 | 3.87 |
| vegetation 2s | 26.10 | 71.30 | 2.60 |

Pedestrians are mainly confused with manmade/sidewalk/driveable_surface; motorcycles and traffic cones are mainly confused with ground/buildings.
Misses into free are more common than confusion between non-free classes, but **there is no paired Baseline dump, so these data cannot quantify the additional miss rate relative to Baseline,
much less distinguish whether free predictions arise from the O threshold, position misalignment, or insufficient point coverage**.

In the 2s GT, the six background classes with ids11–16 account for 92.87%; pedestrians + motorcycles + traffic cones together account for only 0.425%,
yet the latter three classes carry 3/17=17.65% of the weight in mIoU. This directly explains why binary totals can easily conceal small-class failures.
These are GT frequencies only, not training-point/gradient frequencies; they do not establish that gradient imbalance has been measured.

At 2s, AB1 vs occfix11 removes 217508 FP and adds 172198 FP, while recovering 53456
occupied TP and losing 64657 occupied TP; there is a small precision/recall tradeoff, but its magnitude is small.
Where both predictions at the same location are occupied, 16391 classes change from correct→incorrect and 16164 from incorrect→correct, with no large net semantic correction.
The same pattern holds at 3s. A+B currently performs minor recalibration rather than solving small-class recognition.

Sampling cannot replace full results or significance testing; sampled IoU varies especially strongly for rare classes.
A small number of 0s voxels differ across runs, so these cannot be treated as interventions on a strictly identical forward-pass snapshot; the cause has not been separately verified.
This does not establish random sampling in the pipeline: the current LoadMultiViewImageFromMultiSweeps.load_offline
actually uses fixed adjacent indices, so test_mode=False alone does not justify alleging random retrieval.

## 4. Newly confirmed loss–decoding inconsistencies (causal attribution is not yet complete)

### 4.1 SoftVoxel coordinate references are offset by half a voxel

`query_memory.py:sparse_soft_voxel_iou_loss` uses
`coord=(p-origin)/voxel_size`, treating index k as a lower-corner grid point;
GT labels instead represent a voxel, and existing GT point supervision uses its center, `(k+0.5)*voxel_size+origin`.
Consequently, a point exactly at the GT center distributes 1/8 of its mass to each of 8 voxels in the positive directions, instead of placing all its mass on the target center grid point.
A center-indexed trilinear surrogate should be tested with `coord=(p-origin)/voxel_size-0.5`.
This remains a soft surrogate and will not exactly match hard floor; it also does not constitute approval to modify current training.

A CPU forward experiment calling the **actual current loss**: only GT voxel [1,1,1] is pedestrian, voxel=0.4m,
and all others are free. All points below fall in the same correct voxel in the hard evaluator and use identical logits:

| Point coordinates (m) | Current SoftVoxel loss |
| --- | ---: |
| (0.60,0.60,0.60), GT center | 0.9962075 |
| (0.59,0.60,0.60) | 0.9960178 |
| (0.61,0.60,0.60) | 0.9963968 |
| (0.401,0.401,0.401), near the lower-coordinate corner | 0.9632545 |
| (0.799,0.799,0.799), near the higher-coordinate corner | 1.0000000 |

Moving from the center toward lower coordinates thus lowers the loss, creating a preference inconsistent with center-position supervision.
After changing to center indexing only in the diagnostic, the center loss=0.9628175, and shifts to either side both raise it to approximately 0.964248.
These are forward-value comparisons; **no numerical gradient check was performed**.
Values are close to 1 partly because of averaging over 17 classes, with no GT for the other classes in this frame; the absolute loss value cannot directly explain training failure.
Small objects/class boundaries may be more sensitive than large road regions; the actual contribution to the mIoU decline still requires single-variable verification.

### 4.2 Actual decoding depends on more than O

After sigmoid, `opus_head.py:get_occ` additionally multiplies scores for points whose top class is manmade/vegetation
by `clamp(0.1/d,max=1)` according to their distance from the Query-point mean (d here uses normalized coordinates).
Further stages include 3m Query-distance filtering, point filtering by top-class thresholds, scatter_max, closing, and final class thresholds.
These stages are not fully incorporated into occupancy/threshold supervision.

A CPU call to the actual get_occ (padding disabled to isolate the existing class-distance factor):
with the same two point positions and the same maximum logit O=logit(0.4), changing only the top semantic class,
car retains 2 voxels; manmade/vegetation retain 0 voxels; disabling that distance factor in the diagnostic call restores 2 for manmade.
This proves that `max(S_decode)=O` **does not mean that final voxel retention is determined only by O**.
Class thresholds, position, Query range, and postprocessing remain coupled.
This decoder behavior already existed in Baseline and is not a new A+B bug; however, new losses/new logit calibration must still be compatible with it.
It is consistent with the manmade/vegetation degradation, but its contribution has not yet been quantified on a real shared snapshot.

### 4.3 Training logs support "corrections are not small"; decreasing loss alone is insufficient

Means over the last 20 training-log entries of A+B epoch1:

| horizon | G_sem | G_occ | G_pos | Signed O correction | Position correction norm (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1s | .7840 | .7719 | .7768 | -.7183 | .1435 |
| 2s | .7887 | .7783 | .7837 | -.6120 | .2000 |
| 3s | .7933 | .7842 | .7905 | -.5164 | .2507 |

Gating for old Memory has not automatically become small, and the 3s position correction is already substantial relative to a 0.4m voxel.
Negative O corrections may remove false positives or small-class true positives; position shifts can cause misses;
final labels alone cannot separate the two. Similar means across the three Gates do not establish that task separation is ineffective,
nor do different means prove that gradient conflict existed previously.

Semantic CE/Focal aggregates over points, and binary positive/negative balancing is not class balancing; cls_weights=1–3 is also not equivalent to
equal class weighting in mIoU. SoftVoxel uses sigmoid(O)*softmax(S_sem) + accumulation/noisy-OR, whereas actual
decode uses relative logits + sigmoid + scatter_max + closing, leaving differences in probability scale/density/morphology.
These are potential risk mechanisms; actual gradient shares have not been measured, and no loss is claimed to have been proven to dominate the entire decline.

## 5. Next steps: attribute the causes before adding more structure blindly

1. Keep C's run/configuration unchanged and report the five affected classes separately during evaluation, rather than looking only at the mean.
   C only subdivides the semantic gate and cannot directly restore points removed by occupancy or displaced by position;
   C's background group contains both improving road classes and declining manmade/vegetation, so it is not guaranteed to resolve conflicts within the group.
2. Once a GPU becomes available, use the same batch of cached inputs, the same frozen Baseline future snapshot, and the same AB1 weights
   for small-scale branch interventions: restore only baseline O; restore only baseline position; restore only baseline semantics.
   Retain the other two components and the original Memory retrieval; do not resample random inputs; check 0s for strict equality.
   Count per-class TP/FP/FN, point threshold-passing rates, and rejection rates at each filtering stage to identify the main cause.
   This has not yet been run here and is not equivalent to the previously canceled V1 shuffled/context-zero/cache-off controls.
3. Prioritize an independent fix and tests for SoftVoxel center-coordinate alignment, then verify it in a separate experiment directory;
   do not mix it into current C or simultaneously change Gates, radii, or class weights.
4. If restoring O mainly rescues small classes, investigate class-specific occupancy protection/threshold calibration; if restoring position helps,
   prioritize spatial supervision/position-residual corrections; if semantic replacement helps, then target semantic confusion.

GPU0/1 are still occupied by C at approximately 20.7/22.7GB; this round did not preempt GPUs for model interventions.
No original Baseline saved predictions usable for same-token alignment were found for this round, so a Baseline recall control cannot be fabricated.
The current status is therefore "class-level direct causes quantified, two reproducible mechanism issues identified, branch-level causal contributions still awaiting closure,"
not "occupancy alone is conclusively the sole cause"; nor can it be claimed that metrics will necessarily recover after a fix.

## 6. Reproduction and checks for this round

The only added files are this report, two CPU diagnostic tools, and one pure-numpy test file.
The CPU sampling tool accepts repeated `--dump NAME=directory` arguments; the first is the paired reference;
`--samples 256` is used, and `--output` refuses to overwrite existing files.
Saved results from four groups—source occfix11, AB1, AB3, and native11—were processed; outputs appear in Section 3.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/python tools/probe_future_memory_loss_alignment.py --output work_dirs/diagnostics-ab-miou-20261015/alignment_probes.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/pytest -q tests/test_semantic_dump_diagnostics.py
git diff --check
```

Forward-probe assertions passed; 4 pure-function tests passed; the diff check passed.
Output is also saved at `work_dirs/diagnostics-ab-miou-20261015/alignment_probes.json`.
No commits/pushes, checkpoint/cache changes, or historical experiment-value changes were made.