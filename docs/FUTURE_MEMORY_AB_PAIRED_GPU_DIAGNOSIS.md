# A+B single-snapshot GPU branch diagnosis (2026-10-16)

This follows `FUTURE_MEMORY_AB_MIOU_DIAGNOSIS.md`. It ran after the user confirmed GPU availability,
and is neither new training nor full validation. Production models, configurations, checkpoints, and cache were unchanged.
Starting and ending HEAD were both `852977be7f9a734b2d32cef781093c42831350a3`.
No training process was found before the run, and both 3090 cards were idle; this round used only GPU0.

## Method and consistency

- Checkpoint: A+B `epoch_1.pth`, SHA256
  `6d0e968915f28e1fcb14343100eb17a88484af39b94a6fdbb28331496acce597`.
- The same 256 frames and 150 scenes as yesterday, with token/index lists checked item by item.
- Each frame runs a normal A+B forward pass only once; interventions then operate on that output snapshot.
  Memory retrieval, the three gates, and Baseline recurrence are entirely unchanged; branches are not rerun and inputs are not fetched again.
- Five groups: normal AB; restore position only; restore occupancy only; restore semantics only; original Baseline snapshot reference.
- `restore_position=(S_decode_AB, P_base)`.
- `restore_occupancy=(S_sem_AB + max(S_base)-max(S_sem_AB), P_AB)`.
- `restore_semantic=(S_base + O_AB-max(S_base), P_AB)`.
- The snapshot reference directly uses `S_base,P_base`. These frozen outputs already exist within the same forward pass;
  this does not load another model for full Baseline evaluation and is not a cache-off/V1 control.
- All groups call the actual `get_occ()`, retaining class-distance score reduction, thresholds, scatter_max, and closing.
- The A+B checkpoint and original `ckpts/epoch_56.pth` have 679 strictly identical shared tensors,
  with no missing Baseline keys; strict loading of the full A+B model passed.
- All 256 frames passed strict 0s equality and strict equality of replayed AB decoding; all parameters had `grad=None`.
- Cumulative confusion matrices at 0/1/2/3s are also strictly identical to those calculated from yesterday's saved AB1 predictions.

All metrics describe **this subset** and cannot be treated as new full-validation results on 4219 frames.
In particular, subset Future mIoU is AB=13.2491 and snapshot reference=13.2178, whereas the earlier full evaluation had
AB=13.1667 and Baseline=13.2233, so the mean difference has the opposite direction. This subset cannot establish that AB surpasses Baseline overall.
The subset nevertheless reproduces the 2s/3s mIoU shortfall relative to the snapshot reference that motivated this diagnosis.

## 1. Paired metrics

Units are percentages.

| Group | IoU 1s | IoU 2s | IoU 3s | mIoU 1s | mIoU 2s | mIoU 3s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal AB | 23.9487 | 22.9697 | 21.7980 | 15.1101 | 13.0919 | 11.5453 |
| Restore position only | 23.9902 | 23.0111 | 21.7522 | 15.3319 | 13.2747 | 11.5711 |
| Restore occupancy only | 23.3107 | 22.4662 | 21.4376 | 14.8774 | 13.1541 | 11.7196 |
| Restore semantics only | 23.9599 | 22.9820 | 21.8073 | 14.7764 | 12.8035 | 11.2993 |
| Frozen Baseline snapshot reference | 23.3395 | 22.4922 | 21.3749 | 14.8739 | 13.1617 | 11.6180 |

Future mean differences relative to normal AB, in percentage points; brackets give 95% intervals from 1000 paired scene-bootstrap samples:

| Intervention | Future IoU difference | Future mIoU difference |
| --- | ---: | ---: |
| Restore position | +0.0124 [-0.0174,+0.0382] | +0.1434 [+0.0628,+0.2240] |
| Restore occupancy | -0.5006 [-0.7045,-0.3171] | +0.0012 [-0.1686,+0.1710] |
| Restore semantics | +0.0109 [+0.0088,+0.0133] | -0.2894 [-0.3474,-0.2283] |

The intervals describe variation only among selected scenes/frames and exclude uncertainty from training seeds, the full dataset, or retraining;
no multiple-comparison correction was applied to the exploratory per-class analyses.
Restoring position changes 2s mIoU by +0.1827 [+0.0896,+0.2735], but 3s by only +0.0259 [-0.0378,+0.0889].
Restoring occupancy changes 2s mIoU by +0.0622 [-0.1343,+0.2644] and 3s by +0.1743 [-0.0360,+0.3922]:
the latter two point estimates improve, but their overall mIoU intervals cross zero, so stable improvement has not been demonstrated.

## 2. Occupancy: selective recall damage exists, beyond harmless denoising

Holding semantics and position fixed and restoring only O directly rescues several previously affected classes:

| Class | 2s AB | 2s restored O | 3s AB | 3s restored O |
| --- | ---: | ---: | ---: | ---: |
| pedestrian | 2.744 | 5.361 | 1.043 | 3.047 |
| traffic_cone | 4.276 | 5.484 | 2.838 | 3.613 |
| manmade | 10.577 | 11.624 | 9.848 | 11.021 |
| vegetation | 15.730 | 17.059 | 14.886 | 16.264 |
| motorcycle | 5.640 | 5.975 | 4.909 | 5.200 |
| car | 18.355 | 17.569 | 15.241 | 15.143 |
| driveable_surface | 30.820 | 29.847 | 29.149 | 28.163 |

The fraction of pedestrian GT predicted as free drops from 76.96% to 64.07% at 2s and from 78.69% to 68.66% at 3s.
The interval for 2s pedestrian IoU improvement is [+1.5946,+3.7078], for manmade [+0.7524,+1.3442],
and for vegetation [+0.9872,+1.6668]; the direction for these classes is relatively clear.
The interval for restoring O on motorcycles crosses zero, so its small point-estimate gain is not overstated as a stable improvement.

However, restoring O also restores many false positives:

| Metric | 2s AB | 2s restored O | 3s AB | 3s restored O |
| --- | ---: | ---: | ---: | ---: |
| binary precision % | 35.2326 | 29.8784 | 34.0823 | 28.8170 |
| binary recall % | 39.7569 | 47.5233 | 37.6861 | 45.5681 |
| Non-free voxels/frame | 37907.7 | 53432.9 | 36536.6 | 52250.2 |

For example, restoring O at 2s adds a net 667909 occupied TP and a net 3306533 FP.
The occupied IoU decline is therefore consistent: current O has learned more precise, more conservative occupancy filtering,
but suppresses some true positives in classes to which mIoU is sensitive.

Mean O residual grouped by **Baseline-predicted top class** (not statistics restricted to GT-positive samples):

| Class | 1s | 2s | 3s |
| --- | ---: | ---: | ---: |
| pedestrian | -0.5929 | -0.8852 | -1.0013 |
| manmade | -1.5420 | -1.5806 | -1.5468 |
| vegetation | -1.2802 | -1.3059 | -1.2351 |
| driveable_surface | -0.1045 | -0.0200 | +0.0374 |

The overall mean O correction conceals these class differences. Pedestrians are suppressed more strongly as horizon increases, while roads are largely preserved.
These groups also contain Baseline false positives, so negative corrections are not all errors; the paired GT results above provide the basis for judgment.
Restoring O instead changes 1s mIoU by -0.2328, showing that occupancy residuals cannot simply be disabled at all horizons.

## 3. Position: net damage to the 1s/2s macro-average, with class-dependent effects at 3s

Restore only position, retaining AB semantics and O:

- 1s mIoU +0.2217, 2s +0.1827, 3s +0.0259.
- 2s motorcycle IoU +0.9027, with 95% interval [+0.2764,+1.6938]; bicycle point estimate +0.6143, with an interval crossing zero.
- Restoring position also loses road IoU: -0.2559/-0.3074/-0.5850 at 1/2/3s, respectively.
- At 3s, bicycle gains +0.5186, but pedestrians barely improve.

The position branch therefore also entails class tradeoffs, rather than harming every class. Mean gains should not conceal the loss for roads at 3s.
This supports prioritizing position supervision/residuals for the next fix, but **does not directly prove that the SoftVoxel half-cell offset found yesterday
is the training cause of all position errors**. Nearest-neighbor matching, historical positions of dynamic targets, and other losses may also contribute.

## 4. Semantics: the current net effect is positive and should not be removed first

Restoring Baseline semantics while retaining AB O and position lowers 1/2/3s mIoU by
0.3337/0.2884/0.2459, respectively, with all paired intervals below zero.
This indicates that this checkpoint's semantic residual helps overall, rather than gaining binary IoU mainly by damaging classes.
Fixed O does not imply fixed final voxel classes/occupancy: the actual decoder includes class thresholds and cross-point aggregation, so binary IoU also changes slightly.

## 5. Conclusions and next steps

The evidence is now stronger than "examining only final dumps," supporting this statement:

> Output interventions with fixed AB1 inputs and weights show that occupancy calibration provides the main occupied IoU gain,
> while damaging recall for pedestrians, buildings, vegetation, and other classes; position corrections also reduce the 1s/2s macro-average.
> Semantic corrections help overall. Gains/losses across classes and horizons offset one another, producing the 2s/3s mIoU shortfall.

"Main gain" here is a conditional-intervention observation; the deltas cannot simply be added into rigorous percentages of training contribution.
This does not establish any loss term, gradient group, or Memory age as the sole root cause, nor predict values after retraining.

Recommended order of fixes (none modified/launched in this round):

1. Retain the existing semantic branch. Prioritize independently correcting and validating SoftVoxel center-coordinate alignment, and investigate class/time protection
   for position residuals, without mixing other structural changes into the same experiment.
2. The occupancy objective is to protect true positives in affected classes while retaining denoising gains, rather than restoring every false positive;
   candidate directions include class-aware positive-sample balancing/anchors/residual protection and 2s/3s calibration.
   Uniformly lowering thresholds without verification or directly disabling every O residual is not recommended.
3. The next acceptance evaluation should report overall IoU/mIoU alongside TP/FP/FN for the affected classes above, rather than only the mean.
4. C only subdivides the semantic gate and is not guaranteed to resolve the O-filtering and position problems identified here; this diagnosis must not be presented as C's training result.

## Tools, execution, and artifacts

Added:

- `tools/diagnose_future_memory_branches.py`
- `tools/summarize_future_memory_branches.py`
- `tests/test_future_memory_branch_diagnostics.py`
- This report (the previous report also received a link to these results)

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 env/bin/python -u tools/diagnose_future_memory_branches.py --limit 256 --output-dir work_dirs/diagnostics-ab-miou-20261016/paired256
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/python tools/summarize_future_memory_branches.py work_dirs/diagnostics-ab-miou-20261016/paired256 --output-name summary-with-classes.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/pytest -q tests/test_future_memory_branch_diagnostics.py tests/test_semantic_dump_diagnostics.py
```

The tool refuses to overwrite an existing output directory; use a new diagnostic directory for reruns.
The initial 2-frame preflight failed at the CPU/GPU coordinate-range tensor used by diagnostic statistics; explicit device/dtype conversion in the diagnostic tool was fixed.
The failed record remains in `preflight2`, and the passing 2-frame preflight is in `preflight2-v2`.
The 256 frames took approximately 354 seconds (excluding construction), with peak PyTorch allocated memory of approximately 2.23GiB and nvidia-smi usage of approximately 4.2GiB.

Inside `paired256/`: `provenance.json`, `config.py`, `progress.jsonl`, `report.json`,
`per_sample_confusions.npz`, `summary.json`, and `summary-with-classes.json`.
These include input lists, checkpoint/production-source-file hashes, per-sample confusion matrices, filtering counts, per-class residual statistics, and intervals.
The new tests and yesterday's diagnostic tests passed, 9 in total; compilation and git diff --check passed.

No backward pass, optimizer step, training, full nuScenes evaluation, V1 causal control, cache regeneration,
checkpoint rewriting, commit, or push was performed. The diagnostic process exited normally, and the GPU was released.