# SparseWorld Memory Experiment and Debug Summary

Updated: 2026-09-06 UTC
Project: `/data/jxy/projects`
Reference checkpoint: `ckpts/epoch_56.pth`
Cache: fixed epoch-56 schema-v2 query memory cache
Scope: summarize only completed or running Memory-related experiments; do not rewrite historical values
or fill in results for incomplete evaluations.

## Current conclusions

As of this compilation, none of the evaluated Memory-only, joint, Phase2, Phase3, Phase4,
or M0R1/M1R0/M1R1 variants has demonstrated 1s/2s/3s IoU or mIoU above the epoch-56 baseline.

Future Memory Adapter has completed code-level acceptance checks and a 200-iteration smoke run.
Its 12-epoch formal training is running in `work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/`,
with no formal validation results yet. Therefore, improvements in IoU/mIoU cannot currently be claimed for Future Adapter.

## Baseline

The epoch-56 baseline is the current comparison reference.
Do not retrain or reevaluate the baseline for new Memory experiments.

| Horizon | 0s | 1s | 2s | 3s |
| --- | ---: | ---: | ---: | ---: |
| IoU | `25.68` | `23.15` | `22.28` | `21.21` |
| mIoU | `18.20` | `14.96` | `13.18` | `11.53` |

Baseline future mIoU mean: `13.2233`.

## Completed formal results

All results below come from 4,219 nuScenes val samples.
`Future mean` averages 1s/2s/3s mIoU; `Delta` is relative to the baseline future mean of `13.2233`.

| Experiment | Status | mIoU `[0s,1s,2s,3s]` | Mean | Future mean | Delta | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| epoch-56 baseline | Complete | `[18.20, 14.96, 13.18, 11.53]` | `14.4675` | `13.2233` | `0.0000` | Current reference |
| Memory-only epoch 12 | Complete | `[18.20, 14.96, 13.18, 11.53]` | `14.4675` | `13.2233` | `0.0000` | Memory changes continuous tensors, but discrete occupancy metrics remain unchanged |
| Joint finetune epoch 4 | Complete | Full table not retained separately | — | `13.2233` | `0.0000` | Matches the baseline at printed precision; does not exceed it |
| Joint finetune epoch 12 | Complete | Full table not retained separately | — | `13.2033` | `-0.0200` | Does not exceed the baseline |
| Phase2 epoch 4 | Complete | Full table not retained separately | — | `13.2233` | `0.0000` | Matches the baseline at printed precision; does not exceed it |
| Phase2 epoch 12 | Complete | `[18.20, 14.92, 13.14, 11.49]` | `14.4375` | `13.1833` | `-0.0400` | Unfreezing `pts_bbox_head` provides no gain |
| Phase3 epoch 12 | Complete | `[15.48, 14.22, 12.52, 10.95]` | `13.2925` | `12.5633` | `-0.6600` | The old refiner topology degrades 0s and future results |
| Phase4 future-only epoch 12 | Complete | `[18.20, 14.38, 12.75, 11.23]` | `14.1400` | `12.7867` | `-0.4366` | 0s recovers, but future results still decline |
| M1R0 formal epoch 12 | Complete | `[18.20, 14.96, 13.18, 11.53]` | `14.4675` | `13.2233` | `0.0000` | Metrics match the baseline |

Corresponding retained IoU summaries: baseline `[25.68,23.15,22.28,21.21]`;
Phase2 `[25.67,23.15,22.27,21.20]`; Phase3 `[25.18,23.01,22.03,20.86]`;
Phase4 `[25.68,22.74,21.62,20.47]`; M1R0 `[25.68,23.15,22.28,21.21]`.

## 2×2 ablation and smoke records

| Variant | Parameter count | Memory reads / fused queries | Key signals | Current conclusion |
| --- | ---: | ---: | --- | --- |
| M0R1: refiner-only | `801,568` | `0 / 0`, empty forward `1.0` | The refiner has gradients | Trainable, but does not validate Memory |
| M1R0: Memory-only | `924,188` | `6 / 320`, empty forward `0.0` | Extremely weak q/k/v gradients; direct residual approximately `1e-5` | Memory reads work, but their effect on outputs is too weak |
| M1R1: Memory + refiner | `1,725,756` | `6 / 320`, empty forward `0.0` | Both refiner and Memory have gradients; motion residual approximately `1.23 -> 5.51` | Connected, but motion-residual stability is a concern |

M1R1 only establishes that the training graph is connected; it is not evidence of a performance gain.

## Intermediate differences: Baseline vs M1R0

Comparison: epoch-56 baseline vs M1R0 formal epoch-12, covering 4,219 nuScenes val samples.

Raw logits, raw refine points, and 0s outputs are exactly identical:

| Item | mean abs diff | max abs diff |
| --- | ---: | ---: |
| raw logits | `0` | `0` |
| raw refine points | `0` | `0` |

Future refined continuous tensors change:

| Output | 1s | 2s | 3s | 4s | 5s | 6s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logits mean abs | `0.006176` | `0.011827` | `0.016842` | `0.020320` | `0.022441` | `0.023618` |
| points mean abs | `0.0000637` | `0.0001279` | `0.0001760` | `0.0002313` | `0.0002737` | `0.0003166` |

The fraction of discrete occupancy labels that change is very low:

| horizon | changed voxels | changed fraction | baseline non-empty/sample | M1R0 non-empty/sample |
| --- | ---: | ---: | ---: | ---: |
| 0s | `0` | `0` | `52,623.63` | `52,623.63` |
| 1s | `526,540` | `0.0195%` | `54,665.71` | `54,654.54` |
| 2s | `1,148,110` | `0.0425%` | `54,783.32` | `54,770.09` |
| 3s | `1,882,880` | `0.0697%` | `54,054.36` | `54,042.93` |
| 4s | `3,221,811` | `0.1193%` | `54,024.48` | `54,003.26` |
| 5s | `4,523,406` | `0.1675%` | `53,595.03` | `53,574.93` |
| 6s | `6,036,067` | `0.2235%` | `53,119.84` | `53,094.71` |

Direct conclusion: M1R0 was trained and did read Memory,
but most continuous changes caused by Memory did not cross the discrete occupancy decision boundaries.

## Phase2/3/4 summary

Phase2 aimed to expand the trainable scope by unfreezing future/head parameters associated with `pts_bbox_head`.
Its epoch 12 future mean was below the baseline, providing insufficient evidence to further expand this direction.

Phase3 connected the Memory-conditioned refiner to both 0s and future outputs.
It established training-graph connectivity, but 0s declined sharply in formal evaluation,
indicating that the old refiner should not perturb the current-frame output path.

Phase4 changed the refiner to future-only, restoring 0s to the baseline while leaving 1s/2s/3s below it.
This result locates the problem in the old future Memory/refiner topology itself rather than the 0s routing.

## Future Memory Adapter status

Future Memory Adapter is the current new modeling direction.
Its key difference is the separation of the Baseline state stream from the Memory output correction stream:

```text
S_base(t) = ClsBranch(Q_base(t))
S_final(t) = S_base(t) + G(t) * DeltaS_memory(t) + G(t) * DeltaO_memory(t)
P_final(t) = SafeRefine(P_base(t), G(t) * DeltaP_memory(t))
```

Memory corrections are not written back into the next time step's Baseline future recurrence.
Only the evaluation targets 1s, 2s, and 3s are corrected.

Internal-step mapping:

| Target horizon | internal future step | simple_test key | active queries |
| ---: | ---: | --- | ---: |
| 1s | 2 | `semantic_occ_2s` | `840` |
| 2s | 4 | `semantic_occ_4s` | `960` |
| 3s | 6 | `semantic_occ_6s` | `1040` |

The 200-iteration smoke run is complete. Logs contain `mem_1s.*`, `mem_2s.*`, and `mem_3s.*` losses,
and `grad_norm` is nonzero. The smoke run verifies training connectivity only, not performance.

The 12-epoch formal run is currently in progress:

```text
work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/
```

Epoch 1 has been observed progressing normally, with continuing `mem_1s/mem_2s/mem_3s` loss and `grad_norm` output.
Training is not yet complete, and formal validation has not yet been performed.

## Outstanding questions

Resolved engineering errors are not discussed further in this report.
Only issues affecting interpretation of subsequent experiments are retained:

1. Memory perturbations of continuous logits/points may still be too small to change discrete occupancy labels.
   Further diagnostics should examine margins, voxel decision boundaries, and corrected-logit distributions.
2. Whether cache/query semantic matching, spatial matching, and age/reliability weights provide useful predictive
   information still requires Future Adapter formal results plus shuffled/cache-off controls.
3. The old Phase3/Phase4 refiner topology should no longer serve as the main approach;
   the growth of M1R1 motion residuals indicates a continuing stability risk in this path.
4. After Future Adapter formal training completes, compare Memory ON, Memory OFF, and shuffled Memory
   using the same checkpoint to determine whether Memory provides a useful gain.
5. Standard formal training logs primarily record losses and cannot adequately explain flat metrics.
   If formal evaluation shows no improvement, add per-horizon gate, candidate-score,
   semantic-compatibility, and voxel-label-change diagnostics.

## Retention and deletion policy

This file is the sole entry point for experimental data summaries.
Old handoffs, interim handovers, and symlink/summary files under `logs/` will be removed
to avoid maintaining the same results in multiple places.

Raw artifacts are not deleted:

- Checkpoints, full training logs, and TensorBoard files remain in `work_dirs/`.
- The fixed memory cache remains in `data/query_memory/`.
- The running Future Adapter formal directory must not be cleaned, moved, or renamed.
