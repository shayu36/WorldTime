# Release Validation (2026-11-25)

The release directory is separate from the running experiment directory.
The running code, caches, data, and checkpoints were not modified.

## Correspondence between the model and measured results

- Complete seed17 checkpoint SHA256:
  `a09d1d20f2b6049b2d396f4342e416b9c810541745d891e380926866d2404760`.
- Initial T1 SHA256:
  `bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`.
- `artifacts/source_identity.json` verifies, file by file, that the model, data loading,
  and planning evaluator match the actual training/evaluation directories.
- seed17 completed 9865 iterations. The checkpoint audit confirms an unchanged Baseline,
  updated V2, and a nonzero final residual head in the fusion module.
- `artifacts/seed17/all_metrics.json` comes from the existing complete inference and evaluation
  on 4219 frames; its values were not changed.

## Independent environment validation

A new Python3.9 venv (`include-system-site-packages=false`) was created.
The installation script installed torch2.0.1+cu118, the bundled MMCV1.7.0 wheel, torch-scatter,
and requirements, then built the repository's two CUDA extensions from source.
The same server's CUDA toolkit/driver and read-only data were reused;
site-packages from the old environment were not copied.
Fully resolved dependency versions are recorded in `requirements/constraints-py39-cu118.txt`.

Verified:

- `pip check` reports no dependency conflicts.
- Both CUDA extensions compile and import successfully.
- SHA256 checks pass for the release checkpoints and wheel.
- Full-model `load_state_dict(strict=True)` passes; 1722147 trainable parameter scalars in 77 parameter tensors.
- The Baseline is frozen. Only the V2 Adapter and planning fusion are trainable,
  with respective LRs of 2e-5 and 1e-4.
- External validation-input file hashes match the source run.
- Four dedicated test files: 16 passed, in both the original and new environments.
- In the new environment, the release entry point and checkpoint were used for native 4-frame, two-GPU inference,
  paired with the original complete seed17 inference artifacts:
  every occupancy voxel is exactly equal, all four trajectories are exactly equal, and the maximum difference is 0.
  See `artifacts/verification/release_forward_comparison.json`.
- Installation-script shell checks, Python compilation, `--help` for all three public entry points,
  and `git diff --check` pass.

This 4-frame check does not replace complete validation on 4219 frames.
The release preparation did not retrain the entire model or regenerate the cache.
The actual complete evaluation results come from the release checkpoint's existing run;
the small-sample check is not presented as new full-dataset metrics.
Numerical consistency across GPU architectures or build environments has not been verified;
the bundled MMCV wheel contains sm_86 CUDA operators.

## Reproduction boundaries

The repository provides the model, dependencies, build setup, portable entry points, training configurations,
weights, results, and input checksums.
Prepare raw data/preprocessed PKLs/GT/caches as described in the README and reproduction guide;
no data files were pushed to Git.
Public data sources and cache-generation entry points are provided, but this round did not regenerate
the cache and establish byte-for-byte identity.
If external-input hashes differ, check preprocessing and file provenance first;
the result cannot directly be called an exact reproduction of this run.
