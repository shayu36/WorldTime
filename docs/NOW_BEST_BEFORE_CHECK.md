# Code Snapshot of the Best Result Before Repairs

The user-specified branch name `now best before check` contains spaces and is not a valid Git branch name;
`now-best-before-check` is used instead. This branch preserves the original code and run entry points
for T1 epoch 3, without incorporating subsequent causal-ego, Memory-provenance, or base-age safeguards.

## Existing official results (original input protocol)

Model: Future Memory Adapter V2, A+B gating + SoftVoxel-center + T1.
Native checkpoint inference on 4219 frames; no semantic/occupancy/position branch was disabled.

| Horizon | IoU | mIoU |
| --- | ---: | ---: |
| 0s | 25.68 | 18.20 |
| 1s | 23.88 | 15.39 |
| 2s | 22.86 | 13.40 |
| 3s | 21.78 | 11.87 |
| Future Mean | 22.8400 | 13.5533 |

Means are calculated from the saved metrics rounded to two decimal places;
they do not imply four-decimal-place precision in the underlying confusion matrices.

Configuration:
`configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py`

Checkpoint (retained locally only, not uploaded to Git):
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth`

SHA256: `bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`.

The training entry point `tools/run_train_future_memory_occ_t1.sh` resumes from center epoch 2;
it is not a command for a new experiment that continues from this snapshot's epoch 3.
The evaluation entry point is `tools/run_eval_future_memory_occ_t0_t1.sh t1`,
which refuses to overwrite an existing evaluation directory.
The effective run configuration and complete evaluation artifacts remain in the original local work-dir.

## Source identity and scope

`NOW_BEST_BEFORE_CHECK.json` records the original training/evaluation provenance, checkpoint hash, and metrics.
The SHA256 values for `query_memory.py`, `sparseworld_4d_traj.py`, the T1 configuration,
and the training script in this snapshot exactly match the original training provenance.
The dataset, loader, and cache-generation entry points retain their pre-repair versions.
V1 configurations and old checkpoints are unchanged.
Checkpoints, datasets, caches, and logs are not committed.

This is a historical result archive, not a certification of input causality.
Subsequent audits found that legacy external ego fields and the historical Memory generated from them
may contain future trajectory information. The metrics above are retained, but they cannot establish
that future information has been excluded from the added Memory channel.
Repairs and revalidation should use a separate branch and separate cache/work-dir.
