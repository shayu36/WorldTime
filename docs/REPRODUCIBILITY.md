# Input Preparation and Exact Reproduction Protocol

Run all commands from the repository root; see the root README for environment installation.

## External file layout and sources

```text
data/nuscenes/{samples,sweeps,maps,v1.0-trainval,lidarseg}/
data/nuscenes/gts/<scene>/<token>/labels.npz
data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
data/nuscenes/bevdetv2-nuscenes_infos_val.pkl
admlp/fengze_nuscenes_infos_train.pkl
admlp/fengze_nuscenes_infos_val.pkl
admlp/stp3_val/data_nuscene.pkl
occworld/nuscenes_infos_train_temporal_v3_scene.pkl
occworld/nuscenes_infos_val_temporal_v3_scene.pkl
stp3_val/{filter_token.pkl,stp3_traj_gt.pkl,stp3_occupancy.pkl}
data/query_memory/sparseworld_epoch56_schema2_train/<scene>/<token>.pt
data/query_memory/sparseworld_epoch56_schema2_val/<scene>/<token>.pt
```

1. Raw nuScenes and lidarseg: [nuScenes downloads](https://www.nuscenes.org/download).
2. Occupancy GT: [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D), using the nuScenes `gts`.
3. SparseWorld preprocessed infos, auxiliary ADMLP/OccWorld PKLs, and epoch_56:
   [official SparseWorld data and weights](https://huggingface.co/MSunDYY2001/SparseWorld/tree/main).
   See also the existing [prepare_datasets.md](prepare_datasets.md).
   Public availability of these downloads does not automatically establish byte-for-byte identity
   with this experiment's inputs; compare them against `artifacts/input_manifest.json`.
4. `stp3_val` may link to the downloaded `admlp/stp3_val`, provided its three GT/filter files match the manifest.

For example, after placing the files in the specified directories:

```bash
ln -s admlp/stp3_val stp3_val
python tools/reproduce/verify.py --model --data both
```

Reproducing only the final evaluation does not require training-split files; use `--data val`.
PKLs may embed file paths, so retain the official relative directory layout or create symlinks.
Do not change sample filtering to obtain matching metrics.
The source run used train=19730, validation=4219 unique tokens, and a planning candidate filter=4819 with matches=4219.
Checks fail explicitly if external files are missing or hashes differ;
exact reproduction cannot be claimed in that case.

## Memory cache

Cache schema, selection, and alignment implementations are all included in this branch:
`tools/query_memory/precompute_query_memory.py`, `loading_query_memory.py`, and `query_memory.py`.
Historical Query features/points/semantic distributions come from the fixed Baseline epoch56;
each frame has at most 256 Queries.
The model reads at most 3 frames per sample, with target base ages of 2.5/3.5/4.5s;
training and validation caches are separate.

If no existing cache is available, generate it locally with the official epoch56 weights.
The following are executable schema-v2 generation entry points.
This release did not regenerate the existing server cache or verify byte-for-byte identity between a regenerated cache
and the old cache. Differences in regeneration configuration, weights, or preprocessing may change results;
retain generation records and check source paths first.

```bash
python tools/query_memory/precompute_query_memory.py --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py --checkpoint ckpts/epoch_56.pth --split train --output-dir data/query_memory/sparseworld_epoch56_schema2_train --max-queries-per-frame 256 --write-threshold 0.35
python tools/query_memory/precompute_query_memory.py --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py --checkpoint ckpts/epoch_56.pth --split val --output-dir data/query_memory/sparseworld_epoch56_schema2_val --max-queries-per-frame 256 --write-threshold 0.35
```

GT files are used only for loss/evaluation.
For the actual legacy ego-input files, the loader sorts and concatenates all non-`gt` fields.
This release preserves the experimental inputs; it does not silently replace their fields or claim that
an audit for future information has been completed. Field provenance must be established from the data-generation process.

## Planning evaluation consistent with the Baseline

`tools/eval_memory_planning_joint.py predict` uses the native model's `pred_traj [1,6,2]` single-step displacements
and applies `np.cumsum` over time to obtain cumulative XY positions in the current frame.
`audit_planning_pair.metrics` adds z=0 and aligns them with `[:,1:7]` in `stp3_traj_gt.pkl`
and `[:,:6]` in occupancy GT.
Evaluator source and GT SHA256 values must match `artifacts/baseline/metrics.json`.

The retained original evaluator, `AD-MLP/deps/stp3/stp3/planning_metrics.py`, defines:

- A 0.5m grid over XY [-50,50], using GT occupancy as the collision map.
- Obj checks only the reference point; Box checks a 1.85×4.084m ego rectangle
  (with the longitudinal reference offset from the original code).
- Time steps at which the GT rectangle already collides are excluded from the numerator,
  but the denominator remains all matched samples N; out-of-bounds reference-point events are excluded from the numerator.
- Per-step collision rate is the masked event count/N; prefix@h averages the first 2h steps.
  The denominator is not the number of unmasked samples.
- Raw JSON collision rates are fractions; CSV and README display percentages, multiplying by 100 exactly once.
- The meaning of 0% is limited by the GT mask and geometric definitions above.

## Environment and build details

The official MMCV cu118/torch2.0 wheel index does not provide 1.7.0.
The installation script uses the bundled wheel, whose binary hash matches the source run.
It targets Linux x86_64/CPython3.9, and its CUDA operators include sm_86 cubin (RTX 3090);
other architectures require rebuilding MMCV.
On other architectures, build from source with
`MMCV_WITH_OPS=1 pip install --no-binary=mmcv-full mmcv-full==1.7.0`.
Record the compiler/CUDA/torch versions and validation results;
do not directly treat this as bitwise reproduction of the original environment.
The repository's two extensions are built by the root setup.py;
set `TORCH_CUDA_ARCH_LIST` to the actual device architecture.
MMCV uses Apache-2.0 and the wheel includes LICENSE/LICENSES;
see THIRD_PARTY_NOTICES for copyright notices for other upstream source.

## Reproduction scope

This release provides joint-adaptation code from a fixed T1 starting point, the two required checkpoints,
and seed17 results. It does not include the original Baseline training pipeline.
Evaluation with the released weights is the most direct way to check the reported results.
Retraining for one epoch reproduces the training protocol, but nondeterministic CUDA operations
are not guaranteed to produce an identical checkpoint or identical last-digit scores across environments.
Users must prepare all data and caches under the applicable authorizations;
downloading the code alone cannot produce official metrics without these inputs.
