# MemAnchorWorld: Reproducible Release with Planning Supervision

This reproducibility branch releases the complete **T1 + SoftVoxel-center + Memory planning-fusion supervision**
code, seed 17 weights, and reproduction entry points. The Baseline is frozen. The V2 occupancy Adapter
and planning-fusion module are trained with the occupancy loss and original six-step trajectory loss,
respectively. Planning reads the actual Memory Context from the same model forward pass. Planning tokens
retain stop-gradient, so the planning loss updates the fusion module; this must not be described as
end-to-end updating of all of V2.

Each seed independently adapts the same best T1 epoch 3 checkpoint for one complete epoch;
this is not training the entire model from scratch. Each of `17/23/41/73/101` uses a fresh planning
initialization and a new optimizer, with 19730 training samples, two GPUs at batch 1 per GPU, and 9865 iterations.

## Official seed 17 results

This table comes from the public `artifacts/seed17/epoch_1.pth`, using native model inference on 4219 frames
and the original evaluator. No semantic, occupancy, or position residual is disabled.

| Horizon | IoU (%) | mIoU (%) | L2 (m) | Obj Collision (%) | Box Collision (%) |
|---|---:|---:|---:|---:|---:|
| 0s | 25.6800 | 18.2000 | — | — | — |
| 1s | 23.8700 | 15.3800 | 0.1597 | 0.0000 | 0.0948 |
| 2s | 22.8600 | 13.4200 | 0.1947 | 0.0000 | 0.1007 |
| 3s | 21.8000 | 11.9200 | 0.2397 | 0.0000 | 0.1106 |
| Future/Avg. | 22.8433 | 13.5733 | 0.1980 | 0.0000 | 0.1021 |

IoU/mIoU averages include only 1/2/3s. Planning uses the same **prefix averages** as the local Baseline:
1/2/3s average the first 2/4/6 predicted points, respectively; these are not endpoint errors.
The evaluation files also retain six-step and endpoint metrics.
Obj measures collisions at the trajectory reference point; Box measures collisions of the ego rectangle.
Both use the original evaluator's GT box collision mask.
0% means this metric is zero under that mask; it does not establish zero collisions for unmasked events
or under every geometric definition.

Raw results: [all metrics](artifacts/seed17/all_metrics.json),
[planning metrics](artifacts/seed17/planning_metrics.json),
[checkpoint audit](artifacts/seed17/checkpoint_audit.json).

## Download code and weights

```bash
sudo apt-get install -y git-lfs build-essential libgl1 libglib2.0-0
git lfs install
git clone --branch "$(printf '\u5f00\u6e90\u53ef\u590d\u73b0')" https://github.com/shayu36/WorldTime.git
cd WorldTime
git lfs pull
```

The GitHub ZIP download may contain only LFS pointers; use clone + lfs pull above.
Both checkpoints and the MMCV wheel used in the actual run are in Git LFS, totaling approximately 660 MB.
The [checksum manifest](artifacts/weights_manifest.json) includes:

- `artifacts/seed17/epoch_1.pth`: the complete joint model, optimizer, and training meta, 328667873 bytes.
  SHA256 `a09d1d20f2b6049b2d396f4342e416b9c810541745d891e380926866d2404760`.
- `artifacts/initial_t1/epoch_3.pth`: the shared initialization for all seeds, 325954537 bytes.
  SHA256 `bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`.

## Environment installation

Reference platform: Ubuntu 22.04, Python 3.9, CUDA toolkit 11.8, GCC 11, PyTorch 2.0.1+cu118,
and two RTX 3090 24 GB GPUs. The CUDA toolkit must include nvcc; installing only the PyTorch CUDA runtime
is insufficient to compile the extensions. At least 32 GB of host RAM is recommended; data storage is additional.

```bash
conda create -n memanchorworld python=3.9 -y
conda activate memanchorworld
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
bash tools/reproduce/install.sh
```

The installation script installs torch, MMCV, torch-scatter, and [requirements.txt](requirements.txt) in order,
then compiles the BEVPool v2 and MSMV sampling extensions through [setup.py](setup.py) and installs this repository.
The bundled `mmcv-full==1.7.0` wheel is identical to the binary used in the actual run, for CPython 3.9/Linux x86_64/
CUDA 11.8/PyTorch 2.0.1/sm_86. The official cu118/torch2.0 wheel index provides only 1.7.2,
which cannot directly replace the version used here.
For source builds on other platforms, see the [reproduction guide](docs/REPRODUCIBILITY.md).
The historical machine's mixed-environment freeze is retained in `requirements/historical-freeze.txt`;
do not use it in place of the installation script.

## Prepare external inputs

**This repository does not upload data, GT, images, trajectory labels, or Memory caches.**
Required inputs are nuScenes trainval + lidarseg, Occ3D gts, SparseWorld preprocessed infos/ego data,
ST-P3 planning GT, and the fixed schema-v2 Memory cache. Full directory layouts, public sources,
checksum manifests, and cache-generation commands are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
Raw nuScenes images alone cannot replace these inputs.

```bash
python tools/reproduce/verify.py --model --data val
```

This command verifies both checkpoints, strict model loading, parameter scope, external-input SHA256 values,
and cache paths. Before training, check all inputs with `--data both`.
Use `--environment-only` to check the environment first without data.

## Reproduce the complete seed 17 evaluation

```bash
python tools/reproduce/run.py evaluate --gpus 0,1 --output reproduce_outputs/published_seed17
```

The script sequentially runs two-GPU inference on 4219 frames, CPU occupancy evaluation, and planning evaluation
under the Baseline protocol. It writes the complete results to
`reproduce_outputs/published_seed17/eval_epoch_1/all_metrics.json`.
It also retains lossless occupancy predictions, trajectories, per-class confusion matrices,
and checkpoint/configuration/source SHA256 values.
Existing outputs are not overwritten if a process fails; inspect the logs and specify a new output directory.

For initial troubleshooting, run the following 4-frame inference; it does not produce official evaluation scores:

```bash
python tools/reproduce/run.py predict --gpus 0,1 --limit 4 --output reproduce_outputs/probe4
```

## Train independently for one epoch, then evaluate

```bash
python tools/reproduce/run.py train --seed 17 --gpus 0,1 --output reproduce_outputs/retrained_seed17
python tools/reproduce/run.py evaluate --gpus 0,1 --checkpoint reproduce_outputs/retrained_seed17/train/epoch_1.pth --output reproduce_outputs/retrained_seed17_eval
```

Each run starts from the shared T1 initialization, initializes a new planning module according to the seed,
and uses a new optimizer; it does not resume a previous experiment's state.
V2 LR=2e-5, planning LR=1e-4, fixed LR, 100-step linear warmup, and AdamW weight decay=.01.
The Baseline stays in eval mode and frozen. Only 77 parameter tensors (1722147 scalars) are trainable.
AMP dynamic loss scaling, q40, activation checkpointing, and all T1 losses are retained.

Complete five-seed queue:

```bash
python tools/reproduce/run.py five-seeds --gpus 0,1 --output reproduce_outputs/joint_campaign
```

Each seed undergoes complete evaluation after training before the next seed starts.
All horizon metrics and mean/sample SD(ddof=1) are retained after each run;
incomplete seeds are not counted as zero. Use tmux for long-running jobs.
On the machine used here, each epoch took approximately 6.4 hours and a complete evaluation
approximately 40 minutes; timings depend on hardware and I/O.

## Checks and limitations

```bash
python -m pytest -q tests/test_planning_joint_checkpoint.py tests/test_trainable_memory_planning.py tests/test_joint_five_seed_protocol.py deployment/test_planning_details.py
```

The original run included real-sample gradient checks, freeze/update checks, a 40-step two-GPU check,
and the 16 tests above; records are in [artifacts/verification](artifacts/verification).
The release package also underwent independent environment and inference checks;
see [release validation](docs/RELEASE_VALIDATION.md).
Check evaluation of a fixed checkpoint and retraining separately.
Different GPUs, CUDA/library versions, and nondeterministic kernels may affect the last digits;
bitwise consistency across platforms is not guaranteed.

## Source and licensing

Based on SparseWorld / MMDetection3D / BEVDet / OPUS and their upstream dependencies.
Original copyright notices are retained.
ST-P3/UniAD planning-evaluation source is included separately as an exact evaluation dependency;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
This branch does not claim licenses that upstream projects have not granted.
External nuScenes/Occ3D data must be obtained under their respective terms and are not included in this release.
