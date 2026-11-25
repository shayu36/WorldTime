# SparseWorld Code Architecture: A Complete Guide

> Based on the AAAI 2026 paper "SparseWorld: 4D Occupancy World Model with Sparse Dynamic Queries"
> This document describes the role of each project file as a reference for future development.

---

## I. Project Overview

```
SparseWorld/
├── mmdet3d/                    # Core codebase
│   ├── models/                 # Model definitions
│   │   ├── sparsedetectors/    # ★ SparseWorld/OPUS core (new)
│   │   ├── detectors/          # PreWorld detector family (baseline)
│   │   ├── heads/              # Occupancy prediction heads
│   │   ├── necks/              # Feature fusion necks (including view transforms)
│   │   ├── backbones/          # Image backbones
│   │   ├── nerf/               # NeRF rendering head
│   │   ├── modules/            # Deformable attention modules
│   │   └── loss_utils/         # Loss functions
│   ├── datasets/               # Datasets and data pipelines
│   ├── core/                   # Basic components (bbox, eval, hooks, etc.)
│   ├── ops/                    # Operators (BEV pooling, PointNet, etc.)
│   ├── apis/                   # Training/testing APIs
│   └── utils/                  # Utility functions
├── configs/                    # Training/testing configuration files
│   ├── sparseworld/nuscenes-temporal/  # ★ SparseWorld temporal configurations
│   ├── sparseworld/nuscenes/           # PreWorld non-temporal configurations
│   └── _base_/                         # Base configuration templates
├── tools/                      # Training, testing, and data processing scripts
├── AD-MLP/                     # Trajectory planning evaluation module
├── data/nuscenes/              # nuScenes dataset (symbolic link)
├── ckpts/                      # Model weights
├── admlp/                      # pkl files required for AD-MLP evaluation
├── occworld/                   # pkl files required for OccWorld evaluation
└── docs/                       # Documentation
```

---

## II. Model Inheritance Hierarchy

SparseWorld builds on the BEVDet detector family, with the following inheritance chain:

```
BaseDetector (mmdet)
  └── MVXTwoStageDetector        # Multimodal two-stage detector base class
       └── CenterPoint            # Point cloud detector
            └── BEVDet             # BEV detector (single frame)
                 └── BEVDet4D      # Temporal BEV (multi-frame alignment)
                      └── BEVDepth4D     # Depth-supervised BEV
                           └── BEVStereo4D    # Stereo-matching BEV
                                └── BEVStereo4DOCC  # + 3D occupancy prediction
                                     └── PreWorld        # + NeRF pretraining [baseline method]
                                          └── PreWorld4DTraj  # + Temporal trajectory planning [baseline method]

MVXTwoStageDetector (independent branch)
  └── OPUS                       # OPUS sparse query detector
       └── SparseWorld4DTraj      # ★ Complete SparseWorld model
```

**Comparison of the two approaches:**
| Property | PreWorld approach | SparseWorld approach |
|------|-------------|-----------------|
| Feature representation | Dense 3D voxels | Sparse query points |
| Occupancy prediction head | OccHead (3D convolution) | OPUSHead (Transformer) |
| Temporal modeling | Recurrent voxel feature updates | Autoregressive query expansion |
| Image to 3D | LSS view transform | 3D→2D sampling |
| Computational efficiency | Lower (dense computation) | Higher (sparse computation) |

---

## III. Core Model Files in Detail

### 3.1 SparseWorld/OPUS Core (`mmdet3d/models/sparsedetectors/`)

This directory contains the code for the core contributions of the SparseWorld paper.

#### `sparseworld_4d_traj.py` — Top-Level Model
- **Class**: `SparseWorld4DTraj` (inherits from `OPUS`)
- **Purpose**: The complete SparseWorld model, adding future spatiotemporal prediction and ego trajectory planning to OPUS occupancy prediction
- **Core workflow** (`forward_backbone`):
  1. `plan_head` encodes the ego motion state into ego features
  2. `points_scale_branch` dynamically adjusts the initialization scale of query points
  3. Calls the OPUS backbone to extract image features, then obtains current-frame query features through `pts_bbox_head`
  4. Separates queries for the current frame and each future frame using `ind_stamps_all`
  5. **Autoregressive future prediction loop** (runs `num_fu_frames` times):
     - Cross-attention between ego features and scene queries → trajectory prediction
     - Accumulates future queries frame by frame, predicting displacement and semantics through the reg/vel/cls branches
     - Adds velocity offsets for moving objects (classes 2-10)
- **Key submodules**: `plan_head`, `ego_cross_attn`, `traj_head`, `reg_branch`, `vel_branch`, `cls_branch`
- **Development options**: Modify the future prediction strategy, trajectory planning head, or loss weights

#### `opus.py` — OPUS Base Detector
- **Class**: `OPUS` (inherits from `MVXTwoStageDetector`)
- **Purpose**: Image feature extraction, data augmentation, and online/offline inference logic
- **Core methods**:
  - `extract_img_feat`: Extracts features through backbone + neck, with GridMask support
  - `extract_feat`: Complete feature extraction, including color augmentation, normalization, and padding
  - `simple_test_online`: Online inference (caches features frame by frame to save GPU memory)
  - `simple_test_offline`: Offline inference (processes all frames at once)
- **Development options**: Modify the data augmentation strategy or feature extraction pipeline

#### `opus_head.py` — OPUS Detection Head
- **Class**: `OPUSHead` (`pts_bbox_head`)
- **Purpose**: Manages query point initialization, calls the Transformer, computes losses, and generates occupancy predictions
- **Core methods**:
  - `forward`: Initializes queries → Transformer refinement → timestamp assignment
  - `_get_target_single`: KNN bidirectional nearest-neighbor matching (based on the Chamfer Distance idea)
  - `get_occ`: **Inference core** — sparse queries → dense voxels, aggregated with `scatter_max`, with dilation + erosion to fill holes
  - `loss_stack` / `loss_future`: Current-frame/future-frame losses
  - `get_sparse_voxels`: Dense GT → sparse point cloud representation
  - `reset_mask`: Causal attention mask that prevents future queries from seeing the past
- **Key parameters**: `num_query`=720, `num_fu_query`=[720]*N, `empty_dist_thr`, `empty_weights`
- **Development options**: Modify query counts, the matching strategy, or sparse-to-dense aggregation

#### `opus_transformer.py` — Transformer Architecture
- **Classes**:
  - `OPUSTransformer`: Top-level wrapper
  - `OPUSTransformerDecoder`: Manages multiple DecoderLayer instances
  - `OPUSTransformerDecoderLayer`: Single decoder-layer workflow
  - `OPUSSelfAttention`: **Scale-Adaptive Self Attention (SASA)** — a core innovation of the paper
  - `OPUSCrossAttention`: Distance-adaptive cross-attention
  - `OPUSSampling`: Adaptive spatiotemporal sampling
  - `AdaptiveMixing`: Adaptive channel/point mixing (inspired by AdaMixer)
- **Single-layer decoding workflow**: `Positional encoding → sampling → mixing → self-attention → FFN → classification + regression → point-coordinate refinement`
- **SASA core**: Computes Euclidean distances between queries, modulated by a learned τ parameter to form attention biases
- **Development options**: Modify the attention mechanism, sampling strategy, or number of decoder layers

#### `opus_sampling.py` — 4D Spatiotemporal Sampling
- **Function**: `sampling_4d`
- **Purpose**: 3D sampling points → projection into each camera view → bilinear interpolation from multi-scale FPN feature maps
- **Key optimization**: Keeps only one best view per sampling point (`argmax(valid_mask)`)
- **Development options**: Modify the number of sampling points or the multi-view fusion strategy

#### `utils.py` — Utility Functions
- `sparse2dense`: Sparse voxels → dense tensor
- `GridMask`: Grid-mask data augmentation
- `GpuPhotoMetricDistortion`: GPU photometric distortion augmentation
- `pad_multiple`: Pads image dimensions to integer multiples
- `rotation_3d_in_axis`: Rotation about the z-axis
- `inverse_sigmoid`: Inverse sigmoid function

#### `checkpoint.py` — Gradient Checkpointing
- A customized copy of PyTorch `torch.utils.checkpoint`
- Used extensively by SelfAttention/CrossAttention/Sampling/Mixing to trade computation for GPU memory

#### `bbox/` Subdirectory
| File | Purpose |
|------|------|
| `bbox/utils.py` | `encode/decode_points` (normalized coordinate conversion), `get_matched_inds` (query→time-step assignment), `trans_coords` (inter-frame transformation), `dist_loss_weight` (long-distance attenuation), `MultiheadAttention` |
| `bbox/sdnetutils.py` | A simplified version of utils.py |
| `bbox/assigners/hungarian_assigner_3d.py` | Hungarian assigner (for 3D object detection; rarely used for OPUS occupancy prediction) |
| `bbox/coders/nms_free_coder.py` | NMS-free BBox decoder |
| `bbox/match_costs/match_cost.py` | 3D BBox L1/BEV L1/IoU3D matching costs |

#### `csrc/` Subdirectory
- `wrapper.py`: CUDA acceleration for multi-scale multi-view sampling (`msmv_sampling`)
- `setup.py`: CUDA extension build script
- Falls back to PyTorch `F.grid_sample` if CUDA compilation fails

---

### 3.2 PreWorld Detector Family (`mmdet3d/models/detectors/`)

PreWorld is the baseline method on which SparseWorld improves.

| File | Class | Purpose |
|------|-----|------|
| `base.py` | `Base3DDetector` | 3D detector base class |
| `mvx_two_stage.py` | `MVXTwoStageDetector` | Multimodal two-stage detector base class, managing backbone/neck/head |
| `centerpoint.py` | `CenterPoint` | Point cloud detector |
| `bevdet.py` | `BEVDet`, `BEVDet4D`, `BEVDepth4D`, `BEVStereo4D` | BEV detector family implementing 2D→3D view transforms and temporal alignment |
| `bevdet_occ.py` | `BEVStereo4DOCC` | Adds 3D occupancy prediction to BEVStereo4D |
| `preworld.py` | `PreWorld` | NeRF rendering pretraining + OccHead occupancy prediction (two-stage training) |
| `preworld_temporal_traj.py` | `PreWorld4DTraj` | PreWorld + recurrent temporal prediction + trajectory planning |
| `loss.py` | — | Occupancy prediction loss functions: `CE_ssc_loss`, `sem_scal_loss`, `geo_scal_loss`, `l1_loss`, `l2_loss` |
| `lovasz_softmax.py` | — | Lovász-Softmax loss (for occupancy semantic segmentation) |

#### `preworld.py` — PreWorld Detector
- **Two-stage training**:
  - Stage 1 (`is_pretrain=True`): NeRF volume-rendering supervision (2D depth/semantics/color)
  - Stage 2: OccHead 3D occupancy prediction (3D semantic segmentation)
- **Core methods**: `bev_encoder` encodes BEV voxel features, `nerf_head` performs NeRF rendering, and `occupancy_head` predicts occupancy

#### `preworld_temporal_traj.py` — PreWorld4DTraj
- **Temporal recurrence**: Loops over `num_fu_frames` frames, updating voxel features with `single_step_future_prediction` at each frame
- **Future prediction**: Voxel features → ego state fusion → next-frame features → occupancy prediction
- **Trajectory planning**: Ego features → `ego_query_attn` (cross-attention with BEV features) → `traj_branch` outputs a 2D trajectory

---

### 3.3 Feature Extraction (`mmdet3d/models/necks/`, `backbones/`)

| File | Class | Purpose |
|------|-----|------|
| `necks/view_transformer.py` | `LSSViewTransformer`, `LSSViewTransformerBEVDepth`, `LSSViewTransformerBEVStereo` | **LSS view transform**: 2D images → 3D voxels. Depth estimation + outer product + voxel pooling |
| `necks/fpn.py` | `CustomFPN` | Custom FPN with an ASPP module |
| `necks/lss_fpn.py` | `FPN_LSS` | FPN specialized for LSS |
| `necks/second_fpn.py` | `SECONDFPN` | FPN for the SECOND detector |
| `backbones/resnet.py` | `CustomResNet` | Custom ResNet (supports removal of stem stride) |
| `backbones/swin.py` | `SwinTransformer` | Swin Transformer backbone |

#### Core LSS View-Transform Workflow
```
Multi-view images [B,N,C,H,W]
  → 2D backbone feature extraction
  → Depth estimation network → Probability distribution over D depth bins [B,N,D,H,W]
  → Features × depth probabilities → Pseudo point cloud [B,N,D,H,W,C]
  → Projection into 3D voxel space using camera intrinsics and extrinsics
  → BEV pooling aggregation → 3D voxel features [B,C,X,Y,Z]
```

---

### 3.4 Occupancy Prediction Heads (`mmdet3d/models/heads/`)

| File | Class | Purpose |
|------|-----|------|
| `occupancy_head.py` | `OccHead` | Multi-scale 3D convolutional occupancy prediction head (used by PreWorld) |
| | `DownScaleModule3DCustom` | 3D voxel downsampling module |

- **OccHead**: 4 layers of 3D convolution, with layer-by-layer downsampling → upsampling → residual connections, finally outputting [B, num_classes, X, Y, Z]

---

### 3.5 NeRF Rendering (`mmdet3d/models/nerf/`)

| File | Class | Purpose |
|------|-----|------|
| `nerf_head.py` | `NerfHead` | NeRF volume-rendering head: generates 2D depth/semantics/color supervision signals from 3D voxel features |
| `utils.py` | — | Ray generation (`create_rays`), volume rendering (`volume_rendering`), and sampling strategies |

- **NeRF pretraining workflow**: 3D voxel features → sampling along rays → MLP prediction of (density σ, color c, semantics s) → volume rendering → comparison with GT 2D images/depth

---

### 3.6 Loss Functions and Modules (`mmdet3d/models/loss_utils/`, `modules/`)

| File | Purpose |
|------|------|
| `loss_utils/focal_loss.py` | `CustomFocalLoss` (used for the OPUS classification loss) |
| `modules/deformable_self_attn.py` | `DeformableSelfAttention` (deformable self-attention) |
| `modules/multi_scale_deformable_attn_function.py` | CUDA implementation of multi-scale deformable attention |

---

## IV. Datasets and Data Pipelines

### 4.1 Dataset Classes (`mmdet3d/datasets/`)

| File | Class | Purpose |
|------|-----|------|
| `custom_3d.py` | `Custom3DDataset` | 3D dataset base class |
| `nuscenes_dataset.py` | `NuScenesDataset` | Basic nuScenes dataset |
| `nuscenes_dataset_occ.py` | `NuScenesDatasetOccupancy` | + Occupancy GT loading (Occ3D gts) |
| `nuscenes_dataset_occ_trajectory.py` | `NuScenesDatasetOccupancyTrajectory` | ★ + Temporal occupancy + trajectory GT (loads 6 future frames) |
| `occ_metrics.py` | `Metric_mIoU`, `Metric_mIoU_Temporal` | Occupancy prediction evaluation metrics (mIoU/IoU) |
| `ray.py` | `RayGenerator` | NeRF ray generator |
| `builder.py` | — | Dataset builder |

#### Core of `nuscenes_dataset_occ_trajectory.py`
- Loads occupancy GT for the current frame + `num_fu_frames` future frames
- Computes inter-frame ego motion transformation matrices (for coordinate-system alignment)
- Provides GT trajectories (6 ego waypoints over the next 3s)
- Shared by SparseWorld and PreWorld4DTraj

### 4.2 Data Pipelines (`mmdet3d/datasets/pipelines/`)

| File | Purpose |
|------|------|
| `loading.py` | `PrepareImageInputs` (image loading + augmentation), `LoadOccupancy` (loads occupancy GT), `CreateRays` (generates NeRF rays) |
| `loading_traj_temporal.py` | ★ `LoadOccupancyTemporalTrajectory`: loads temporal occupancy + trajectory GT, including inter-frame transforms |
| `transforms_3d.py` | 3D data augmentation (flipping, rotation, scaling, etc.) |
| `formating.py` | `DefaultFormatBundle3D`: formats data into the standard mmdet format |
| `compose.py` | Pipeline composer |

---

## V. Configuration Files in Detail

### 5.1 SparseWorld Temporal Configurations (`configs/sparseworld/nuscenes-temporal/`)

| Configuration file | Purpose | Model class |
|----------|------|--------|
| `bevstereo-occ-traj.py` | BEVStereo baseline (temporal occupancy + trajectory) | `PreWorld4DTraj` |
| `preworld-7frame-pretrain-traj.py` | PreWorld NeRF pretraining | `PreWorld4DTraj` (is_pretrain=True) |
| `preworld-7frame-finetune-traj.py` | PreWorld occupancy fine-tuning | `PreWorld4DTraj` (is_pretrain=False) |
| **`sparseworld-traj-finetune.py`** | **★ Main SparseWorld configuration** | `SparseWorld4DTraj` |
| `sparseworld-traj-finetune-72pts.py` | SparseWorld 72-point variant | `SparseWorld4DTraj` |

#### Key Parameters in `sparseworld-traj-finetune.py`

```python
model = dict(
    type='SparseWorld4DTraj',
    img_backbone=dict(type='ResNet', depth=50),       # ResNet-50 backbone
    pts_bbox_head=dict(
        type='OPUSHead',
        num_query=720,                                 # Number of queries per frame
        num_fu_query=[720, 720, 720, 720, 720, 720],  # Number of queries for each future frame
        transformer=dict(
            type='OPUSTransformer',
            decoder=dict(
                num_layers=6,                          # 6 Transformer decoder layers
                num_refines=[1,1,1,1,1,1],            # Number of refinements per layer
            )
        ),
    ),
    num_fu_frames=6,                                   # Predict 6 future frames
)
# Training: 64 epochs, AdamW lr=2e-4, CosineAnnealing
# Data: 7 temporal frames (1 current +6 future), ResNet-50 backbone
```

### 5.2 PreWorld Non-Temporal Configurations (`configs/sparseworld/nuscenes/`)

| Configuration file | Purpose |
|----------|------|
| `bevstereo-occ.py` | BEVStereo occupancy baseline |
| `preworld-7frame-pretrain.py` | PreWorld NeRF pretraining (non-temporal) |
| `preworld-7frame-finetune.py` | PreWorld occupancy fine-tuning (non-temporal) |

### 5.3 Base Configurations (`configs/_base_/`)

- `datasets/nus-3d.py`: nuScenes dataset parameters
- `default_runtime.py`: Logging and checkpoint-saving policy
- `schedules/cosine.py`: CosineAnnealing learning-rate policy

---

## VI. Training and Testing Tools

### 6.1 Core Scripts (`tools/`)

| File | Purpose |
|------|------|
| `train.py` | Single-machine training entry point |
| `dist_train.sh` | Distributed training launch script |
| `test.py` | Single-machine testing entry point (occupancy prediction mIoU + output_data.pkl generation) |
| `test_temporal.py` | Temporal testing entry point |
| `dist_test.sh` / `dist_test_temporal.sh` | Distributed testing scripts |
| `get_flops.py` | Computes model FLOPs |
| `create_data_bevdet.py` | Generates data info pkl files in BEVDet format |

### 6.2 Data Generation (`tools/gen_data/`)

| File | Purpose |
|------|------|
| `gen_depth_gt.py` | Generates depth GT |
| `gen_seg_gt_from_lidarseg.py` | Generates semantic segmentation GT from lidarseg |
| `gen_seg_gt_from_occ.py` | Generates semantic segmentation GT from occupancy |

### 6.3 Visualization (`tools/visualization/`)

| File | Purpose |
|------|------|
| `vis_tool.py` | Occupancy visualization tool |
| `visual.py` | General-purpose visualization |

---

## VII. Evaluation Module (`AD-MLP/`)

Trajectory planning evaluation uses the STP3/UniAD evaluation framework.

| File | Purpose |
|------|------|
| `AD-MLP/deps/stp3/evaluate_for_mlp.py` | Evaluation entry point: loads predicted trajectories and computes L2/collision rates |
| `AD-MLP/deps/stp3/stp3/planning_metrics.py` | `PlanningMetric` class: L2 error, point collisions, and BBox collisions |

**Evaluation metrics**:
- `plan_L2_{1,2,3}s`: Trajectory L2 error at 1/2/3 seconds
- `plan_obj_col_{1,2,3}s`: Point-level object collision rates
- `plan_obj_box_col_{1,2,3}s`: BBox-level object collision rates

**Evaluation workflow**:
1. `tools/test.py` performs inference to generate `admlp/output_data.pkl` (token → predicted trajectory)
2. `evaluate_for_mlp.py` loads `output_data.pkl` + GT and computes planning metrics

---

## VIII. Required Data Files

| Directory/file | Contents |
|-----------|------|
| `data/nuscenes/samples/` | Keyframe images (~53G) |
| `data/nuscenes/sweeps/` | Intermediate-frame images (~342G) |
| `data/nuscenes/v1.0-trainval/` | Metadata JSON |
| `data/nuscenes/maps/` | Map data |
| `data/nuscenes/gts/` | Occ3D occupancy GT |
| `data/nuscenes/bevdetv2-nuscenes_infos_{train,val}.pkl` | Data info in BEVDet format |
| `occworld/nuscenes_infos_{train,val}_temporal_v3_scene.pkl` | Temporal scene info |
| `admlp/fengze_nuscenes_infos_{train,val}.pkl` | Data info in AD-MLP format |
| `admlp/stp3_val/` | GT required for evaluation (filter_token, stp3_occupancy, stp3_traj_gt) |
| `ckpts/epoch_56.pth` | SparseWorld pretrained weights |

---

## IX. Core Data Flow

### 9.1 Training Data Flow

```
nuScenes data
  ↓ PrepareImageInputs (loads images from 6 views + augmentation)
  ↓ LoadOccupancyTemporalTrajectory (loads current-frame + 6 future-frame occupancy GT + trajectory GT)
  ↓ DefaultFormatBundle3D (formatting)
  ↓
SparseWorld4DTraj.forward_train()
  ↓ extract_feat() → Multi-scale image features [B, TN, C, H, W]
  ↓ pts_bbox_head.forward() → Initializes 720 query points → 6 OPUSTransformer refinement layers
  ↓                           ↓ Each layer: 3D→2D sampling → AdaptiveMixing → SASA → FFN
  ↓                           ↓ Outputs: query_feat, cls_scores, refine_pts
  ↓ forward_backbone() → Autoregressive future prediction loop (6 frames)
  ↓                       ↓ Ego features × scene queries → trajectory prediction
  ↓                       ↓ reg/vel/cls branches → future query positions + semantics
  ↓ Loss computation:
  ↓   - Current frame: cls_loss (Focal) + pts_loss (L1)
  ↓   - Future frames: fu_cls_loss + fu_pts_loss (per frame)
  ↓   - Trajectory:  traj_loss (L2)
```

### 9.2 Inference Data Flow

```
Test image sequence
  ↓ simple_test_online() → Extracts and caches features frame by frame
  ↓ pts_bbox_head.forward() → Transformer query refinement
  ↓ forward_backbone() → Autoregressive prediction of 6 future frames
  ↓ get_occ() → Sparse queries → dense voxel occupancy [B, 200, 200, 16]
  ↓            (sigmoid activation → score filtering → scatter_max aggregation → dilation to fill holes)
  ↓ Outputs: semantic_occ_{0-6}s, geo_occ_{0-6}s, pred_traj
  ↓
evaluate_for_mlp.py → Computes L2/collision-rate evaluation metrics
```

---

## X. Development Guide

### 10.1 Common Modification Points

| Modification goal | Relevant files | Description |
|----------|----------|------|
| Increase/decrease query counts | `opus_head.py`, configuration files | Modify `num_query`, `num_fu_query` |
| Modify the number of Transformer layers | `opus_transformer.py`, configuration files | Modify `num_layers`, `num_refines` |
| Modify attention mechanisms | `opus_transformer.py` | `OPUSSelfAttention`, `OPUSCrossAttention` |
| Modify the sampling strategy | `opus_sampling.py` | `sampling_4d`, sampling point counts/view selection |
| Modify loss functions | `opus_head.py`, `sparseworld_4d_traj.py`, `loss.py` | Various loss weights and types |
| Modify the number of temporal prediction frames | `sparseworld_4d_traj.py`, configuration files | `num_fu_frames`, `num_fu_query` |
| Modify trajectory planning | `sparseworld_4d_traj.py` | `plan_head`, `traj_head`, `ego_cross_attn` |
| Modify the backbone | Configuration files | Replace `img_backbone` with Swin, etc. |
| Modify data augmentation | `opus.py`, `utils.py`, `pipelines/` | GridMask, PhotoMetric, 3D transforms |
| Add new evaluation metrics | `occ_metrics.py`, `evaluate_for_mlp.py` | Add new Metric classes |

### 10.2 Training Commands

```bash
# SparseWorld training (8 GPUs)
bash ./tools/dist_train.sh ./configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py 8

# SparseWorld testing (occupancy metrics)
python tools/test.py --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py --checkpoint ckpts/epoch_56.pth

# Trajectory planning evaluation
cd AD-MLP/deps/stp3 && python evaluate_for_mlp.py
```

### 10.3 Two-Stage Training Workflow

```
Stage 1: NeRF pretraining (preworld-7frame-pretrain-traj.py)
  → Learn to extract 3D representations from 2D images
  → Supervision signals: 2D depth, semantics, and color

Stage 2: Occupancy fine-tuning (sparseworld-traj-finetune.py)
  → Load Stage 1 weights
  → Train occupancy prediction + trajectory planning end to end
  → Supervision signals: 3D occupancy GT, trajectory GT
```

---

## XI. 16 Semantic Classes (nuScenes-Occ3D)

| ID | Class | Motion type |
|----|------|--------|
| 0 | others | Static |
| 1 | barrier | Static |
| 2 | bicycle | **Dynamic** |
| 3 | bus | **Dynamic** |
| 4 | car | **Dynamic** |
| 5 | construction_vehicle | **Dynamic** |
| 6 | motorcycle | **Dynamic** |
| 7 | pedestrian | **Dynamic** |
| 8 | traffic_cone | Static |
| 9 | trailer | **Dynamic** |
| 10 | truck | **Dynamic** |
| 11 | driveable_surface | Static |
| 12 | other_flat | Static |
| 13 | sidewalk | Static |
| 14 | terrain | Static |
| 15 | manmade | Static |
| 16 | vegetation | Static |
| 17 | empty (free) | — |

> In `sparseworld_4d_traj.py`, classes 2-10 are treated as moving objects, with velocity offsets added for future prediction.
