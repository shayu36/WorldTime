# STAC-QM 第二阶段交接文档（GPT Codex 用）

> 生成时间: 2026-08-19
> 目的: 在另一 AI 端（GPT Codex）继续第二阶段训练。本文是唯一需要完整阅读的入口文档。
> 完整对话记录（可选查阅，勿全文粘贴给模型）: `docs/archive/conversation_transcript.md`
> 权威项目文档: `docs/STAC_QM_Modeling_Repair.md`、`docs/STAC_QM_Implementation.md`

---

## 1. 一句话背景

SparseWorld 4D occupancy 预测 + 轨迹预测仓库，新增了 STAC-QM（历史 Query Memory）模块。
第一阶段（Memory-only 与受限联合微调）证明：**在当前冻结边界内，历史 Query Memory 无法提升 occupancy IoU/mIoU，也无法提升规划 L2/碰撞**。
第二阶段目标：**解冻 occupancy 输出路径（`pts_bbox_head`），让模型真正能利用 Memory 信息，追求 IoU/mIoU 提升。**

---

## 2. 仓库与环境

```text
仓库:        /data/jxy/projects  （master 分支，远程 GitHub: shayu36/new-work）
Python 环境: /data/jxy/projects/env/bin/python  （torch 2.0.1+cu118, mmcv, mmdet3d 本仓库）
GPU:         2 × 24 GB（CUDA_VISIBLE_DEVICES=0,1）
数据:        nuScenes（data/ 目录，已配置）
基础权重:    ckpts/epoch_56.pth（原 64-epoch cosine 训练的第 56 个 epoch，已接近收敛）
缓存:
  train: data/query_memory/sparseworld_epoch56_schema2_train （19,730 条，epoch-56 生成）
  val:   data/query_memory/sparseworld_epoch56_schema2_val   （4,219 条，epoch-56 生成，永不改动）
  test:  路由到 val 缓存
```

当前 git 状态（交接时）：
- HEAD = `76b3c99 Add STAC-QM joint training configs and SparseWorld evaluation pipeline`
- 工作树状态是历史交接快照；完整对话记录现归档于 `docs/archive/conversation_transcript.md`。

硬性约束（用户明确要求，必须遵守）：
1. **所有长时间命令（训练/评估/缓存生成）由用户自己运行，AI 只给命令，不得后台执行。**
2. **不得执行 `git push`；不得 force push；不做破坏性 git 操作。**
3. 不得提交：nuScenes 数据、`.pt` 缓存、checkpoint、`output_data.pkl`、大日志、评估产物。
4. 不得修改原始基线配置 `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py`。
5. 代码追求正确性，不需要追求小 diff。

---

## 3. 第一阶段已完成的工作（Codex 不需要重做）

### 3.1 STAC-QM 建模修复（已提交）

历史 Query 记忆模块，位于 `mmdet3d/models/sparsedetectors/query_memory.py`：

- 每个 Query 最多读一次真实历史 Memory：
  - 720 个 observation queries 在 SCF 前读一次（future_offset=0）；
  - 未来 6 组 [60,60,60,60,40,40] 各自在引入时读一次（offset 0.5~3.0）；
  - 已激活 Query 永不重读。恒等式：**7 次读、1040 个融合 Query**。
- future-aware age：`effective_age = base_age + future_offset`，缓存时间戳不可变。
- `QueryMotionCompensator` 零初始化（初始行为 = 纯 ego-pose 对齐）。
- 每 Query 语义可靠性（schema-v2 缓存字段：semantic distribution/label/margin/entropy/reliability）。
- 统一确定性多样性选择器（缓存预计算 / 缓存加载 / 在线 bank 共用）。
- target-age 历史选择：`history_target_ages=[2.5,3.5,4.5]`、tolerance 0.35，严格过去、同 scene、每 slot 一帧、不重复。

### 3.2 训练安全机制（已提交）

- `SparseWorld4DTraj` 显式训练策略（`mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py`）：
  - `memory_finetune_mode`（仅 `query_memory.*` 可训，已完成 12-epoch 正式训练）
  - `memory_joint_finetune_mode`（联合：`query_memory/position_encoder/reg_branch/vel_branch/cls_branch/ego_cross_attn`，已完成 12-epoch 正式训练）
  - 两种模式互斥；都要求 `query_memory enabled + source='cache' + freeze_base_model=True`
  - 冻结策略：先全冻结，再按权威 allowlist 精确解冻
  - `train()` 覆写：根模型 training=True，仅 allowlist 模块进入 train 模式，其余子模块保持 eval（冻结 BN/dropout）
  - **TASS 状态冻结**：加载 ckpt 后一次性归一化 `num_stamps_all`、派生 `ind_stamps_all`、重建 RAP masks、克隆后断言不可变（`freeze_tass_state=True`）；`set_epoch()` 只记录 epoch 并断言不变
  - 空历史 batch 安全：数值恒等 + 零值 autograd 桥（`_training_identity()`），backward 合法且梯度为显式 0
- 优化器（`mmdet3d/core/optimizer/trainable_only_optimizer_constructor.py`）：
  - `TrainableOnlyOptimizerConstructor`：只收 `requires_grad=True` 参数；支持 custom_keys（最长匹配、lr_mult/decay_mult）、bypass_duplicate；拒绝不支持的 paramwise 选项；空集报错
  - `mmdet3d/apis/train.py`：joint 模式强制要求该 constructor；加载 ckpt 后调用 `validate_query_memory_training_setup(optimizer, optimizer_cfg, logger)` 校验训练集/优化器成员/LR 档位
  - **LR 校验已兼容 resume**：校验 `initial_lr` 绝对档位 + 各参数组当前 LR 的公共调度器比例（cosine 衰减后恢复不再误报）
- 连通性 smoke hook（`mmdet3d/core/hook/query_memory_training_hook.py`）：
  - `QueryMemoryConnectivityOptimizerHook`（Memory-only）与 `QueryMemoryJointConnectivityOptimizerHook`（joint）
  - 校验：优化器成员、冻结参数零梯度、目标模块非零梯度、7 读/1040 融合、TASS 不可变、冻结参数/buffer 前后 hash 一致
- 评估管线修复（第二阶段直接可用）：
  - `SparseWorld4DTraj.uses_sparseworld_eval_api = True`
  - `mmdet3d/apis/train.py::_select_detector_eval_hook`：SparseWorld 用专用 hook，其他模型用 mmdet 通用 hook
  - `mmdet3d/apis/sparseworld_test.py`：字典结果解析（不再 `result[0]`）、DDP sampler 顺序恢复、padding 裁剪到精确 4219 样本
  - `mmdet3d/core/evaluation/sparseworld_eval_hooks.py`：指标列表展开为标量（`IoU_0s..3s/mean/future_mean`、`mIoU_0s..3s/mean/future_mean`），TensorBoard 不再崩溃；支持 `save_best`（指定 `mIoU_future_mean` 之类的标量 key）
  - `tools/test.py` 的 `multi_gpu_test` 指向新实现；`eval_hook.interval` 支持逐 epoch 验证

### 3.3 第一阶段实验结果（全部为 4219 样本 val，occupancy 0s/1s/2s/3s）

```text
epoch-56 基线（Memory OFF）:
  IoU  [25.68, 23.15, 22.27, 21.21]   mIoU [18.20, 14.96, 13.18, 11.53]
  mIoU future mean = 13.2233

Memory-only epoch 12（只训 query_memory）:
  IoU  [25.68, 23.14, 22.28, 21.21]   mIoU [18.20, 14.95, 13.17, 11.51]
  future mean = 13.2100  （-0.0133，无提升）

联合微调（joint，6 个未来模块 + query_memory）逐 epoch:
  epoch 4:  mIoU [18.20, 14.96, 13.18, 11.53]  future mean 13.2233 ← 最佳，与基线打平
  epoch 12: mIoU [18.20, 14.94, 13.16, 11.51]  future mean 13.2033
  其余 epoch 在 13.19~13.21 之间波动（噪声级）

规划指标（AD-MLP，L2 ↓ / box 碰撞 ↓）:
  基线:      L2 0.1596/0.1945/0.2393   碰撞 0.0948%/0.1007%/0.1106%
  joint e4:  L2 0.16034/0.19570/0.24088 碰撞 0.0830%/0.0948%/0.1067%
  joint e8:  L2 0.16015/0.19540/0.24067 碰撞 0.0830%/0.0948%/0.1146%
  joint e10: L2 0.16022/0.19558/0.24084 碰撞 0.0948%/0.1007%/0.1146%
  joint e12: L2 0.16020/0.19557/0.24087 碰撞 0.0948%/0.1007%/0.1146%
  （Memory-only e8 曾恶化碰撞到 0.27~0.31%，joint 已消除该恶化）

结论: 在「冻结观测 Query 生成器 + 冻结 pts_bbox_head」边界内，Memory 对任何指标都无增益。
```

**为什么第一阶段失败（第二阶段的理论依据）**：Memory 读入发生在 observation queries（SCF 前）与各组未来 queries（引入时），但消费这些融合特征的解码路径（`pts_bbox_head` 的 query refine 与 occupancy 语义输出）全程冻结，模型结构上无法利用新信息。0s mIoU 恒为 18.20 即是证据（0s 输出路径完全冻结）。第二阶段必须解冻 `pts_bbox_head`。

---

## 4. 第二阶段方案（待 Codex 实施）

### 4.1 目标

在保留全部第一阶段安全机制的前提下，解冻 `pts_bbox_head`，让 occupancy 解码路径与 Memory 协同适应，追求 IoU/mIoU（尤其是 1s/2s/3s）提升。0s 解冻后允许变化，但需监控不要倒退。

### 4.2 新增模式：`memory_phase2_finetune_mode`

在 `SparseWorld4DTraj` 增加第三种调优模式，训练集合 = 第一阶段 joint 集合 + `pts_bbox_head.*`：

```text
可训练:
  query_memory.*         lr 5e-5  （lr_mult 5.0）
  position_encoder.*     lr 1e-5
  reg_branch.*           lr 1e-5
  vel_branch.*           lr 1e-5
  cls_branch.*           lr 1e-5
  ego_cross_attn.*       lr 5e-6  （lr_mult 0.5）
  pts_bbox_head.*        lr 5e-6  （lr_mult 0.5，最大模块，保守）

保持冻结（且在 eval 模式）:
  img_backbone.*（ResNet，缓存对齐红线）
  img_neck.*   （缓存对齐红线）
  plan_head.*
  points_scale_branch.*
  traj_head.*
  TASS 状态（num_stamps_all / ind_stamps_all / RAP masks，继续走现有冻结断言）
```

**为什么 backbone/neck 继续冻结**：schema-v2 历史缓存是 epoch-56 模型生成的。当前帧与历史帧的观测 Query 必须处于同一特征空间；解冻 backbone/neck 会让当前 Query 漂移而缓存 Query 不动，破坏对齐。`pts_bbox_head` 是查询生成与语义解码的最后一环，解冻它引入的漂移限于 head 层，用低 LR 控制。

**TASS 必须保持冻结**：TASS 时间戳分配决定 7 读/1040 融合的结构不变量与 RAP 因果 mask。现有 `_freeze_memory_finetune_temporal_state()` / `set_epoch()` 断言机制直接复用（它由 `_memory_tuning_active()` 驱动，扩展后自动生效，但必须补测试确认）。

### 4.3 实施清单（按顺序）

1. **模型策略**（`mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py`）：
   - 新增 `_MEMORY_PHASE2_TRAINABLE_PREFIXES`（joint + `pts_bbox_head.`）与 `_MEMORY_PHASE2_TRAIN_MODULES`
   - 解析 `memory_phase2_finetune_mode`；三种模式互斥；同样要求 enabled + cache + freeze_base_model
   - 更新 `_memory_tuning_active()` / `memory_tuning_trainable_prefixes()` / `memory_tuning_train_modules()` / `is_memory_tuning_parameter()` / `train()` 覆写
2. **优化器路径**（`mmdet3d/apis/train.py`）：joint 分支的 constructor 强制检查同样覆盖 phase2（TrainableOnlyOptimizerConstructor）
3. **smoke hook**（`mmdet3d/core/hook/query_memory_training_hook.py`）：新增 `QueryMemoryPhase2ConnectivityOptimizerHook`（或泛化现有 joint hook），梯度组增加 `pts_bbox_head`；沿用 7/1040、TASS 不可变、冻结参数/buffer hash 校验
4. **配置**：
   - `configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py`（继承 joint 配置，换模式 + custom_keys 增加 `pts_bbox_head: lr_mult 0.5`；`load_from='ckpts/epoch_56.pth'`、`resume_from=None`、12 epochs、interval=1）
   - `...-phase2-smoke.py`（IterBasedRunner 200 iters，`QueryMemoryPhase2ConnectivityOptimizerHook`，`log_diagnostics=True`）
5. **测试**（`tests/test_query_memory_integration.py` + `tests/test_sparseworld_evaluation.py`）：
   - `_make_sparseworld_shell` 支持 phase2；三种模式互斥参数化测试
   - phase2 精确可训练集 / LR 档位 / 模块 train-eval 模式
   - phase2 下 TASS 跨 epoch 冻结、7 读/1040、严格缓存强制
   - 现有 47 个测试全部保持通过
6. **文档**：在 `docs/STAC_QM_Implementation.md` 增加 phase2 小节（模式、冻结边界、LR 档位、命令、结果表）

### 4.4 缓存对齐策略（先做 A，视结果决定是否做 B）

- **A（默认）**：train 缓存保持 epoch-56 生成不动，接受 head 层漂移，靠低 LR（5e-6）控制。val 永远用 epoch-56 缓存保证公平对比。
- **B（可选，A 效果不理想时）**：训练中途刷新 train 缓存（用当前 checkpoint 重新生成，见 §5.4 命令；`--overwrite` 需明确意图）。**val 缓存绝不重新生成。**

### 4.5 对比基线

phase2 必须与以下数字对比（均为 4219 样本 val）：

```text
epoch-56 基线: mIoU [18.20, 14.96, 13.18, 11.53], future mean 13.2233
joint 最佳(epoch 4): 与基线打平 13.2233
```

判断标准：phase2 任意 epoch 的 mIoU future mean 超过 13.2233 即视为有效；同时检查 0s 与 1s/2s/3s 逐项变化。若 phase2 依然无提升，结论为「epoch-56 基线的 occupancy 已饱和，需要架构级改动（如 Memory 同时喂给 0s 输出路径、增大融合强度、多尺度缓存）」，届时停下与用户讨论，不要擅自扩大实验。

---

## 5. 命令参考（用户自己运行；AI 只提供）

### 5.1 短验证（Codex 可自行运行）

```bash
cd /data/jxy/projects
/data/jxy/projects/env/bin/python -m pytest -q tests/test_query_memory.py \
  tests/test_query_memory_integration.py tests/test_sparseworld_evaluation.py
/data/jxy/projects/env/bin/python -m py_compile mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py \
  mmdet3d/apis/train.py mmdet3d/core/evaluation/sparseworld_eval_hooks.py
git diff --check
```

### 5.2 phase2 smoke（200 迭代，单卡，用户运行）

```bash
cd /data/jxy/projects
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2-smoke.py \
  --work-dir work_dirs/sparseworld-traj-memory-phase2-smoke \
  --gpu-id 0 --deterministic
```

通过标准：hook 完整通过（优化器成员、pts_bbox_head 非零梯度、TASS 不可变、7 读/1040、冻结参数/buffer hash 一致），不能只看 iter_200.pth 是否存在。

### 5.3 phase2 正式训练（12 epochs，双卡，用户运行）

```bash
cd /data/jxy/projects
mkdir -p work_dirs/sparseworld-traj-memory-phase2
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 --master_port=29520 \
  tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py \
  --work-dir work_dirs/sparseworld-traj-memory-phase2 \
  --launcher pytorch --validate --deterministic \
  2>&1 | tee work_dirs/sparseworld-traj-memory-phase2/train.log
```

恢复（如中断，从最新 epoch_N.pth，端口换新）：

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 --master_port=29521 \
  tools/train.py configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py \
  --work-dir work_dirs/sparseworld-traj-memory-phase2 \
  --resume-from work_dirs/sparseworld-traj-memory-phase2/epoch_N.pth \
  --launcher pytorch --validate --deterministic \
  2>&1 | tee -a work_dirs/sparseworld-traj-memory-phase2/train.log
```

预期时长：约 1.3 h/epoch（含 ~20 min 验证）；12 epochs ≈ 16 h。解冻 head 后显存可能上升，若 OOM 先降 `samples_per_gpu=1`。

### 5.4 （可选 B）训练中途刷新 train 缓存（用户运行）

```bash
cd /data/jxy/projects
/data/jxy/projects/env/bin/python tools/query_memory/precompute_query_memory.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py \
  --checkpoint work_dirs/sparseworld-traj-memory-phase2/epoch_N.pth \
  --split train \
  --output-dir data/query_memory/sparseworld_epoch56_schema2_train \
  --max-queries-per-frame 256 --write-threshold 0.35 --min-reliability 0.0 \
  --spatial-cell-size 4.0 --max-per-spatial-cell 16 --max-per-class 64 \
  --workers-per-gpu 2 --overwrite
```

刷新前必须备份原 train 缓存目录；刷新后跑审计：

```bash
/data/jxy/projects/env/bin/python tools/query_memory/audit_query_memory_cache.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py \
  --split train --expected-schema-version 2 \
  --expected-source-checkpoint work_dirs/sparseworld-traj-memory-phase2/epoch_N.pth \
  --json-out work_dirs/stacqm_cache_audit_phase2.json
```

### 5.5 occupancy 与规划评估（epoch 4/8/10/12 或最佳 epoch，用户运行）

```bash
cd /data/jxy/projects
mkdir -p work_dirs/sparseworld-traj-memory-phase2/planning_eval
set -o pipefail
[ -f admlp/output_data.pkl ] && cp -n admlp/output_data.pkl admlp/output_data.pkl.phase2.bak
PORT=29530
for EPOCH in 4 8 10 12; do
  CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
    --nproc_per_node=2 --master_port=${PORT} \
    tools/test.py \
    --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py \
    --checkpoint work_dirs/sparseworld-traj-memory-phase2/epoch_${EPOCH}.pth \
    --launcher pytorch --eval segm --deterministic \
    2>&1 | tee work_dirs/sparseworld-traj-memory-phase2/planning_eval/eval_epoch${EPOCH}.log
  cp admlp/output_data.pkl work_dirs/sparseworld-traj-memory-phase2/planning_eval/output_data_epoch${EPOCH}.pkl
  /data/jxy/projects/env/bin/python AD-MLP/deps/stp3/evaluate_for_mlp.py \
    2>&1 | tee work_dirs/sparseworld-traj-memory-phase2/planning_eval/planning_epoch${EPOCH}.log
  PORT=$((PORT + 1))
done
```

说明：`tools/test.py --eval segm` 会覆盖 `admlp/output_data.pkl`（评估器写入的轨迹 cumsum），因此逐 epoch 归档。规划评估脚本硬编码读取该 pkl，cwd 必须在 `/data/jxy/projects`（其相对路径 `stp3_val/` 在此解析）。占用评估日志内 `{'IoU': [...], 'mIoU': [...], 'classes': 17}` 即为 occupancy 结果。

---

## 6. 关键文件地图（Codex 快速定位）

```text
模型与训练策略:
  mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py    # SparseWorld4DTraj：模式、allowlist、train()、TASS 冻结、validate_query_memory_training_setup
  mmdet3d/models/sparsedetectors/query_memory.py           # STACQueryMemory / QueryMemoryBank / 选择器
  mmdet3d/models/sparsedetectors/opus.py                   # OPUS 基类；pts_bbox_head 为 RAP head（查询生成 + get_occ 语义输出）
  mmdet3d/models/heads/occupancy_head.py                   # OccHead（辅助，确认 pts_bbox_head 的占用输出子模块归属时再看）
训练/优化器/评估:
  mmdet3d/apis/train.py                                    # 优化器目标选择、TrainableOnly 强制、_select_detector_eval_hook
  mmdet3d/core/optimizer/trainable_only_optimizer_constructor.py
  mmdet3d/core/evaluation/sparseworld_eval_hooks.py        # 指标标量展开 + save_best 支持
  mmdet3d/apis/sparseworld_test.py                         # 字典结果解析 / DDP 收集 / 4219 裁剪
  mmdet3d/core/hook/query_memory_training_hook.py          # 两个连通性 smoke hook
缓存工具:
  tools/query_memory/precompute_query_memory.py            # --checkpoint 可任意指定来源权重
  tools/query_memory/audit_query_memory_cache.py
  mmdet3d/datasets/pipelines/loading_query_memory.py       # 严格缓存加载
配置:
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py / -smoke.py
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint.py / -smoke.py   ← phase2 配置从 joint 继承
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py            ← 修复版基座（勿改 baseline finetune.py）
测试:
  tests/test_query_memory.py / test_query_memory_integration.py / test_sparseworld_evaluation.py
```

---

## 7. 已知坑（Codex 必读，避免重蹈覆辙）

1. **CRLF**：仓库部分文件为 CRLF，编辑后 `git diff --check` 会报 trailing whitespace；新文件一律 LF。
2. **训练时验证走 `_select_detector_eval_hook`**：`SparseWorld4DTraj.uses_sparseworld_eval_api=True` 保证走专用 hook；通用 mmdet hook 会对字典结果做 `result[0]` 崩掉（第一阶段的 KeyError: 0）。
3. **指标必须是标量**：IoU/mIoU 列表直接进 `log_buffer` 会让 TensorBoard 崩（NotImplementedError: Got <class 'list'>）。现在由 eval hook 展开，**新增指标时也必须走 `_flatten_eval_results`**。
4. **DDP 验证是 4220 → 4219**：sampler padding 由 `_interleave_result_parts(part_list, size)` 裁剪；不要改成 mmdet 的 zip 收集。
5. **恢复训练用 `--resume-from`**，LR 校验看 `initial_lr`，cosine 衰减后恢复不会误报（不要改回严格 `actual_lr == expected_lr` 校验）。
6. **TASS 冻结断言**：任何模式下 `set_epoch()` 会断言 `num_stamps_all`/`ind_stamps_all`/RAP mask 逐位不变；phase2 解冻 head 后尤其要确认 OPUS loss 不会把 `num_stamps_all` 当作可训练状态累积（现有 guard 已处理，加测试守护）。
7. **7 读 / 1040 融合是结构不变量**，任何改动后跑集成测试的 forward_backbone 计数断言。
8. **空历史 batch**：必须保持数值恒等 + 零值 autograd 桥（`_training_identity`），否则 scene 边界 batch 会无 grad_fn 崩溃；零梯度不算连通性证据。
9. **`admlp/output_data.pkl` 每次占用评估都会覆盖**，逐 epoch 归档（§5.5）。
10. **评估 `--eval segm` 会打印 `{'IoU': ..., 'mIoU': ..., 'classes': 17}`**，0s 是数组第一项（历史上有标签误读）。
11. **不要提交产物**；`git diff --check` 每次改动后跑；测试全绿再给用户命令。
12. 若 `samples_per_gpu=2` 显存不够：先试 `samples_per_gpu=1`；不要改 global batch 口径而不改 LR 档位说明。

---

## 8. 交接验收清单（Codex 开始动手前执行）

```text
[ ] git status / git log 确认 HEAD=76b3c99，工作树仅 transcript.md 与 install.md 两处无关改动
[ ] 读完 docs/STAC_QM_Modeling_Repair.md 与 docs/STAC_QM_Implementation.md
[ ] 读完 mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py 的训练策略相关函数（§6 列出的）
[ ] 跑 §5.1 短验证：47 passed、py_compile 通过、git diff --check 通过
[ ] 确认理解 §7 的 12 条坑
[ ] 按 §4.3 实施清单顺序动手，每步后跑测试
[ ] smoke 命令交给用户前，先自查配置可加载（Config.fromfile 不报错）
```

---

## 9. 结果记录模板（每次训练/评估后回填本文件，保持 Codex 会话间上下文连续）

```text
阶段: phase2 | 日期:
命令: （用户实际运行的命令）
状态: 完成/失败+原因
occupancy: epoch→mIoU[0s,1s,2s,3s] / future mean
规划:    L2 1s/2s/3s, box碰撞 1s/2s/3s
与基线差: future mean Δ vs 13.2233
结论/下一步:
```

### 9.1 Phase2 实现与静态验证

```text
阶段: phase2 implementation | 日期: 2026-08-19
命令: Codex 运行 §5.1 pytest、py_compile、两份 phase2 Config.fromfile、git diff --check
状态: 完成；修改前 47 passed，修改后 59 passed（均为 19 warnings）；其余检查通过
occupancy: 未运行（禁止由 Codex 执行长时间评估）
规划:    未运行
与基线差: 未产生实验结果；基线 future mean 仍为 13.2233
结论/下一步: memory_phase2_finetune_mode、TrainableOnly optimizer 路径、phase2 smoke hook、正式/smoke 配置和测试已实现。用户下一步运行 §5.2 的 200-iteration smoke，并回填 hook 的实际通过/失败结果；smoke 完整通过后才运行 §5.3 正式训练。
```

### 9.2 Phase2 200-Iteration Smoke

```text
阶段: phase2 smoke | 日期: 2026-08-19
命令: CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/train.py configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2-smoke.py --work-dir work_dirs/sparseworld-traj-memory-phase2-smoke --gpu-id 0 --deterministic
状态: 完成；200/200 iterations，QueryMemoryPhase2ConnectivityOptimizerHook 在 iteration 100 与 after_run 均通过；iter_200.pth 已生成
occupancy: 未评估（smoke 只验证训练连通性与冻结边界）
规划:    未评估
与基线差: 不适用；尚无 4219-sample val 指标
结论/下一步: 49,812,743 个允许参数进入 optimizer；pts_bbox_head 梯度显著非零（iter 100: 57.9960，iter 200: 55.4997），joint 各必需梯度组均通过 ever-nonzero 门槛。全程保持 7 reads / 1040 fused queries，TASS 不可变，冻结参数与 buffer hash 一致。epoch-56 缺失 query_memory.* keys 为新增 STAC-QM 的预期加载提示。下一步可运行 §5.3 的 12-epoch 双卡正式训练并逐 epoch 验证。
```

### 9.3 Phase2 Formal Training Progress

```text
阶段: phase2 formal training | 日期: 2026-08-19 至 2026-08-20
命令: CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun --nproc_per_node=2 --master_port=29520 tools/train.py configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-phase2.py --work-dir work_dirs/sparseworld-traj-memory-phase2 --launcher pytorch --validate --deterministic 2>&1 | tee work_dirs/sparseworld-traj-memory-phase2/train.log
状态: 已完成；双卡 DDP（WORLD_SIZE=2），12 epochs 均完成训练与 4219 样本验证，进程正常退出
occupancy:
  epoch 1 -> mIoU [18.19, 14.96, 13.13, 11.48] / future mean 13.1900
  epoch 2 -> mIoU [18.22, 14.91, 13.10, 11.45] / future mean 13.1533
  epoch 3 -> mIoU [18.17, 14.91, 13.12, 11.48] / future mean 13.1700
  epoch 4 -> mIoU [18.20, 14.97, 13.17, 11.53] / future mean 13.2233
  epoch 5 -> mIoU [18.23, 14.96, 13.16, 11.51] / future mean 13.2100
  epoch 6 -> mIoU [18.21, 14.94, 13.13, 11.51] / future mean 13.1933
  epoch 7 -> mIoU [18.18, 14.90, 13.13, 11.47] / future mean 13.1667
  epoch 8 -> mIoU [18.19, 14.92, 13.13, 11.49] / future mean 13.1800
  epoch 9 -> mIoU [18.22, 14.95, 13.17, 11.51] / future mean 13.2100
  epoch 10 -> mIoU [18.21, 14.93, 13.14, 11.51] / future mean 13.1933
  epoch 11 -> mIoU [18.19, 14.92, 13.15, 11.50] / future mean 13.1900
  epoch 12 -> mIoU [18.20, 14.92, 13.14, 11.49] / future mean 13.1833
规划:    未运行
与基线差: epoch 1: -0.0333；epoch 2: -0.0700；epoch 3: -0.0533；epoch 4: +0.0000；epoch 5: -0.0133；epoch 6: -0.0300；epoch 7: -0.0566；epoch 8: -0.0433；epoch 9: -0.0133；epoch 10: -0.0300；epoch 11: -0.0333；epoch 12: -0.0400 vs 13.2233
结论/下一步: phase2 全部 12 epochs 均未超过 13.2233；最佳 epoch 4 仅打平 epoch-56 基线与 joint 最佳，解冻 pts_bbox_head 未产生有效增益。按 §4.5 停止继续扩大现有微调范围，转入架构级讨论：优先解除 gate/out_proj 双重梯度抑制，并让 Memory-conditioned refinement 直接参与 0s 与未来 occupancy 输出；未经用户确认不启动新实验。
```
