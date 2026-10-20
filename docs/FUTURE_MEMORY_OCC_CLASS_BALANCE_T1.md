# T0/T1：仅 3s 占据正样本 BCE 类别平衡

## 起点与证据（2026-10-18）

起点是 **A+B + SoftVoxel-center epoch 2 的原生推理 checkpoint**：

`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-from-ab1/epoch_2.pth`

已有 4219 帧正式成绩为 IoU `[25.68,23.85,22.84,21.68]`、mIoU
`[18.20,15.28,13.21,11.39]`，Future Mean **22.7900 / 13.2933**。
这些不是本轮 T1 成绩，不关闭任何时刻的位置修正。

用户运行的 D0 256 帧、150 场景配对诊断位于
`work_dirs/diagnostics-center-occ-20261018-074738/summary.json`。
只恢复 Baseline occupancy、保留当前语义和位置时：

| 时刻 | 二值 IoU 差值（百分点） | mIoU 差值 | mIoU 差值的场景 bootstrap 95% CI |
| --- | ---: | ---: | --- |
| 1s | -0.6288 | -0.2524 | [-0.4218,-0.0789] |
| 2s | -0.4870 | +0.0616 | [-0.1242,+0.2637] |
| 3s | -0.3202 | +0.1947 | [-0.0145,+0.4220] |

3s 的 pedestrian/manmade/vegetation 分别 +2.0268/+1.0605/+1.3963，
driveable_surface -1.0283。恢复语义使 Future mIoU -0.3533；恢复位置的
Future mIoU -0.0041，且 3s -0.0639。故保留语义、占据、位置、A+B gates，
只试验 3s 监督的类别聚合。D0 支持“类别间取舍”的假设，**并未证明训练正样本
类别不平衡是唯一原因**；3s 整体 mIoU 干预区间仍跨零。它不是完整 validation，
也不刻画不同训练种子的波动。

## 唯一损失改动

原有正负平衡 BCE 已分别计算正点均值和负点均值；本轮不改变正负标签规则。
令 P 为当前 rank/local batch 中所有有效正点，P_c 为其中 GT target label=c
的点集合，C_present 为 P 中实际出现的类别集合：

```
L_pos_point = mean_{i in P} BCE(O_i, 1)
L_pos_class = mean_{c in C_present} mean_{i in P_c} BCE(O_i, 1)
L_pos_mix   = (1-beta) * L_pos_point + beta * L_pos_class
L_occ      = (w_pos * I_pos * L_pos_mix + w_neg * I_neg * L_neg)
             / max(w_pos * I_pos + w_neg * I_neg, EPS)
```

配置：`occupancy_positive_class_balance_beta={'1s':0., '2s':0., '3s':0.25}`。
0.25 是保守的首轮混合比例：保留 75% 原逐点目标，只让 25% 正样本项对每个
实际出现的 GT 类别等权。全部 17 个非空类别都可参与；没有特定小类名单、
没有使用预测类别或修改已有 semantic class weights，缺席类别不计入分母。
类别不平衡被减弱而非消除；稀有类若标签噪声更大，也可能被放大，需实测。

正样本仍为“落在 GT 非空体素”或“最近非空中心在原 radius 内”；前者的
GT 类别优先，后者使用原 nearest-label。points mask、越界过滤完全沿用。
负样本项、正负组权重、外层 `loss_occ_weight=0.25`、threshold、anchor、
semantic、SoftVoxel-center、原 future classification/position losses **不变**。
T1 仅替换 `mem_3s.loss_occ` 内部正样本均值，保留全部 loss 名称。

空正样本时只剩原负样本项；空负样本时只剩正样本项；全空时返回有限的零。
新路径按 FP32 累加，避免 FP16 大样本数量/求和溢出（显式 double 输入保留
double）。beta=0 且不要求新统计时严格走原 reducer，保持原运算顺序与 dtype。
只新增 C 维 scatter/bincount 和点级向量，不生成 dense points×classes 张量。

DDP 与原 loss 一样采用 **每 rank 的 local batch 聚合，再平均 rank loss**，
不是全局 batch 每类计数归一化。固定相同 GPU 数、samples_per_gpu、seed 和
采样顺序，避免实验间改变这个含义。beta 只作用 3s，但共享参数仍可能影响
1s/2s 的后续推理，不能承诺这两个时刻训练后逐位不变。

## 诊断日志

T1 默认 `occupancy_class_balance_diagnostics=True`，且仅 beta>0 的 3s 记录：

- `mem_3s.occ_positive_bce_point_mean / class_mean / mixed_mean`
- `mem_3s.occ_negative_bce_mean`
- `mem_3s.occ_class_balance_beta / occ_positive_present_classes`
- `mem_3s.occ_positive_count / occ_negative_count`
- `mem_3s.occ_positive_voxel_count / occ_positive_proximity_only_count`
- 各类别 `occ_positive_count_c0` 至 `c16`，以及对应 `voxel_count_c*`、
  `proximity_only_count_c*`。实现从 logits 类别数推导，不硬编码 17。

这些是被当前预测点匹配到的 GT 类别计数，**不是 GT 体素体积**。
exact 与 proximity-only 互斥，逐类满足二者之和等于 positive_count；不能直接
把可能重叠的 voxel_positive 和 proximity_positive 两个原 mask 相加。
统计 tensor 全部 detach，没有 `loss` 字样，因此不进入优化目标。
空类别、空正样本也保留相同 scalar keys，避免 DDP rank 的日志 collective
数量不同。MMDetection 对日志做 rank 平均，logger 又按窗口平均，因此打印的
count 允许为小数，表示平均每 rank 每迭代数量，不是全局总数。

## 匹配续训及运行命令

两组均从 **center epoch 2** 连同 optimizer/runner 状态恢复，各追加一个 epoch
到 **epoch 3**；不是从 T0 训练后的 epoch 3 接着训练 T1。

| 项目 | T0 | T1 |
| --- | --- | --- |
| 模型 / 推理 | 原 center | 完全相同 |
| 起点 | center-from-ab1/epoch_2.pth | 同一个 checkpoint |
| trainable | 原 `future_memory_adapter_v2.*` | 不增删参数 |
| 3s 正 BCE | 原逐点均值 | 75%逐点 + 25%逐类 |
| seed / DDP / batch | 17 / 2 GPU / 每卡1 | 相同 |
| LR / scheduler | 恢复原 AdamW，fixed 2e-5 | 相同 |
| q chunk / AMP | 48 / dynamic FP16 | 相同 |
| 总 runner epoch | 3（只新增1） | 3（只新增1） |

T0 已由用户手工启动在
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t0-from-center2`。
T0 使用原 center 配置，CLI 覆盖 max_epochs=3、独立 work-dir，且**必须显式**
`--resume-from .../epoch_2.pth`；本项目 tools/train.py 的 CLI 默认值可能覆盖
配置中的 resume_from，不能只设置配置路径。原配置文件未修改。

T1 手工执行前检查（CPU，不启动训练）：

```bash
bash /data/jxy/projects/tools/run_train_future_memory_occ_t1.sh --check
```

等 T0 完成、两张 GPU 空闲后，在 tmux 执行：

```bash
bash /data/jxy/projects/tools/run_train_future_memory_occ_t1.sh
```

脚本检查 epoch2/optimizer/匹配配置、拒绝已有输出目录、拒绝忙碌 GPU，使用
与 T0 相同的环境与 seed、独立 port 29555，显式 resume 同一个 epoch2。
记录 checkpoint/source SHA256 与配置，不记录环境变量/密钥，不修改旧 checkpoint。
输出为 `work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth`。
不创建自动启动或等待 GPU 空闲后自动运行的任务。

## 验收与后续判断

本轮只做 CPU 静态、配置/独立 Adapter 构建和前向/纯函数验收；不由助手启动训练、
backward、optimizer step、GPU smoke 或完整评估。测试覆盖 beta=0 恒等、
精确混合公式、缺席类/空样本、有效点 mask、FP16 累加、B=1/2、其他损失
严格不变、真实 horizon 接线、日志不参与 loss、配置与 checkpoint key 兼容。
这不是 T1 梯度验收或训练/指标提升结论。现有 T0 训练由用户启动，保持不动。

完整模型的 CPU 构建尝试**未完成**：旧 `opus_head.py:85` 在初始化中直接调用
`torch.tensor(...).cuda()`，隐藏 GPU 的检查环境报 `No CUDA GPUs are available`。
不改 Baseline、不占用 T0 的 GPU 绕过；完整 T1 model build/GPU smoke/DDP 启动
验收留到 GPU 空闲后。`--check` 仅验证 CPU Adapter 严格载入和优化器 state
分组/形状恢复，不把这称为完整模型通过。

本轮已实际执行：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
env/bin/python -m pytest -q \
  tests/test_future_memory_occ_class_balance.py \
  tests/test_future_memory_adapter_v2.py \
  tests/test_future_memory_adapter_ab.py::test_migration_preserves_nonzero_outputs_all_horizons \
  tests/test_future_memory_adapter_ab.py::test_horizon_and_task_gate_isolation \
  tests/test_future_memory_adapter_ab.py::test_no_candidate_gates_and_outputs_are_exact_baseline \
  tests/test_future_memory_adapter_ab.py::test_decode_independent_occupancy_gate \
  tests/test_future_memory_softvoxel_center.py::test_centre_is_optimal_and_neighbours_symmetric \
  tests/test_future_memory_softvoxel_center.py::test_boundary_stays_finite_and_out_of_bounds_safe \
  -k 'not cuda'
```

最终结果 **94 passed, 1 deselected**；AB/SoftVoxel 文件中的梯度测试均未选择。
初次测试发现旧 CPU Half reducer 不支持 `clamp_min`，因此 legacy 严格相等
仅以支持的 FP32/FP64 检查；新 reducer 的 80,000 点 FP16 输入升 FP32 检查
已通过。这不是 CUDA AMP 运行验收。另修正了测试调用 `_parse_losses` 时遗漏
实例参数的问题，随后真实 MMDetection parser 已验证新统计不计入总 loss。

`bash -n`、启动脚本 `--check`、Python compile、`git diff --check` 均通过。
另把 HEAD 中原 loss 函数加载为只读对照，对 B=1/B=2 的正常、空 GT、越界、
全 mask、空预测共 10 种输入检查：所有默认 loss 均 `torch.equal`。
V1/V2 Adapter 类、解码、位置、occupancy target 与 SoftVoxel 函数 AST
和 HEAD 完全一致；来源 epoch2 SHA256 与 D0 provenance 一致。
HEAD 保持 `3c057d61ab2c7d8ea68ceb4ddd319d9bf7e1bf73`，未提交、推送或改动
已有 checkpoint/cache；用户原有 diagnostic launcher 与位置复查文档改动保留。

后续首先比较 **T1 epoch3 与 T0 epoch3**，并保留 center epoch2 作起点参照。
目标是改善 3s 及 Future mIoU，同时保留占据 IoU 增益；可把 Future IoU
相对起点下降不超过 0.05 个百分点当预先约定的探索预算（22.7400），不是
显著性界限或允许忽略 T0 的标准。检查逐时刻、逐类 TP/FP/FN，尤其行人、
植被、人造物与道路的取舍。1 epoch 仅作筛选，不自动追加 epoch 或组合 C。
正式结论仍需匹配的完整验证；碰撞率/L2 要对最终选中 checkpoint 单独评估。
