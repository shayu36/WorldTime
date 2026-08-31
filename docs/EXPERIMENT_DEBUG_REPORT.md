# STAC-QM Phase 2/3 实验与 Debug 报告

更新时间：2026-08-31 UTC  
项目：`/data/jxy/projects`  
比较基准：`ckpts/epoch_56.pth` + 固定 epoch-56 schema-v2 cache  
验证集：nuScenes val，4,219 个样本  
约束：不刷新 cache；不修改原始 baseline config；不解冻 backbone、neck 或原 OPUS 六层。

## 结论先行

截至目前，没有任何 Phase 2/3 变体在 occupancy 上超过 epoch-56 baseline。

- Phase 2 解冻 `pts_bbox_head` 后，最终 future mIoU 为 `13.1833`，比 baseline `13.2233` 低 `0.0400`。
- Phase 3 formal 把 Memory-conditioned refiner 接入 0s 和 future，最终 future mIoU 为 `12.5633`，比 baseline 低 `0.6600`；0s 也从 `18.20` 降到 `15.48`。
- Phase 4 future-only 保留 raw 0s 路径后恢复了 0s，但 future mIoU 仍为 `12.7867`，比 baseline 低 `0.4366`。
- M1R0 formal 证明 Memory 确实读取、训练并改变了 future 连续张量，但 occupancy 离散化后改变比例最高只有 6s 的 `0.2235%`，因此 aggregate mIoU 与 baseline 完全一致。
- M1R1 smoke 的 Memory 与 refiner 都有梯度，但 motion residual 从约 `1.23` 增长到 `5.51`，暂不适合继续扩大联合训练。

因此，当前问题已从“Memory 是否接通”缩小为：future Memory 信号的匹配/几何对齐/运动补偿以及 residual 到 occupancy 离散决策的有效性。

## 1. Occupancy 正式结果

以下均为 4,219-sample validation；`future mean` 是 1s/2s/3s 的 mIoU 平均值，delta 相对 baseline future mean `13.2233`。

| 实验 | mIoU `[0s,1s,2s,3s]` | mIoU mean | Future mean | Delta | 解释 |
| --- | --- | ---: | ---: | ---: | --- |
| epoch-56 baseline | `[18.20, 14.96, 13.18, 11.53]` | `14.4675` | `13.2233` | `0.0000` | 当前比较基准 |
| Phase 2 formal，epoch 12 | `[18.20, 14.92, 13.14, 11.49]` | `14.4375` | `13.1833` | `-0.0400` | 解冻 `pts_bbox_head` 仍无收益 |
| Phase 3 formal，epoch 12 | `[15.48, 14.22, 12.52, 10.95]` | `13.2925` | `12.5633` | `-0.6600` | 0s refiner 路由破坏当前帧，future 也下降 |
| Phase 4 future-only，epoch 12 | `[18.20, 14.38, 12.75, 11.23]` | `14.1400` | `12.7867` | `-0.4366` | raw 0s 恢复；future Memory 仍无收益 |
| M1R0 formal，epoch 12 | `[18.20, 14.96, 13.18, 11.53]` | `14.4675` | `13.2233` | `0.0000` | 与 baseline 完全一致 |

对应 IoU（非 mIoU）为：baseline `[25.68,23.15,22.28,21.21]`；Phase 2 `[25.67,23.15,22.27,21.20]`；Phase 3 `[25.18,23.01,22.03,20.86]`；Phase 4 `[25.68,22.74,21.62,20.47]`；M1R0 `[25.68,23.15,22.28,21.21]`。

### Phase 2

目录：`work_dirs/sparseworld-traj-memory-phase2/`。12 epochs 完成，训练和逐 epoch validation 正常结束。最佳 epoch 4 只是在打印精度上追平 baseline，没有超过 `13.2233`；epoch 12 为 `13.1833`。因此 Phase 2 的负结论成立，不应继续扩大其解冻边界。

### Phase 3 connectivity smoke

目录：`work_dirs/sparseworld-traj-memory-phase3-smoke/`。这是连通性验收，不是性能实验：

- `200/200` 完成；`7 reads / 1040 fused queries` 不变量通过；冻结参数/buffer hash、TASS 不可变检查通过。
- iteration 100→200：fusion out `0.0049→0.0093`，fusion gate `0.0017→0.0041`，fusion alpha 梯度 `0.2853→0.6673`；attention q/k/v 梯度约 `0.0001/0.0001/0.0019→0.0001/0.0001/0.0050`。
- refiner、cls/reg/vel 分支梯度均非零；refiner residual ratio `0.0062→0.0068`，direct fusion residual ratio `0.0002→0.0007`。

结论：训练图连接正常，但 smoke 没有 occupancy evaluation，不能声称有性能收益。

### Phase 3 formal

目录：`work_dirs/sparseworld-traj-memory-phase3/`。0s 和 future 都走 Memory-conditioned refiner，0s mIoU 下降 `2.72` 个点，future mean 下降 `0.6600`。这说明当前帧路径被不必要地扰动，且 future residual 本身也未形成有效预测信号。

### Phase 4 future-only

目录：`work_dirs/sparseworld-traj-memory-phase4-future-only/`。raw OPUS observation Query 保留给 0s，Memory/refiner 仅作用于 future groups。0s 恢复到 `18.20`，隔离并确认了 Phase 3 的 0s degradation 来源；但 1s/2s/3s 仍分别低 `0.58/0.43/0.30`，所以 future 路径仍未解决。

## 2. 严格 2×2 smoke 消融

| 变体 | 参数量 | Memory reads / fused queries | 关键梯度与状态 | 结论 |
| --- | ---: | ---: | --- | --- |
| M0R1：refiner-only | `801,568` | `0 / 0`，empty forward `1.0` | refiner total grad 约 `0.215` | refiner 可训练；无 Memory 读取 |
| M1R0：Memory-only | `924,188` | `6 / 320`，empty forward `0.0` | q/k/v 几乎为 0；memory residual 约 `1e-5` | Memory 读取正常但直接融合极弱 |
| M1R1：Memory + refiner | `1,725,756` | `6 / 320`，empty forward `0.0` | qm grad total 约 `0.226`；refiner residual 约 `0.006` | 两条路径均接通，但 motion residual 有发散趋势 |

M1R1 smoke 中 motion residual 约 `1.23→5.51`，fusion alpha `0.0014→0.0042`；这是稳定性警报，不是性能提升证据。

## 3. Baseline vs M1R0 中间层差异

原始报告：[`logs/diagnostics/baseline-vs-M1R0-intermediates.json`](../logs/diagnostics/baseline-vs-M1R0-intermediates.json)，对应 `work_dirs/sparseworld-traj-memory-ablation-M1R0/baseline_vs_m1r0_intermediates.json`。比较覆盖全部 4,219 个验证样本。

指标定义：raw logits=`pts_bbox_head all_cls_scores[-1]`；raw refine points=`all_refine_pts[-1]`；future refined=`forward_backbone forecast_*_list`；occupancy=`semantic_occ_{0..6}s`。

### 连续张量

| 输出 | 0s / raw | 1s | 2s | 3s | 4s | 5s | 6s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw logits mean abs | `0` | — | — | — | — | — | — |
| raw refine points mean abs | `0` | — | — | — | — | — | — |
| future refined logits mean abs | — | `0.006176` | `0.011827` | `0.016842` | `0.020320` | `0.022441` | `0.023618` |
| future refined points mean abs | — | `0.0000637` | `0.0001279` | `0.0001760` | `0.0002313` | `0.0002737` | `0.0003166` |

raw logits 和 raw refine points 在全部 queries 上 `max abs=0`；当前 0s refined logits/points 也完全一致。M1R0 的参数确实更新：`fusion.alpha 0.01717→0.05004`，`q_proj norm 9.251→9.513`，`fusion.out_proj norm 2.104→3.346`。因此“没有 mIoU 变化”不能解释为 Memory 没训练。

### Occupancy 离散化后

每个 horizon 共 `2,700,160,000` voxels；changed fraction 是 baseline 与 M1R0 semantic occupancy label 不同的比例。

| horizon | changed voxels | changed fraction | baseline non-empty/sample | M1R0 non-empty/sample |
| --- | ---: | ---: | ---: | ---: |
| 0s | `0` | `0` | `52,623.63` | `52,623.63` |
| 1s | `526,540` | `0.0195%` | `54,665.71` | `54,654.54` |
| 2s | `1,148,110` | `0.0425%` | `54,783.32` | `54,770.09` |
| 3s | `1,882,880` | `0.0697%` | `54,054.36` | `54,042.93` |
| 4s | `3,221,811` | `0.1193%` | `54,024.48` | `54,003.26` |
| 5s | `4,523,406` | `0.1675%` | `53,595.03` | `53,574.93` |
| 6s | `6,036,067` | `0.2235%` | `53,119.84` | `53,094.71` |

因果解释：Memory 不改变原始 OPUS；它改变 future refined 连续值，但改变幅度大部分没有跨过 occupancy 的离散类别边界，所以 aggregate mIoU 不动。

## 4. 已出现的问题与定位

### 4.1 初始融合梯度过弱

早期同时存在接近零的 `out_proj`、很小的 gate 和小 alpha，导致 q/k/v 第一阶段几乎拿不到梯度。后续通过非零投影、较大的初始 gate 和 alpha 改善了 connectivity；但 M1R0 formal 的 direct fusion residual 仍约 `1e-5`，说明“接通”不等于“信号足够强”。

### 4.2 Memory read count 曾与预期不符

早期 observation queries 与 future queries 都在不合适的时机读取，读取数和 hook 预期不一致。当前修正为普通 Phase 3 `7 reads / 1040 fused queries`，future-only 严格消融为 `6 reads / 320 fused queries`。

### 4.3 Phase 3 0s refiner 路由破坏当前帧

Phase 3 formal 的 0s mIoU 从 `18.20` 降到 `15.48`。Phase 4 future-only 将 raw 0s 路径固定回 baseline 后恢复到 `18.20`，因此该问题已定位为当前帧 refiner 路由，而不是 baseline 或 cache 本身。

### 4.4 Future residual 被弱化或被离散化消除

M1R0 显示 future refined logits/points 的连续值确实变化，但 occupancy changed fraction 仅 `0.0195%→0.2235%`。需要继续检查 threshold/margin、points 的米制位移、future decoder 是否覆盖/衰减 residual，以及 cache query matching 和 motion compensation。

### 4.5 M1R1 motion residual 稳定性不足

联合 smoke 的 motion residual 约 `1.23→5.51`，Phase 3 smoke 也曾到约 `7.06`。在解释清楚 residual 的尺度、坐标系和 loss 约束前，不应继续扩大联合训练或追加长训。

### 4.6 评估管线曾有两个工程错误

1. SparseWorld 返回 dictionary，而通用 MMDetection test path 使用 `result[0]`；已改为显式 SparseWorld evaluation hook，并恢复分布式 sampler 顺序、裁掉 padding。
2. evaluator 返回四元素 IoU/mIoU list，而 TensorBoard 只接受 scalar；已 flatten 为逐 horizon、all-horizon mean 和 future-horizon mean。

### 4.7 中间层双 GPU 线程并发失败

为了加速 baseline/M1R0 前向比较，曾尝试两个 GPU 的线程并行，触发 `CUDA error: an illegal memory access was encountered`，位置在 `opus_sampling.py` CUDA sampling。随后改用单 GPU 串行，完整 4,219 样本比较成功。不要复用该线程并行方案。

### 4.8 非致命运行 warning

训练中反复出现 OMP/MKL 线程数自动设为 1、`fork`/`spawn` start method、`torch.meshgrid` 缺少 indexing，以及新 Query Memory 参数相对 epoch-56 checkpoint 的 missing keys。这些没有导致训练失败；新增模块缺失 keys 是预期的随机初始化行为。

## 5. Cache 与训练安全证据

- train schema-v2 cache：`19,730 / 19,730`；val：`4,219 / 4,219`。
- train/val audit：`failure_count=0`，没有 missing/corrupt/orphan/noncausal/scene/temporal/shape/schema/source-checkpoint failure。
- C0/C1 zero-init identity：通过；空历史 batch 可 backward，且输出恒等、梯度显式为零。
- Phase 3 smoke：`7/1040`、冻结状态 hash、TASS 不变和目标梯度检查通过。
- M1R0 formal：base model 未改变；仅 Query Memory 参数更新，且同一 checkpoint 的中间层比较已完成。

## 6. 当前运行状态

截至本报告检查时，tmux session `0` 仍存在，但其中运行的是独立的 `dsqe-ddp-32-baseline56-b2` baseline 训练，约在 epoch 28，不属于本报告的 Phase 2/3 结果。没有正在运行的 Phase 2/3/Memory ablation 任务；不要将该 tmux pane 的新输出混入本报告，除非另行记录实验身份和配置。

## 7. 文件入口

- Phase 2：[`logs/phase2/formal-train.log`](../logs/phase2/formal-train.log) → `work_dirs/sparseworld-traj-memory-phase2/train.log`
- Phase 3 smoke/formal：[`logs/phase3/`](../logs/phase3/)
- future-only：[`logs/phase3/future-only-phase4-train.log`](../logs/phase3/future-only-phase4-train.log)
- 2×2 smoke 与 M1R0 formal：[`logs/ablations/`](../logs/ablations/)
- cache audit 与中间层比较：[`logs/diagnostics/`](../logs/diagnostics/)
- 旧 epoch-51/早期诊断：[`logs/archive/legacy/`](../logs/archive/legacy/)

原始 checkpoint、cache 和完整日志均仍在各自 `work_dirs/` 目录；本报告只引用，不复制大型产物。
