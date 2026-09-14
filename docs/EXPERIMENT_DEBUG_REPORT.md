# SparseWorld Memory 实验与 Debug 汇总

更新时间：2026-09-06 UTC
项目：`/data/jxy/projects`
基准：`ckpts/epoch_56.pth`
cache：固定 epoch-56 schema-v2 query memory cache
范围：只汇总已经完成或正在运行的 Memory 相关实验，不改写历史数值，不补填未完成评估结果。

## 当前结论

截至当前整理点，已经完成评估的 Memory-only、joint、Phase2、Phase3、Phase4、M0R1/M1R0/M1R1 变体均未证明 1s/2s/3s IoU 或 mIoU 超过 epoch-56 baseline。

Future Memory Adapter 已完成代码级验收和 200-iteration smoke；12-epoch formal 训练正在 `work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/` 运行中，尚无正式 validation 结果。因此目前不能声称 Future Adapter 已提升 IoU/mIoU。

## Baseline

epoch-56 baseline 是当前比较基准，禁止为了新 Memory 实验重新训练或重新评估 baseline。

| Horizon | 0s | 1s | 2s | 3s |
| --- | ---: | ---: | ---: | ---: |
| IoU | `25.68` | `23.15` | `22.28` | `21.21` |
| mIoU | `18.20` | `14.96` | `13.18` | `11.53` |

Baseline future mIoU mean：`13.2233`。

## 已完成正式结果

以下结果均来自 nuScenes val 4,219 samples。`Future mean` 是 1s/2s/3s mIoU 平均值，`Delta` 相对 baseline future mean `13.2233`。

| 实验 | 状态 | mIoU `[0s,1s,2s,3s]` | Mean | Future mean | Delta | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| epoch-56 baseline | 完成 | `[18.20, 14.96, 13.18, 11.53]` | `14.4675` | `13.2233` | `0.0000` | 当前基准 |
| Memory-only epoch 12 | 完成 | `[18.20, 14.96, 13.18, 11.53]` | `14.4675` | `13.2233` | `0.0000` | Memory 连续张量有变化，但离散 occupancy 指标不变 |
| Joint finetune epoch 4 | 完成 | 未单独保留完整表 | — | `13.2233` | `0.0000` | 打印精度上追平 baseline，未超过 |
| Joint finetune epoch 12 | 完成 | 未单独保留完整表 | — | `13.2033` | `-0.0200` | 未超过 baseline |
| Phase2 epoch 4 | 完成 | 未单独保留完整表 | — | `13.2233` | `0.0000` | 打印精度上追平 baseline，未超过 |
| Phase2 epoch 12 | 完成 | `[18.20, 14.92, 13.14, 11.49]` | `14.4375` | `13.1833` | `-0.0400` | 解冻 `pts_bbox_head` 没有带来收益 |
| Phase3 epoch 12 | 完成 | `[15.48, 14.22, 12.52, 10.95]` | `13.2925` | `12.5633` | `-0.6600` | 旧 refiner 拓扑会破坏 0s，并降低 future |
| Phase4 future-only epoch 12 | 完成 | `[18.20, 14.38, 12.75, 11.23]` | `14.1400` | `12.7867` | `-0.4366` | 0s 恢复，但 future 仍下降 |
| M1R0 formal epoch 12 | 完成 | `[18.20, 14.96, 13.18, 11.53]` | `14.4675` | `13.2233` | `0.0000` | 与 baseline 指标一致 |

对应已保留的 IoU 汇总：baseline `[25.68,23.15,22.28,21.21]`；Phase2 `[25.67,23.15,22.27,21.20]`；Phase3 `[25.18,23.01,22.03,20.86]`；Phase4 `[25.68,22.74,21.62,20.47]`；M1R0 `[25.68,23.15,22.28,21.21]`。

## 2×2 消融与 smoke 记录

| 变体 | 参数量 | Memory reads / fused queries | 关键信号 | 当前结论 |
| --- | ---: | ---: | --- | --- |
| M0R1：refiner-only | `801,568` | `0 / 0`，empty forward `1.0` | refiner 有梯度 | 可训练，但不验证 Memory |
| M1R0：Memory-only | `924,188` | `6 / 320`，empty forward `0.0` | q/k/v 梯度极弱；direct residual 约 `1e-5` | Memory 读取正常，但输出影响太弱 |
| M1R1：Memory + refiner | `1,725,756` | `6 / 320`，empty forward `0.0` | refiner 和 Memory 都有梯度；motion residual 约 `1.23 -> 5.51` | 连通，但存在 motion residual 稳定性风险 |

M1R1 只能证明训练图接通，不能作为性能收益证据。

## Baseline vs M1R0 中间层差异

比较对象：epoch-56 baseline 与 M1R0 formal epoch-12。覆盖 nuScenes val 4,219 samples。

raw logits、raw refine points 与 0s 输出完全一致：

| 项 | mean abs diff | max abs diff |
| --- | ---: | ---: |
| raw logits | `0` | `0` |
| raw refine points | `0` | `0` |

future refined 连续张量有变化：

| 输出 | 1s | 2s | 3s | 4s | 5s | 6s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logits mean abs | `0.006176` | `0.011827` | `0.016842` | `0.020320` | `0.022441` | `0.023618` |
| points mean abs | `0.0000637` | `0.0001279` | `0.0001760` | `0.0002313` | `0.0002737` | `0.0003166` |

离散 occupancy label 改变比例很低：

| horizon | changed voxels | changed fraction | baseline non-empty/sample | M1R0 non-empty/sample |
| --- | ---: | ---: | ---: | ---: |
| 0s | `0` | `0` | `52,623.63` | `52,623.63` |
| 1s | `526,540` | `0.0195%` | `54,665.71` | `54,654.54` |
| 2s | `1,148,110` | `0.0425%` | `54,783.32` | `54,770.09` |
| 3s | `1,882,880` | `0.0697%` | `54,054.36` | `54,042.93` |
| 4s | `3,221,811` | `0.1193%` | `54,024.48` | `54,003.26` |
| 5s | `4,523,406` | `0.1675%` | `53,595.03` | `53,574.93` |
| 6s | `6,036,067` | `0.2235%` | `53,119.84` | `53,094.71` |

直接结论：M1R0 不是“没训练”或“没读取 Memory”，而是 Memory 造成的连续变化大多没有跨过 occupancy 离散决策边界。

## Phase2/3/4 汇总

Phase2 的目标是扩大可训练边界，解冻 `pts_bbox_head` 相关 future/head 参数。结果显示 epoch 12 future mean 低于 baseline，继续扩大该方向的收益依据不足。

Phase3 把 Memory-conditioned refiner 接到 0s 和 future。它证明了训练图连通，但正式评估中 0s 大幅下降，说明当前帧输出路径不应被旧 refiner 扰动。

Phase4 改成 future-only 后 0s 恢复 baseline，但 1s/2s/3s 仍低于 baseline。该结果把问题定位到旧 future Memory/refiner 拓扑本身，而不是 0s 路由。

## Future Memory Adapter 状态

Future Memory Adapter 是当前新的建模方向。核心差异是分离 Baseline state stream 与 Memory output correction stream：

```text
S_base(t) = ClsBranch(Q_base(t))
S_final(t) = S_base(t) + G(t) * DeltaS_memory(t) + G(t) * DeltaO_memory(t)
P_final(t) = SafeRefine(P_base(t), G(t) * DeltaP_memory(t))
```

Memory 修正不会写回下一时刻的 Baseline future recurrence。只修正评估目标 1s、2s、3s。

内部 step 映射：

| 目标 horizon | internal future step | simple_test key | active queries |
| ---: | ---: | --- | ---: |
| 1s | 2 | `semantic_occ_2s` | `840` |
| 2s | 4 | `semantic_occ_4s` | `960` |
| 3s | 6 | `semantic_occ_6s` | `1040` |

200-iteration smoke 已完成，日志中出现 `mem_1s.*`、`mem_2s.*`、`mem_3s.*` loss，`grad_norm` 非零。smoke 只验证训练接线，不代表性能。

12-epoch formal 当前运行中：

```text
work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/
```

当前已观察到 epoch 1 正常推进，`mem_1s/mem_2s/mem_3s` loss 和 `grad_norm` 持续输出；尚未完成训练，也尚未执行正式 validation。

## 仍需关注的问题

已解决的工程错误不再在本报告展开。当前仍需关注的问题只保留会影响后续实验判断的部分：

1. Memory 对连续 logits/points 的扰动幅度可能仍不足以改变 occupancy 离散标签，后续需要对 margin、voxel decision boundary 和 corrected logits 分布做诊断。
2. cache/query 的语义匹配、空间匹配、age/reliability 权重是否真的提供正向预测信息，还需要 Future Adapter formal + shuffled/cache-off 对照来验证。
3. 旧 Phase3/Phase4 refiner 拓扑不应继续作为主线使用；M1R1 的 motion residual 增长说明该路径仍有稳定性风险。
4. Future Adapter formal 完成后，必须按相同 checkpoint 进行 Memory ON、Memory OFF、shuffled Memory 对照，才能判断 Memory 是否提供有效增益。
5. 标准 formal train log 主要记录 loss，不足以解释 flat metrics；若正式评估无提升，需要补充 per-horizon gate、candidate score、semantic compatibility 和 voxel label change diagnostics。

## 保留与删除策略

本文件是唯一实验数据汇总入口。旧 handoff、阶段性交接、`logs/` 下的软链接/摘要文件会删除，避免同一结果在多个入口重复维护。

原始产物不删除：

- checkpoints、完整训练日志、TensorBoard 文件仍保留在 `work_dirs/`；
- fixed memory cache 仍保留在 `data/query_memory/`；
- 正在运行的 Future Adapter formal 目录不清理、不移动、不重命名。
