# 补充材料：T1 时序 Adapter 首轮结果

## 结论边界

这轮实验在 1s/2s/3s 的 Memory Adapter 之间加入 Query-ID 对齐的隐状态通路；
新增状态只影响后续 Adapter，不写回 Baseline 未来递推。
对已评估的 epoch 2/3/4，IoU 与 mIoU 相对起始 checkpoint 基本持平，未显示稳定增益。
更准确的归类是“当前结果没有证明这项改动有效”，而不是“已证明改动导致指标显著下降”。

评估对象为完整 4219 帧验证集。该轮训练在 epoch 5 的 1650/9865 iter 中途停止，
没有完成配置中的 12 个 epoch，因此以下结论只适用于已评估 checkpoint，不能代表
12 epoch 收敛结果。

## IoU / mIoU

起始参考是 T1 + SoftVoxel-center 的 epoch 3；实验组由该 checkpoint 初始化。
数值来自各自 `metrics.json`，每个时刻按验证器输出精度保留两位小数；Future Mean
为 1s/2s/3s 的平均值。

| checkpoint | IoU [0s, 1s, 2s, 3s] | Future IoU Mean | 相对起点 | mIoU [0s, 1s, 2s, 3s] | Future mIoU Mean | 相对起点 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 起始 T1 + SoftVoxel-center epoch 3 | [25.68, 23.88, 22.86, 21.78] | 22.8400 | — | [18.20, 15.39, 13.40, 11.87] | 13.5533 | — |
| 时序 Adapter epoch 2 | [25.68, 23.84, 22.83, 21.77] | 22.8133 | -0.0267 | [18.20, 15.34, 13.38, 11.92] | 13.5467 | -0.0067 |
| 时序 Adapter epoch 3 | [25.68, 23.87, 22.86, 21.76] | 22.8300 | -0.0100 | [18.20, 15.35, 13.39, 11.91] | 13.5500 | -0.0033 |
| 时序 Adapter epoch 4 | [25.68, 23.88, 22.84, 21.74] | 22.8200 | -0.0200 | [18.20, 15.37, 13.37, 11.90] | 13.5467 | -0.0067 |

差值很小，且由于逐时刻结果已四舍五入，Future Mean 的末位差值不应过度解释。
epoch 2/3/4 没有呈现相对起点持续上升的趋势；当前不应据此把时序 Adapter
标记为新的最佳模型。

## L2 与碰撞率

三个时序 Adapter checkpoint 的规划推理结果一致，并与记录的 Baseline 参考值在
输出精度内相同：

| 时刻 | L2 (m) | box collision (%) |
| --- | ---: | ---: |
| 1s | 0.159596 | 0.094809 |
| 2s | 0.194547 | 0.100735 |
| 3s | 0.239348 | 0.110611 |

这与架构一致：新通路不更新 `pred_traj`，所以这些结果不是轨迹/碰撞改善证据。

## 可复现路径与限制

- 起始 checkpoint：
  `work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth`
- 实验 checkpoint：
  `work_dirs/t1-temporal-from-best3-12ep/epoch_2.pth`、`epoch_3.pth`、`epoch_4.pth`
- IoU/mIoU：各 checkpoint 对应目录 `eval_epoch_{2,3,4}/metrics.json`
- L2/碰撞率：各 checkpoint 对应目录 `planning_epoch_{2,3,4}/planning_metrics.json`
- 训练配置：`work_dirs/t1-temporal-from-best3-12ep/sparseworld-traj-memory-future-adapter-v2-t1-temporal.py`

这不是完整的匹配对照结论：虽然仓库有 temporal-control 配置，但没有对应的
同起点、同预算 control 训练与完整评估。因此不能把 epoch 间的小差值单独归因于
新时序通路。另一个重要限制是，本轮配置沿用旧 ego/cache 输入协议；它不能作为
修复后因果输入协议下的验证结果。相关来源审计及限制见
[`FUTURE_MEMORY_T1_EGO_PROVENANCE_FOLLOWUP.md`](FUTURE_MEMORY_T1_EGO_PROVENANCE_FOLLOWUP.md)。

因此本结果只作为没有观测到稳定增益的补充材料；不改写历史最佳模型指标，
不据此声称时序机制必然无效，也不声称已完成 12 epoch 实验。
