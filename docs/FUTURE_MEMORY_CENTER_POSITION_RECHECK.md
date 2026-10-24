# SoftVoxel-center A+B checkpoint 的位置修正复查

2026-10-18，仅诊断；没有训练或修改模型、占据分支、Memory cache 和 checkpoint。

## 对象与方法

- 配置：`configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center.py`
- checkpoint：`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-from-ab1/epoch_2.pth`
- 输出：`work_dirs/diagnostics-center-position-20261018/full4219/`
- 4219 帧、150 场景；每帧只执行一次模型前向。同一次前向的 A+B 输出与 `position_12_base` 配对比较。后者仅在最终体素解码时将 1s、2s 位置替换为 Baseline 快照；0s、3s、语义 logits、占据 logits、Memory 检索和 Baseline 递推均不变。
- 使用实际 `get_occ()` 评估，并以场景为单位做 1000 次配对 bootstrap。区间仅度量固定 checkpoint 和验证集上的场景差异，不代表训练随机种子的不确定性，也不能预测重新训练的收益。

原 A+B 重放的 IoU 为 `[25.6831, 23.8549, 22.8412, 21.6817]`，mIoU 为 `[18.1999, 15.2792, 13.2094, 11.3880]`（0s、1s、2s、3s）；四舍五入后与已有 `eval_epoch_2/metrics.json` 一致。4219 帧完整、0s 重放一致、A+B 解码重放一致、冻结 Baseline 参数一致。

## 全量结果

| 指标 | A+B 原输出 | 关闭 1s/2s 位置修正 | 差值 | 配对 95% 区间 |
| --- | ---: | ---: | ---: | ---: |
| 1s IoU | 23.8549 | 23.8333 | -0.0217 | [-0.0380, -0.0048] |
| 2s IoU | 22.8412 | 22.8285 | -0.0127 | [-0.0295, 0.0054] |
| Future IoU 均值 | 22.7926 | 22.7811 | -0.0115 | [-0.0217, -0.0006] |
| 1s mIoU | 15.2792 | 15.3089 | +0.0297 | [-0.0172, 0.0669] |
| 2s mIoU | 13.2094 | 13.2237 | +0.0144 | [-0.0373, 0.0553] |
| Future mIoU 均值 | 13.2922 | 13.3069 | +0.0147 | [-0.0167, 0.0403] |

0s 与 3s 的 IoU/mIoU 在两条路径中严格相同。

关闭 1s/2s 位置修正有明显的类别交换：1s/2s motorcycle 分别 +0.6718/+0.5490，bicycle +0.4324/+0.4102，traffic_cone +0.2951/+0.2219；但 driveable_surface 分别 -0.3942/-0.4763，other_flat -0.3634/-0.3985。正负作用在总体 mIoU 中基本抵消。固定 256 帧子集同样仅得到 Future mIoU +0.0172，区间跨 0。

旧 A+B epoch 1 checkpoint 上，相同“关闭 1s/2s 位置”全量干预的 Future mIoU 曾为 +0.1354、IoU +0.0136；它是**不同 checkpoint**，不能将其收益外推到本次 checkpoint，也不能把两次差异单独归因于 SoftVoxel 代码（中间还进行了一个 epoch 的训练）。

## 决策

当前 checkpoint 不支持“全局关闭 1s/2s 位置修正可稳定提升 mIoU”的判断；同时它会轻微但稳定地降低 Future IoU。因此**本轮不修改、不重训通用 Position Gate，保持占据分支不变**。若下一阶段仍要研究位置门控，应单独设计基于预测类别或点级证据的选择性干预，并先做同 checkpoint 的配对诊断；不能使用 GT 类别作为推断时门控条件。任何新门控训练都需独立 checkpoint 和正式验证，不能把本次输出干预成绩称为新模型指标。

复查命令：`bash tools/run_next_future_memory_diagnostics.sh center-position-full`。脚本会拒绝覆盖已有输出目录；重跑时用 `FUTURE_MEMORY_DIAG_OUTPUT_DIR=work_dirs/<新的独立目录> bash tools/run_next_future_memory_diagnostics.sh center-position-full`。原始结果见输出目录的 `provenance.json`、`report.json`、`summary.json` 和 `per_sample_confusions.npz`。
