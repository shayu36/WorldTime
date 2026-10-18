# A+B 后续逐步实验：先诊断，再改损失

状态（2026-10-16）：这里只完成代码、配置和 CPU 静态/小规模测试。两张 3090 正运行另一条 `sparseworld-traj-prescf.py` 训练，本轮没有启动 A+B 诊断、完整 validation 或新训练。下述干预是固定 A+B epoch 1 checkpoint 的输出实验，不是重新训练的结果。

## 已有依据与边界

A+B epoch 1 完整 validation：Future IoU 22.7900，Future mIoU 13.1667。此前 Baseline 约为 22.2133 / 13.2233。256 帧配对诊断显示恢复 Baseline 位置后 Future mIoU +0.1434（95% CI +0.0628 至 +0.2240）；恢复 Baseline occupancy 后二值 Future IoU -0.5006；恢复 Baseline semantic 后 Future mIoU -0.2894。这些是输出干预，不证明重新训练后的因果效果；完整记录见 `FUTURE_MEMORY_AB_PAIRED_GPU_DIAGNOSIS.md`。

## 阶段 1：输出级诊断，不改权重

固定同一 A+B epoch 1 checkpoint、同一前向。比较：原始 A+B、仅 1s/2s 恢复 Baseline 位置、以及仅 2s/3s 对 occupancy **负**残差进行软保护（保留原残差的 25%、50%、75%）。保护质量由 Baseline 和修正后语义分布在 motorcycle、pedestrian、traffic cone、manmade、vegetation 上的概率并集计算，不用 hard argmax。正占据残差、Semantic 分支、Memory 检索与未来递推均不变。三个系数只是诊断网格，不是推荐的训练超参数。

先运行 256 帧扫描。`summarize_future_memory_scan.py` 用场景 SHA256 的奇偶拆成筛选集/留出集；仅在筛选集上选择 2s/3s mIoU 较优且二值 IoU 降幅不超过 0.10 个百分点的方案，再看留出集。留出集也不能替代完整验证。

```bash
bash /data/jxy/projects/tools/run_next_future_memory_diagnostics.sh scan
```

无论扫描结果如何，1s/2s 位置干预只在已有 256 帧显著；需单独完成 4219 帧验证。该脚本只比较原始 A+B 与关闭 1s/2s 位置修正，避免多方案完整评估浪费 GPU。评估会持续较久，应在 GPU 空闲后运行：

```bash
bash /data/jxy/projects/tools/run_next_future_memory_diagnostics.sh position-full
```

两个输出目录分别是 `work_dirs/diagnostics-ab-miou-20261016/scan256` 和 `position_full4219`。脚本拒绝覆盖已有目录。比较 1s/2s/3s 二值 IoU、mIoU、逐类 TP/FP/FN，重点核对 2s/3s 的 pedestrian、vegetation 和 manmade。只有在扫描的留出集方向一致且完整验证未伤害关键指标后，才考虑真正改变模型配置；此时需要新的训练/完整验证，不能把输出干预直接当新模型成绩。

## 阶段 2：SoftVoxel 中心对齐单变量实验

旧 SoftVoxel 将索引 `k` 当 `(origin+k·voxel_size)`，GT 的索引 `k` 实际代表体素中心 `(origin+(k+0.5)·voxel_size)`。新增可选 `soft_voxel_center_offset=0.5` 让训练损失的三线性插值围绕 GT 中心对称；边界处丢弃无效角点并归一化有效角点权重，超出场景的点不投回边界。原 V2 默认仍为 0/False，不改变已有 checkpoint 的推理。

独立配置从 A+B epoch 1 恢复 optimizer/runner，在固定 `2e-5` 学习率下只追加一个 epoch，产生本地 epoch 2；原 A+B epoch 2 是对照。注意：这项实验检验的是训练损失，不会让已训练 checkpoint 的推理直接变化，不能用输出干预替代。完成阶段 1 诊断并确认 GPU 空闲后，用户本地运行：

```bash
bash /data/jxy/projects/tools/run_train_future_memory_softvoxel_center.sh
```

保存到独立目录 `work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center`，不覆盖 A+B 或 V1 checkpoint。训练结束后用现有 dump-only + 离线 evaluator 对新 epoch 2 做完整验证，并与原 A+B epoch 2、同一个 Baseline 比较 Future IoU/mIoU 和逐类召回。若新 loss 未改善 2s/3s mIoU，或二值 IoU 明显下降，则停止该支路；若方向有利，再考虑下一轮更长期训练。训练之前只可说“代码接通”，不能说已提升指标。

## 可能的后续（尚未实现或启动）

若输出扫描证实 occupancy 保护有益，再设计可训练的类别/时刻条件或 Baseline anchor；避免直接把输出级 `alpha` 当训练结果。若位置全量验证证实 1s/2s 伤害，再单独调位置 gate 或位置损失。每次只改一个因素，不同时改 Top-K、阈值与多个损失权重。

## 2026-10-18进展

256帧扫描已经完成：筛选集选中`alpha=0.25`，但留出集2s/3s二值IoU下降
0.2367个百分点，因此不直接采用。`alpha=0.75`在留出集mIoU +0.1851、
IoU +0.0603个百分点，仍只是诊断。完整4219帧位置输出干预（A+B epoch1）
把1s/2s位置恢复Baseline后，Future IoU 22.7907→22.8043、mIoU
13.1653→13.3007；这不是新训练checkpoint。

SoftVoxel中心对齐从A+B epoch1**正确恢复**并仅续训一个epoch；结果checkpoint
与来源的706个非V2张量完全一致。完整4219帧评估得到IoU
`[25.68,23.85,22.84,21.68]`、mIoU `[18.20,15.28,13.21,11.39]`，
Future均值22.7900/13.2933；同轮次旧SoftVoxel对照为22.7433/13.1067。
此前一次漏传`--resume-from`的错误训练结果保留在无`-from-ab1`后缀的目录，
不得用于比较。后续优先在中心对齐checkpoint上复查位置分支干预，
再决定是否改可训练Position Gate；不要把输出干预和训练结果合并记为同一checkpoint。
