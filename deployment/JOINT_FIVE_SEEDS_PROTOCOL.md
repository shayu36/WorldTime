# 2026-11-24：五种子、每种子完整 1 epoch 的联合适配

用户请求在当前服务器运行不冻结占据 Adapter 的五种子实验。独立目录
`/data/jxy/WorldTime-planning-joint-5seed-20261124`，原目录及上一轮规划-only
结果不被覆盖。新模型类为 `SparseWorld4DTrajMemoryPlanningJoint`。

## 固定协议

- 种子 17 / 23 / 41 / 73 / 101。每种子完整 19730 训练样本，双卡每卡 batch 1，9865 次迭代，一个 epoch。
- 每种子从同一最好 T1 epoch_3 开始，SHA256
  `bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`。
  新规划融合模块按该种子初始化；不载入旧 optimizer、AMP scaler 或迭代状态。
- Baseline 参数与运行状态保持冻结；训练 77 个参数张量、1722147 个标量：
  `future_memory_adapter_v2.*`（1498145）与 `memory_planning_fusion.*`（224002）。
- 保持 T1 原占据目标和六步原轨迹监督。占据损失更新 V2，轨迹损失更新融合模块。
  Scene token packing 的 stop-gradient 保留，因此本实验不是规划损失端到端更新 V2。
- 学习率：V2 为 2e-5（沿用 T1 最后续训率），新规划模块 1e-4（沿用规划-only率）。
  AdamW weight_decay=.01，固定 LR，100 iter linear warmup，起点比例 .5。
- V2 q40，activation checkpoint=True，save_on_cpu=False，FP16 dynamic loss scale。
  保留同一次前向的真实 Memory Context 与 candidate mask 供规划融合。
  联合训练恢复原规划-only路径跳过的 future point masks，用于占据目标。
- 无跨时刻修正写回 Baseline recurrence。没有重建 Memory cache。
- 初始新规划残差为零；T1 占据残差保持源 checkpoint 的已训练值。

## 检查与执行

专项 CPU 测试验证 checkpoint 重算的值/梯度与 RNG、无候选边界、原规划-only兼容性、
双模块 optimizer 范围/学习率、checkpoint 基线隔离与 V2 更新、逐时刻结果保存。
真实 TRAIN index=64 检查包含三次 backward、两次临时更新；单独证明两类梯度路由、
融合上游梯度、Baseline 不变、两个模块更新。该检查不保存训练 checkpoint。
其日志及 acceptance.json 位于 deployment/joint_acceptance_v2.log、work_dirs/joint_acceptance_v2。
先前掩码缺失的失败日志保留供审计，已修复后重跑。

双卡性能检查使用前 40 次真实训练迭代；不保存 checkpoint，正式五种子重新从固定源启动。
正式队列依次完成训练→checkpoint审计→双卡推理→CPU评估，再进入下一个种子。
单作业最多使用每卡90%的PyTorch allocator预算。没有同时挤入两个联合训练作业。

运行控制器：`tools/run_memory_planning_joint_five_seeds.py` 的 prepare/check/run。
启动前 registry.json 固化 Python 源码、五份完整配置、源 checkpoint 和 Baseline 评估依据的 SHA256。
只读输入共享，代码和输出隔离；修改已登记代码会阻止队列混用版本。
正式会话名 `planning-joint-five-seeds`，可脱离 SSH 持续运行。

## 每个种子必须保留

- epoch_1.pth、训练日志、配置、两 rank 的训练计时与参数状态摘要。
- 4219 个唯一 token 的原生占据预测（uint8 无损）和轨迹增量累计位置。
- IoU/mIoU：0s、1s、2s、3s，以及仅 1/2/3s 的 Future Mean；原始 confusion matrices。
- L2 / Obj Collision / Box Collision：0.5s 到 3s 的六个时刻、1/2/3s 前缀均值和末端值。
  规划 evaluator 与 GT SHA256 必须与既有 Baseline 一致，独立记录规划有效样本数。
- checkpoint/预测/配置/评估器来源摘要和完成标记。没有完成的种子不填零、不算入统计。
- 总目录 results_by_horizon.json / .csv / statistics_by_horizon.csv 保留五次各自结果及均值、样本标准差(ddof=1)。

本实验衡量固定最好 T1 起点下一个 epoch 联合适配的种子差异，不能称为整个方法从头训练的方差。
不以训练loss变化声明 IoU/mIoU 或规划改善；以完整每种子评估为准。
