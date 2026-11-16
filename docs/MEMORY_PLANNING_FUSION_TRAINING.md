# 用已有轨迹监督训练 Memory 规划融合模块

2026-11-16：完成独立模块、接线、配置、CPU 单测及单训练样本 GPU 梯度验收。
下述梯度验收阶段没有启动正式训练、200-iteration smoke、完整评估或多卡任务。
随后用户授权的一轮训练启动记录见文末；验收数据与训练数据分开记录。

## 目标与原有模型的关系

此前 T1 原生规划路径不消费 Memory 修正，规划头又被冻结，因此虽然训练代码
保留 `loss_traj`，该损失无法训练 Memory。此前两轮推理连接试验只验证了
固定连接规则，不是规划训练。本轮新增可训练融合模块，复用现有轨迹监督。

独立模型类 `SparseWorld4DTrajMemoryPlanning`，独立配置：
`configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-planning-fusion.py`。
原 V1/T1 配置、参数 key、默认前向行为不变。原基类只增加一个默认关闭的
快照导出分支；主工作区已有的内部消融改动保留。

初始化来源为最佳 T1：
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth`。
`init_weights()` 严格加载所有原参数，只允许新的 `memory_planning_fusion.*`
缺失；原参数缺失、多余 key、shape 错误都会失败。原 checkpoint 不写入。

## 可训练和冻结范围

- 唯一白名单：`memory_planning_fusion.*`，19 个参数 Tensor、224,002 个标量。
- Baseline、原规划头和整个 T1 Memory 模块均 `requires_grad=False`、eval mode。
- 冻结前向在 `torch.no_grad()` 内；新融合模块在该作用域之外执行。
- 冻结快照 detach 是有意的；**融合模块输出和轨迹损失之间不 detach**。
- 优化器复用 `TrainableOnlyOptimizerConstructor`，只包含白名单参数。

新配置从 T1 的权重开始，使用新的 AdamW 状态，不恢复 occupancy 训练优化器。
lr=1e-4、weight_decay=1e-2 是新小模块的初始试验设置，不是经过验证的最优值。
配置的 max_epochs=1 是初筛预算；完成的训练和评估记录见文末。

## 融合输入和计算

在 1/2/3s 获取各自冻结 T1 快照。对每个 Query，仅用有点级 Memory 候选的点
做加权均值，padding/无候选点不参与。每个 Query 得到一个 token：

| 输入 | 维数 |
| --- | ---: |
| Baseline Query feature、Point Memory Context | 2C |
| Baseline 逐点语义概率、修正语义概率及差值的汇总 | 3K |
| 原/新占据概率与占据 logit 差值 | 3 |
| 原/新位置中心、位置位移 | 9 |
| Semantic / Occupancy / Position gate 均值 | 3 |
| 有候选点比例 | 1 |

总宽度 `2C+3K+16`，当前 C=256、K=17，即 579。R、Q 不写死。
位置中心使用原模型线性归一化坐标；位移按 pc_range 转为米，再除以配置的
0.4m voxel size，以体素为单位。语义输入使用 `S_sem`，不用 `S_decode`。
既有 task gate 已经用于输出修正，不再重复乘到语义/占据/位置残差上。

```
冻结 T1 → 每 Query 579维 token → 可训练 token encoder → 128维 tokens
冻结 ego feature + 可训练 horizon embedding → 规划 query
规划 query → 4头 Attention 读取 tokens → 轨迹残差头 → [B,1,2] XY增量
```

128 维、4 heads 为低开销的初始设计，约 22.4 万新参数。没有重做全部点×点
注意力；T1 内部仍按原逻辑检索。Fusion 自身使用 `[B,1,Q+1]` 的 ego-to-Query
读取，多出的零 null token 防止全无效候选导致 NaN。

最终残差层权重和 bias 零初始化，其余编码/Attention 正常初始化：

```
delta_xy = fusion(frozen_memory_evidence, frozen_ego, horizon)
pred_traj_step = native_pred_traj_step + delta_xy
```

无 Memory 的样本强制 delta=0，但保留 autograd 图，避免 DDP 的使用参数集合
因空候选而变化。0.5/1.5/2.5s 的原增量保持不变；六步按原导出流程 cumsum
一次得到累计坐标。新轨迹不反馈到场景 recurrence，occupancy 输出保持原 T1。

## 使用哪个 loss

`forward_train` 调用新 `forward_backbone` 得到融合后的 `pred_trajs_list`，随后
逐步调用原 `SparseWorld4DTraj.loss_traj`：

```
sum_xy mean_batch((pred_increment - temporal_trajs_increment)^2)
```

GT 直接沿用原 `temporal_trajs` 张量和原坐标约定，不自行再次 cumsum。GT 只进入
loss，不进入融合 token、Attention 或 ego query。冻结 occupancy 不计算其训练
loss，避免不参与本轮优化的昂贵点匹配/SoftVoxel 计算。

保留原日志键 `loss_traj_1s` ... `loss_traj_6s`，其中数字是原代码的六个内部步
编号，物理时刻仍为 0.5/1/1.5/2/2.5/3s；并非预测到了 6 秒。
本轮没有增加新的碰撞 loss 或改变原 loss 权重。以后必须继续同时评估 L2 和
碰撞率，梯度连通不等于规划指标改善。

## 验收结果

命令：

```
env/bin/python -m pytest -q tests/test_trainable_memory_planning.py tests/test_full_memory_planning_connection.py tests/test_memory_planning_connection.py
env/bin/python -m py_compile mmdet3d/models/sparsedetectors/memory_planning_fusion.py mmdet3d/models/sparsedetectors/sparseworld_memory_planning.py tools/check_memory_planning_gradients.py
git diff --check
```

34 passed。覆盖零初始化恒等、B=1/B=2、可变 Q/R/K、无候选安全退化、无效 token
隔离、三种信息输入列的梯度、真实 recurrence/loss 方法接线、GT 不参与输入、
严格 checkpoint 缺失检查以及此前两种只读推理连接兼容性。CPU 测试不执行
optimizer step；非零残差头测试用显式参数赋值检验后续梯度。

真实 GPU 验收命令（本轮已执行一次，不是训练启动命令）：

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 env/bin/python -u tools/check_memory_planning_gradients.py --index 64 --output work_dirs/memory-planning-gradient-20261116-kq117oik/acceptance
```

仅加载 **训练集 index 64 一个样本**，走真实 `forward_train`，随后复用该样本的
冻结快照进行第二次反向传播。总计两次 backward、一次微型 optimizer step，
无 runner/epoch 循环、无 validation；更新只存在于进程内，未保存 checkpoint。

| 模块 | 第一次 backward 梯度范数之和 | 第二次 backward 梯度范数之和 |
| --- | ---: | ---: |
| token_encoder | 0 | 0.000478059 |
| ego_encoder | 0 | 0.001640469 |
| horizon_embedding | 0 | 0.000000381 |
| attention | 0 | 0.000327241 |
| residual_head | 0.658105105 | 0.630373964 |

第一次上游为零是最后一层零初始化的预期结果；经过唯一的一次微型更新，
第二次反向传播在所有上游组都有非零有限梯度。原有模块所有参数始终 grad=None、
requires_grad=False、eval mode；冻结参数及 state_dict buffer 哈希逐项不变。
初始六步轨迹与原生 Baseline 严格相同。

实际 Query 数量 `[840,960,1040]`，有候选 Query 数量 `[754,851,898]`。
峰值 allocated 显存 2013.86 MiB，总验收耗时 30.10 秒。该峰值属于单样本和
第二次复用快照的检查，不是多卡正式训练显存保证。
同一样本 loss 从 0.111520842 变为 0.111125037，仅是优化接线检查，不能当作
验证集收益或收敛结果。

机器记录：
`work_dirs/memory-planning-gradient-20261116-kq117oik/acceptance/gradient_acceptance.json`。

## 梯度验收阶段尚未进行的事项

连续迭代 smoke、正式规划适配训练、多卡运行、训练后 4219 帧评估均未执行。
因此本轮结论仅为“已有轨迹监督能训练新增融合模块，且冻结边界验证通过”。
没有修改原 checkpoint、重建 cache、提交或推送；没有声称 L2/碰撞率或 IoU 提升。

## 后续训练启动：2026-11-16 02:00 UTC

用户授权“启动训练”后，在 `tmux memory-planning-train` 启动独立规划融合训练。
双卡 RTX 3090，每卡 batch=1、worker=1，global batch=2；沿用上述 AdamW、
动态 FP16 和 100 iteration warmup，1 epoch 后自动结束。只有融合模块可训练，
没有打开 validation，也没有恢复旧的 occupancy 优化器状态。

工作目录：`work_dirs/memory-planning-fusion-from-t1-e3`。
目录内 `launch.sh` 是实际启动命令，`launch_manifest.json` 记录参数、源 checkpoint
SHA256 和 HEAD，`source_snapshot` 保存启动时的配置与相关实现，`console.log`
保存日志。源 T1 checkpoint SHA256：
`bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`。

本节只记录训练启动，不代表完成该 epoch 或获得新的规划评估结果。

## 训练完成与规划评估启动：2026-11-16

训练于 UTC 04:24 完成 9865 次迭代，进程 exit code=0，保存
`work_dirs/memory-planning-fusion-from-t1-e3/epoch_1.pth`。
保存后逐 Tensor 对比源 T1 的 766 项 state dict，全部严格相同；新融合模块
残差头已非零，参数及训练日志中的 loss/grad_norm 均有限。
检查记录见同目录 `completion_check.json`。

用户要求评估后，在 `tmux memory-planning-eval` 启动 epoch_1 原生推理。
使用未改动的 `tools/audit_planning_pair.py predict/metrics`；主模型为新规划
融合类，严格加载本次 checkpoint，未安装此前两种固定规则的推理连接。
评估覆盖配置中的完整验证 token 集合，保存逐样本输入摘要、轨迹与 provenance。
自动与既有 Baseline 的 token/输入摘要、TASS、指标代码和 GT 哈希核对后，
生成前缀平均主指标及末端值对照。

评估输出：同训练目录的 `eval_epoch_1/`；任务脚本、日志、阶段与退出码：
`eval_epoch_1_job/`。依次执行导出、计分、对比；全部通过后 stage=complete。
本条为启动记录，未预填任何 L2 或碰撞率结果。

## Epoch 1 完整评估结果

随后已完成 4219 帧原生推理、计分和逐 token 输入核对，任务 exit code=0。
验证 token 顺序、图像/ego 输入摘要、TASS、GT 和指标源码哈希均与 Baseline 一致。
主表沿用 Baseline 的 1/2/3s 前缀平均口径；不把末端指标混入主表。

| 指标 | 1s | 2s | 3s |
| --- | ---: | ---: | ---: |
| Baseline L2 (m) | 0.15959594 | 0.19454673 | 0.23934847 |
| 规划融合 epoch1 L2 (m) | 0.15968642 | 0.19473332 | 0.23966271 |
| L2 差值 (m) | +0.00009048 | +0.00018659 | +0.00031424 |
| Box Collision (%)，两者相同 | 0.09480920 | 0.10073477 | 0.11061073 |
| Obj Collision (%)，两者相同 | 0 | 0 | 0 |

平均 L2 从 0.1978303790 到 0.1980274816m；本轮没有规划指标提升。
3470/4219 条轨迹变化，说明训练后的融合模块确实改变规划输出。
碰撞指标使用实际原 ST-P3 代码及 GT-box 碰撞掩码，不代表不带掩码的真实世界
事故率。Checkpoint SHA256：
`e5626f31d566c3090a0b7ad381e558a9ddcadfb7a1ffcba02172a9f0e29d840e`。

机器记录：`work_dirs/memory-planning-fusion-from-t1-e3/eval_epoch_1/` 下的
`metrics.json`、`comparison.json`、`provenance.json` 及 `trajectories.pkl`。
权重、数据和运行目录不提交到 Git。已将验收、训练完成检查和评估机器记录
汇总为 `docs/validation/memory_planning_fusion_20261116.json`，供核对来源。

在具备同样本地数据/依赖的项目根目录，可使用同一导出与指标入口复算：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 env/bin/python tools/audit_planning_pair.py predict --source-root . --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-planning-fusion.py --checkpoint work_dirs/memory-planning-fusion-from-t1-e3/epoch_1.pth --output work_dirs/memory-planning-fusion-from-t1-e3/eval_reproduce
env/bin/python tools/audit_planning_pair.py metrics --metric-root . --prediction work_dirs/memory-planning-fusion-from-t1-e3/eval_reproduce/trajectories.pkl --output work_dirs/memory-planning-fusion-from-t1-e3/eval_reproduce/metrics.json
```

`eval_reproduce` 必须是新目录。仅运行第一个命令不会自动计分。
测试辅助保留了此前两种固定规则推理连接的代码，用于回归测试；新的模型类、
配置和上述评估命令均不调用这些连接器。

## 发布分支检查

`规划监督` 分支基于 c562513，未纳入主工作区已有的 RecallOcc 消融改动。
发布版本重新运行上述 34 个测试全部通过；严格加载训练后的 epoch1，并在
验证集前 4 个样本上检查图像/ego 输入一致，4 条轨迹与完整评估保存值均
`np.array_equal`，最大绝对差为 0。这里仅声明 4 样本发布检查，不重复声称
已在发布工作区执行一遍全量评估。
