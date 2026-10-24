# 第一轮：T1 + SoftVoxel-center 的 Adapter 内部时序通路

## 范围与起点

本轮只落实独立代码/配置及静态、CPU 前向验收，不启动训练、验证集评估、
反向传播或参数更新。历史最佳归档 `now-best-before-check@c562513` 不修改；
使用其本地 T1 epoch 3 权重作为两组实验共同起点：

`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth`

已有 IoU/mIoU 数值属于历史模型，不是本实验的结果。当前工作区的 Memory
因果输入修复仍保留。本实验配置沿用 T1 的旧输入/cache 配置，不自动切到
`t1-memory-causal`，不生成 cache，也不能据此宣称输入已完全满足因果协议。
必须让对照组与实验组使用同一代码版本、数据和 cache；若改用修复协议，应
同时修改两组的共同配置并重新建立该协议的起点，不能跨协议直接归因。

## 要检验的假设

2s/3s Adapter 能否利用较早目标时刻已经提取的信息，提高长时刻语义质量，
而不改写 Baseline 递推或规划输入。第一轮不是全模型状态反馈，也不累加
各时刻的 semantic/occupancy logits 或位置残差。

```
Baseline: state(1s) -> state(1.5s) -> state(2s) -> ... -> state(3s)
              |                        |                  |
Adapter:     A1 -> detached H1 --------> A2 -> detached H2 -> A3
              |                        |                  |
           output(1s)               output(2s)          output(3s)
```

仅新增 Adapter -> Adapter 通路。0s、0.5s、1.5s、2.5s 保持原路径。
对固定权重，修改新时序通路不改变 1s 输出；但训练会更新共享 V2 参数，
所以续训后的 1s 质量仍需评估，不能承诺训练后与起点逐位相同。

## 状态、对齐与更新

原 Point Attention 的 context 为 `[B,Q,R,C]`，完全保留。令原 Point Adapter
MLP 输出为 `A_t[b,q,r,:]`，形状同上。新增隐状态 **不是**这份 context 的全局
池化，而是按 Query 分别对有有效点级 Memory 支持的点汇总：

```
H_t[b,q] = mean_{r: has_candidate[b,q,r]} A'_t[b,q,r]
```

每条状态包含：

- `features`: `[B,Q,C]`；点维用 FP32 累加，再转回原 dtype。
- `valid`: `[B,Q]`；该 Query 至少有一个当前有效点级候选。
- `query_ids`: `[Q]`，原 OPUS Query 槽位索引，同 batch 使用同一调度。
- `horizon_id`: 0/1/2，只允许读取紧邻的前一目标时刻。

递推函数维护实际 Query ID：从 `ind_stamps_all==0` 的原始索引开始，随各
内部时刻追加 scheduled Query 的原始索引。用 sort/searchsorted 对齐前一
时刻状态，而不是假设当前张量前 Qprev 项一定对应，或混合 Batch 维。
新加入、身份未匹配或上一时刻无有效支持的 Query 得到零历史状态。

48 个点不是经验证的固定物体/轨迹身份。本轮不做 r 对 r 的跨时刻匹配，
Query 内状态汇总对点顺序置换不敏感。该策略丢失部分点级空间细节，也可能
混合一个 Query 内不同类别的信息；这是待实验检验的限制，不能预设收益。
原逐点检索、语义分歧特征、三个任务 Gate 仍保留，不退化为 Query 级预测。

2s/3s 的新增读取公式：

```
Z_q = LN(stop_gradient(H_previous[matched_query_id]))
D_q = max_delta * tanh(W_output GELU(W_value Z_q))
g_qr = sigmoid(W_gate [A_t, Z_q] + b_gate)
g_qr = g_qr * current_point_has_candidate * previous_query_valid
A'_t = A_t + g_qr * D_q
```

`g` 为 `[B,Q,R,1]`，`D` 为 `[B,Q,C]`，按自己的 Query 广播到点。
原 residual heads 和 A+B task/horizon gates 从 `A'_t` 产生输出。
返回的 `point_context` 仍只代表原历史 Memory 读取，不把新的隐状态伪装进去。

1s 不读取前序隐状态，只有产生 H1 的操作。H2 汇总融合后的 A'2，因此可以
在前向上把 H1 的信息传到 3s。无当前有效 Memory 时，即使有上一步状态也
不能启用 Gate；该 Query 的新状态同时清零，不穿过无支持时刻延续旧状态。
现有无 Memory 回退、effective age 和 base-age 防御检查保留。

## 初始化、梯度与生命周期

- `temporal_adapter_enabled=False` 是模块默认值，不新增旧模型参数键或改变
  旧推理路径。V1 完全不进入此分支。
- 启用时新参数全部在 `future_memory_adapter_v2.temporal_read.*` 下，仍受
  已有 V2-only 冻结/优化器白名单约束。
- 仅 W_output 零初始化；其他投影正常初始化，门控 bias 为 -2。这样迁移
  **训练过、非零残差的 T1** 也能保持初始输出恒等，而不只是保持 Baseline。
- bias=-2 在其他门控输入为零时对应 sigmoid≈0.119，并不意味着所有点实际
  初始 Gate 都等于该值。`max_delta=1.0` 限制特征更新每维幅度，是特征单位，
  不是米或 logits 阈值；真正初始更新严格为零。
- 新模块初始化隔离 CPU RNG，不改变匹配对照后续初始化/Dropout 随机流。
- 首轮固定使用跨 horizon stop-gradient，保留本时刻融合层的普通可训练
  计算路径；不做跨 1s/2s/3s 的完整 BPTT。**本轮没有验证实际梯度**。
- 隐状态是一次 `_forward_backbone_future_memory_adapter` 调用内的局部变量，
  不挂在 model 属性上，不跨 sample、batch 或 scene 持久化，不写 cache。
- activation checkpoint wrapper 显式返回状态及小型统计，并在每次调用内部
  绑定自己的输入状态，避免反向重计算捕获后来更新的循环变量。

Baseline 的 `state_feat/state_pos`、轨迹分支输入与 freeze/eval 策略均不变。
新隐状态扰动可改变后续 **Adapter** 输出，但不得改变后续 Baseline 快照、
非目标时刻输出和 `pred_trajs_list`。

## 独立配置与公平对照

| 项目 | 对照 | 实验 |
| --- | --- | --- |
| 配置尾缀 | `v2-t1-temporal-control.py` | `v2-t1-temporal.py` |
| 输出目录 | `work_dirs/t1-temporal-control-from-best3` | `work_dirs/t1-temporal-from-best3` |
| 起点 | T1 epoch 3 | 同一 checkpoint |
| 新通路 | 关闭 | 开启，零贡献初始化 |
| 优化器 | 新 AdamW | 新 AdamW |
| 预算 | 3 个本地 epoch | 3 个本地 epoch |
| LR / warmup | fixed 2e-5 / 100 iter | 相同 |
| seed / q chunk / AMP | 17 / 48 / dynamic FP16 | 相同 |
| 数据、所有损失、A+B、SoftVoxel-center、T1 beta | 不变 | 相同 |

两组都使用 `load_from`，`resume_from=None, auto_resume=False`。新增参数后
不能直接恢复旧优化器的参数组；不把旧 optimizer moments 只保留给对照。
epoch_1/2/3 是新的本地编号，不覆盖源 checkpoint。三个 epoch 是首轮观察
窗口而非收敛保证；最终判断对照各时刻曲线，不只挑实验组最好的单个 epoch。

CPU 非写入检查入口（本身不会训练）：

```bash
/data/jxy/projects/env/bin/python /data/jxy/projects/tools/check_future_memory_temporal.py
```

检查源权重必须完整匹配旧 V2/T1 参数，只允许新增 temporal_read 参数缺失；
随后用真实已训练权重、小型 B=2 合成张量检查三个 horizon 初始输出严格一致。
它不保存迁移 checkpoint、不创建优化器、不构建完整 Baseline，也不启动 GPU。

训练仍使用现有 tools/train.py；本轮没有新增自动执行训练的入口，也没有
启动任何一个实验。后续启动前应先做单独的梯度与实际训练显存/smoke 验收，
确认所选输入协议和两个独立 work-dir 尚未有结果，再使用同样的 DDP 参数。

## 日志与资源

每个目标时刻有三个 detached、非 loss 日志（包括 checkpoint 训练路径）：

- `mem_*s.temporal_previous_query_fraction`：同时有当前支持及匹配前态的 Query 比例。
- `mem_*s.temporal_gate_mean`：有当前点支持和有效前态的点上的平均新 Gate。
- `mem_*s.temporal_update_abs`：所有当前点/特征上的平均绝对特征更新，含无效零项。

空 Memory 返回相同日志键的零值，避免 DDP rank 日志集合不一致。1s 更新值
恒为零；零初始化时三个 horizon 的更新均为零。评估 diagnostics 另含逐点
Gate 和按 Query 的 previous-valid mask，可定位“门开了但实际更新仍很小”。

默认 C=256 新增 **132,353** 参数，V2 总计 **1,630,498** 参数；不新增
全局点对点注意力。状态在 Q=1040、B=1、FP32 下约 1.02 MiB，B=2 翻倍。
传递中的几份状态/快照会同时存在，不能把单状态大小当作总峰值。
新增融合仍按 q48 分块：主要逐点门输入 `[B,48,48,512]`，FP32 约
4.5 MiB/B/chunk；加上投影、激活和优化器状态有额外开销。原 Point Attention
和 Baseline 内存需求未消失，CPU 检查不能证明 24GB GPU 训练一定可运行。

评估应同时比较 1/2/3s IoU/mIoU、重点类别召回、上述 Gate/更新统计，并核对
0s/非目标输出与规划路径的隔离。模型结构保持规划不回写，不等于已经取得
该 checkpoint 的正式 L2/碰撞率结果。没有本实验 IoU/mIoU 提升结论。

## 本轮验收

新增测试文件 `tests/test_future_memory_adapter_temporal.py` 为纯前向/结构检查：

- 非零 T1 残差下初始恒等；旧配置 keys/RNG 不变。
- B=1/B=2、Query 数增加、ID 重排、新 Query 零历史、不混合 Batch。
- 点池化置换不变、无效点忽略、FP16 汇总与 CPU BF16 autocast 有限。
- 无 Memory、无效/过期/半径外 Memory 回退，状态清零，局部点 mask。
- 开启通路后隐状态影响后续 Adapter，而 logits/位置输出本身不被累加。
- Baseline/规划/0s/非目标隔离、跨前向重置、chunk 一致性。
- checkpoint 开关的训练模式纯前向结果、固定日志键一致。
- 完整 3s 循环结束后逆序重放 checkpoint 回调，确认各自绑定原来的 horizon
  和前态；这是显式前向重放，不是反向重计算或梯度验收。
- 实际冻结策略/优化器成员检查（仅创建优化器，未更新），源权重缺失拒绝。
- 两组配置除新通路外的数据、损失、学习率和预算一致。

未做梯度验收、训练 smoke、正式训练、完整评估、checkpoint 写入、cache
重建或 Git 提交/推送。

### 实际执行记录（2026-10-22）

在 `/data/jxy/projects` 执行下列显式前向测试选择，结果 **119 passed**，
6.93 秒；19 个 warning 为已有依赖的弃用通知。新增文件内 22 项测试全部通过。
没有运行 AB 的两个求导测试、SoftVoxel 的位置梯度测试或 V1 优化测试。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. env/bin/pytest -q \
  tests/test_future_memory_adapter_temporal.py \
  tests/test_future_memory_adapter_v2.py \
  tests/test_future_memory_occ_class_balance.py \
  tests/test_future_memory_softvoxel_center.py::test_centre_is_optimal_and_neighbours_symmetric \
  tests/test_future_memory_softvoxel_center.py::test_boundary_stays_finite_and_out_of_bounds_safe \
  tests/test_future_memory_adapter_ab.py::test_migration_preserves_nonzero_outputs_all_horizons \
  tests/test_future_memory_adapter_ab.py::test_horizon_and_task_gate_isolation \
  tests/test_future_memory_adapter_ab.py::test_no_candidate_gates_and_outputs_are_exact_baseline \
  tests/test_future_memory_adapter_ab.py::test_decode_independent_occupancy_gate \
  tests/test_future_memory_adapter.py::test_zero_init_and_no_candidate_are_strict_baseline_identity \
  tests/test_future_memory_adapter.py::test_active_queries_and_corrected_logits_never_feed_baseline_state
```

`OMP_NUM_THREADS=1 env/bin/python tools/check_future_memory_temporal.py`：PASS。
实际读取上述 T1 epoch 3，旧 Adapter 参数覆盖完整；新增权重为零贡献，
B=2、三个 horizon 的已训练残差及 gates 严格相等。

六个本轮 Python 文件的 `python -m py_compile`：PASS。
本轮修改文件的 `git diff --check`：PASS。全仓默认检查存在用户已有
`nuscenes_dataset_occ_trajectory.py` 的 7 处 CRLF 行尾告警，本轮未改该文件。
允许 CRLF 的 `git -c core.whitespace=cr-at-eol diff --check`：PASS。
初始/最终 HEAD 均为 `c562513beb0ba2f919c62d524f468d58ddb3bf85`，未提交。

这只证明静态/前向与白名单接线。完整模型 build、真实 GPU AMP/显存、
checkpoint 反向重计算、梯度有效性和训练效果仍需后续单独验收。

后续实际训练与完整评估的补充记录见
[T1 时序 Adapter 结果补充材料](FUTURE_MEMORY_T1_TEMPORAL_RESULTS_SUPPLEMENT.md)。
