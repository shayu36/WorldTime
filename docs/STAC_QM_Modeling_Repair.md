# STAC-QM 建模修复与训练安全边界

本文只记录建模修复、cache schema 和训练安全边界。实验数值、debug 结论和当前未解决问题统一见 [EXPERIMENT_DEBUG_REPORT.md](EXPERIMENT_DEBUG_REPORT.md)。命令入口见 [STAC_QM_Implementation.md](STAC_QM_Implementation.md)。

## 修复目标

STAC-QM 的目标是在 SparseWorld trajectory forecasting 中引入 causal query memory，同时保证：

- Memory 只读取真实过去帧；
- cache 记录的语义、位置、可靠性和时间信息可审计；
- 新增 Memory 分支可训练；
- baseline 复现路径不因 Memory 训练发生隐式漂移；
- 空历史或无候选 batch 退化为数值恒等。

## 六项建模修复

### 1. 每个 query 只读取真实历史一次

Observation queries 在 SCF 前读取一次 Memory；每个 scheduled future query group 在被引入时读取一次 Memory。已经 active 的 query 不重复读取。

默认调度：

```text
observation queries: 720
future scheduled groups: [60, 60, 60, 60, 40, 40]
future offsets: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
```

旧 STAC-QM integration 的默认期望是 `7` 次 Memory calls 和 `1040` 个 fused queries。

### 2. Future-aware effective age

cache 中保留历史帧相对当前帧的基础时间差：

```text
base_age = current_timestamp - history_timestamp
```

future step 读取 Memory 时使用：

```text
effective_age = base_age + future_offset
```

causal filtering、max-age filtering、temporal attention penalty、motion compensation 和 diagnostics 均使用 `effective_age`。缓存中的原始 timestamp 不被改写。

### 3. 零初始化运动补偿

`QueryMotionCompensator` 根据 memory feature 和 time feature 预测有界速度：

```text
velocity = v_max * tanh(MLP([LN(memory_feature), time_features]))
aligned_points = ego_aligned_points + effective_age * velocity
```

最后一层初始化为零，使初始行为等价于仅做 ego-pose alignment。

### 4. Per-query semantic reliability

schema-v2 cache 为每个 selected query 保存：

```text
query_semantic_distribution [M, C_sem]
query_label                 [M]
query_margin                [M]
query_entropy               [M]
query_reliability           [M]
```

reliability 来自 top-1 probability、top-1/top-2 margin 和 normalized entropy 的组合，并 clamp 到 `[0, 1]`。schema-v1 仍可 fallback：

```text
query_reliability = query_conf
query_label = -1
```

formal Memory 实验应使用 schema-v2 cache。

### 5. 共享确定性 diversity selector

cache precompute、cache loading 和 online memory bank 共享同一个 `select_diverse_memory_queries(...)`。选择逻辑按 reliability 稳定排序，优先覆盖不同类别和空间 cell，再按容量补齐。

### 6. target-age history selection

默认 repaired config 使用：

```python
history_selection_mode = 'target_age'
history_target_ages = [2.5, 3.5, 4.5]
history_age_tolerance = 0.35
```

历史帧必须满足：

- strictly past；
- same scene；
- 每个 target slot 最多一个 frame；
- 同一 frame 不重复填多个 slot；
- 不能选择 cache 无法生成的 split 边界样本。

## Schema-v2 cache 校验

每条记录的关键字段：

```text
query_feat
query_points_metric
query_conf
query_semantic_distribution
query_label
query_margin
query_entropy
query_reliability
valid_mask
ego2global
timestamp
frame_idx
scene_id
sample_idx
pc_range
embed_dims
num_points
num_classes
source_config
source_checkpoint
schema_version
```

loader 必须检查：

- 必要字段存在；
- feature、points、label、semantic distribution、reliability、time 字段 shape 一致；
- semantic distribution 类别维度合法；
- 数值 finite；
- probability 非负且行和有效；
- reliability/margin 范围合法；
- scene 和时间因果关系合法。

缺字段或 shape 错误必须显式报错，不能静默 fallback。schema-v1 fallback 只用于兼容旧记录，不作为 formal 主路径。

## 训练安全边界

### Memory-only 旧路径

旧 Memory-only mode 使用：

```python
query_memory_cfg = dict(
    enabled=True,
    source='cache',
    memory_finetune_mode=True,
    freeze_base_model=True,
)
```

只允许 `query_memory.*` 参数训练。base model、OPUS head、TASS assignment 和 frozen buffers 必须保持不变。

### Joint 旧路径

joint finetune 曾用于验证扩大训练边界是否能让 Memory 信号进入 future head。它属于历史实验入口，不是当前主线。

### Future Memory Adapter 当前路径

Future Adapter mode 使用独立模块：

```python
future_memory_adapter_enabled=True
future_memory_adapter_finetune_mode=True
future_memory_target_horizons=[1.0, 2.0, 3.0]
```

安全边界：

- baseline image backbone/neck 冻结；
- 原 OPUS/SparseWorld head 冻结；
- 原始 query encoding、future recurrence、classification/regression branches 冻结；
- 只训练 `future_memory_adapter.*`；
- Memory corrected logits、points、features 不写回 baseline future recurrence；
- 0s 和非目标 future step 不走 Future Adapter。

## Future Adapter 建模不变量

当前 Future Adapter 修复旧 Phase3/Phase4 拓扑中的跨时刻 logits 累加问题。每个目标 horizon 独立计算：

```text
S_base(t) = ClsBranch(Q_base(t))
S_final(t) = S_base(t) + G(t) * DeltaS_memory(t) + G(t) * DeltaO_memory(t)
P_final(t) = SafeRefine(P_base(t), G(t) * DeltaP_memory(t))
```

禁止：

- `S(t) = S_final(t-1) + ClsBranch(Q(t))`；
- 把 1s corrected logits 传给 2s；
- 把 2s corrected logits 传给 3s；
- 用 Memory correction 改写下一步 baseline recurrence state。

目标 horizon 映射：

| horizon | internal future step | output key |
| ---: | ---: | --- |
| 1s | 2 | `semantic_occ_2s` |
| 2s | 4 | `semantic_occ_4s` |
| 3s | 6 | `semantic_occ_6s` |

active query 数由实际 schedule 张量得到，默认验收为 `840 / 960 / 1040`。

## 初始化策略

新增 Memory 路径需要同时满足可训练性和 baseline 等价：

- q/k/v、memory projection、semantic projection 使用正常非零初始化；
- gate bias 约为 `-1`；
- semantic/occupancy/position residual head 最后一层权重和 bias 初始化为零；
- 不使用 `1e-3` 级全局 alpha；
- 无有效 Memory candidate 时 gate 强制为零，输出严格退化为 baseline。

## 验收原则

代码级验收应覆盖：

- zero-init baseline identity；
- no-candidate baseline fallback；
- horizon embedding 在 attention query 前生效；
- memory feature、semantic distribution、label、reliability、age、relative position 对候选分数或输出具有因果影响；
- baseline 参数无梯度；
- residual head 和上游 adapter 参数在合成 backward/微型 optimizer step 中可获得梯度；
- cache schema/collate 字段不丢失。

正式 IoU/mIoU 结论必须等 formal training 和完整 validation 完成后再写入实验汇总。
