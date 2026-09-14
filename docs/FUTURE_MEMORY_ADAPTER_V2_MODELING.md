# Future Memory Adapter V2 建模说明

## 1. V1 基线与保留策略

V1 的正式结果保持原样并继续可复现：epoch 4 的 Future Mean mIoU 为
13.2500（相对 Baseline +0.0267），epoch 6 的 Future Mean IoU 为 22.2467
（相对 Baseline +0.0333）。`FutureMemoryAdapter`、其配置和 checkpoint 路径
没有改名或覆盖；V2 通过独立的 `FutureMemoryAdapterV2` 类和独立配置启用。

本阶段只做代码级验收，没有 V2 正式训练或 NuScenes 完整验证，因此没有新的
IoU/mIoU 结论。

## 2. V2 解决的问题

V2 修复五个相互关联的问题：候选支持统计量不再被 top-k 数量错误缩小；预测
时刻使用未来有效年龄；Query 粗检索后在 Query 内做 Point 检索；语义匹配受
不确定性、双路候选和 Semantic Dropout 约束，避免 Baseline 错误确认偏差；
训练同时监督 semantic、occupancy、阈值边界和稀疏可微软体素目标，使训练信号
更接近最终 IoU/mIoU 解码流程。

## 3. Query → Point 两阶段数据流

对每个目标时刻（1s、2s、3s）读取 Baseline 快照 `Q_base/P_base/S_base`：

1. Query 粗检索计算 feature/geometry 路径和 semantic 增强路径，分别取
   `coarse_topk_geometry=8`、`coarse_topk_semantic=8`，稳定去重后形成并集。
2. 对每个当前 Query，只展开并集中的历史 Query 及其 48 个历史点；历史
   Query 语义分布广播到点，但每个历史点保留自己的空间位置编码。
3. 在 `query_chunk_size=64` 的块内计算 Point Attention，取最多
   `point_topk=8`，得到 `H_qr`：`[B,Q,R,C]`，以及 `G_qr`：`[B,Q,R,1]`。
4. Point Adapter 输入包含 Query、当前点位置编码、Context、Query-Context
   差异、horizon embedding、候选数、可靠性、有效年龄、距离、不确定性和
   当前/历史语义分歧，输出 semantic/occupancy/position residual 和 Point Gate。

默认规模的粗分数最大形状为 `[B,Q,768]`（最多 3×256 历史 Query），Point
Attention 最大形状为 `[B,64,48,8,768]`（64 个当前 Query、48 点、8 个 head、
最多 16 个粗候选×48 历史点）。不存在全部当前点×全部历史点的全局矩阵。

## 4. 不确定性感知语义匹配与双路候选

当前每个点使用 `softmax(S_base)`。归一化熵为 `U_qr`，top1-top2 margin 同时
记录；显式语义权重为 `(1-U_qr)^gamma`，并由 `semantic_weight_floor` 限制
下界。高熵点会降低语义匹配，但不会删除空间/特征合理的 Memory。

粗检索固定包含两条路径：

```text
A_geometry = A_feature - A_distance - A_effective_age + A_reliability
A_semantic = A_geometry + w_sem * A_semantic_compatibility
M_q = stable_unique(TopK(A_geometry) ∪ TopK(A_semantic))
```

geometry 路径始终保留，显式语义错误时仍可进入候选并集。训练模式可按
`semantic_dropout_probability` 将当前语义替换为均匀分布；evaluation 严格关闭。
Point Adapter 额外接收 JS divergence（当前点分布与历史分布的对称分歧），
因此语义不一致的近邻 Memory 有机会用于纠错，而不是在检索阶段硬删除。

## 5. Effective age

缓存只保存基础年龄 `base_age = t_current - t_history`。V2 对每个目标时刻创建
新张量，不修改 cache：

```text
effective_age = base_age + horizon_seconds
1s → +1.0s, 2s → +2.0s, 3s → +3.0s
```

最大年龄筛选、时间分数、support age、Adapter diagnostics 都使用
`effective_age`；diagnostics 同时保留 `base_age`、`effective_age` 和
`horizon_seconds`。

候选统计使用真正的权重质量：

```python
valid_weights = weights * top_valid.to(weights.dtype)
weight_mass = valid_weights.sum(-1).clamp_min(EPS)
support = (valid_weights * selected).sum(-1) / weight_mass
```

不再除以有效候选数量，部分无效候选、dropout、无候选和多 head 汇总都保持有限。

## 6. Semantic / occupancy 解耦

V2 产生：

```text
S_sem = S_base + G * DeltaS
O     = max_c(S_base) + G * DeltaO
S_decode = S_sem + (O - max_c(S_sem))
```

因此 `argmax(S_decode) == argmax(S_sem)` 且 `max(S_decode) == O`。类别排序只由
semantic 分支决定，是否越过类别阈值只由 occupancy 分支决定；现有 `get_occ()`
仍可使用 `S_decode`。零残差或无有效 Memory 时严格退化为 Baseline。

## 7. 新损失

每个目标时刻只增加以下损失，权重全部在 V2 配置中：

```text
mem_1s.loss_base_cls (1.0)
mem_1s.loss_sem       (0.25)
mem_1s.loss_occ       (0.25)
mem_1s.loss_threshold (0.10)
mem_1s.loss_soft_voxel(0.10)
mem_1s.loss_pts       (0.50)
```

`mem_2s.*`、`mem_3s.*` 同构。原始 future 分类损失作为 `loss_base_cls` 稳定锚点
保留，原始 point matching 作为 `loss_pts` 保留。Occupancy 使用由 0.4m voxel size 推导的正样本
半径（默认 0.20m）和 BCE-with-logits；semantic 只在可靠占据匹配点上使用现有
类别权重；threshold loss 从 `pts_bbox_head.test_cfg.score_thr` 动态转换为
logit 阈值。SoftVoxel 使用每个点周围 8 个连续坐标体素的三线性权重和 noisy-OR，
只在预测体素与 GT 非空体素的稀疏并集上计算 Soft-IoU，保持对 semantic logits、
occupancy logits 和连续点位置的梯度。

0s 没有新增 Memory loss，0.5s、1.5s、2.5s 也不接入 V2 loss。

## 8. Recurrence 隔离与冻结边界

V2 只读取每个目标时刻的 Baseline 快照。Context、gate、修正 logits、occupancy
logits、修正点和修正 query feature 都不会写回 OPUS/SparseWorld future recurrence；
改变 1s 输出不会改变 2s Baseline state，改变 2s 输出不会改变 3s state。

V2 optimizer 白名单只有：

```text
future_memory_adapter_v2.*
```

Baseline、原始 query encoding、OPUS head、recurrence、原始分类/回归分支均冻结，
冻结模块保持 eval mode。V2 q/k/v、位置和语义投影使用正常 Xavier 初始化，gate
bias 为 -1，三个 residual head 的最后一层零初始化；没有 `1e-3` 全局缩放。
