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
Point Adapter 接收完整的当前/历史逐点语义概率、语义差向量、余弦相似度和
JS divergence。因此语义不一致的近邻 Memory 有机会用于纠错，而不是在检索
阶段硬删除。

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

## 9. 候选隔离与候选内粗 Context

早期 V2 曾将全部 Memory Query 的 coarse_k.mean 与 coarse_v.mean 合成
coarse_summary，并注入所有当前 Query。这会绕过 memory_valid、effective age、
最大年龄、空间半径和 Top-K 并集，现已彻底删除。

现在每个 Query 的粗读取严格限定为：

    U_q = stable_unique(TopK(A_geometry) union TopK(A_semantic))
    alpha_qm = MaskedSoftmax(A_semantic[q,m]), m in U_q
    H_coarse[q] = sum(m in U_q) alpha_qm * V_m

H_coarse 的 shape 为 [B,Q,C]。重复入选的候选只保留第一次出现的位置；原始连续
候选分数在并集内部参与 softmax。全部候选无效时 alpha 与 H_coarse 严格为零。
未选中、memory_valid=False、effective age 超限或粗半径外的 Memory 不会进入
粗 Context，也不会展开到 Point Attention。当前点自身的 qpoint 仅作为 Point
Attention 的 Query 条件，不会累加进历史 Point Memory Context。

## 10. 完整逐点语义分歧特征

Semantic Dropout 只修改检索条件 P_retrieval 和 w_retrieval，不修改 Baseline
logits，也不替换 Adapter 使用的 P_current。历史 Query 级语义广播到候选内部
的历史点，经 Point Attention 后形成每个当前点独立的 P_memory。

Adapter 显式接收以下 shape：

    P_current:             [B,Q,R,17]
    P_memory:              [B,Q,R,17]
    P_current-P_memory:    [B,Q,R,17]
    cosine_similarity:     [B,Q,R,1]
    JS_divergence:         [B,Q,R,1]
    current_entropy:       [B,Q,R,1]
    top1_top2_margin:      [B,Q,R,1]
    semantic_weight:       [B,Q,R,1]

无有效候选时 P_memory、语义差向量、cosine 和 JS 均安全置零，Point Gate 为零。
Adapter 输入宽度由模块参数推导为
5*embed_dims + 3*num_classes + 10，默认是 1341；没有写死 Query 数、点数或
类别数。语义不同仍不是 hard mask，geometry 分支持终保留。

## 11. eligible 与 selected 候选计数

- eligible_point_count [B,Q,R]：通过 Memory 有效性、effective age、粗候选
  限制和 point radius 后，进入 Point Top-K 选择范围的历史点数量。
- selected_point_count [B,Q,R]：完成 Point Top-K 后实际参与 Point Attention
  的有效历史点数量，不超过 point_topk。

Adapter 同时使用两者：eligible 除以本 Query 的粗候选展开容量
Ku*R_memory，selected 除以 point_topk。Padding 与无效候选不计数；空候选时
两者均为零。

## 12. 位置残差的严格恒等

已核实 SparseWorld point proposal 使用 encode_points/decode_points 定义的
线性归一化坐标。V2 只处理有候选且 gate 非零的点，并计算相对编码变化：

    decoded_base = decode(base_pos)
    encoded_base = encode(decoded_base)
    encoded_shifted = encode(decoded_base + delta_position_metric)
    encoded_delta = encoded_shifted - encoded_base
    output_pos = base_pos + encoded_delta

当 delta_position_metric 为零时，两次 encode 输入相同，encoded_delta 严格为
零；无候选或 gate 为零时直接返回原始 base_pos。非零残差仍按 pc_range 转换，
输入不被原地修改，位置学习路径也没有 detach。

## 13. cls_weights 统一转换

权重来源优先级不变：首先读取 pts_bbox_head.train_cfg['cls_weights']，不存在时
回退模型 class_weights。list、tuple、CPU Tensor 和 CUDA Tensor 最终统一使用
torch.as_tensor，并指定 S_sem 的 device 与 dtype，然后 flatten 为一维。

转换后严格检查元素数量等于 num_classes；异常同时报告实际数量与期望数量。
原始配置对象不被修改。同一转换函数也用于 sparse SoftVoxel loss。

## 14. 本轮验收边界

本轮仅完成静态检查、Python 编译、配置解析和不涉及反向传播的小型纯函数/前向
验收，包括 B=1/B=2、Memory 少于/多于 Top-K、候选隔离、语义特征、位置严格
恒等以及 loss forward 边界输入。

本轮没有执行 backward、torch.autograd.grad、optimizer step、梯度验收、训练
数据迭代、200-iteration smoke、V1 对照实验、V2 正式训练或完整 nuScenes
评估。因此当前只说明已知代码问题已修正且静态/前向验收通过；尚未进行梯度
验收和训练前 smoke，不能据此声称 V2 已具备正式训练结论，也没有新的
IoU/mIoU 提升结论。
