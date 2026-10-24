# 轨迹派生 ego 与 T1 Memory：端到端复核（2026-10-21）

## 结论边界

学长的原则正确：使用过去轨迹估计速度/角速度不等于未来泄漏，也不要求必须使用
独立传感器测量。判断标准是实际参与估计的数据时间、读取时的样本身份，以及是否
进入预测器；不能仅凭字段名、数组名、数据中存在未来 GT 或历史记录时间戳作结论。

本地旧输入不属于“只截取过去速度”的情况：同一样本的 `a/v` 与未来位姿差存在
精确代数关系，模型实际读取这组字段，真实训练前向的预测也对它响应。

T1 Memory 的证据必须分层：

| 问题 | 实测结论 |
| --- | --- |
| 旧验证缓存是否可以复现 | 抽查一个记录，特征/位置/语义分布/可靠性均严格相等 |
| 该记录的未来相关输入能否影响当前 T1 | 可以；固定其他输入，1/2/3s 最终体素输出改变 |
| 历史 Memory 是否进入训练预测及损失 | 真实训练样本、原模型 forward/loss 路径验证通过；无参数更新 |
| 旧训练缓存是否已经逐项证明来源 | **未完成**；随机和确定性图像预处理探针均未匹配抽查旧记录 |
| 原训练的全部 Memory 是否污染、提升多少来自该通道 | **未证明/未量化**，不得由上述探针推断 |

因此不能再将“全部旧训练缓存已证实泄漏”或“指标提升来自泄漏”作为结论。
另一方面，至少一个原验证缓存的实际未来依赖已经形成证据链，也不能称为纯理论风险。

## 1. Baseline：不是因为“由轨迹算速度”而判定

合法的例子是 `v(t) = (p(t)-p(t-dt))/dt`。先计算一段轨迹的速度，再选择历史段，
只要每个保留值的计算依赖不晚于预测起点，也是可以的。

旧 `data_nuscene.pkl` 的实测关系则为：

```text
v = s_v * (g[3] - g[1])
a = s_a * (g[6] - g[1]) / 2.5
```

其中 `g[0]` 是该记录当前位姿，`g[1/3/6]` 为名义上 +0.5/+1.5/+3s 的位姿。
`s_v/s_a` 为每条记录的单个拟合标量，约在 [0.95,1.05] 内。这里 `a` 只是字段名，
上述形式不能解释为常规的历史速度差分加速度；坐标分量为平面位置/偏航角，不能
随意称为三轴线速度或加速度。

已有全量审计 `work_dirs/future_memory_t1_ego_provenance_20261020_verified.json`
覆盖实际训练 19,730 条和验证 4,219 条，全部满足这些关系，误差最大 7.11e-15。
独立使用 nuScenes metadata 位姿重建 `g`，全部匹配，最大误差 1.82e-12；不是只
因为字段叫 `gt` 就把它当未来。旧数据 SHA256 为
`287c3fe4ddc0928b4fdd1174b348b4c67c4316826eeb16c549f43d7fd4533f0d`。

本轮又检查 train/test 公共索引 20（原始索引 25）及对应历史源帧：

- 图像当前 token、dataset 当前 token、ego 记录 token 一致。
- `temporal_ego_states[0]` 等于该 token 的 `a/cmd/v/x` 拼接结果，不是前移到过去
  某个 token。数组里的 `0` 表示当前记录，不代表字段内部的信息截止于当前。
- 真实训练 forward hook 捕获 `plan_head` 的实际输入，与上述 21 维向量相同。
- 固定图像、Memory、GT，只依照上述关系改变当前记录未来终点对应的 `a[0]`，
  Baseline 当前/未来预测及 T1 输出发生变化。因此字段不只是加载后未使用。

实际路径：

```text
同 token 的 a/cmd/v/x
  → temporal_ego_states[0] → plan_head
  → points_scale_branch → RAP 初始点 → Query/点/语义
  → ego_cross_attn / Baseline 未来递推
  → T1 输出与训练损失
```

`x` 可由过去四次位姿增量重建，不应归为本项泄漏。未来占据/轨迹 GT 用作损失监督
本身也是正常的。`cmd` 与未来终点分类规则吻合，但如果任务明确允许提前导航指令，
还需按导航协议单独讨论；本轮端到端扰动只改 `a`，不依赖对 `cmd` 的定性。

上游检索也有相关历史，但不是本地结论的替代证据：

- [AD-MLP 官方 issue #4](https://github.com/E2E-AD/AD-MLP/issues/4) 记录了旧数据问题与作者修订回复。
- [官方 README](https://github.com/E2E-AD/AD-MLP#news) 有 2023-10-20 数据修复公告。
- [官方 GT 生成代码](https://github.com/E2E-AD/AD-MLP/blob/main/pytorch/admlp/generate_fengze.py)
  从当前帧沿 `next` 构造轨迹；它不是 `a/v/cmd` 的完整生成器。

本地 pickle 的完整原始生成器/随机种子仍未找到；上述是全量数值关系与独立时间
重建，不应描述成“已经找到了完整生成源码”。

## 2. T1：必须相对于最终预测起点判断

不能因为字段使用了“相对历史源帧的未来”就断言超过最终当前时刻。例如历史源在
`t-2.5s`，它的 `v` 用到源帧 +1.5s，通常仍在最终当前 `t` 之前。

实测验证样本 `df25b3ecaf994001ab5d5b92b9de89fd`：

| 事件 | 相对当前预测起点 t |
| --- | ---: |
| slot0 Memory 源帧 `08d95280dd2f42d98a4a6e33dff8e815` | -2.499926s |
| 源记录 g[1] | -2.000056s |
| 源记录 g[3] | -0.999748s |
| 源记录 g[6] | +0.500433s |

因此此处需要警惕的是 `a` 的未来终点依赖（以及需单独界定协议的 `cmd`），不能
笼统说每个历史字段、每个 Memory slot 都超出当前时刻。

完整代码通路：

```text
历史源帧的旧 ego 字段
  → precompute 的 plan_head / points_scale
  → RAP observation Query（ind_stamps_all == 0）
  → schema-v2 cache
  → 当前样本 loader 的历史选择、坐标对齐、age/valid mask
  → 双路粗检索 → Point Attention → T1 分支残差
  → 当前样本 1/2/3s 预测 → 训练损失或最终体素解码
```

缓存的记录时间戳在过去、RAP mask 正确，均不能屏蔽已经进入一个合法 Query
特征的未来相关条件。这不是把未来图像直接喂给网络，也不是把预测标签作为 Memory。

## 3. 实际 GPU 检验及限制

复核使用归档提交 `c562513beb0ba2f919c62d524f468d58ddb3bf85` 的 dataset getter、
V2 memory reshape、cache prefix 原函数，在进程内恢复，避免未完成修复改变审计对象。
checkpoint 为原 T1 epoch3，SHA256
`bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`。
其 679 个 Baseline 共有 tensor 与 `ckpts/epoch_56.pth` 严格相等。

### 验证集：旧缓存 → 实际输出，证据闭合到单个样本

1. 读取原缓存，按原源帧数据、原 prefix 和权重在 RAM 中重放。
2. `query_feat/query_points_metric/query_semantic_distribution/query_reliability`
   与旧缓存逐项 `torch.equal=True`，重复前向也严格一致。
3. 将源记录 `g[6].x` 增加 1m 对应的变化仅应用到 `a[0]` 输入；采用已验证的
   代数关系重构这一输入变化，不声称调用了未找回的原始生成器。
4. 源图像、过去位姿、其他字段均不变。只在 RAM 中替换当前样本 slot0 Memory。
5. 当前 ego、Baseline 全部未来快照、0s 输出严格不变；1/2/3s logits、点位置和
   原解码器生成的最终体素标签均改变；0.5/1.5/2.5s 输出严格不变。

这证明该旧验证缓存的未来相关输入不是“虽加载但被 mask 完全屏蔽”的无效字段。
扰动幅度只用于依赖检测，不是现实驾驶反事实、性能实验或泄漏收益估计。

### 训练集：真实训练路径生效，但旧缓存历史来源仍未闭合

抽查训练样本 `e1b5fd2f3e5f4059ab7527781bb652a3`，slot0 源帧的 g[6] 位于
当前之后 +0.499887s。训练随机图像增强的历史 RNG 未保存；按当前随机增强重放，
以及另试确定性图像处理，都未匹配旧缓存。**这不能简单归因于随机增强，也不能
把重放得到的记录当成原训练缓存。**

在原 `model(return_loss=True)` 路径上，用同一真实 batch、图像、当前 ego、标签
和随机种子比较两条仅源 `a[0]` 不同的内存重建记录：

- Baseline 快照、0s、非目标时刻严格不变。
- 1/2/3s T1 预测变化，语义/占据/位置损失也变化。
- 例如重建对的 2s `loss_sem` 为 0.06451661 → 0.06426831；3s `loss_occ`
  为 0.12437536 → 0.12377477。这不是 IoU/mIoU，也不是训练改进。
- 原生重复前向的预测 tensor 摘要严格一致；完整 loss/diagnostic 字典未逐位一致，
  所以未把整个字典严格不等当作唯一因果证据。

这证明“可疑条件 → Memory → 实际训练预测/损失”是可达且生效的路径，仍不能
证明原历史训练中每一条旧缓存都由完全相同条件生成，更不能证明所有优化步骤的影响。
本轮所有运行都在 `no_grad` 下，未进行梯度或参数更新检验。

## 4. 修正之前的表述与下一步边界

- “轨迹派生速度就是泄漏”：错误；应依据具体时刻和实际读入字段。
- “历史帧的未来必然是当前帧的未来”：错误；必须做上述绝对时间比较。
- “旧训练缓存已全部验证”：未完成，不能这么说。
- “IoU/mIoU 增益由泄漏造成”：没有量化证据，不能这么说。
- “没有未来图像且 mask 正确，所以一定干净”：不成立，字段内容也需核查。

此次只完成复核，不推动 Baseline 重训。后续若只修 T1 Memory，应明确是保持
Baseline 原协议的条件实验；消除新增 Memory 通道与证明全模型严格因果是不同目标。
旧模型换新缓存后的指标变化还混有输入分布变化，不能直接称为“泄漏贡献”。

## 5. 可复现产物与实际验收

新增只读诊断入口：`tools/recheck_ego_memory_dependency.py`。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128 env/bin/python tools/recheck_ego_memory_dependency.py --gpu --output work_dirs/ego_memory_recheck_end_to_end_20261021.json
```

命令成功退出；已有 JSON 拒绝覆盖，复跑需指定新的输出文件。
额外训练源预处理探针：`--splits train --source-eval-images`，结果保存在
`work_dirs/ego_memory_recheck_train_eval_images_20261021.json`。

- `env/bin/pytest -q tests/test_future_memory_ego_provenance.py`：5 passed。
- 新诊断脚本 Python 编译通过。
- 两个读取的旧 cache 和 T1 checkpoint 检验前后 SHA256 不变。
- 未启动训练、backward、optimizer step、完整 validation 或批量 cache 重建。
- 未修改模型/loader 修复逻辑，之前暂停的未提交修复仍保留；未提交/推送。
- 全工作区 `git diff --check` 仍报告此前 dataset 修复的 7 处 CRLF/尾部空白，
  不能宣称整个未完成修复工作区验收通过；本轮新诊断文件未引入该问题。
