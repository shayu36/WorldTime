# T1 外部自车输入溯源：发现未来轨迹编码（2026-10-20 UTC）

## 结论与修正之前的审计边界

**不能确认当前 IoU/mIoU 增益不受未来信息影响。实际发现外部 ego 输入中存在
可全量复现的未来 GT 编码关系；不再只是“生成来源未证实”。**

此前图像时序、Memory 记录时间戳、RAP mask、训练/验证场景隔离检查仍成立，但这些
检查不覆盖“当前时间戳的字段内部编码未来轨迹”。此前未来 GT 扰动测试没有重新生成
外部 pickle 的 `a/v/cmd`，因此测试通过不排除这种预处理泄漏。

没有改动模型、训练配置、原始 pickle、Memory cache 或 checkpoint。现有指标保留，
应标记为原输入协议下的历史结果，不能据此宣称严格因果输入下的同等增益。也不能反向
断言增益全部来自泄漏：贡献大小尚未通过干净协议的对照实验量化。

## 上游检索与文件身份

本地两个文件：

- `admlp/stp3_val/data_nuscene.pkl`
- `stp3_val/data_nuscene.pkl`

SHA256 均为 `287c3fe4ddc0928b4fdd1174b348b4c67c4316826eeb16c549f43d7fd4533f0d`，
大小 25,507,056 字节，与 SparseWorld 公开数据发布中的 LFS SHA256/size 完全一致。
仓库 `docs/prepare_datasets.md` 第 6 步也指向该发布。这个一致性证明文件身份，
不证明数据在生成时遵守了因果约束。

来源：

- [SparseWorld 数据发布及 LFS 哈希](https://huggingface.co/MSunDYY2001/SparseWorld/commit/8a9a8f16223d64c520e57646211e6523ca51ed90)
- [AD-MLP 官方 issue #4](https://github.com/E2E-AD/AD-MLP/issues/4)：报告过未来
  waypoint 被误作速度/加速度；作者回复已修订训练流程，后续评论仍质疑旧下载文件。
- [AD-MLP 官方说明](https://github.com/E2E-AD/AD-MLP)：有 2023-10-20 数据修复公告。
- [固定上游版本的 GT 生成代码](https://github.com/E2E-AD/AD-MLP/blob/4b93ba085ee47474152f282177865796ea577fc0/pytorch/admlp/generate_fengze.py)：
  生成当前及未来位姿相对轨迹，不是 `data_nuscene.pkl` 四个输入字段的完整生成器。
- [固定上游版本的读取代码](https://github.com/E2E-AD/AD-MLP/blob/4b93ba085ee47474152f282177865796ea577fc0/pytorch/admlp/planner.py)：
  排除 `gt` 后按键排序读取其他字段。

没有找到本地这份 pickle 的完整原始生成器/随机种子，不能把 issue 中描述的旧错误
公式直接当作本地公式。下面的公式来自本地文件独立数值核查。

## 全量数值证据

令 `g[k] = record['gt'][k]`，`g[0]` 是当前位姿，`g[1/3/6]` 分别是名义上
未来 0.5/1.5/3 秒（实际时间戳有抖动）。全文件 34,149 条记录中 29,049 条具有
完整 7 点 GT，余下是场景尾部不完整目标，没有混入完整目标的统计分母。

完整记录均满足：

```text
v = s_v × (g[3] - g[1])
a = s_a × (g[6] - g[1]) / 2.5
```

`s_v/s_a` 是每条记录各自的单个标量，由三个坐标同时拟合。全部拟合系数在
约 `[0.95, 1.05]` 内，最大绝对残差 `7.11e-15`；全部参考向量非退化。
这是代数编码关系，不是一般意义上“速度和未来运动有相关性”。缩放范围像幅值扰动，
但没有生成器证据，**不声称已证实某个随机采样算法**。2.5 是上述匹配公式中的
名义时间间隔；用实际样本时间差直接做物理速度重建，并不能精确还原这些字段。

`cmd` 则全部等于根据未来终点横向坐标判定的 one-hot：

```text
g[6,0] <= -2 → [1,0,0]
g[6,0] >=  2 → [0,0,1]
其他          → [0,1,0]
```

| 实际模型使用的集合 | 样本数 | v 关系符合 | a 关系符合 | cmd 规则符合 |
| --- | ---: | ---: | ---: | ---: |
| train | 19730 | 19730 | 19730 | 19730 |
| test/validation | 4219 | 4219 | 4219 | 4219 |

另外，不依赖 pickle 的 `gt` 字段，用 nuScenes metadata 中 ego/global 和
lidar/ego 变换独立重建相对位姿：23,949 个实际样本全部匹配，最大误差
`1.82e-12`。因此这些确实是对应样本的未来轨迹，而非误命名的历史数组。

### x 字段

`x` 在全部 23,949 个实际样本中可以仅由过去四次单步位姿变化的负累计和重建，
最大误差 `7.28e-12`；不需要当前之后的位姿。它的各步增量使用各自起始帧坐标系，
不是把所有过去位置统一变换到当前坐标系。这里验证的是时间来源，不替这个坐标
表示方式作建模正确性认证；没有证据要求把它也归为本次未来泄漏。

## 进入占据预测和 cache 的路径

实际代码路径：

```text
data_nuscene.pkl 的 a/cmd/v/x
  → temporal_ego_states[0]  # “0”只表示所属样本，并不净化字段内容
  → plan_head
      → points_scale_branch → RAP 初始点 → Query/点/语义输出
      → ego_cross_attn → fused_ego_feat → 未来占据类别与位置分支
  → Baseline 快照 → V2 检索与残差
```

源码位置：`nuscenes_dataset_occ_trajectory.py:584`、
`sparseworld_4d_traj.py:1666`、`:661`、`:681`、`opus_head.py:108`。
所以这不是仅影响 L2/碰撞率、与 IoU/mIoU 无关的输入。

CPU 加载 T1 epoch3 checkpoint 中原生 `plan_head` 和 `points_scale_branch` 权重，
按实际拓扑 strict load；固定抽取 256 个验证样本，只清零一组输入：

| 纯前向探针 | ego feature 改变样本数 | point scale 改变样本数 | scale 最大绝对变化 |
| --- | ---: | ---: | ---: |
| a 置零 | 256/256 | 256/256 | 0.50658 |
| v 置零 | 256/256 | 256/256 | 0.14705 |
| cmd 置零 | 256/256 | 256/256 | 0.14521 |

这是真实权重的子网络依赖检验，不是完整 GPU 推理、IoU 评估或训练。
输入置零有分布偏移，不能把这个探针称作“修复后的性能”。

### Memory 的时间戳合格不等于内容合格

当前 cache 生成入口 `tools/query_memory/precompute_query_memory.py:239–247`
同样读取源帧的当前 ego 字段，计算 plan feature/point scale，再保存 observation
Query。若 cache 按该入口和当前外部数据生成，它携带的依赖时间可以晚于记录时间。

例如记录时刻为当前的 `t−2.5s`，该记录的 `a/cmd` 涉及其 `+3s` GT，
依赖即可延伸至当前的 `t+0.5s`。全量所选 history metadata 中：

| 关系集合 | 所有 Memory 选择关系 | 源记录的 g[6] 时间晚于当前 |
| --- | ---: | ---: |
| train | 44523 | 16218 |
| validation | 9514 | 3470 |

均出现在最近的 slot0；超出当前时间 train `0.394377–1.050062s`，
validation `0.398234–1.000319s`。这是对源输入依赖和时间关系的检查，
**未重放历史 cache 内容，不能冒称对每个旧 cache tensor 完成了污染定量证明**。
旧记录只保存 source 路径、缺少生成时源码/ego 数据 hash，不能认证旧 cache 清白；
干净协议不应继续复用这些未经证明的记录。

## 对指标解释的影响

- `base_age > 0` 缺口：正式 loader 路径此前未观察到触发。加防御对合法输入应是恒等，
  不能把它与这里的预处理未来信息混为一谈。
- RAP mask：屏蔽较晚 Query 的检查仍通过。但 mask 不可能屏蔽已嵌入“当前 Query”
  的未来自车信息。
- Baseline/V2 共享错误输入不能证明错误会在指标差值中抵消。V2 用非线性门控、
  不同损失，且多读历史 Memory；增益可能依赖这种输入。
- T1/T0 同输入、同训练起点的结果仍是**旧输入协议下的条件对比**。不能据此保证
  改成干净输入之后仍有同样的 T1 增益。
- 历史 IoU/mIoU 和碰撞率日志不删除、不改数值；增加“外部 ego 输入含未来编码，
  因果复验待完成”标记。当前不能认证 `+0.6267 IoU / +0.3300 mIoU` 在无泄漏
  条件下成立，也不能宣称这些差值全部是泄漏贡献。

## 建议修复与复验顺序（尚未执行）

### R0：补防御和来源约束，不先盲目重训

1. 模块内独立检查原始 `base_age` 有限且大于 0，再判断 effective_age。
   保留 V1 路径；合法输入严格恒等测试；无效输入候选/上下文/门控全部为 0。
   单独这项不需要重新训练，前提是确认正式路径从未触发。
2. 新建版本化 ego 数据，禁止覆盖原文件；显式字段 schema，禁止“除 gt 之外全读”。
   每个字段保存 source 时间戳、来源、生成器版本与哈希。
3. `v/a` 首选原始 CAN 消息中测量时间 `utime <= current_timestamp` 的最近一条；
   禁止最近邻不分前后、跨当前时刻插值、双向平滑。没有历史 CAN 时，用仅过去位姿
   和真实时间差作后向差分，并记录缺失/估计标记，禁止从未来补齐。
   坐标系、角速度和线速度/加速度的单位要明确，不能继续用混合 `[x,y,yaw]`
   位移冒充三轴加速度。[官方 CAN 字段与时间戳说明](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md)
4. `cmd` 有两种独立协议：可追溯的提前给定导航路线；或者无导航/unknown 条件。
   若沿用未来 GT 派生的 route command，只能明确报告为 oracle-command 条件，不能
   宣称纯过去输入。本轮未证明本地 cmd 来自提前给定路线。
5. 源帧构建 Memory 时，所有信息必须在**该历史帧时刻**已可用，而不只是晚于源帧、
   但早于最终当前时刻就算合法。保存 max_input_timestamp 并在线断言。

### R1：低成本因果扰动与配对敏感性诊断

- 新生成器在删除/任意替换 `timestamp>t` 的原始记录后，当前全部输入严格不变；
  历史 cache prefix 对晚于源帧的扰动严格不变。这是应补的真正端到端预处理隔离测试。
- 用固定且按场景分组的小样本，比较 Baseline/T1 在旧输入与因果输入下的输出。
  因果 V2 要用对应重算的**独立小样本 cache**，不能只改当前 ego 输入保留旧 cache。
- 比较每个 horizon 的 IoU/mIoU/precision/recall、按类 IoU 与配对场景 bootstrap CI。
  若想声称“影响可忽略”，预先定义等价容忍界，而非把“CI 包含 0”当成等价证明。
- 旧权重直接换输入属于分布外敏感性诊断，性能降幅不能直接等同于泄漏收益；
  这一步不替代干净训练。

### R2：建立可以正式引用的干净对照

最有说服力：使用干净输入，重训一个匹配 Baseline，再固定该 Baseline、重建独立
train/val Memory cache、重新训练 V2/T1；两者采用相同数据划分、评估 mask、阈值
和采样协议，明确 Memory 额外历史窗口。V2 从同一干净 Baseline 零残差起点训练。
若采用预训练初始化，优先图像/检测等不含该错误 ego 输入的权重。

只把 pickle 改掉，然后从旧 T1 继续一个 epoch、Baseline 全冻结，不足以排除
已经依赖错误字段的冻结 Baseline 和旧 cache 的影响。资源有限时可先统一修复输入、
对 Baseline 作匹配适配训练再比较 V2，但应明确初始化继承与局限；不能把该低成本
迁移实验冒称“完全重新训练的干净结果”。

正式复验至少保留 matched Baseline/V2；若要单独确认 T1 class-balanced loss 的
贡献，再增加同起点同预算的 T0。多 seed 或预先锁定 checkpoint 选择规则可降低
验证集挑选最优带来的偏差。这里的重训、cache 重建、完整评估需用户另行授权，
本轮没有启动。

## 可复现产物与命令

新增 `tools/audit_future_memory_ego_provenance.py`、
`tests/test_future_memory_ego_provenance.py`。完整机器结果：

`work_dirs/future_memory_t1_ego_provenance_20261020_verified.json`

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 env/bin/python tools/audit_future_memory_ego_provenance.py --config work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py --checkpoint work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth
```

`--output` 可写新 JSON，拒绝覆盖已有路径。这个命令只做 CPU 元数据/代数审计和
ego 子网前向，不调用完整 dataset `__getitem__` 或 GPU occupancy 模型。
原始生成器、旧 cache 的实际运行环境仍不可从现有 provenance 完整恢复；
这不妨碍确认输入代数上的未来编码，但限制了对旧 tensor 的归因强度。

本轮指定 CPU 前向/纯函数测试 **105 passed, 1 deselected**（排除 CUDA），其中新增
5 项检查比例签名、拒绝不同方向、排除零向量退化、GT/cmd 规则、过去位姿重建的
未来扰动不变性；其余为上轮同一组前向审计/V2/T1/loader/TASS/采样对齐测试。
Python 编译通过，`git diff --check` 通过。未执行 backward、optimizer step、
训练、完整验证、cache 重建、checkpoint 修改、提交或推送，未停止其他进程。
