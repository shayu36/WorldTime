# T1 epoch 3 时间因果审计与 planning 入口（2026-10-20 UTC）

> **后续重要更新（同日）：** 外部 ego 数据溯源已发现 `a/v/cmd` 中可全量复现的
> 未来 GT 编码关系；不能再把下文“正式数据路径未发现未来输入”的有限检查理解为
> 完整无泄漏结论。记录时间戳/Query mask 合格不代表字段内部内容合格。
> 详见 [外部输入溯源与复验方案](FUTURE_MEMORY_T1_EGO_PROVENANCE_FOLLOWUP.md)。
> 下文保留原审计事实与当时的范围，历史指标不修改。

## 审计对象与结论边界

对象：`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth`。
SHA256：`bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`。
HEAD：`3c057d61ab2c7d8ea68ceb4ddd319d9bf7e1bf73`，保留既有未提交 T1 修改。

当前 strict cache 数据路径未发现未来图像、未来 Memory 或跨场景输入。
这不是对任意手工输入、外部 pickle 生成过程、全部 cache 内容或完整 GPU 模型的
无条件“无泄漏认证”。本轮没有修改模型实现、配置、cache 或 checkpoint。

## 全量元数据与 cache 抽查

使用真实 dataset 的 `temp2nusc_map`、`get_adj_info`、
`_build_query_memory_history_infos`，不调用 `dataset.__getitem__`：

| 检查 | train | test/validation |
| --- | ---: | ---: |
| 有效样本 | 19730 | 4219 |
| 场景 | 700 | 150 |
| 当前+历史图像帧记录 | 98650 | 21095 |
| 实际选出的历史 Memory 关系 | 44523 | 9514 |
| 时间/场景/重复候选异常 | 0 | 0 |
| 抽查 cache 文件 | 128 | 128 |
| 基础年龄范围（秒） | 2.15045–4.75214 | 2.15100–4.75158 |

训练/验证 token 和 scene 的交集均为 0。图像按每个 camera 自己的时间戳比较，
避免把 camera 与 lidar 的传感器不同步误判为未来帧。
256 个抽查 cache 均通过真实 loader schema/scene/sample/frame/timestamp 校验；
声明来源均为 `ckpts/epoch_56.pth`、`sparseworld-traj-memory-only.py`。
未遍历所有 cache tensor 内容，也没有重新生成 cache。

现有 precompute 代码只运行 RAP prefix，从 `ind_stamps_all==0` 选择 observation
Query；语义、可靠性来自模型预测，不来自 GT 标签。对齐目标是当前 ego pose，
不是未来 GT pose。cache 历史生成时的 source hash 未保存在每条记录中，因此来源
字段检查不等于对历史生成过程的独立重放证明。

## 时间掩码与监督隔离

- 从 T1 checkpoint 的 `num_stamps_all` 重建真实 `reset_mask`：1040×1040，
  各时刻 Query 数 `[720,60,60,60,60,40,40]`。六层相同，屏蔽 272800 条
  `key_time > query_time` 边，屏蔽位置的测试 softmax 权重为零。
- 真实 `OPUSSelfAttention` CPU 前向：修改较晚时刻 Query 特征，较早输出严格不变。
- 训练入口在 checkpoint resume 之后冻结 TASS/重建 mask；冻结状态的现有测试通过。
- 实际 `forward_backbone`/未来递推方法 + 小规模响应式前端/真实 V2：替换未来 GT、
  未来 ego states、未来 pose 为不可用对象，eval 预测严格不变；修改当前 ego state
  则预测变化，排除了“测试分支始终返回常量”的伪隔离。
- 训练纯前向中修改未来 GT trajectory：监督点 mask 会变化，预测不变。
  该 mask 是 loss 的有效空间区域筛选，不是给预测器看的未来特征。
- T1 的未来 GT 类别只用于占据正样本 BCE 聚合；未来标签用于监督本身是合法的，
  不能把“计算未来目标 loss”误称为“提前读取未来作为输入”。
- 1/2/3s effective_age 是 base_age+1/2/3；无效/过期候选、递推隔离、
  mask 对 loss 生效等前向测试通过。
- 与原 epoch56 共有的 679 个 tensor 全部严格相同，没有缺失；与直接起点 center2
  的所有非 V2 tensor 也未变化。T1 state 中未发现非有限浮点 tensor。
- T0/T1 各有 4219 个唯一 token 的 occupancy dump；评估脚本按 token 取文件、缺失即报错，
  内部 step 2/4/6 对应 1/2/3s，不把进度条 DDP padding 当额外样本。

## 仍存在的风险/未完成项

1. **模块内部缺少独立 base_age>0 防御**：绕过 loader，构造
   memory_valid=True、base_age=-0.5、horizon=1，会以 effective_age=0.5 进入候选。
   当前 strict loader 会先拒绝这种当前/未来来源，并设置 age>0 有效掩码；本次正式
   数据检查未发现该输入。它是已复现的接口防御缺口，不是已证实本次成绩泄漏。
   审计没有擅自改变模型；后续修复应增加独立基础年龄检查并验证正常输入恒等。
2. 当前 ego 输入字段包含 `a/cmd/v/x`，显式 `gt` 被排除，模型只取第 0 时刻。
   这些字段来自外部 `data_nuscene.pkl`，仓库未找到其完整生成流程。
   尤其 `cmd` 是外部导航条件还是从未来 GT 轨迹提取，尚未独立证实。
   Baseline 与 V2 均使用该输入，但“共同使用”不能代替严格因果来源证明。
3. 没有实际完整 GPU 网络的未来输入扰动重放；当前 GPU 被其他任务占用。
   已完成的是元数据、checkpoint、真实子模块与真实方法的 CPU 前向检查。
4. 样本按名义 0.5s 帧索引构造目标，实际时间戳有抖动；不是每帧精确间隔 0.500000 秒。
   与当前 Baseline 口径相同，但不应宣称精确连续时间插值评估。

## 2.5/3.5/4.5 秒的含义

当前相机分支配置 `num_frames=5`、`sweeps_num=4`、`multi_adj_frame_id_cfg=(0,5,1)`，
加载 t、t−0.5、t−1、t−1.5、t−2 的关键帧（每帧六相机）。3 秒是未来预测范围。
历史 Memory 另外选 t−2.5、t−3.5、t−4.5 的 observation Query，允许 ±0.35 秒匹配误差。
Query 属于该历史帧，不是“半个 Query”，也不是预测出的未来 Query。
每个历史 Query 自身还来自该历史帧及其过去约 2 秒图像；名义最远信息可到 t−6.5 秒。
若任务协议限定只能用一段 2/3 秒视频，额外 Memory 超出那个输入预算；这属于
输入窗口/公平性说明问题，不是读取未来。当前模式需要保留这个区别。

## 碰撞率评估入口

已新增 `tools/eval_future_memory_planning.py` 与
`tools/run_eval_future_memory_t1_planning.sh`。手工启动：

```bash
bash /data/jxy/projects/tools/run_eval_future_memory_t1_planning.sh --check
bash /data/jxy/projects/tools/run_eval_future_memory_t1_planning.sh
```

脚本锁定已正式评估的 T1 epoch3 checkpoint SHA256，并校验 occupancy 评估时的模型
源文件 hash。双卡调用原生 simple_test，不关闭任何 V2 分支；只收集 pred_traj，
丢弃 occupancy 大数组，按 sample token 去除 DDP padding。对 6 个相对位移沿时间
累计一次，与旧 dataset.evaluate 完全同一转换，不读取/覆盖 admlp/output_data.pkl。

独立输出目录：`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/planning_epoch_3`。
产生 trajectories.pkl、manifest.json、provenance.json、planning_metrics.json。
拒绝覆盖已存在目录；GPU 忙时退出而不停止现有进程。推理已成功、仅 CPU metric
中断时可执行同一 launcher 的 `--metrics-only`。

采用原仓库 `PlanningMetric`，不增加贴零或阈值修正。filter_token 有 4819 个 token，
与当前 4219 帧 dataset 交集为 4219；与此前 A+B planning 使用的 token 集合完全相同。
所谓 1/2/3s 数值是前 2/4/6 个半秒步的**累计窗口平均**，不是只取端点。
碰撞率 raw fraction 乘 100 后才是百分数；报告中对比用户提供的 Baseline：
L2 `[0.1596,0.1945,0.2393]` m；box collision `[0.0948,0.1007,0.1106]` %。
这份 Baseline 是用户提供的四舍五入参考，不冒称本轮重新评估。

当前架构的 V2 修正不回写轨迹递推，planning 仍由冻结 Baseline state 生成。
合成前向验证修改 V2 occupancy/position 输出会改变占据输出、不会改变 pred_traj；
因此不能把 occupancy IoU 提升自动解释为碰撞率提升。T1 专属 planning 尚待用户执行。

## 实际验收

CPU 指定前向/纯函数测试：**100 passed, 1 deselected**（排除 CUDA），包括新审计测试、
V2/T1 前向测试、loader/history/TASS 选定测试和 4219 帧 DDP padding 对齐测试。
未选择任何 backward/optimizer-step 测试。
初次新增 RAP 测试误把 attention 返回的 tuple 当 tensor，修正测试解包后通过；
审计计数脚本最初未把浮点类型 stamp ID 转 long，修正 bincount 输入后全量重跑通过。
这些修正均只在审计脚本/测试中，未改变模型。

Python compile、bash -n、launcher --check、git diff --check 通过。
PlanningMetric CPU 零误差/空障碍合成前向通过；未启动完整 planning 推理或评分，
未执行训练、反传、优化器更新、cache 重建、checkpoint 修改、提交或推送。
