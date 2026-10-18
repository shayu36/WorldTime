# A+B 任务与时刻门控续训

起点：occupancy 修正版 `occfix-q48/epoch_11.pth`，对应 Future IoU
22.7967、Future mIoU 13.1100。实现前 HEAD 为
`852977be7f9a734b2d32cef781093c42831350a3`，原工作区干净。

## 唯一结构变更

A：原 `gate_head` 保留为语义头，复制其训练后的 weight/bias 为
`gate_occ_head` 和 `gate_pos_head`。B：新增零初始化
`task_horizon_bias[3,3]`（行=1/2/3s，列=sem/occ/pos）。
每个 Gate 为 `sigmoid(head_task(F) + bias[horizon, task])`，输出
`[B,Q,R,1]`。C 未启用，也未新增类别组门控。

语义、占据、位置分别应用自己的 Gate；decode 仍保留类别排序，最大
logit 等于 occupancy。位置更新沿用米制相对编码残差。
无候选时三门控归零；Baseline 冻结、0s/非目标步不变、修正不反馈递推。
默认 `gate_mode=shared`，旧 V1/V2 配置和参数键兼容。

迁移不清零已训练的残差头、不重置原 Gate bias。初始输出应等于训练后
的源 checkpoint，而非等于冻结 Baseline。迁移工具生成新文件，拒绝覆盖，
保留全部原 state_dict 张量，记录源 SHA256，丢弃旧 optimizer/epoch 状态。
完整模型严格加载和真实 Adapter 权重的 B=2 三时刻合成前向验证用于启动检查；
这不是全 validation 等价性评估。

## 本轮运行设置

独立配置 `sparseworld-traj-memory-future-adapter-v2-ab.py`，独立目录
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11`。
仅启动 A+B 一条训练，后续根据结果决定 C，不自动启动 C。

- 双 RTX3090，batch/card=1，worker/card=1，chunk=48。
- activation checkpoint=True，save_on_cpu=False，dynamic FP16，seed=17。
- AdamW 新优化器，学习率 2e-5，weight decay 沿用0.01。
- 100 iter 线性 warmup（0.5→1倍），随后固定学习率。
- 选择2e-5是低于源epoch11约3.09e-5的保守短程适配值，未经调参优化。
- 运行3个本地epoch，保存epoch_1/2/3，对应源epoch11之后的三个续训阶段。
  不是原始runner的epoch12/13/14精确resume，数据采样计数也从本地epoch0开始。
- 原 occupancy标签、正负平衡、anchor和其余损失/检索参数继承修正版。

训练日志追加各horizon有效候选比例、三Gate在有效候选上的mean/std、
语义和占据修正绝对均值、位置米制修正范数均值及有符号occupancy shift。
这些detach统计不参与总loss。activation checkpoint返回全部门控；DDP
图锚点覆盖三门控，空Memory分支保留零梯度连接。

## 验收与归因

先比较迁移前后精确输出、独立分支/时刻控制、无候选回退、B=2、
checkpoint重计算梯度和全空Memory参数连接，然后检查真实训练首批迭代。
不覆盖旧checkpoint，不提交/推送。

训练后逐时刻报告IoU/mIoU及逐类结果，并检查占据precision/recall。
目标是保持IoU收益并改善2s/3s语义，不预先声称成功。
本轮仅运行A+B探索；同起点、同优化器重置、同学习率/采样/预算的共享Gate
续训对照尚未运行。因此对源epoch11的提升包含额外训练的影响，不能单独
归因于门控结构。未来开启C也应采用相同预算对照。

## 正式评估结果与当前选择（2026-10-16）

以下结果均来自4219帧完整validation；Future Mean只平均1s、2s、3s。

| checkpoint | IoU [0s,1s,2s,3s] | Future IoU | mIoU [0s,1s,2s,3s] | Future mIoU |
| --- | --- | ---: | --- | ---: |
| A+B epoch1 | [25.68, 23.83, 22.84, 21.70] | **22.7900** | [18.20, 15.08, 13.07, 11.35] | **13.1667** |
| A+B epoch2 | [25.68, 23.80, 22.79, 21.64] | 22.7433 | [18.20, 15.05, 13.02, 11.25] | 13.1067 |
| A+B epoch3 | [25.68, 23.84, 22.82, 21.64] | 22.7667 | [18.20, 15.08, 13.05, 11.26] | 13.1300 |

A+B epoch1是本轮V2路线保留的最新最佳结果。相对历史Baseline Future IoU
22.2133提高0.5767，但Future mIoU 13.1667仍低于Baseline 13.2233，差0.0566。
因此这里的“最佳”指当前V2实验中以occupied IoU为主的综合选择，不代表17类
语义mIoU已经超过Baseline，也不改变历史V1记录。

原始结果保存在
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/eval_epoch_{1,2,3}/metrics.json`。

## 已放弃的C方案：类别组Semantic Gate

C从A+B epoch1出发，保留语义/占据/位置三任务Gate与horizon bias，只把语义Gate
进一步拆成三个预定义类别组：背景类、交通参与者、道路设施。新增零初始化的
`Linear(256,3)`，为每个点预测三个组偏置，再按固定类别到组的映射展开成
`[B,Q,R,17]` Semantic Gate；occupancy和position Gate仍为`[B,Q,R,1]`。
迁移时零组偏置与A+B严格等价，随后所有V2参数共同续训。

完整评估结果如下：

| checkpoint | Future IoU | Future mIoU | 相对A+B最佳IoU | 相对A+B最佳mIoU |
| --- | ---: | ---: | ---: | ---: |
| C epoch1 | 22.7900 | 13.1667 | +0.0000 | +0.0000 |
| C epoch2 | 22.7600 | 13.1133 | -0.0300 | -0.0534 |
| C epoch3 | 22.7700 | 13.1400 | -0.0200 | -0.0267 |

C被放弃并从生产代码中移除，原因是：

1. 三轮没有超过A+B epoch1；继续训练没有形成上升趋势。
2. C没有解决关键取舍：occupied IoU保持较高，但2s/3s语义mIoU仍低于Baseline。
3. 固定粗类别组内部同时包含改善与退化类别。例如背景组同时包含道路类与
   manmade/vegetation，单个组偏置无法表达组内相反需求。
4. 类别组Semantic Gate只能调节已有点的语义残差，不能恢复被occupancy分支过滤
   或被位置残差移出正确体素的点，和已观察到的主要错误机制不完全匹配。
5. 它增加参数、checkpoint迁移和类别级Gate维护成本，却没有得到可测的指标收益。

C的训练checkpoint与评估产物暂留在
`work_dirs/sparseworld-traj-memory-future-adapter-v2-c-from-ab1-5ep`作为实验依据，
但C配置、迁移工具、训练/评估脚本、专属测试和类别组模型路径均已撤销。
后续开发以A+B为代码与checkpoint起点，不再从C继续训练。
