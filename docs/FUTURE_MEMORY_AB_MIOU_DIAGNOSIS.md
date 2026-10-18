# A+B：2s/3s mIoU 下降诊断（2026-10-15）

2026-10-16补充：GPU可用后已完成256帧单快照分支干预，见
[GPU配对诊断](FUTURE_MEMORY_AB_PAIRED_GPU_DIAGNOSIS.md)。以下保留10月15日当时的证据和限制，
后续配对实验支持occupancy的类别选择性召回损伤，并发现位置分支的净损伤；语义分支总体有益。

> 2026-10-16状态更新：C已完成三轮完整评估，Future IoU/mIoU分别为
> 22.7900/13.1667、22.7600/13.1133、22.7700/13.1400，均未超过A+B epoch1。
> C类别组Gate方案已放弃并从模型代码与运行入口移除；当前以A+B epoch1作为
> 最新保留的V2结果。下文关于“C正在运行”的文字是诊断当时的环境快照。

本轮只做代码阅读、既有日志分析、CPU 已保存预测抽样和微型前向实验。
没有改动生产模型/损失/配置、没有 backward/optimizer step、没有新训练、
没有重新运行完整 validation、没有停止正在进行的 C 续训。
起点 HEAD：`852977be7f9a734b2d32cef781093c42831350a3`。
已有 A+B/C 未提交改动保持不变。

## 1. 先分清比较对象

以下是已有 **4219 帧完整评估**，不是本轮重新评估。IoU/mIoU 单位为百分数，差值为百分点。

| 模型 | Future IoU | Future mIoU | mIoU 1s | mIoU 2s | mIoU 3s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline（历史正式记录） | 22.2133 | 13.2233 | 14.96 | 13.18 | 11.53 |
| occfix V2 epoch11，A+B 的源模型 | 22.7967 | 13.1100 | 15.06 | 13.04 | 11.23 |
| A+B epoch1 | 22.7900 | 13.1667 | 15.08 | 13.07 | 11.35 |
| A+B epoch2 | 22.7433 | 13.1067 | 15.05 | 13.02 | 11.25 |
| A+B epoch3 | 22.7667 | 13.1300 | 15.08 | 13.05 | 11.26 |

下降已经存在于源 V2。A+B epoch1 对源模型改善了 mIoU（尤其3s +0.12），
但未恢复原始 Baseline。不能把全部退化归因于 A+B，也不能说是 A+B 本身带来 +0.58 IoU。
三轮没有持续改善趋势；也没有同预算共享 Gate 续训对照，不能单独归因于拆 Gate。

来源：`docs/优秀实验记录.md`；
`work_dirs/sparseworld-traj-memory-future-adapter-v2-occfix-q48/eval_epoch_11/metrics.json`；
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/eval_epoch_{1,2,3}/metrics.json`。

逐类 Baseline 使用既有 `work_dirs/sparseworld-traj-memory-only/eval_epoch56_memory_off.log`。
注意该旧日志 2s binary IoU 是22.27，历史正式汇总是22.28；本报告不改写历史数值，
也不把0.01的记录差异解释为模型收益。两处 mIoU 一致。

## 2. 完整评估：究竟哪些类别拖低 mIoU

以 A+B 最佳 mIoU 的 epoch1 比较 Baseline，下面是逐类 IoU 差值：

| 类别 | 2s delta | 3s delta |
| --- | ---: | ---: |
| pedestrian | -2.35 | -1.83 |
| motorcycle | -1.05 | -1.22 |
| traffic_cone | -0.59 | -0.82 |
| manmade | -0.93 | -1.12 |
| vegetation | -1.22 | -1.34 |
| driveable_surface | +1.32 | +1.67 |
| sidewalk | +0.75 | +0.83 |
| terrain | +0.82 | +0.88 |
| car | +0.87 | +0.28 |

不能概括成“所有背景改善、只有动态类受损”：manmade/vegetation 同样受损。
2s 所有改善类别对平均值贡献约 +0.2924，所有退化类别贡献 -0.3965；
3s 分别 +0.2712 / -0.4524（由日志四舍五入后的逐类值计算）。
表中五个受损类别已贡献约 -0.3612 / -0.3724，足以抵消其余类别收益。

二值 occupied IoU 只区分 free/occupied。把行人预测成道路，在二值上仍可能算 TP，
但在语义上同时增加行人的 FN 和道路的 FP；把行人预测为空则两项都受损。
mIoU 对17个类别等权，binary IoU 不对类别等权。
因此“部分大类/区域改善 + 小类漏检/误分类”可以使 binary IoU 上升、mIoU 下降。
这是当前输出层面的取舍，不意味着两个指标必然互斥，也不是已经证明某个损失主动牺牲语义。

## 3. CPU 保存预测诊断：256帧，覆盖150场景

脚本 `tools/diagnose_future_memory_semantics.py` 按真实 dataset.temp2nusc_map
均匀选择256个索引；同一 token、同一 GT、0/1/2/3s，对比 native11、occfix11、AB1、AB3。
按现有 evaluator 不使用 camera/lidar mask，计算18类 confusion matrix，mIoU 排除free。
只读取已有 dump，未推理新预测。结果与 token 清单保存在：

`work_dirs/diagnostics-ab-miou-20261015/paired_subset256.json`

| 模型/时刻 | binary precision % | binary recall % | binary IoU % | mIoU % |
| --- | ---: | ---: | ---: | ---: |
| native11 2s | 37.6673 | 28.4023 | 19.3212 | 11.0913 |
| occfix11 2s | 35.1434 | 39.8871 | 22.9750 | 13.0562 |
| AB1 2s | 35.2326 | 39.7569 | 22.9697 | 13.0919 |
| native11 3s | 36.6076 | 26.0472 | 17.9505 | 9.4292 |
| occfix11 3s | 33.9689 | 37.8342 | 21.8008 | 11.4442 |
| AB1 3s | 34.0823 | 37.6861 | 21.7980 | 11.5453 |

这反证了把**旧版0.20m正样本错误直接当作当前 A+B 原因**的做法：
occfix 已恢复大量 recall，A+B 保存的配置也确实是 positive_radius=None、
正负组各自平均、occupancy anchor=0.05。当前仍有损失，但不是未经修复的原始配置。
native→occfix 包含重新训练等变化，不能当作单一标签修复的严格消融。

AB1 的真实 GT 类别去向（每行按该类 GT 体素数量归一化）：

| 类别/时刻 | 类别正确 % | 预测为空 % | 预测成其他非空类 % |
| --- | ---: | ---: | ---: |
| pedestrian 2s | 3.12 | 76.96 | 19.92 |
| pedestrian 3s | 1.13 | 78.69 | 20.18 |
| motorcycle 2s | 6.64 | 66.17 | 27.19 |
| motorcycle 3s | 5.63 | 68.69 | 25.68 |
| traffic_cone 2s | 5.45 | 59.95 | 34.59 |
| traffic_cone 3s | 3.44 | 58.95 | 37.61 |
| manmade 2s | 17.56 | 78.57 | 3.87 |
| vegetation 2s | 26.10 | 71.30 | 2.60 |

行人主要错到 manmade/sidewalk/driveable_surface；摩托车、锥桶主要错到地面/建筑。
漏成free比非空类混淆更常见，但**没有配对 Baseline dump，因此不能据此计算相对 Baseline
新增漏检的比例，更不能区分是 O 阈值、位置错位还是点覆盖不足造成free**。

2s GT 中 ids11–16的六种背景占92.87%；行人+摩托车+锥桶合计仅0.425%，
但后面三类占 mIoU 的3/17=17.65%权重。这直接解释小类失败为何容易被 binary 总数掩盖。
这只是 GT 频率，不等于训练点/梯度频率；不能由此声称已测得梯度失衡。

AB1 vs occfix11 在2s中移除了217508个FP、增加172198个FP，同时恢复53456个
occupied TP、丢失64657个occupied TP；有小幅precision/recall取舍，但幅度很小。
同位置两边都occupied时，正确→错误类别16391，错误→正确16164，并无净大幅语义纠正。
3s同类现象。A+B 当前主要是小幅重新校准，而不是解决了小类识别。

抽样不能替代全量结果或显著性检验；稀有类的抽样 IoU 波动尤其明显。
跨运行0s有少量voxel差异，不能当成严格同一前向快照的干预实验；其原因尚未单独验证。
没有据此认定流水线随机采样：当前 LoadMultiViewImageFromMultiSweeps.load_offline
实际使用固定 adjacent 索引，不能只看 test_mode=False 就指控随机检索。

## 4. 新确认的损失—解码不一致（不是已经完成因果归因）

### 4.1 SoftVoxel 的坐标参考点错开半体素

`query_memory.py:sparse_soft_voxel_iou_loss` 使用
`coord=(p-origin)/voxel_size`，将 index k 当作下角点格点；
GT标签却表示一个体素，现有GT点监督使用 `(k+0.5)*voxel_size+origin` 的中心。
因而一个恰在GT中心的点，会向正方向8个体素各分1/8质量，而不是完整落在目标中心格点。
中心索引的三线性代理应检验 `coord=(p-origin)/voxel_size-0.5`。
这仍是软代理，不会与 hard floor 完全相同，也不等于已经批准修改现有训练。

调用**真实当前 loss** 的CPU前向实验：仅GT体素[1,1,1]为行人，voxel=0.4m，
其余为空。所有下面的点在hard evaluator中都落在同一个正确体素，logits相同：

| 点坐标(m) | 当前SoftVoxel loss |
| --- | ---: |
| (0.60,0.60,0.60)，GT中心 | 0.9962075 |
| (0.59,0.60,0.60) | 0.9960178 |
| (0.61,0.60,0.60) | 0.9963968 |
| (0.401,0.401,0.401)，低坐标角附近 | 0.9632545 |
| (0.799,0.799,0.799)，高坐标角附近 | 1.0000000 |

即中心向低坐标移动损失更低，与中心位置监督形成不一致偏好。
仅在诊断中换成中心索引后，中心loss=0.9628175，左右偏移均升至约0.964248。
这些是前向值比较，**没有执行梯度数值检查**。
数值接近1部分因为17类平均、其余类该帧无GT，不能拿loss绝对值直接解释训练失败。
对细小物体/类别边界可能比大片道路敏感；实际造成多少mIoU下降仍需单变量验证。

### 4.2 真实解码不只依赖 O

`opus_head.py:get_occ` 在sigmoid后，对top类为manmade/vegetation的点，
还按其距Query点均值的距离乘 `clamp(0.1/d,max=1)`（这里d来自归一化坐标）。
之后还有3m Query距离筛选、按top类别阈值筛点、scatter_max、closing、最终类别阈值。
这些步骤没有完全进入 occupancy/threshold 的监督。

CPU调用真实 get_occ（关闭padding，仅隔离已有类别距离因子）：
相同两点位置、相同最大logit O=logit(0.4)，只换top语义：
car保留2体素；manmade/vegetation保留0体素；诊断调用禁用该距离因子后manmade恢复2。
这证明 `max(S_decode)=O` **不等于最终voxel是否保留只由O决定**。
类别阈值、位置、Query范围及后处理仍耦合。
这段decoder是Baseline已有行为，不是A+B新增bug；但新损失/新的logit校准仍须与它兼容。
它与manmade/vegetation退化相符，但尚未在真实同快照上量化贡献。

### 4.3 训练日志支持“修正并不小”，不能只看loss下降

A+B epoch1最后20条训练日志均值：

| horizon | G_sem | G_occ | G_pos | O有符号修正 | 位置修正范数(m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1s | .7840 | .7719 | .7768 | -.7183 | .1435 |
| 2s | .7887 | .7783 | .7837 | -.6120 | .2000 |
| 3s | .7933 | .7842 | .7905 | -.5164 | .2507 |

旧Memory门控没有自动变成小值，3s位置修正相对0.4m体素已不小。
负O修正可能剔除误报，也可能剔除小类真阳性；位置移动可造成漏检；
只有最终标签不能分离两者。不能因三个Gate均值接近就断言任务分离无效，
也不能因均值不同就证明原来有梯度冲突。

语义CE/Focal按点聚合、binary正负平衡不是类别平衡；cls_weights=1–3也不等同
mIoU的类别等权。SoftVoxel用sigmoid(O)*softmax(S_sem)+累积/noisy-OR，而实际
decode用相对logits+sigmoid+scatter_max+closing，仍有概率标度/密度/形态差异。
这些属于风险机制，未测量实际梯度占比，不声称某项loss已被证明主导全部下降。

## 5. 下一步：先归因，避免再盲目加结构

1. 保持C运行/配置不变，评估时单列上述五个受损类别，不只看mean。
   C只细分semantic gate，不能直接恢复被occupancy删掉或位置移走的点；
   C的背景组同时包含获益道路和受损manmade/vegetation，不保证解决组内冲突。
2. GPU可用后，同一批缓存输入、同一次冻结Baseline未来快照、同一AB1权重，
   做小规模分支干预：只恢复baseline O；只恢复baseline位置；只恢复baseline语义。
   保留其它两项和原Memory检索；不重算随机输入；0s严格核对。
   统计每类TP/FP/FN、point过阈值率、各过滤阶段淘汰率，才能判断主因。
   这里尚未运行，不等同原来被取消的V1 shuffled/context-zero/cache-off对照。
3. 优先为SoftVoxel中心坐标对齐做独立修复与测试，之后才用独立实验目录验证；
   不混进当前C，也不顺手同时改Gate、半径、类别权重。
4. 若恢复O主要救回小类，再研究按类别的占据保护/阈值校准；若恢复位置有效，
   优先修正空间监督/位置残差；若语义替换有效，再针对语义混淆。

当前GPU0/1仍被C占用约20.7/22.7GB，本轮没有抢占GPU做模型干预。
没有找到本轮可用于同token对齐的原始Baseline保存预测，不能伪造Baseline召回对照。
因此现阶段是“类别级直接原因已量化，两个可复现机制问题已发现，分支因果贡献待闭环”，
不是“已经铁证occupancy单分支是唯一原因”，也不能宣称修复后指标一定回升。

## 6. 本轮复现与检查

新增文件只有本报告、两个CPU诊断工具、一个纯numpy测试文件。
CPU抽样工具的 `--dump NAME=directory` 可重复传入；首个是配对参考，
`--samples 256`，`--output` 拒绝覆盖已有文件。
已运行源occfix11、AB1、AB3、native11四组保存结果，输出见第3节。

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/python tools/probe_future_memory_loss_alignment.py --output work_dirs/diagnostics-ab-miou-20261015/alignment_probes.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/pytest -q tests/test_semantic_dump_diagnostics.py
git diff --check
```

前向探针断言通过；4项纯函数测试通过；diff检查通过。
输出另存 `work_dirs/diagnostics-ab-miou-20261015/alignment_probes.json`。
未提交/推送、未改checkpoint/cache、未修改历史实验数值。
