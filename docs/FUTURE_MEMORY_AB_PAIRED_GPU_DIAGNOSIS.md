# A+B 单快照 GPU 分支诊断（2026-10-16）

承接 `FUTURE_MEMORY_AB_MIOU_DIAGNOSIS.md`。用户确认 GPU 可用后运行，
不是新训练，也不是完整 validation。没有改动生产模型、配置、checkpoint、cache。
开始和结束 HEAD 均为 `852977be7f9a734b2d32cef781093c42831350a3`。
运行前未发现训练进程，两张3090空闲；本轮只使用GPU0。

## 方法与一致性

- checkpoint：A+B `epoch_1.pth`，SHA256
  `6d0e968915f28e1fcb14343100eb17a88484af39b94a6fdbb28331496acce597`。
- 与昨日相同的256帧、150场景，token/index清单逐一核对。
- 每帧只执行一次正常A+B前向，再对该次输出快照做干预。
  Memory检索、三个门控、Baseline递推完全不改，不重新运行分支，也不重取输入。
- 五组：正常AB；仅恢复位置；仅恢复occupancy；仅恢复semantic；原始Baseline快照参考。
- `恢复位置=(S_decode_AB, P_base)`。
- `恢复occupancy=(S_sem_AB + max(S_base)-max(S_sem_AB), P_AB)`。
- `恢复semantic=(S_base + O_AB-max(S_base), P_AB)`。
- 快照参考直接用 `S_base,P_base`。这是同次前向中已经存在的冻结输出，
  不是加载另一个模型做完整Baseline评估，也不是cache-off/V1对照。
- 所有组都调用真实 `get_occ()`，保留类别距离降分、阈值、scatter_max和closing。
- A+B checkpoint 与原始 `ckpts/epoch_56.pth` 的679个共有张量严格相等，
  没有缺失的Baseline键；全模型A+B严格加载通过。
- 256帧全部通过0s严格一致、重放AB解码严格一致检查，所有参数 `grad=None`。
- 与昨日保存AB1预测计算得到的0/1/2/3s累计混淆矩阵也全部严格相等。

所有指标为**本次子集**，不能当作新的4219帧完整验证结果。
特别是子集Future mIoU：AB=13.2491、快照参考=13.2178；此前完整评估则
AB=13.1667、Baseline=13.2233，平均差的方向不同。不能用本子集宣称AB全面超越Baseline。
本子集仍复现了关注的2s/3s mIoU低于快照参考的现象。

## 1. 配对指标

单位为百分数。

| 组别 | IoU 1s | IoU 2s | IoU 3s | mIoU 1s | mIoU 2s | mIoU 3s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正常AB | 23.9487 | 22.9697 | 21.7980 | 15.1101 | 13.0919 | 11.5453 |
| 仅恢复位置 | 23.9902 | 23.0111 | 21.7522 | 15.3319 | 13.2747 | 11.5711 |
| 仅恢复occupancy | 23.3107 | 22.4662 | 21.4376 | 14.8774 | 13.1541 | 11.7196 |
| 仅恢复semantic | 23.9599 | 22.9820 | 21.8073 | 14.7764 | 12.8035 | 11.2993 |
| 冻结Baseline快照参考 | 23.3395 | 22.4922 | 21.3749 | 14.8739 | 13.1617 | 11.6180 |

相对正常AB的Future均值差，单位百分点；括号为1000次配对场景bootstrap的95%区间：

| 干预 | Future IoU差 | Future mIoU差 |
| --- | ---: | ---: |
| 恢复位置 | +0.0124 [-0.0174,+0.0382] | +0.1434 [+0.0628,+0.2240] |
| 恢复occupancy | -0.5006 [-0.7045,-0.3171] | +0.0012 [-0.1686,+0.1710] |
| 恢复semantic | +0.0109 [+0.0088,+0.0133] | -0.2894 [-0.3474,-0.2283] |

区间只描述已选场景/帧的变化，不包括训练种子、完整数据集或重训练的不确定性；
多类别探索也未做多重比较修正。
位置恢复的2s mIoU差为+0.1827 [+0.0896,+0.2735]，3s仅+0.0259 [-0.0378,+0.0889]。
occupancy恢复的2s mIoU差+0.0622 [-0.1343,+0.2644]，3s+0.1743 [-0.0360,+0.3922]：
后两项点估计改善，但总体mIoU区间跨零，不能说稳定提升已经得到证明。

## 2. occupancy：存在选择性的召回损伤，而不是只有无害去噪

保持语义和位置不变，仅恢复O，直接救回了先前受损的多个类别：

| 类别 | 2s AB | 2s恢复O | 3s AB | 3s恢复O |
| --- | ---: | ---: | ---: | ---: |
| pedestrian | 2.744 | 5.361 | 1.043 | 3.047 |
| traffic_cone | 4.276 | 5.484 | 2.838 | 3.613 |
| manmade | 10.577 | 11.624 | 9.848 | 11.021 |
| vegetation | 15.730 | 17.059 | 14.886 | 16.264 |
| motorcycle | 5.640 | 5.975 | 4.909 | 5.200 |
| car | 18.355 | 17.569 | 15.241 | 15.143 |
| driveable_surface | 30.820 | 29.847 | 29.149 | 28.163 |

2s行人GT预测为空比例从76.96%降到64.07%，3s从78.69%降到68.66%。
2s行人IoU改善的区间为[+1.5946,+3.7078]，manmade为[+0.7524,+1.3442]，
vegetation为[+0.9872,+1.6668]；这些类别的方向比较清楚。
摩托车恢复O的区间跨零，不把其小幅点估计收益夸大成稳定改善。

但是恢复O也恢复大量假阳性：

| 指标 | 2s AB | 2s恢复O | 3s AB | 3s恢复O |
| --- | ---: | ---: | ---: | ---: |
| binary precision % | 35.2326 | 29.8784 | 34.0823 | 28.8170 |
| binary recall % | 39.7569 | 47.5233 | 37.6861 | 45.5681 |
| 非空体素/帧 | 37907.7 | 53432.9 | 36536.6 | 52250.2 |

例如2s恢复O净增加667909个occupied TP，同时净增加3306533个FP。
所以occupied IoU下降并不矛盾：当前O学到了更精确、更保守的占据筛选，
但它压掉了一部分mIoU敏感类别的真阳性。

按**Baseline预测top类**分组的平均O残差（不是GT正样本专属统计）：

| 类别 | 1s | 2s | 3s |
| --- | ---: | ---: | ---: |
| pedestrian | -0.5929 | -0.8852 | -1.0013 |
| manmade | -1.5420 | -1.5806 | -1.5468 |
| vegetation | -1.2802 | -1.3059 | -1.2351 |
| driveable_surface | -0.1045 | -0.0200 | +0.0374 |

全体平均O修正不能反映这种类别差异。行人随horizon被更强抑制，道路则基本保留。
这些分组也包含Baseline误报，负修正并不全是错；判断依据是上面的GT配对结果。
恢复O对1s mIoU反而是-0.2328，说明不能简单把所有horizon的occupancy残差全关掉。

## 3. 位置：对1s/2s宏平均有净损伤，3s因类别而异

仅恢复位置，保留AB语义与O：

- 1s mIoU +0.2217，2s +0.1827，3s +0.0259。
- 2s摩托车IoU +0.9027，95%区间[+0.2764,+1.6938]；自行车点估计+0.6143，区间跨零。
- 恢复位置同时损失道路IoU：1/2/3s分别-0.2559/-0.3074/-0.5850。
- 3s自行车+0.5186，但行人基本没有改善。

因此位置分支也有类别取舍，而不是对所有类别都有害。平均收益不应掩盖3s道路的损失。
这支持把位置监督/残差作为下一步修复重点，但**不能直接证明昨日发现的SoftVoxel半格偏差
就是全部位置误差的训练原因**。仍可能包含最近邻匹配、动态目标历史位置、其它损失的影响。

## 4. 语义：目前净作用是正面的，不应先砍掉

恢复Baseline语义、保持AB的O和位置，1/2/3s mIoU分别下降
0.3337/0.2884/0.2459，配对区间均在零以下。
这说明该checkpoint的语义残差总体在帮助，而不是主要靠损坏类别来换binary IoU。
O固定不代表最终voxel类别/占据固定：真实decoder有类别阈值和跨点聚合，故binary IoU也有微小变化。

## 5. 结论与下一步

现在证据比“只看最终dump”更强，可以表述为：

> 对AB1固定输入与权重的输出干预表明，occupancy校准提供了主要的occupied IoU收益，
> 同时损伤行人、建筑、植被等类别召回；位置修正也拖累1s/2s宏平均。
> 语义修正总体有益。不同类别、不同horizon的收益/损失相互抵消，造成2s/3s mIoU不足。

这里“主要收益”是条件干预观察，不能把几个delta直接相加当作严格的训练贡献百分比。
它没有证明某个损失项、某组梯度或Memory年龄是唯一根因，也不能预测重新训练后的数值。

建议修复顺序（本轮尚未修改/启动）：

1. 保留现有semantic分支。优先单独修正并验证SoftVoxel中心坐标对齐，研究位置残差的
   类别/时刻保护，不与其它结构变更混在一次实验中。
2. occupancy目标不是恢复所有误报，而是在保留去噪收益的同时保护受损类别真阳性；
   候选方向是类别感知的正样本平衡/anchor/残差保护及2s/3s校准。
   不建议未经验证统一降低阈值或直接关闭所有O残差。
3. 下一轮验收同时报告总体IoU/mIoU和上述受损类别TP/FP/FN，不只看mean。
4. C仅细分semantic gate，不能保证解决本轮定位的O筛选和位置问题；不将本诊断宣称为C的训练结果。

## 工具、运行与产物

新增：

- `tools/diagnose_future_memory_branches.py`
- `tools/summarize_future_memory_branches.py`
- `tests/test_future_memory_branch_diagnostics.py`
- 本报告（上一份报告追加了本次结果入口）

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 env/bin/python -u tools/diagnose_future_memory_branches.py --limit 256 --output-dir work_dirs/diagnostics-ab-miou-20261016/paired256
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/python tools/summarize_future_memory_branches.py work_dirs/diagnostics-ab-miou-20261016/paired256 --output-name summary-with-classes.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/bin/pytest -q tests/test_future_memory_branch_diagnostics.py tests/test_semantic_dump_diagnostics.py
```

输出目录已存在时工具拒绝覆盖；复跑请使用新的诊断目录。
初次2帧预检在诊断统计的CPU/GPU坐标范围张量处失败，已修复诊断工具显式device/dtype转换；
失败记录保留在 `preflight2`，通过的2帧预检在 `preflight2-v2`。
256帧约354秒（不含构建），峰值PyTorch allocated约2.23GiB，nvidia-smi占用约4.2GiB。

`paired256/` 内：`provenance.json`、`config.py`、`progress.jsonl`、`report.json`、
`per_sample_confusions.npz`、`summary.json`、`summary-with-classes.json`。
包含输入清单、checkpoint/生产源文件hash、逐样本混淆矩阵、过滤计数、逐类残差统计和区间。
新增及昨日诊断测试合计9项通过；编译和git diff --check通过。

未执行backward、optimizer step、训练、完整nuScenes评估、V1因果对照、cache重新生成、
checkpoint改写、提交或推送。诊断进程已正常退出，GPU已释放。
