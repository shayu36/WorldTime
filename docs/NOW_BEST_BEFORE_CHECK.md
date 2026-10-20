# 修复前最佳结果代码快照

用户指定分支名 `now best before check` 含空格，不是合法 Git 分支名；使用
`now-best-before-check`。此分支保存 T1 epoch 3 的原代码与运行入口，
不混入后续 causal ego、Memory provenance 或 base-age 防御修复。

## 已有正式结果（原输入协议）

模型：Future Memory Adapter V2，A+B 门控 + SoftVoxel-center + T1。
4219 帧原生 checkpoint 推理；没有关闭任何 semantic/occupancy/position 分支。

| 时刻 | IoU | mIoU |
| --- | ---: | ---: |
| 0s | 25.68 | 18.20 |
| 1s | 23.88 | 15.39 |
| 2s | 22.86 | 13.40 |
| 3s | 21.78 | 11.87 |
| Future Mean | 22.8400 | 13.5533 |

均值由已保存的两位小数指标计算，不代表底层混淆矩阵具有四位小数精度。

配置：
`configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py`

checkpoint（仅保留本地，不上传 Git）：
`work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth`

SHA256：`bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`。

训练入口 `tools/run_train_future_memory_occ_t1.sh` 从 center epoch 2 恢复；
它不是从本快照 epoch 3 继续训练新实验的命令。评估入口为
`tools/run_eval_future_memory_occ_t0_t1.sh t1`，会拒绝覆盖已有评估目录。
有效运行配置及完整评估产物仍保存在本地原 work-dir。

## 源码身份与范围

`NOW_BEST_BEFORE_CHECK.json` 收录原训练/评估 provenance、checkpoint 哈希及指标。
快照中的 `query_memory.py`、`sparseworld_4d_traj.py`、T1 配置和训练脚本 SHA256
与原训练 provenance 完全一致。数据集、loader 和缓存生成入口保留修复前版本。
V1 配置和旧 checkpoint 不变。checkpoint、数据集、cache、日志不提交。

这是历史结果归档，不是对输入因果性的认证。后续审计发现，旧外部 ego 字段及由其
生成的历史 Memory 可能包含未来轨迹信息；上述指标保留，但不能据此声明新增
Memory 通道已排除未来信息。修复和复验应在独立分支、独立 cache/work-dir 进行。
