# MemAnchorWorld — 规划监督

本分支归档该实验的完整项目源码、配置、训练/评估入口、测试和已有结果。没有在发布时重训或重评。

从最好 T1 epoch3 初始化新 Memory 规划融合模块，Baseline/原规划头/V2保持冻结。轨迹监督训练 memory_planning_fusion；epoch1–4及全量评估均保留。为使完整代码与实际训练一致，本次将原始训练快照中的 Python 文件恢复到分支根目录；原快照另完整保存在 publication/training_source_snapshot.tar.gz，未混入新的联合监督版。既有40e16d7发布版曾移除未启用的其他消融代码，本次恢复的是训练时已有代码，不是新建模实验。占据结果使用原记录精度，规划主口径为前缀平均。

## 实际结果

单位为百分比。Future Mean 仅平均 1s、2s、3s，不含 0s。完整精度与逐类结果以 `results/` 的原文件为准。

| Seed | Epoch | mIoU 1s | 2s | 3s | Future mIoU | IoU 1s | 2s | 3s | Future IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 15.3900 | 13.4000 | 11.8700 | 13.5533 | 23.8800 | 22.8600 | 21.7800 | 22.8400 |
| 17 | 2 | 15.3900 | 13.4000 | 11.8700 | 13.5533 | 23.8800 | 22.8600 | 21.7800 | 22.8400 |
| 17 | 3 | 15.3900 | 13.4000 | 11.8700 | 13.5533 | 23.8800 | 22.8600 | 21.7800 | 22.8400 |
| 17 | 4 | 15.3900 | 13.4000 | 11.8700 | 13.5533 | 23.8800 | 22.8600 | 21.7800 | 22.8400 |

## 代码与证据

- 原始基础提交：`40e16d7d8f2ec5fca9adf413ca845846207ffde4`（WorldTime）。本仓库提交号是发布身份，不替代训练时身份。
- 训练时源码哈希见 `publication/manifest.json`；对应源码逐文件核验后放在项目原位置。
- `results/` 保留原始 metrics、展开配置、训练/评估记录及可获得的混淆矩阵、轨迹结果。大型文本日志无损 gzip 压缩。
- `results/summary.json` / `summary.csv` 汇总本分支全部已完成轮次；不补造未运行结果。
- 每个原始结果文件的路径及 SHA256 见 `publication/evidence.json`。历史记录中的绝对路径保留，复跑时需映射到本机目录。
- 相关说明：`docs/MEMORY_PLANNING_FUSION_TRAINING.md`。

## 环境与复现边界

原环境为 Python 3.9、PyTorch 2.0.1+cu118、MMCV 1.7.0、MMDetection 2.28.0。依赖见 `requirements.txt`，项目准备说明在 `docs/`。CUDA 扩展源码随项目保留，二进制扩展需按本机 GPU 编译。

数据集、GT 文件、固定 Memory cache、Python 环境、checkpoint 权重、RNG 权重 sidecar、逐帧稠密 occupancy dump、SSH 凭据均未上传。它们不是代码缺失：数据路径、起点权重身份和可获得的权重哈希保留在 study/registry/run/provenance 文件中。复跑需要自行提供匹配的数据与权重，不能只下载仓库便声称复现成功。

各组实际训练命令、展开配置与评估命令见 `results/` 中的 run/launch/registry 记录。旧队列会使用原服务器绝对路径或前序初始化目录；迁移机器时须映射这些路径。不要把共享队列默认的五轮预算当作本分支实际完成轮次，也不要更改学习率日程分母来伪造相同预算。

仅核验发布文件（无训练、无 GPU）：

```bash
python tools/verify_published_experiment.py
```
