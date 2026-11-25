# MemAnchorWorld：含规划监督的可复现版本

本分支 `开源可复现` 发布完整 **T1 + SoftVoxel-center + Memory 规划融合监督**代码、
seed 17 权重与复现入口。Baseline 冻结，V2 占据 Adapter 和规划融合模块分别由占据损失和
原六步轨迹损失训练。规划读取同一次模型前向的真实 Memory Context；规划 token 保留
stop-gradient，因此规划损失更新融合模块，不能表述为端到端更新整个 V2。

本实验是从同一最好 T1 epoch 3 出发、各种子独立适配一个完整 epoch，
不是整个模型从头训练。`17/23/41/73/101` 每次均使用新规划初始化和新 optimizer，
19730 个训练样本，双卡每卡 batch 1，共 9865 次迭代。

## seed 17 的正式结果

该表来自公开的 `artifacts/seed17/epoch_1.pth`，4219 帧原生模型推理与原评估器；
未关闭任何语义、占据或位置残差。

| 时距 | IoU (%) | mIoU (%) | L2 (m) | Obj Collision (%) | Box Collision (%) |
|---|---:|---:|---:|---:|---:|
| 0s | 25.6800 | 18.2000 | — | — | — |
| 1s | 23.8700 | 15.3800 | 0.1597 | 0.0000 | 0.0948 |
| 2s | 22.8600 | 13.4200 | 0.1947 | 0.0000 | 0.1007 |
| 3s | 21.8000 | 11.9200 | 0.2397 | 0.0000 | 0.1106 |
| Future/Avg. | 22.8433 | 13.5733 | 0.1980 | 0.0000 | 0.1021 |

IoU/mIoU 平均仅含 1/2/3s。规划表使用与本地 Baseline 一致的 **前缀平均**：
1/2/3s 分别平均前 2/4/6 个预测点；不是末端误差。评估文件也保留六步及末端指标。
Obj 是轨迹参考点碰撞，Box 是自车矩形碰撞；两者使用原评估器的 GT box collision mask。
0% 指该指标在该掩码下为零，不代表未掩码事件或所有几何定义下都无碰撞。

原始结果：[全部指标](artifacts/seed17/all_metrics.json)、
[规划指标](artifacts/seed17/planning_metrics.json)、
[checkpoint 审计](artifacts/seed17/checkpoint_audit.json)。

## 下载代码和权重

```bash
sudo apt-get install -y git-lfs build-essential libgl1 libglib2.0-0
git lfs install
git clone --branch '开源可复现' https://github.com/shayu36/WorldTime.git
cd WorldTime
git lfs pull
```

GitHub 的 ZIP 下载可能只包含 LFS 指针；使用上面的 clone + lfs pull。
两份 checkpoint 及实际使用的 MMCV wheel 均在 Git LFS 中，合计约 660 MB。
[校验清单](artifacts/weights_manifest.json)包括：

- `artifacts/seed17/epoch_1.pth`：完整联合模型、optimizer 和训练 meta，328667873 bytes。
  SHA256 `a09d1d20f2b6049b2d396f4342e416b9c810541745d891e380926866d2404760`。
- `artifacts/initial_t1/epoch_3.pth`：所有种子共同的初始化，325954537 bytes。
  SHA256 `bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`。

## 环境安装

参考平台：Ubuntu 22.04、Python 3.9、CUDA toolkit 11.8、GCC 11、PyTorch 2.0.1+cu118，
两张 RTX 3090 24 GB。CUDA toolkit 必须包含 nvcc，单装 PyTorch CUDA runtime 不足以编译扩展。
建议至少 32 GB 主机内存，数据盘另计。

```bash
conda create -n memanchorworld python=3.9 -y
conda activate memanchorworld
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
bash tools/reproduce/install.sh
```

安装脚本按顺序安装 torch、MMCV、torch-scatter、[requirements.txt](requirements.txt)，
再用 [setup.py](setup.py) 编译 BEVPool v2 和 MSMV sampling 两个扩展并安装当前仓库。
这里包含一个与实际运行二进制完全一致的 `mmcv-full==1.7.0` wheel，适用 CPython 3.9/Linux x86_64/
CUDA 11.8/PyTorch 2.0.1/sm_86；官方 cu118/torch2.0 wheel 索引只有 1.7.2，不能直接替换本次版本。
其他平台从源码构建方法见[复现说明](docs/REPRODUCIBILITY.md)。
历史机器的混合环境 freeze 留在 `requirements/historical-freeze.txt`，不要用它代替本安装脚本。

## 准备外部输入

**本仓库不上传数据、GT、图片、轨迹标签或 Memory cache。**
需要 nuScenes trainval + lidarseg、Occ3D gts、SparseWorld 预处理 infos/自车数据、
ST-P3 规划 GT 与固定 schema-v2 Memory cache。完整目录、公开来源、校验清单与 cache 生成命令
见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。这些输入不是仅有原始 nuScenes 图片即可替代的。

```bash
python tools/reproduce/verify.py --model --data val
```

该命令校验两个 checkpoint、模型严格加载、参数范围、外部输入 SHA256 与 cache 路径。
在训练前用 `--data both` 检查所有输入。可先只用 `--environment-only` 检查环境，不需要数据。

## 复现 seed 17 的完整评估

```bash
python tools/reproduce/run.py evaluate --gpus 0,1 --output reproduce_outputs/published_seed17
```

脚本串行执行双卡 4219 帧推理、CPU 占据评估和 Baseline 同口径规划评估，
在 `reproduce_outputs/published_seed17/eval_epoch_1/all_metrics.json` 输出完整结果。
同时保留无损占据预测、轨迹、每类 confusion matrices、checkpoint/配置/源码 SHA256。
进程失败时已有输出不被覆盖；请检查日志并指定新输出目录。

首次排障可执行下面 4 帧推理；它不会产生正式评估成绩：

```bash
python tools/reproduce/run.py predict --gpus 0,1 --limit 4 --output reproduce_outputs/probe4
```

## 独立训练一个 epoch，随后评估

```bash
python tools/reproduce/run.py train --seed 17 --gpus 0,1 --output reproduce_outputs/retrained_seed17
python tools/reproduce/run.py evaluate --gpus 0,1 --checkpoint reproduce_outputs/retrained_seed17/train/epoch_1.pth --output reproduce_outputs/retrained_seed17_eval
```

每次从共同 T1 初始化，新规划模块按 seed 初始化，新 optimizer；不恢复先前实验状态。
V2 LR=2e-5，planning LR=1e-4，固定 LR，100-step linear warmup，AdamW weight decay=.01。
Baseline 保持 eval/frozen，只有 77 个参数张量（1722147 标量）可训练。
AMP dynamic loss scaling、q40、activation checkpoint 和全部 T1 损失保留。

完整五种子队列：

```bash
python tools/reproduce/run.py five-seeds --gpus 0,1 --output reproduce_outputs/joint_campaign
```

每种子训练后完整评估再进入下一种子；逐次保留所有时距指标和 mean/sample SD(ddof=1)，
不会把未完成种子当零。请使用 tmux 运行长任务。在本次机器上每 epoch 约 6.4 小时，
完整评估约 40 分钟；时间随设备和 I/O 变化。

## 检查与边界

```bash
python -m pytest -q tests/test_planning_joint_checkpoint.py tests/test_trainable_memory_planning.py tests/test_joint_five_seed_protocol.py deployment/test_planning_details.py
```

原运行已有真实样本梯度、冻结/更新检查、40 步双卡检查和上述 16 项测试；记录在
[artifacts/verification](artifacts/verification)。发布包另有独立环境与推理检查，
详见 [发布验收](docs/RELEASE_VALIDATION.md)。固定 checkpoint 评估和重新训练应分别核对；
不同 GPU、CUDA、库版本及非确定性 kernel 可能影响末位数，不能保证跨平台逐比特一致。

## 源码与许可

基于 SparseWorld / MMDetection3D / BEVDet / OPUS 及其上游依赖。保留原始版权声明。
ST-P3/UniAD 规划评估源码作为精确评估依赖单独附带，归属说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本分支未声称取得上游未授予的许可；
外部 nuScenes/Occ3D 数据按各自条款获取，不随本发布提供。
