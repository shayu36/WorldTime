# 发布验收（2026-11-25）

发布目录与正在运行的实验目录独立。未修改正在运行的代码、cache、数据或checkpoint。

## 模型与实测结果对应

- seed17完整checkpoint SHA256：
  `a09d1d20f2b6049b2d396f4342e416b9c810541745d891e380926866d2404760`。
- 初始化T1 SHA256：
  `bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c`。
- `artifacts/source_identity.json`逐文件核对模型、数据加载和规划评估器与真实训练/评估目录一致。
- seed17完成9865次迭代，checkpoint审计确认Baseline不变、V2更新，融合最终残差头非零。
- `artifacts/seed17/all_metrics.json`来自既有4219帧完整推理与评估，未更改数值。

## 独立环境验收

使用新建Python3.9 venv（`include-system-site-packages=false`），按安装脚本安装torch2.0.1+cu118、
随附MMCV1.7.0 wheel、torch-scatter和requirements，再从源码构建本仓库两个CUDA扩展。
复用同一服务器CUDA toolkit/驱动和只读数据；没有复制旧环境site-packages。
依赖完整解析版本记录在`requirements/constraints-py39-cu118.txt`。

已验证：

- `pip check`无依赖冲突。
- 两个CUDA扩展成功编译、导入。
- 发布checkpoint与wheel SHA256校验通过。
- 完整模型`load_state_dict(strict=True)`通过；1722147个可训练参数标量，77个参数张量。
- Baseline冻结，训练模块仅为V2 Adapter和planning fusion，两个LR分别2e-5、1e-4。
- 外部validation输入文件hash与源运行相符。
- 四个专项测试文件：16 passed；原环境和新环境分别运行通过。
- 新环境下用发布入口、发布checkpoint进行4帧双卡原生推理，与原seed17完整推理产物配对：
  所有占据体素逐项严格相等，四条轨迹严格相等，最大差值0。
  记录见`artifacts/verification/release_forward_comparison.json`。
- 安装脚本shell检查、Python编译、三个公开入口`--help`、`git diff --check`通过。

本次4帧检查不代替4219帧完整validation。发布整理期间没有重新训练整个模型或重新生成cache。
实际完整评估结果来自发布checkpoint原有运行，未把小样本检查冒充新的全量指标。
跨GPU架构或编译环境的数值一致性尚未验证；附带MMCV wheel包含sm_86 CUDA算子。

## 可复算边界

仓库提供模型、依赖、构建、可移植入口、训练配置、权重、结果和输入校验值。
原始数据/预处理PKL/GT/cache按README和复现文档自行准备；未将任何数据文件推送Git。
已给出公开数据来源与cache生成入口，但本轮没有重新生成并证明cache逐字节相同。
若外部输入hash不同，应先核对预处理及文件来源，不能直接称为本次运行的精确复算。
