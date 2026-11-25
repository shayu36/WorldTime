# 输入准备与精确复现协议

所有命令从仓库根目录运行；环境安装见根 README。

## 外部文件目录和来源

```text
data/nuscenes/{samples,sweeps,maps,v1.0-trainval,lidarseg}/
data/nuscenes/gts/<scene>/<token>/labels.npz
data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
data/nuscenes/bevdetv2-nuscenes_infos_val.pkl
admlp/fengze_nuscenes_infos_train.pkl
admlp/fengze_nuscenes_infos_val.pkl
admlp/stp3_val/data_nuscene.pkl
occworld/nuscenes_infos_train_temporal_v3_scene.pkl
occworld/nuscenes_infos_val_temporal_v3_scene.pkl
stp3_val/{filter_token.pkl,stp3_traj_gt.pkl,stp3_occupancy.pkl}
data/query_memory/sparseworld_epoch56_schema2_train/<scene>/<token>.pt
data/query_memory/sparseworld_epoch56_schema2_val/<scene>/<token>.pt
```

1. 原始 nuScenes 与 lidarseg：[nuScenes downloads](https://www.nuscenes.org/download)。
2. 占据 GT：[Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)，取 nuScenes `gts`。
3. SparseWorld 预处理 infos、ADMLP/OccWorld 辅助 PKL 和 epoch_56：
   [SparseWorld 官方数据与权重](https://huggingface.co/MSunDYY2001/SparseWorld/tree/main)。
   也见仓库已有 [prepare_datasets.md](prepare_datasets.md)。这些公开下载并不等于自动证明
   与本实验输入逐字节相同，必须对照 `artifacts/input_manifest.json`。
4. `stp3_val` 可链接到已下载的 `admlp/stp3_val`，前提是其三个 GT/过滤文件与校验清单一致。

例如已经按目录放好文件后：

```bash
ln -s admlp/stp3_val stp3_val
python tools/reproduce/verify.py --model --data both
```

只复现最终评估时不用训练 split 文件，使用 `--data val`。
PKL 可能内嵌文件路径，需保持官方相对目录或建立软链接；不应为凑指标改变样本过滤。
源运行 train=19730，validation=4219 个唯一 token，planning 候选 filter=4819、匹配=4219。
外部文件不存在或 hash 不匹配时检查会明确失败，不能据此声明精确复现。

## Memory cache

缓存 schema、选择和对齐实现都在本分支：
`tools/query_memory/precompute_query_memory.py`、`loading_query_memory.py`、`query_memory.py`。
历史 Query 的特征/点/语义分布来自固定 Baseline epoch56；每帧最多256个 Query。
模型每样本读取最多3帧，目标基础年龄2.5/3.5/4.5s，训练/验证缓存分离。

若无现成缓存，可在本地使用官方 epoch56 权重生成；以下是可执行的 schema-v2 生成入口。
本次发布没有重新生成服务器已有缓存，也没有验证重新生成缓存与旧缓存逐字节一致。
重新生成的配置、权重和预处理不同可能改变结果，应保存生成记录并先核对源路径。

```bash
python tools/query_memory/precompute_query_memory.py --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py --checkpoint ckpts/epoch_56.pth --split train --output-dir data/query_memory/sparseworld_epoch56_schema2_train --max-queries-per-frame 256 --write-threshold 0.35
python tools/query_memory/precompute_query_memory.py --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py --checkpoint ckpts/epoch_56.pth --split val --output-dir data/query_memory/sparseworld_epoch56_schema2_val --max-queries-per-frame 256 --write-threshold 0.35
```

GT 文件只用于损失/评估；实际旧自车输入文件按 loader 排序拼接所有非 `gt` 字段。
此发布保持实验输入，不静默替换其字段或声称已完成未来信息审计；字段来源需要以数据生成流程为准。

## 与 Baseline 一致的规划评估

`tools/eval_memory_planning_joint.py predict` 使用原生模型的 `pred_traj [1,6,2]` 单步位移，
沿时间 `np.cumsum` 得到当前帧 XY 累计位置。`audit_planning_pair.metrics` 加入z=0，
与 `stp3_traj_gt.pkl` 的 `[:,1:7]` 和占据 GT `[:,:6]` 对齐。
评估器源码、GT SHA256 必须与 `artifacts/baseline/metrics.json` 一致。

保留的原评估器 `AD-MLP/deps/stp3/stp3/planning_metrics.py` 定义：

- 0.5m 网格，XY范围[-50,50]，使用GT占据作为碰撞图。
- Obj 只检测参考点，Box 检测1.85×4.084m自车矩形（前后参考偏移沿原代码）。
- GT矩形已碰撞的时刻从分子排除，但分母仍为全部匹配样本N；参考点越界事件不计入分子。
- 每步碰撞率为掩码后的事件数/N，前缀@h平均前2h步；不是除以未掩码样本数。
- 原JSON碰撞率为fraction，CSV和README显示百分比，乘100仅一次。
- 0%的含义受上述GT掩码和几何定义限定。

## 环境与构建说明

MMCV官方 cu118/torch2.0 wheel索引没有1.7.0，安装脚本使用附带的、和源运行binary hash一致的wheel。
其平台为Linux x86_64/CPython3.9，CUDA算子包含sm_86 cubin（RTX 3090）；其他架构需要重编译MMCV。
在其他架构可自行从源码构建 `MMCV_WITH_OPS=1 pip install --no-binary=mmcv-full mmcv-full==1.7.0`，
记录编译器/CUDA/torch和验证结果，不把它直接当作原环境逐位复现。
本仓库两个扩展由根setup.py构建，`TORCH_CUDA_ARCH_LIST`需要设成实际设备架构。
MMCV为Apache-2.0，wheel中携带LICENSE/LICENSES；其他上游源码版权见THIRD_PARTY_NOTICES。

## 复现范围

发布的是固定T1起点的联合适配代码、两个必要checkpoint和seed17的结果，不含原始Baseline训练链。
直接用发布权重评估是最直接的结果复核路径。重新训练一个epoch可复现训练协议，
不保证非确定性CUDA在不同环境下生成完全相同的checkpoint或末位成绩。
所有数据和cache都由使用者按授权准备；缺少这些输入不能只下载代码就产生正式指标。
