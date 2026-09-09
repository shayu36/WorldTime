# STAC-QM / Future Memory Runbook

本文只保留命令、配置和操作边界。实验数据、debug 结论和未解决问题统一见 [EXPERIMENT_DEBUG_REPORT.md](EXPERIMENT_DEBUG_REPORT.md)。建模修复和训练安全边界见 [STAC_QM_Modeling_Repair.md](STAC_QM_Modeling_Repair.md)。Future Memory Adapter 的独立建模说明见 [FUTURE_MEMORY_ADAPTER_MODELING.md](FUTURE_MEMORY_ADAPTER_MODELING.md)。

## 基本环境

默认项目路径：

```bash
cd /data/jxy/projects
export PROJECT_ROOT=/data/jxy/projects
export PYTHON_BIN=/data/jxy/projects/env/bin/python
export TORCHRUN_BIN=/data/jxy/projects/env/bin/torchrun
```

关键前置文件：

```bash
test -f ckpts/epoch_56.pth
test -d data/query_memory/sparseworld_epoch56_schema2_train
test -d data/query_memory/sparseworld_epoch56_schema2_val
```

默认不重新生成 fixed memory cache。需要替换 cache 时必须明确记录新来源、checkpoint、schema version 和生成命令。

## 配置入口

| 配置 | 用途 | 备注 |
| --- | --- | --- |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py` | 原始 baseline | 不为 Memory 实验修改行为 |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py` | STAC-QM 数据/cache plumbing | train split 使用 schema-v2 train cache |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm-val.py` | STAC-QM validation plumbing | val/test 使用 schema-v2 val cache |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py` | 旧 Memory-only formal | 历史实验入口 |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only-smoke.py` | 旧 Memory-only smoke | 历史连通性检查 |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint.py` | 旧 joint formal | 历史实验入口 |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-joint-smoke.py` | 旧 joint smoke | 历史连通性检查 |
| `configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter.py` | 当前 Future Memory Adapter | 当前主线 |

## 代码级检查

运行本阶段相关静态检查：

```bash
${PYTHON_BIN} -m py_compile \
  mmdet3d/models/sparsedetectors/query_memory.py \
  mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py \
  mmdet3d/datasets/pipelines/loading_query_memory.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter.py \
  tests/test_future_memory_adapter.py

git diff --check
```

运行相关快速测试：

```bash
${PYTHON_BIN} -m pytest -q \
  tests/test_query_memory.py \
  tests/test_query_memory_integration.py \
  tests/test_sparseworld_evaluation.py \
  tests/test_future_memory_adapter.py
```

配置解析和 model build：

```bash
${PYTHON_BIN} - <<'PY'
from mmcv import Config
from mmdet3d.models import build_model

cfg = Config.fromfile(
    'configs/sparseworld/nuscenes-temporal/'
    'sparseworld-traj-memory-future-adapter.py')
model = build_model(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
print(type(model).__name__)
print(type(model.future_memory_adapter).__name__)
print(model.query_memory)
PY
```

预期：模型为 `SparseWorld4DTraj`，`future_memory_adapter` 存在，dedicated Future Adapter 配置下 `query_memory` 为 `None`。

## Future Memory Adapter smoke

smoke 只用于检查训练接线和短程稳定性，不作为性能结论。

```bash
export FUTURE_CFG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter.py

mkdir -p work_dirs/sparseworld-traj-memory-future-adapter-smoke
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1 ${TORCHRUN_BIN} \
  --nproc_per_node=2 \
  --master_port=29514 \
  tools/train.py \
  "${FUTURE_CFG}" \
  --work-dir work_dirs/sparseworld-traj-memory-future-adapter-smoke \
  --launcher pytorch \
  --deterministic \
  --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=2 \
    runner._delete_=True \
    runner.type=IterBasedRunner \
    runner.max_iters=200 \
    lr_config._delete_=True \
    lr_config.policy=CosineAnnealing \
    lr_config.by_epoch=False \
    lr_config.warmup=linear \
    lr_config.warmup_iters=50 \
    lr_config.warmup_ratio=0.3333333333333333 \
    lr_config.min_lr_ratio=0.001 \
    checkpoint_config._delete_=True \
    checkpoint_config.by_epoch=False \
    checkpoint_config.interval=100 \
    checkpoint_config.max_keep_ckpts=2 \
    checkpoint_config.save_last=True \
    evaluation.interval=1000000000 \
    log_config.interval=20 \
  2>&1 | tee work_dirs/sparseworld-traj-memory-future-adapter-smoke/train.log
```

检查要点：

- train log 中存在 `mem_1s.loss_*`、`mem_2s.loss_*`、`mem_3s.loss_*`；
- `grad_norm` 非零；
- 没有触发 Memory cache schema/shape 异常；
- 没有执行 full evaluation。

## Future Memory Adapter formal

正式训练使用 fixed cache 和 baseline checkpoint，只训练新增 Future Adapter 参数。

```bash
export FUTURE_CFG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter.py

mkdir -p work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1 ${TORCHRUN_BIN} \
  --nproc_per_node=2 \
  --master_port=29519 \
  tools/train.py \
  "${FUTURE_CFG}" \
  --work-dir work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix \
  --launcher pytorch \
  --deterministic \
  --cfg-options \
    runner.max_epochs=12 \
    log_config.interval=50 \
  2>&1 | tee work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/train.log
```

如果该任务已经在 tmux 中运行，不要重复启动。先检查：

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'sparseworld-traj-memory-future-adapter|tools/train.py|torchrun'
tail -n 80 work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/train.log
```

## Future Adapter evaluation

formal 训练完成后再评估。评估应至少包括：

1. Future Adapter ON：使用训练得到的 checkpoint。
2. Future Adapter OFF：同一配置关闭 adapter 或使用 baseline checkpoint 对照。
3. shuffled/cache-control：确认 Memory 内容是否提供因果贡献。

不要在 formal 训练未完成前填写 IoU/mIoU 结论。

示例命令需要按实际 checkpoint 调整：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 ${TORCHRUN_BIN} \
  --nproc_per_node=2 \
  --master_port=29520 \
  tools/test.py \
  --config "${FUTURE_CFG}" \
  --checkpoint work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/epoch_12.pth \
  --launcher pytorch \
  --eval segm \
  --deterministic \
  2>&1 | tee work_dirs/sparseworld-traj-memory-future-adapter-formal-batch4fix/eval_epoch12_adapter_on.log
```

## Cache audit

cache audit 不会重新生成 cache，可用于确认 schema 和 split routing：

```bash
${PYTHON_BIN} tools/query_memory/audit_query_memory_cache.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --split train \
  --expected-schema-version 2 \
  --expected-source-checkpoint ckpts/epoch_56.pth \
  --json-out work_dirs/stacqm_cache_audit_train.json

${PYTHON_BIN} tools/query_memory/audit_query_memory_cache.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --split val \
  --expected-schema-version 2 \
  --expected-source-checkpoint ckpts/epoch_56.pth \
  --json-out work_dirs/stacqm_cache_audit_val.json
```

## 产物策略

不要提交：

- `work_dirs/` 下的 checkpoint、日志、TensorBoard event；
- `data/query_memory/` 下的 `.pt` cache；
- evaluation pkl、临时 diagnostic json；
- tmux 捕获日志。

可以提交：

- model/dataset/config/test 代码；
- `docs/` 下的简洁建模和实验汇总文档。
