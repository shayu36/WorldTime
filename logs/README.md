# 实验日志索引

`logs/` 只保存分类入口；大型原始日志、checkpoint 和评估产物仍在 `/data/jxy/projects/work_dirs/`，这里使用相对符号链接避免复制。

## 目录

| 目录 | 内容 |
| --- | --- |
| [`phase2/`](phase2/) | Phase 2 formal 训练和 runner 原始日志 |
| [`phase3/`](phase3/) | Phase 3 connectivity smoke、formal，以及 future-only Phase 4 |
| [`ablations/`](ablations/) | M0R1、M1R0、M1R1 smoke 和 M1R0 formal |
| [`diagnostics/`](diagnostics/) | baseline/M1R0 中间层比较和 cache audit |
| [`archive/legacy/`](archive/legacy/) | 早期 epoch-51 Memory eval、旧诊断和日志片段；不作为当前结论 |

## 结果来源

当前结论和数值统一见 [`../docs/EXPERIMENT_DEBUG_REPORT.md`](../docs/EXPERIMENT_DEBUG_REPORT.md)。

日志入口与原始目录的对应关系：

- `phase2/formal-train.log` → `work_dirs/sparseworld-traj-memory-phase2/train.log`
- `phase3/formal-train.log` → `work_dirs/sparseworld-traj-memory-phase3/train.log`
- `phase3/future-only-phase4-train.log` → `work_dirs/sparseworld-traj-memory-phase4-future-only/train.log`
- `ablations/M1R0-formal-train.log` → `work_dirs/sparseworld-traj-memory-ablation-M1R0/train.log`
- `diagnostics/baseline-vs-M1R0-intermediates.json` → `work_dirs/sparseworld-traj-memory-ablation-M1R0/baseline_vs_m1r0_intermediates.json`

请勿把 `archive/legacy/` 中的旧日志与当前 epoch-56 baseline 混用。
