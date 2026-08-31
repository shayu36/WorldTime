# STAC-QM 文档索引

当前实验结果以 [`EXPERIMENT_DEBUG_REPORT.md`](EXPERIMENT_DEBUG_REPORT.md) 为准；它统一记录了 epoch-56 baseline、Phase 2、Phase 3、Phase 4 future-only，以及 M0R1/M1R0/M1R1 消融和中间层差异检查。

## 当前有效文档

| 文档 | 用途 | 状态 |
| --- | --- | --- |
| [`EXPERIMENT_DEBUG_REPORT.md`](EXPERIMENT_DEBUG_REPORT.md) | 当前实验数据、debug 证据、问题与结论 | **权威结果** |
| [`STAC_QM_Modeling_Repair.md`](STAC_QM_Modeling_Repair.md) | STAC-QM 建模修复、缓存验收和训练安全边界 | 有效设计/验收记录 |
| [`STAC_QM_Implementation.md`](STAC_QM_Implementation.md) | 实现说明、配置和运行手册 | 有效 runbook；结果表以本目录报告为准 |
| [`PHASE2_HANDOFF.md`](PHASE2_HANDOFF.md) | Phase 2 交接、命令和原始记录 | 历史交接，结果已汇总 |
| [`STAC_QM_CODEX_API_HANDOFF.md`](STAC_QM_CODEX_API_HANDOFF.md) | 跨会话 handoff 和设计演进 | 历史 handoff；Phase 3 旧提案部分已被实际结果取代 |
| [`QueryMemory_Implementation_Analysis.md`](QueryMemory_Implementation_Analysis.md) | 早期调用链和模块设计分析 | 设计参考 |
| [`SparseWorld_Code_Architecture.md`](SparseWorld_Code_Architecture.md) | 项目代码架构 | 通用参考 |
| [`getting_started.md`](getting_started.md)、[`prepare_datasets.md`](prepare_datasets.md) | 环境与数据准备 | 通用参考 |

## 归档

完整对话原文已移至 [`archive/conversation_transcript.md`](archive/conversation_transcript.md)，仅用于追溯，不作为实验结论来源。

大型 checkpoint、cache 和完整训练日志仍保留在 `/data/jxy/projects/work_dirs/`；`logs/` 只提供分类后的入口和符号链接，避免重复占用磁盘。
