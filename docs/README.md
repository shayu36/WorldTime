# Project Documentation

当前实验结论以 [EXPERIMENT_DEBUG_REPORT.md](EXPERIMENT_DEBUG_REPORT.md) 为唯一入口。旧 handoff、阶段性实验记录和 `logs/` 下的重复索引已清理；原始日志和 checkpoint 仍在 `work_dirs/`。

## 保留文档

| 文档 | 主要内容 |
| --- | --- |
| [EXPERIMENT_DEBUG_REPORT.md](EXPERIMENT_DEBUG_REPORT.md) | Memory-only、joint、Phase2、Phase3、Phase4、M0R1/M1R0/M1R1、Future Adapter smoke/formal 状态的统一实验汇总 |
| [FUTURE_MEMORY_ADAPTER_MODELING.md](FUTURE_MEMORY_ADAPTER_MODELING.md) | Future Memory Adapter 建模说明：独立 output correction、1s/2s/3s 映射、active query 覆盖、candidate scoring、初始化和冻结策略 |
| [STAC_QM_Implementation.md](STAC_QM_Implementation.md) | 当前可用 runbook：环境检查、配置入口、代码验收、smoke/formal/eval 命令、产物策略 |
| [STAC_QM_Modeling_Repair.md](STAC_QM_Modeling_Repair.md) | STAC-QM 建模修复、schema-v2 cache 校验、训练安全边界和 Future Adapter 不变量 |
| [SparseWorld_Code_Architecture.md](SparseWorld_Code_Architecture.md) | SparseWorld 代码架构参考 |
| [getting_started.md](getting_started.md) | 环境与基础使用说明 |
| [prepare_datasets.md](prepare_datasets.md) | 数据集准备说明 |

## 维护规则

- 新实验结果只写入 `EXPERIMENT_DEBUG_REPORT.md`。
- 命令和 runbook 只写入 `STAC_QM_Implementation.md`。
- 建模边界和验收原则只写入对应 modeling 文档。
- 不在 `docs/` 中复制大型日志；大型产物保留在 `work_dirs/`。
