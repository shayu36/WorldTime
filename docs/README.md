# Project Documentation

Use [EXPERIMENT_DEBUG_REPORT.md](EXPERIMENT_DEBUG_REPORT.md) as the sole entry point for current experimental conclusions.
Old handoffs, interim experiment records, and duplicate indexes under `logs/` have been removed;
raw logs and checkpoints remain in `work_dirs/`.

## Retained documents

| Document | Main contents |
| --- | --- |
| [EXPERIMENT_DEBUG_REPORT.md](EXPERIMENT_DEBUG_REPORT.md) | Unified summary of Memory-only, joint, Phase2, Phase3, Phase4, M0R1/M1R0/M1R1, and Future Adapter smoke/formal experiment status |
| [FUTURE_MEMORY_ADAPTER_MODELING.md](FUTURE_MEMORY_ADAPTER_MODELING.md) | Future Memory Adapter modeling: independent output correction, 1s/2s/3s mapping, active-query coverage, candidate scoring, initialization, and freezing strategy |
| [STAC_QM_Implementation.md](STAC_QM_Implementation.md) | Current runbook: environment checks, configuration entry points, code acceptance checks, smoke/formal/eval commands, and artifact policy |
| [STAC_QM_Modeling_Repair.md](STAC_QM_Modeling_Repair.md) | STAC-QM modeling repairs, schema-v2 cache validation, training safeguards, and Future Adapter invariants |
| [SparseWorld_Code_Architecture.md](SparseWorld_Code_Architecture.md) | SparseWorld code architecture reference |
| [getting_started.md](getting_started.md) | Environment and basic usage |
| [prepare_datasets.md](prepare_datasets.md) | Dataset preparation |

## Maintenance rules

- Add new experimental results only to `EXPERIMENT_DEBUG_REPORT.md`.
- Add commands and runbook procedures only to `STAC_QM_Implementation.md`.
- Add modeling boundaries and acceptance principles only to the relevant modeling documents.
- Do not copy large logs into `docs/`; retain large artifacts in `work_dirs/`.
