_base_ = ['./sparseworld-traj-memory-phase4-future-only.py']

# Strict 2x2 cell M=OFF, R=ON.  The future-only route is held constant: raw
# OPUS is retained for 0s while the refiner is applied to scheduled future
# Query groups.  Only refiner parameters enter the optimizer.
model = dict(
    query_memory_cfg=dict(
        enabled=False,
        memory_phase3_finetune_mode=True,
        memory_conditioned_refiner=True,
        memory_ablation_memory_enabled=False,
        memory_ablation_refiner_enabled=True,
        freeze_base_model=True,
        source='cache',
        log_diagnostics=False))

load_from = 'ckpts/epoch_56.pth'
resume_from = None
