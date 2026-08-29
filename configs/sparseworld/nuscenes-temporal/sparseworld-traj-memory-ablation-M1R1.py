_base_ = ['./sparseworld-traj-memory-phase4-future-only.py']

# Strict 2x2 cell M=ON, R=ON.  This is the complete Phase-4 future-only path.
model = dict(
    query_memory_cfg=dict(
        enabled=True,
        memory_phase3_finetune_mode=True,
        memory_conditioned_refiner=True,
        memory_ablation_memory_enabled=True,
        memory_ablation_refiner_enabled=True,
        freeze_base_model=True,
        source='cache',
        log_diagnostics=False))

load_from = 'ckpts/epoch_56.pth'
resume_from = None
