_base_ = ['./sparseworld-traj-memory-phase4-future-only.py']

# Strict 2x2 cell M=ON, R=OFF.  Keep the Phase-4 future-only route and train
# only STAC-QM; the refiner is disabled both in forward and in the optimizer.
model = dict(
    query_memory_cfg=dict(
        enabled=True,
        memory_phase3_finetune_mode=True,
        memory_conditioned_refiner=False,
        memory_ablation_memory_enabled=True,
        memory_ablation_refiner_enabled=False,
        freeze_base_model=True,
        source='cache',
        log_diagnostics=False))

load_from = 'ckpts/epoch_56.pth'
resume_from = None
