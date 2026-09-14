_base_ = ['./sparseworld-traj-memory-phase4-future-only.py']

# Strict 2x2 cell M=OFF, R=OFF.  This cell is the fixed epoch-56 checkpoint
# baseline and is evaluated without a new fine-tuning run.
model = dict(
    query_memory_cfg=dict(
        enabled=False,
        memory_phase3_finetune_mode=False,
        memory_conditioned_refiner=False,
        memory_ablation_memory_enabled=False,
        memory_ablation_refiner_enabled=False,
        freeze_base_model=False,
        memory_phase3_future_only=True))

load_from = 'ckpts/epoch_56.pth'
resume_from = None
