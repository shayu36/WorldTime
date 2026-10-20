"""T1: one matched continuation epoch; class-balance positive BCE at 3s only.

Compare against T0 resumed from the SAME center epoch 2 with the same optimizer,
seed, LR, batch size and epoch budget. This does not change inference or gates.
"""
_base_ = ['./sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center.py']

work_dir = ('work_dirs/'
            'sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2')
load_from = None
resume_from = ('work_dirs/'
               'sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-from-ab1/'
               'epoch_2.pth')
auto_resume = False

model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    # Conservative first experiment: 75% existing point mean, 25% mean of
    # per-class positive means. All present GT classes participate equally.
    # No change to the negative BCE, threshold loss, or outer loss weights.
    occupancy_positive_class_balance_beta={'1s': 0.0, '2s': 0.0, '3s': 0.25},
    occupancy_class_balance_diagnostics=True)))

# Resume after epoch 2; exactly one new epoch, numbered 3. Do not reset LR,
# optimizer moments, FP16 scaler or runner iteration count with load_from.
runner = dict(type='EpochBasedRunner', max_epochs=3)
