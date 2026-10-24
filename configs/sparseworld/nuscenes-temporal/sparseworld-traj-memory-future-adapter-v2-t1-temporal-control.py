"""Matched T1 continuation: no temporal read, fresh optimizer, 3 local epochs.

Legacy input/cache protocol, not a temporal-causality repair. Both arms must
use the same data/code revision; do not mix with the memory-causal experiment.
"""
_base_ = ['./sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py']

load_from = ('work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-'
             'center-t1-from-center2/epoch_3.pth')
resume_from = None
auto_resume = False
work_dir = 'work_dirs/t1-temporal-control-from-best3'
runner = dict(type='EpochBasedRunner', max_epochs=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
seed = 17
model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    temporal_adapter_enabled=False)))
# Inherit fixed LR 2e-5, 100-step warmup, AMP, q48, batch=1/GPU, unchanged
# T1 and SoftVoxel-center losses. Use load_from, NOT resume_from: treatment
# adds parameters and both arms intentionally start fresh AdamW/scaler state.
temporal_experiment_protocol = 'legacy_inputs__matched_fresh_optimizer'
