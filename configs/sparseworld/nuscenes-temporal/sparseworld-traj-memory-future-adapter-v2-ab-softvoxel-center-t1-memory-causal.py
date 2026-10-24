"""Memory-only causal repair; current Baseline inputs and all T1 losses unchanged.

Start from the best old T1 weights; fresh AdamW for 3 local adaptation epochs.
This is NOT a fully causal Baseline retrain and NOT a new reported result.
"""
from tools.query_memory.memory_only_protocol import memory_only_data

_base_ = ['./sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py']
data = memory_only_data()
work_dir = 'work_dirs/t1-memory-causal-from-best3'
load_from = ('work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-'
             't1-from-center2/epoch_3.pth')
resume_from = None
auto_resume = False
runner = dict(type='EpochBasedRunner', max_epochs=3)
# Inherit fixed LR 2e-5, 100 warmup iters, batch=1/GPU, q48, AMP,
# task/horizon gates, and T1 3s beta=.25. Preserve every local checkpoint.
checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
memory_input_protocol = 'causal_history_only__legacy_current_baseline'
