"""Portable release config for the exact one-epoch seed-17 protocol."""
_base_ = ['./sparseworld-traj-memory-planning-joint.py']
planning_source_checkpoint = 'artifacts/initial_t1/epoch_3.pth'
model = dict(memory_planning_source_checkpoint=planning_source_checkpoint)
load_from = None
resume_from = None
auto_resume = False
work_dir = 'reproduce_outputs/seed_17/train'
seed = 17
