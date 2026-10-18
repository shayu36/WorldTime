"""One local epoch from A+B epoch 1: only SoftVoxel grid alignment changes.

Resume preserves the A+B optimizer and learning-rate state. The old loss
remains the default in every other V2 configuration.
"""
_base_ = ['./sparseworld-traj-memory-future-adapter-v2-ab.py']

work_dir = 'work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center'
load_from = None
resume_from = ('work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/'
               'epoch_1.pth')
auto_resume = False

model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    soft_voxel_center_offset=0.5,
    soft_voxel_renormalize_valid_corners=True)))

# The source checkpoint has completed local epoch 1. Runner epoch 2 performs
# precisely one further training epoch, using the resumed optimizer state.
runner = dict(type='EpochBasedRunner', max_epochs=2)
