"""D5 control: one matched epoch after T1, all three fixed history slots."""

_base_ = ['./sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py']

work_dir = 'work_dirs/future-memory-v2-internal-d5-3slots-from-t1e3'
load_from = None
resume_from = ('work_dirs/'
               'sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-'
               't1-from-center2/epoch_3.pth')
auto_resume = False
runner = dict(type='EpochBasedRunner', max_epochs=4)
