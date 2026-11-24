"""RecallOcc 2x2: fresh adapters, fixed epoch56 base, original input protocol.

Default is the full E4 model. All arms train from epoch zero, with a fresh
optimizer; the archived multi-stage T1 adapter is never an initializer.
"""
_base_ = ['../nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py']
load_from = 'ckpts/epoch_56.pth'
resume_from = None
auto_resume = False
recallocc = dict(experiment='E4', seeds=[0, 1, 2],
                 checkpoint_selection='final_epoch', effective_batch_size=2,
                 base_checkpoint='ckpts/epoch_56.pth',
                 protocol='existing_epoch56_schema2')
model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    memory_read_mode='point', memory_gate_mode='task_horizon',
    history_slot_limit=None)))
# Keep T1 optimizer, fixed LR=2e-5, warmup=100, losses, SoftVoxel-center,
# class balance, cache, 1/2/3s supervision and thresholds identical in all arms.
# Three complete NEW epochs; this is not the archived 2->3 continuation.
runner = dict(type='EpochBasedRunner', max_epochs=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=3, save_last=True)
find_unused_parameters = False
