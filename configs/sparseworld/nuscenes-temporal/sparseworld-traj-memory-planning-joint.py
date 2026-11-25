"""Five independent one-epoch joint continuations; frozen Baseline, trainable V2 + fusion."""
_base_ = ['./sparseworld-traj-memory-planning-fusion.py']
model = dict(type='SparseWorld4DTrajMemoryPlanningJoint',
             query_memory_cfg=dict(future_memory_adapter_v2=dict(
                 activation_checkpoint=True, save_on_cpu=False, query_chunk_size=40)))
# Keep T1's last continuation LR for the pretrained occupancy adapter. Use the
# established planning-only LR for the fresh fusion, with identical warmup.
optimizer = dict(paramwise_cfg=dict(custom_keys={
    'future_memory_adapter_v2': dict(lr_mult=0.2),
    'memory_planning_fusion': dict(lr_mult=1.0)}))
runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
work_dir = 'work_dirs/memory-planning-joint-five-seeds-1ep-20261124'
