"""Planning-only adaptation of frozen best T1; no training launched by this file.

Fresh optimizer for the NEW fusion, not resume of the occupancy optimizer.
One epoch is an initial screen, not a claim of convergence or planning benefit.
"""
_base_ = ['./sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py']

planning_source_checkpoint = (
    'work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/'
    'epoch_3.pth')
model = dict(
    type='SparseWorld4DTrajMemoryPlanning',
    memory_planning_source_checkpoint=planning_source_checkpoint,
    memory_planning_cfg=dict(
        embed_dims=256, num_classes=17, hidden_dims=128, num_heads=4,
        horizon_count=3, voxel_size=(0.4, 0.4, 0.4)),
    query_memory_cfg=dict(freeze_base_model=True,
                         future_memory_adapter_v2=dict(
                             activation_checkpoint=False, save_on_cpu=False)))

work_dir = 'work_dirs/memory-planning-fusion-from-t1-e3'
# init_weights strictly loads T1 with only memory_planning_fusion.* absent.
load_from = None
resume_from = None
auto_resume = False
seed = 17
optimizer = dict(_delete_=True, type='AdamW',
                 constructor='TrainableOnlyOptimizerConstructor',
                 lr=1e-4, weight_decay=1e-2,
                 paramwise_cfg=dict(bypass_duplicate=True))
lr_config = dict(_delete_=True, policy='fixed', warmup='linear',
                 warmup_iters=100, warmup_ratio=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=1)
data = dict(samples_per_gpu=1, workers_per_gpu=1)
fp16 = dict(loss_scale='dynamic')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
find_unused_parameters = False
