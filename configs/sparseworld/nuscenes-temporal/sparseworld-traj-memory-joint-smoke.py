_base_ = ['./sparseworld-traj-memory-joint.py']

# Short user-run connectivity and frozen-state gate for joint fine-tuning.
model = dict(query_memory_cfg=dict(log_diagnostics=True))

data = dict(samples_per_gpu=1, workers_per_gpu=2)
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=200)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
optimizer_config = dict(
    _delete_=True,
    type='QueryMemoryJointConnectivityOptimizerHook',
    grad_clip=dict(max_norm=5, norm_type=2),
    connectivity_check_iter=100,
    log_interval=20,
    expected_memory_reads=7,
    expected_fused_queries=1040)
checkpoint_config = dict(
    _delete_=True,
    by_epoch=False,
    interval=100,
    max_keep_ckpts=2,
    save_last=True)
evaluation = dict(interval=10**9, by_epoch=False)
custom_hooks = []
log_config = dict(interval=20)

load_from = 'ckpts/epoch_56.pth'
resume_from = None
