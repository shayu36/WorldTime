"""A+B: 3 local continuation epochs from occfix epoch 11; C disabled.

Gate heads inherit the trained shared gate, horizon offsets start at zero.
This is a fresh-optimizer continuation, not a resume of the original runner.
"""
_base_ = ['./sparseworld-traj-memory-future-adapter-v2.py']

work_dir = 'work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11'
load_from = work_dir + '/init_ab_from_occfix_epoch11.pth'
resume_from = None
auto_resume = False
model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    gate_mode='task_split', horizon_gate_bias=True,
    query_chunk_size=48, activation_checkpoint=True, save_on_cpu=False)))
data = dict(samples_per_gpu=1, workers_per_gpu=1)
optimizer = dict(lr=2e-5)
# Below source epoch-11 LR (~3.09e-5). Constant LR makes this short
# adaptation window and a later matched shared-gate control easy to compare.
lr_config = dict(_delete_=True, policy='fixed', warmup='linear',
                 warmup_iters=100, warmup_ratio=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=3)
fp16 = dict(loss_scale='dynamic')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=50)
workflow = [('train', 1)]
