_base_ = ['./sparseworld-traj-finetune.py']

model = dict(
    memory_enabled=True,
    memory_bank_size=5,
    memory_embed_dims=256,
    memory_num_heads=8,
    memory_dropout=0.1,
    memory_confidence_threshold=0.3,
    memory_lambda_pos=0.01,
    memory_lambda_time=0.1,
    memory_lambda_conf=0.5,
    memory_self_noise=0.1,
)

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=1e-2)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-3
)

runner = dict(type='EpochBasedRunner', max_epochs=42)
data = dict(samples_per_gpu=4, workers_per_gpu=2)
load_from = 'ckpts/epoch_56.pth'

# Evaluate every 2 epochs, overriding interval=24 in the base config.
evaluation = dict(interval=2)
