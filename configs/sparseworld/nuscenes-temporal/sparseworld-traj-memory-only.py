_base_ = ['./sparseworld-traj-finetune-stacqm.py']

# Formal STAC-QM Memory-only fine-tuning. The inherited train/val/test pipelines
# use split-specific schema-v2 cache roots with strict loading.
model = dict(
    query_memory_cfg=dict(
        enabled=True,
        source='cache',
        memory_finetune_mode=True,
        freeze_base_model=True,
        log_diagnostics=False))

# Two-GPU formal run: 2 samples per GPU, global batch size 4.
data = dict(samples_per_gpu=2)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=-1,
    save_last=True)

load_from = 'ckpts/epoch_56.pth'
resume_from = None
