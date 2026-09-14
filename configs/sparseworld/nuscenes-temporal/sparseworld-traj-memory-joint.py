_base_ = ['./sparseworld-traj-finetune-stacqm.py']

# Formal clean-start joint fine-tuning. Fixed epoch-56 Query caches remain aligned
# because the image backbone, neck, and pts_bbox_head stay frozen and in eval.
model = dict(
    query_memory_cfg=dict(
        enabled=True,
        source='cache',
        memory_finetune_mode=False,
        memory_joint_finetune_mode=True,
        freeze_base_model=True,
        log_diagnostics=False))

# Two-GPU formal run: 2 samples per GPU, global batch size 4.
data = dict(samples_per_gpu=2)

# Base LR applies to position_encoder/reg/vel/cls. Query Memory learns faster;
# ego cross-attention adapts conservatively.
optimizer = dict(
    type='AdamW',
    constructor='TrainableOnlyOptimizerConstructor',
    lr=1e-5,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys={
            'query_memory': dict(lr_mult=5.0),
            'ego_cross_attn': dict(lr_mult=0.5),
        },
        bypass_duplicate=True))
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
evaluation = dict(interval=1)

load_from = 'ckpts/epoch_56.pth'
resume_from = None
