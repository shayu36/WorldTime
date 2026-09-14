_base_ = ['./sparseworld-traj-memory-joint.py']

# Phase 3: keep the six original OPUS decoder layers and the image feature
# space frozen.  Only STAC-QM plus the shared Memory-conditioned refiner and
# its current/future output branches are optimized.
model = dict(
    query_memory_cfg=dict(
        memory_joint_finetune_mode=False,
        memory_phase2_finetune_mode=False,
        memory_phase3_finetune_mode=True,
        memory_conditioned_refiner=True,
        memory_phase3_base_aux_weight=0.25,
        fusion_gate_bias=-1.0,
        fusion_alpha_init=1e-3,
        fusion_out_proj_gain=0.1,
        freeze_base_model=True,
        log_diagnostics=False))

data = dict(samples_per_gpu=2)

optimizer = dict(
    type='AdamW',
    constructor='TrainableOnlyOptimizerConstructor',
    lr=1e-5,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys={
            'query_memory': dict(lr_mult=5.0),
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
