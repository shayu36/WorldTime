_base_ = ['./sparseworld-traj-memory-joint.py']

# Phase2 keeps the epoch-56 image feature space fixed while allowing the
# occupancy query-refinement and semantic-output head to adapt to STAC-QM.
model = dict(
    query_memory_cfg=dict(
        memory_joint_finetune_mode=False,
        memory_phase2_finetune_mode=True))

# Keep the joint LR tiers and adapt the large occupancy head conservatively.
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'pts_bbox_head': dict(lr_mult=0.5),
        }))

runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1)

load_from = 'ckpts/epoch_56.pth'
resume_from = None
