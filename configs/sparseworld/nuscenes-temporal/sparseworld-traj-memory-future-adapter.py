"""Code-level FutureMemoryAdapter configuration.

This configuration is intentionally a finetuning/build fixture.  It loads the
existing OPUS baseline checkpoint and fixed schema-v2 memory cache, but is not
intended to be launched as a training or full-evaluation job in this phase.
"""

_base_ = ['./sparseworld-traj-finetune-stacqm.py']

load_from = 'ckpts/epoch_56.pth'
resume_from = None

future_memory_target_horizons = [1.0, 2.0, 3.0]

# Keep the dataset/cache/evaluator inherited from the validated trajectory
# baseline.  The model override explicitly disables every legacy STAC-QM and
# Phase2/3/4 refiner route and enables only the independent output adapter.
model = dict(
    query_memory_cfg=dict(
        enabled=False,
        source='cache',
        freeze_base_model=True,
        future_memory_adapter_enabled=True,
        future_memory_adapter_finetune_mode=True,
        future_memory_target_horizons=future_memory_target_horizons,
        future_memory_adapter=dict(
            enabled=True,
            embed_dims=256,
            num_classes=17,
            num_points=48,
            num_heads=8,
            horizon_count=3,
            topk=16,
            spatial_radius=6.0,
            max_age=8.0,
            dropout=0.0,
            gate_bias=-1.0,
        ),
        # Explicitly disable old MemoryConditionedRefiner/Phase routes.
        memory_conditioned_refiner=False,
        memory_finetune_mode=False,
        memory_joint_finetune_mode=False,
        memory_phase2_finetune_mode=False,
        memory_phase3_finetune_mode=False,
        memory_phase3_future_only=False,
    ),
)

optimizer = dict(
    type='AdamW',
    constructor='TrainableOnlyOptimizerConstructor',
    lr=1e-4,
    weight_decay=1e-2,
    paramwise_cfg=dict(bypass_duplicate=True))
