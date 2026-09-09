_base_ = ['../../_base_/datasets/nus-3d.py', '../../_base_/default_runtime.py']

# Global
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT'
    ],
    'Ncams':
    6,
    'input_size': (512, 1408),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [0.4, 0.4, 0.4]

numC_Trans = 32
use_checkpoint = False
occ_size = [200, 200, 16]
voxel_out_channel = 32
empty_idx = 17
num_cls = 18



depth_gt_path = './data/depth_gt'
semantic_gt_path = './data/seg_gt_lidarseg'


object_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

embed_dims = 256
query_memory_train_cache_root = (
    './data/query_memory/sparseworld_epoch56_schema2_train')
query_memory_val_cache_root = (
    './data/query_memory/sparseworld_epoch56_schema2_val')
query_memory_history_frames = 3
query_memory_max_queries_per_frame = 256
query_memory_write_threshold = 0.35
query_memory_embed_dims = embed_dims
query_memory_num_heads = 8
# STAC-QM modeling-repair configuration (Problems 2/3/5/6).
query_memory_frame_interval = 0.5
query_memory_history_selection_mode = 'target_age'
query_memory_history_target_ages = [2.5, 3.5, 4.5]
query_memory_history_age_tolerance = 0.35
query_memory_visual_history_window = 2.0
query_memory_retention_seconds = 5.0
query_memory_max_bank_entries = 16
query_memory_min_reliability = 0.0
query_memory_spatial_cell_size = 4.0
query_memory_max_per_spatial_cell = 16
query_memory_max_per_class = 64
num_layers = 6
num_query = 720
num_fu_query = [60,60,60,60,40,40]
# num_query = 720
# num_fu_query = [30,30,40,40,50,50]

# num_query = 1440
# num_fu_query = [60,60,60,60,40,40]

num_frames = 5
num_fu_frames = 6
num_out_query=700

num_levels = 4
num_points = 4
# num_refines = [1, 4, 16, 32, 48, 64]
num_refines = [1, 4, 16, 24, 32, 48]
query_memory_num_points = num_refines[-1]
multi_adj_frame_id_cfg = (0, num_frames, 1)

pretrain = False
finetune_epoch = 5
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

model = dict(
    type='SparseWorld4DTraj',
    final_softplus=True,
    out_dim=256,
    use_grid_mask=False,
    num_out_query = num_out_query,
    query_memory_cfg=dict(
        enabled=True,
        source='cache',
        # --- future-aware age (Problem 2) ---
        frame_interval=query_memory_frame_interval,
        # --- history selection (Problem 6) ---
        history_selection_mode=query_memory_history_selection_mode,
        history_frames=query_memory_history_frames,
        history_target_ages=query_memory_history_target_ages,
        history_age_tolerance=query_memory_history_age_tolerance,
        visual_history_window=query_memory_visual_history_window,
        retention_seconds=query_memory_retention_seconds,
        max_bank_entries=query_memory_max_bank_entries,
        # --- diversity selection (Problem 5) ---
        max_queries_per_frame=query_memory_max_queries_per_frame,
        write_threshold=query_memory_write_threshold,
        min_reliability=query_memory_min_reliability,
        spatial_cell_size=query_memory_spatial_cell_size,
        max_per_spatial_cell=query_memory_max_per_spatial_cell,
        max_per_class=query_memory_max_per_class,
        # --- attention / fusion ---
        embed_dims=query_memory_embed_dims,
        num_heads=query_memory_num_heads,
        spatial_radius=6.0,
        topk=16,
        max_age=8.0,
        lambda_position=1.0,
        lambda_time=1.0,
        lambda_reliability=1.0,
        dropout=0.0,
        # --- motion compensation (Problem 3) ---
        motion_compensation=True,
        max_velocity=20.0,
        # --- misc ---
        log_diagnostics=True,
        freeze_base_model=True),
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    is_pretrain = pretrain,
    finetune_epoch=finetune_epoch,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type='OPUSHead',
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        num_fu_frames = num_fu_frames,
        num_fu_query = num_fu_query,
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        is_pretrain = pretrain,
        transformer=dict(
            type='OPUSTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            scales=[0.5],
            pc_range=point_cloud_range,
            num_query=num_query,
            num_fu_query=num_fu_query),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5)),
    train_cfg=dict(
        pts=dict(
            cls_weights=[
                3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 2, 1, 1, 1, 1],
            )
        ),
    test_cfg=dict(
        pts=dict(
            score_thr=[0.35]*15 + [0.25,0.3],
            padding=True)
    )
)


# Data
dataset_type = 'NuScenesDatasetOccpancy4DTraj'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

pretrain = False
# find_unused_parameters = True

ida_aug_conf = {
    'resize_lim': (0.38,0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',to_float32=False,color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames-1),
    dict(type='LoadOccGTFromFile4DTraj'),
    dict(
        type='LoadQueryMemoryFromFiles',
        cache_root=query_memory_train_cache_root,
        history_frames=query_memory_history_frames,
        max_queries_per_frame=query_memory_max_queries_per_frame,
        strict=True,
        embed_dims=query_memory_embed_dims,
        num_points=query_memory_num_points,
        history_selection_mode=query_memory_history_selection_mode,
        history_target_ages=query_memory_history_target_ages,
        min_reliability=query_memory_min_reliability,
        spatial_cell_size=query_memory_spatial_cell_size,
        max_per_spatial_cell=query_memory_max_per_spatial_cell,
        max_per_class=query_memory_max_per_class),
    dict(type='RandomTransformImage',ida_aug_conf=ida_aug_conf,training=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect4D', keys=['img', 'voxel_semantics',
                                'mask_lidar','mask_camera',
                                 'rays', 'temporal_semantics', 'temporal_rays', 'temporal_ego_states', 'temporal_trajs','temporal2ego','temporal_ego2global',
                                 'memory_query_feat', 'memory_points_metric', 'memory_conf', 'memory_reliability', 'memory_label', 'memory_semantic_distribution', 'memory_valid', 'memory_source_ego2global', 'memory_age',
                               ],meta_keys = ('filename','ori_shape','img_shape','pad_shape','lidar2img','img_timestamp','timestamp','ego2lidar','ego2global','sample_idx','scene_token','scene_name','frame_idx',))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles',to_float32=False,color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames-1,test_mode=False),
    dict(type='LoadOccGTFromFile4DTraj'),  # For visualization...
    dict(
        type='LoadQueryMemoryFromFiles',
        cache_root=query_memory_val_cache_root,
        history_frames=query_memory_history_frames,
        max_queries_per_frame=query_memory_max_queries_per_frame,
        strict=True,
        embed_dims=query_memory_embed_dims,
        num_points=query_memory_num_points,
        history_selection_mode=query_memory_history_selection_mode,
        history_target_ages=query_memory_history_target_ages,
        min_reliability=query_memory_min_reliability,
        spatial_cell_size=query_memory_spatial_cell_size,
        max_per_spatial_cell=query_memory_max_per_spatial_cell,
        max_per_class=query_memory_max_per_class),
    dict(type='RandomTransformImage',ida_aug_conf=ida_aug_conf,training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect4D', keys=['img', 'voxel_semantics',
                                        'mask_lidar','mask_camera','temporal_semantics',
                                        'temporal_ego_states', 'temporal_trajs', 'temporal_agent_boxes', 'temporal_agent_feats',
                                        'memory_query_feat', 'memory_points_metric', 'memory_conf', 'memory_reliability', 'memory_label', 'memory_semantic_distribution', 'memory_valid', 'memory_source_ego2global', 'memory_age',],
                 meta_keys = ['filename','box_type_3d','ori_shape','img_shape','pad_shape','sample_idx',
                              'lidar2img','img_timestamp','timestamp','ego2lidar','ego2global','scene_token','scene_name','frame_idx','gt_boxes','gt_labels','occ_gt_path'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    query_memory_history_frames=query_memory_history_frames,
    query_memory_max_queries_per_frame=query_memory_max_queries_per_frame,
    query_memory_history_selection_mode=query_memory_history_selection_mode,
    query_memory_history_target_ages=query_memory_history_target_ages,
    query_memory_history_age_tolerance=query_memory_history_age_tolerance,
    query_memory_visual_history_window=query_memory_visual_history_window,
)

test_data_config = dict(
    pipeline=test_pipeline,
    query_memory_cache_root=query_memory_val_cache_root,
    query_memory_strict=True,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        query_memory_cache_root=query_memory_train_cache_root,
        query_memory_strict=True,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

checkpoint_config = dict(
    interval=1,  
    max_keep_ckpts=-1,
    save_last=True     
)

runner = dict(type='EpochBasedRunner', max_epochs=64)
# custom hooks
custom_hooks = [dict(type='CustomSetEpochInfoHook')]

log_config = dict(
    interval=50,
)
load_from='ckpts/epoch_56.pth'

revise_keys = [(r'^module.', '')]

# fp16 = dict(loss_scale='dynamic')
