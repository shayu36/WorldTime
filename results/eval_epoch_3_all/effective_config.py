point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDatasetOccpancy4DTraj'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=4),
    dict(type='LoadOccGTFromFile4DTraj'),
    dict(
        type='LoadQueryMemoryFromFiles',
        cache_root='./data/query_memory/sparseworld_epoch56_schema2_train',
        history_frames=3,
        max_queries_per_frame=256,
        strict=True,
        embed_dims=256,
        num_points=48,
        history_selection_mode='target_age',
        history_target_ages=[2.5, 3.5, 4.5],
        min_reliability=0.0,
        spatial_cell_size=4.0,
        max_per_spatial_cell=16,
        max_per_class=64),
    dict(
        type='RandomTransformImage',
        ida_aug_conf=dict(
            resize_lim=(0.38, 0.55),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect4D',
        keys=[
            'img', 'voxel_semantics', 'mask_lidar', 'mask_camera', 'rays',
            'temporal_semantics', 'temporal_rays', 'temporal_ego_states',
            'temporal_trajs', 'temporal2ego', 'temporal_ego2global',
            'memory_query_feat', 'memory_points_metric', 'memory_conf',
            'memory_reliability', 'memory_label',
            'memory_semantic_distribution', 'memory_valid',
            'memory_source_ego2global', 'memory_age'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'lidar2img', 'img_timestamp', 'timestamp', 'ego2lidar',
                   'ego2global', 'sample_idx', 'scene_token', 'scene_name',
                   'frame_idx'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='color'),
    dict(
        type='LoadMultiViewImageFromMultiSweeps',
        sweeps_num=4,
        test_mode=False),
    dict(type='LoadOccGTFromFile4DTraj'),
    dict(
        type='LoadQueryMemoryFromFiles',
        cache_root='./data/query_memory/sparseworld_epoch56_schema2_val',
        history_frames=3,
        max_queries_per_frame=256,
        strict=True,
        embed_dims=256,
        num_points=48,
        history_selection_mode='target_age',
        history_target_ages=[2.5, 3.5, 4.5],
        min_reliability=0.0,
        spatial_cell_size=4.0,
        max_per_spatial_cell=16,
        max_per_class=64),
    dict(
        type='RandomTransformImage',
        ida_aug_conf=dict(
            resize_lim=(0.38, 0.55),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect4D',
                keys=[
                    'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                    'temporal_semantics', 'temporal_ego_states',
                    'temporal_trajs', 'temporal_agent_boxes',
                    'temporal_agent_feats', 'memory_query_feat',
                    'memory_points_metric', 'memory_conf',
                    'memory_reliability', 'memory_label',
                    'memory_semantic_distribution', 'memory_valid',
                    'memory_source_ego2global', 'memory_age'
                ],
                meta_keys=[
                    'filename', 'box_type_3d', 'ori_shape', 'img_shape',
                    'pad_shape', 'sample_idx', 'lidar2img', 'img_timestamp',
                    'timestamp', 'ego2lidar', 'ego2global', 'scene_token',
                    'scene_name', 'frame_idx', 'gt_boxes', 'gt_labels',
                    'occ_gt_path'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    train=dict(
        type='NuScenesDatasetOccpancy4DTraj',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=False,
                color_type='color'),
            dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=4),
            dict(type='LoadOccGTFromFile4DTraj'),
            dict(
                type='LoadQueryMemoryFromFiles',
                cache_root=
                './data/query_memory/sparseworld_epoch56_schema2_train',
                history_frames=3,
                max_queries_per_frame=256,
                strict=True,
                embed_dims=256,
                num_points=48,
                history_selection_mode='target_age',
                history_target_ages=[2.5, 3.5, 4.5],
                min_reliability=0.0,
                spatial_cell_size=4.0,
                max_per_spatial_cell=16,
                max_per_class=64),
            dict(
                type='RandomTransformImage',
                ida_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect4D',
                keys=[
                    'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                    'rays', 'temporal_semantics', 'temporal_rays',
                    'temporal_ego_states', 'temporal_trajs', 'temporal2ego',
                    'temporal_ego2global', 'memory_query_feat',
                    'memory_points_metric', 'memory_conf',
                    'memory_reliability', 'memory_label',
                    'memory_semantic_distribution', 'memory_valid',
                    'memory_source_ego2global', 'memory_age'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'lidar2img', 'img_timestamp', 'timestamp',
                           'ego2lidar', 'ego2global', 'sample_idx',
                           'scene_token', 'scene_name', 'frame_idx'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        query_memory_cache_root=
        './data/query_memory/sparseworld_epoch56_schema2_train',
        query_memory_strict=True,
        use_valid_flag=True,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(0, 5, 1),
        query_memory_history_frames=3,
        query_memory_max_queries_per_frame=256,
        query_memory_history_selection_mode='target_age',
        query_memory_history_target_ages=[2.5, 3.5, 4.5],
        query_memory_history_age_tolerance=0.35,
        query_memory_visual_history_window=2.0),
    val=dict(
        type='NuScenesDatasetOccpancy4DTraj',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=False,
                color_type='color'),
            dict(
                type='LoadMultiViewImageFromMultiSweeps',
                sweeps_num=4,
                test_mode=False),
            dict(type='LoadOccGTFromFile4DTraj'),
            dict(
                type='LoadQueryMemoryFromFiles',
                cache_root=
                './data/query_memory/sparseworld_epoch56_schema2_val',
                history_frames=3,
                max_queries_per_frame=256,
                strict=True,
                embed_dims=256,
                num_points=48,
                history_selection_mode='target_age',
                history_target_ages=[2.5, 3.5, 4.5],
                min_reliability=0.0,
                spatial_cell_size=4.0,
                max_per_spatial_cell=16,
                max_per_class=64),
            dict(
                type='RandomTransformImage',
                ida_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect4D',
                        keys=[
                            'img', 'voxel_semantics', 'mask_lidar',
                            'mask_camera', 'temporal_semantics',
                            'temporal_ego_states', 'temporal_trajs',
                            'temporal_agent_boxes', 'temporal_agent_feats',
                            'memory_query_feat', 'memory_points_metric',
                            'memory_conf', 'memory_reliability',
                            'memory_label', 'memory_semantic_distribution',
                            'memory_valid', 'memory_source_ego2global',
                            'memory_age'
                        ],
                        meta_keys=[
                            'filename', 'box_type_3d', 'ori_shape',
                            'img_shape', 'pad_shape', 'sample_idx',
                            'lidar2img', 'img_timestamp', 'timestamp',
                            'ego2lidar', 'ego2global', 'scene_token',
                            'scene_name', 'frame_idx', 'gt_boxes', 'gt_labels',
                            'occ_gt_path'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        query_memory_cache_root=
        './data/query_memory/sparseworld_epoch56_schema2_val',
        query_memory_strict=True,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(0, 5, 1),
        query_memory_history_frames=3,
        query_memory_max_queries_per_frame=256,
        query_memory_history_selection_mode='target_age',
        query_memory_history_target_ages=[2.5, 3.5, 4.5],
        query_memory_history_age_tolerance=0.35,
        query_memory_visual_history_window=2.0),
    test=dict(
        type='NuScenesDatasetOccpancy4DTraj',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=False,
                color_type='color'),
            dict(
                type='LoadMultiViewImageFromMultiSweeps',
                sweeps_num=4,
                test_mode=False),
            dict(type='LoadOccGTFromFile4DTraj'),
            dict(
                type='LoadQueryMemoryFromFiles',
                cache_root=
                './data/query_memory/sparseworld_epoch56_schema2_val',
                history_frames=3,
                max_queries_per_frame=256,
                strict=True,
                embed_dims=256,
                num_points=48,
                history_selection_mode='target_age',
                history_target_ages=[2.5, 3.5, 4.5],
                min_reliability=0.0,
                spatial_cell_size=4.0,
                max_per_spatial_cell=16,
                max_per_class=64),
            dict(
                type='RandomTransformImage',
                ida_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect4D',
                        keys=[
                            'img', 'voxel_semantics', 'mask_lidar',
                            'mask_camera', 'temporal_semantics',
                            'temporal_ego_states', 'temporal_trajs',
                            'temporal_agent_boxes', 'temporal_agent_feats',
                            'memory_query_feat', 'memory_points_metric',
                            'memory_conf', 'memory_reliability',
                            'memory_label', 'memory_semantic_distribution',
                            'memory_valid', 'memory_source_ego2global',
                            'memory_age'
                        ],
                        meta_keys=[
                            'filename', 'box_type_3d', 'ori_shape',
                            'img_shape', 'pad_shape', 'sample_idx',
                            'lidar2img', 'img_timestamp', 'timestamp',
                            'ego2lidar', 'ego2global', 'scene_token',
                            'scene_name', 'frame_idx', 'gt_boxes', 'gt_labels',
                            'occ_gt_path'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        query_memory_cache_root=
        './data/query_memory/sparseworld_epoch56_schema2_val',
        query_memory_strict=True,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(0, 5, 1),
        query_memory_history_frames=3,
        query_memory_max_queries_per_frame=256,
        query_memory_history_selection_mode='target_age',
        query_memory_history_target_ages=[2.5, 3.5, 4.5],
        query_memory_history_age_tolerance=0.35,
        query_memory_visual_history_window=2.0),
    train_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    val_dataloader=dict(workers_per_gpu=1),
    test_dataloader=dict(workers_per_gpu=1))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
checkpoint_config = dict(interval=1, max_keep_ckpts=3, save_last=True)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/memory-planning-fusion-from-t1-e3'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT'
    ],
    Ncams=6,
    input_size=(512, 1408),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    x=[-40, 40, 0.4],
    y=[-40, 40, 0.4],
    z=[-1, 5.4, 0.4],
    depth=[1.0, 45.0, 0.5])
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
query_memory_train_cache_root = './data/query_memory/sparseworld_epoch56_schema2_train'
query_memory_val_cache_root = './data/query_memory/sparseworld_epoch56_schema2_val'
query_memory_history_frames = 3
query_memory_max_queries_per_frame = 256
query_memory_write_threshold = 0.35
query_memory_embed_dims = 256
query_memory_num_heads = 8
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
num_fu_query = [60, 60, 60, 60, 40, 40]
num_frames = 5
num_fu_frames = 6
num_out_query = 700
num_levels = 4
num_points = 4
num_refines = [1, 4, 16, 24, 32, 48]
query_memory_num_points = 48
multi_adj_frame_id_cfg = (0, 5, 1)
pretrain = False
finetune_epoch = 5
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
    out_channels=256,
    num_outs=4)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
model = dict(
    type='SparseWorld4DTrajMemoryPlanning',
    final_softplus=True,
    out_dim=256,
    use_grid_mask=False,
    num_out_query=700,
    query_memory_cfg=dict(
        enabled=False,
        source='cache',
        frame_interval=0.5,
        history_selection_mode='target_age',
        history_frames=3,
        history_target_ages=[2.5, 3.5, 4.5],
        history_age_tolerance=0.35,
        visual_history_window=2.0,
        retention_seconds=5.0,
        max_bank_entries=16,
        max_queries_per_frame=256,
        write_threshold=0.35,
        min_reliability=0.0,
        spatial_cell_size=4.0,
        max_per_spatial_cell=16,
        max_per_class=64,
        embed_dims=256,
        num_heads=8,
        spatial_radius=6.0,
        topk=16,
        max_age=8.0,
        lambda_position=1.0,
        lambda_time=1.0,
        lambda_reliability=1.0,
        dropout=0.0,
        motion_compensation=True,
        max_velocity=20.0,
        log_diagnostics=False,
        freeze_base_model=True,
        future_memory_adapter_enabled=True,
        future_memory_adapter_finetune_mode=True,
        future_memory_target_horizons=[1.0, 2.0, 3.0],
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
            gate_bias=-1.0),
        memory_conditioned_refiner=False,
        memory_finetune_mode=False,
        memory_joint_finetune_mode=False,
        memory_phase2_finetune_mode=False,
        memory_phase3_finetune_mode=False,
        memory_phase3_future_only=False,
        future_memory_adapter_version='v2',
        future_memory_adapter_v2=dict(
            enabled=True,
            embed_dims=256,
            num_classes=17,
            num_points=48,
            num_heads=8,
            coarse_topk_geometry=8,
            coarse_topk_semantic=8,
            point_topk=8,
            coarse_radius=6.0,
            point_radius=6.0,
            max_effective_age=8.0,
            semantic_uncertainty_gamma=2.0,
            semantic_weight_floor=0.1,
            semantic_dropout_probability=0.1,
            query_chunk_size=48,
            activation_checkpoint=False,
            save_on_cpu=False,
            coarse_geometry_context_weight=0.5,
            point_geometry_head_fraction=0.5,
            gate_bias=-1.0,
            dropout=0.0,
            voxel_size=(0.4, 0.4, 0.4),
            loss_occ_weight=0.25,
            loss_occ_anchor_weight=0.05,
            loss_sem_weight=0.25,
            loss_threshold_weight=0.1,
            loss_soft_voxel_weight=0.1,
            loss_base_cls_weight=1.0,
            loss_pts_weight=0.5,
            threshold_margin=0.1,
            positive_radius=None,
            occupancy_positive_weight=1.0,
            occupancy_negative_weight=1.0,
            threshold_positive_weight=1.0,
            threshold_negative_weight=1.0,
            gate_mode='task_split',
            horizon_gate_bias=True,
            soft_voxel_center_offset=0.5,
            soft_voxel_renormalize_valid_corners=True,
            occupancy_positive_class_balance_beta=dict({
                '1s': 0.0,
                '2s': 0.0,
                '3s': 0.25
            }),
            occupancy_class_balance_diagnostics=True)),
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    is_pretrain=False,
    finetune_epoch=5,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    pts_bbox_head=dict(
        type='OPUSHead',
        num_classes=17,
        in_channels=256,
        num_query=720,
        num_fu_frames=6,
        num_fu_query=[60, 60, 60, 60, 40, 40],
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        voxel_size=[0.4, 0.4, 0.4],
        is_pretrain=False,
        transformer=dict(
            type='OPUSTransformer',
            embed_dims=256,
            num_frames=5,
            num_points=4,
            num_layers=6,
            num_levels=4,
            num_classes=17,
            num_refines=[1, 4, 16, 24, 32, 48],
            scales=[0.5],
            pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
            num_query=720,
            num_fu_query=[60, 60, 60, 60, 40, 40]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5)),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            score_thr=[
                0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                0.35, 0.35, 0.35, 0.35, 0.35, 0.25, 0.3
            ],
            padding=True)),
    future_memory_adapter_version='v2',
    memory_planning_source_checkpoint=
    'work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth',
    memory_planning_cfg=dict(
        embed_dims=256,
        num_classes=17,
        hidden_dims=128,
        num_heads=4,
        horizon_count=3,
        voxel_size=(0.4, 0.4, 0.4)),
    pretrained=None,
    align_after_view_transfromation=True)
bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
ida_aug_conf = dict(
    resize_lim=(0.38, 0.55),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
share_data_config = dict(
    type='NuScenesDatasetOccpancy4DTraj',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(0, 5, 1),
    query_memory_history_frames=3,
    query_memory_max_queries_per_frame=256,
    query_memory_history_selection_mode='target_age',
    query_memory_history_target_ages=[2.5, 3.5, 4.5],
    query_memory_history_age_tolerance=0.35,
    query_memory_visual_history_window=2.0)
test_data_config = dict(
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles',
            to_float32=False,
            color_type='color'),
        dict(
            type='LoadMultiViewImageFromMultiSweeps',
            sweeps_num=4,
            test_mode=False),
        dict(type='LoadOccGTFromFile4DTraj'),
        dict(
            type='LoadQueryMemoryFromFiles',
            cache_root='./data/query_memory/sparseworld_epoch56_schema2_val',
            history_frames=3,
            max_queries_per_frame=256,
            strict=True,
            embed_dims=256,
            num_points=48,
            history_selection_mode='target_age',
            history_target_ages=[2.5, 3.5, 4.5],
            min_reliability=0.0,
            spatial_cell_size=4.0,
            max_per_spatial_cell=16,
            max_per_class=64),
        dict(
            type='RandomTransformImage',
            ida_aug_conf=dict(
                resize_lim=(0.38, 0.55),
                final_dim=(256, 704),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                H=900,
                W=1600,
                rand_flip=True),
            training=False),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect4D',
                    keys=[
                        'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                        'temporal_semantics', 'temporal_ego_states',
                        'temporal_trajs', 'temporal_agent_boxes',
                        'temporal_agent_feats', 'memory_query_feat',
                        'memory_points_metric', 'memory_conf',
                        'memory_reliability', 'memory_label',
                        'memory_semantic_distribution', 'memory_valid',
                        'memory_source_ego2global', 'memory_age'
                    ],
                    meta_keys=[
                        'filename', 'box_type_3d', 'ori_shape', 'img_shape',
                        'pad_shape', 'sample_idx', 'lidar2img',
                        'img_timestamp', 'timestamp', 'ego2lidar',
                        'ego2global', 'scene_token', 'scene_name', 'frame_idx',
                        'gt_boxes', 'gt_labels', 'occ_gt_path'
                    ])
            ])
    ],
    query_memory_cache_root=
    './data/query_memory/sparseworld_epoch56_schema2_val',
    query_memory_strict=True,
    ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
    type='NuScenesDatasetOccpancy4DTraj',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(0, 5, 1),
    query_memory_history_frames=3,
    query_memory_max_queries_per_frame=256,
    query_memory_history_selection_mode='target_age',
    query_memory_history_target_ages=[2.5, 3.5, 4.5],
    query_memory_history_age_tolerance=0.35,
    query_memory_visual_history_window=2.0)
key = 'test'
optimizer = dict(
    type='AdamW',
    constructor='TrainableOnlyOptimizerConstructor',
    lr=0.0001,
    weight_decay=0.01,
    paramwise_cfg=dict(bypass_duplicate=True))
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='fixed', warmup='linear', warmup_iters=100, warmup_ratio=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=1)
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
revise_keys = [('^module.', '')]
future_memory_target_horizons = [1.0, 2.0, 3.0]
future_memory_adapter_version = 'v2'
future_memory_adapter_v2 = dict(
    enabled=True,
    embed_dims=256,
    num_classes=17,
    num_points=48,
    num_heads=8,
    coarse_topk_geometry=8,
    coarse_topk_semantic=8,
    point_topk=8,
    coarse_radius=6.0,
    point_radius=6.0,
    max_effective_age=8.0,
    semantic_uncertainty_gamma=2.0,
    semantic_weight_floor=0.1,
    semantic_dropout_probability=0.1,
    query_chunk_size=40,
    activation_checkpoint=True,
    save_on_cpu=False,
    coarse_geometry_context_weight=0.5,
    point_geometry_head_fraction=0.5,
    gate_bias=-1.0,
    dropout=0.0,
    voxel_size=(0.4, 0.4, 0.4),
    loss_occ_weight=0.25,
    loss_occ_anchor_weight=0.05,
    loss_sem_weight=0.25,
    loss_threshold_weight=0.1,
    loss_soft_voxel_weight=0.1,
    loss_base_cls_weight=1.0,
    loss_pts_weight=0.5,
    threshold_margin=0.1,
    positive_radius=None,
    occupancy_positive_weight=1.0,
    occupancy_negative_weight=1.0,
    threshold_positive_weight=1.0,
    threshold_negative_weight=1.0)
auto_resume = False
fp16 = dict(loss_scale='dynamic')
planning_source_checkpoint = 'work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth'
seed = 17
find_unused_parameters = False
cfg_name = 'sparseworld-traj-memory-planning-fusion'
