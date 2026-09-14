# Copyright (c) Phigent Robotics. All rights reserved.
from mmdet3d.models.detectors.bevdet_occ import BEVStereo4DOCC
from .opus import OPUS
import torch.nn.functional as F
import torch
import time
import warnings
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import MultiheadAttention
from torch import nn
import numpy as np
from mmdet3d.models import builder
from .opus_transformer import OPUSSelfAttention, OPUSCrossAttention
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import get_dist_info
from mmdet3d.models.detectors.loss import CE_ssc_loss, sem_scal_loss, geo_scal_loss, l1_loss, l2_loss
from mmdet3d.models.detectors.lovasz_softmax import lovasz_softmax
from IPython import embed
from mmdet3d.models.sparsedetectors.bbox.utils import decode_points, encode_points, trans_coords,get_matched_inds
from mmdet3d.models.sparsedetectors.query_memory import (
    STACQueryMemory, QueryMemoryBank, decode_points_metric,
    logits_to_query_confidence, FutureMemoryAdapter, FutureMemoryAdapterV2,
    decode_semantic_occupancy_logits, sparse_soft_voxel_iou_loss
)
from mmdet3d.models.heads import DownScaleModule3DCustom
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
device = torch.device('cuda')
# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947,
                                   497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031,
                                   141625221, 2307405309])
import time
# from ptflops import get_model_complexity_info
from thop import profile

def Scatter(src_dict):
    for key, value in src_dict.items():
        if isinstance(value, torch.Tensor):
            src_dict[key] = value.cuda()
        if isinstance(value, dict):
            src_dict[key] = Scatter(value)
        if isinstance(value, list):
            if isinstance(value[0], dict):
                src_dict[key] = [Scatter(v) for v in value]
            if isinstance(value[0], torch.Tensor):
                src_dict[key] = [v.cuda() for v in value]
    return src_dict


class MemoryConditionedRefiner(nn.Module):
    """Shared feature refiner used by current and future occupancy paths.

    The position input is metric-space Query center coordinates.  A small
    fixed residual scale keeps the new branch conservative at initialization
    while leaving a live gradient path through every refiner layer.
    """

    def __init__(self, embed_dims, residual_scale=0.1):
        super().__init__()
        self.position_proj = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True))
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)
        self.residual_scale = float(residual_scale)

    def forward(self, query_feat, query_pos_metric, horizon_embedding):
        input_dtype = query_feat.dtype
        query_float = query_feat.float()
        pos_float = self.position_proj(query_pos_metric.float())
        if horizon_embedding.dim() == 1:
            horizon_embedding = horizon_embedding.view(1, 1, -1)
        horizon_float = horizon_embedding.to(
            device=query_feat.device, dtype=torch.float32)
        horizon_float = horizon_float.expand(query_feat.shape[0],
                                             query_feat.shape[1], -1)
        delta = self.out_proj(self.fuse(torch.cat(
            [query_float, pos_float, horizon_float], dim=-1)))
        return query_feat + self.residual_scale * delta.to(input_dtype)


@DETECTORS.register_module()
class SparseWorld4DTraj(OPUS):
    uses_sparseworld_eval_api = True
    _MEMORY_ONLY_TRAINABLE_PREFIXES = ('query_memory.',)
    _MEMORY_JOINT_TRAINABLE_PREFIXES = (
        'query_memory.',
        'position_encoder.',
        'reg_branch.',
        'vel_branch.',
        'cls_branch.',
        'ego_cross_attn.',
    )
    _MEMORY_JOINT_TRAIN_MODULES = (
        'query_memory',
        'position_encoder',
        'reg_branch',
        'vel_branch',
        'cls_branch',
        'ego_cross_attn',
    )
    _MEMORY_PHASE2_TRAINABLE_PREFIXES = (
        *_MEMORY_JOINT_TRAINABLE_PREFIXES,
        'pts_bbox_head.',
    )
    _MEMORY_PHASE2_TRAIN_MODULES = (
        *_MEMORY_JOINT_TRAIN_MODULES,
        'pts_bbox_head',
    )
    _MEMORY_PHASE3_TRAINABLE_PREFIXES = (
        'query_memory.',
        'memory_refiner.',
        'memory_cls_branch.',
        'memory_reg_branch.',
        'memory_vel_branch.',
        'memory_horizon_embedding.',
    )
    _MEMORY_PHASE3_TRAIN_MODULES = (
        'query_memory',
        'memory_refiner',
        'memory_cls_branch',
        'memory_reg_branch',
        'memory_vel_branch',
        'memory_horizon_embedding',
    )
    # The repaired future-only topology has one auditable trainable prefix.
    # Every original OPUS/image/trajectory parameter is frozen in this mode.
    _FUTURE_MEMORY_TRAINABLE_PREFIXES = ('future_memory_adapter.',)
    _FUTURE_MEMORY_TRAIN_MODULES = ('future_memory_adapter',)
    _FUTURE_MEMORY_V2_TRAINABLE_PREFIXES = ('future_memory_adapter_v2.',)
    _FUTURE_MEMORY_V2_TRAIN_MODULES = ('future_memory_adapter_v2',)

    def __init__(self,
                 out_dim=32,
                 dataset_type='Nuscenes',
                 num_classes=18,
                 test_threshold=8.5,
                 drop_out=0.1,
                 use_3d_loss=True,
                 if_pretrain=False,
                 if_render=True,
                 if_post_finetune=False,
                 finetune_epoch = 0,
                 num_out_query=600,
                 empty_idx=17,
                 use_focal_loss=True,
                 balance_cls_weight=True,
                 final_softplus=True,
                 memory_enabled=False,
                 memory_bank_size=5,
                 memory_embed_dims=256,
                 memory_num_heads=8,
                 memory_dropout=0.1,
                 memory_confidence_threshold=0.3,
                 memory_lambda_pos=0.01,
                 memory_lambda_time=0.1,
                 memory_lambda_conf=0.5,
                 query_memory_cfg=None,
                 future_memory_adapter_version=None,
                 **kwargs):
        self.memory_self_noise = kwargs.pop('memory_self_noise', 0.0)
        super(SparseWorld4DTraj, self).__init__(**kwargs)
        self.dataset_type = dataset_type
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.num_refines = self.pts_bbox_head.transformer.num_refines[-1]
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus
        # self.if_pretrain = if_pretrain
        self.if_render = if_render
        self.if_post_finetune = if_post_finetune
        self.empty_idx = empty_idx
        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            self.semantic_loss = nn.CrossEntropyLoss(
                weight=self.class_weights, reduction="mean"
            )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))

        self.velocity_dim = 3
        self.past_frame = 5
        self.pc_range = self.pts_bbox_head.pc_range

        self.plan_head = nn.Sequential(
            nn.Linear(self.velocity_dim * (self.past_frame + 2), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim)
        )
        self.ego_cross_attn = OPUSCrossAttention(self.out_dim, 8, drop_out, self.pts_bbox_head.pc_range)

        self.position_encoder = nn.Sequential(
            nn.Linear(4 * self.num_refines, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
        )

        self.reg_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 3)
        )

        self.vel_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 2)
        )

        self.cls_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 17)
        )
        self.points_scale_branch = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
        )

        self.traj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, 2),
        )
        self.l2_loss = l2_loss()

        self.box_mode_3d = Box3DMode.LIDAR
        self.planning_metric = None
        self.finetune_epoch = finetune_epoch

        self.pred_num = torch.zeros(18).cuda()

        self.gt_traj = list()
        self.tau = list()

        legacy_memory_cfg = dict(
            enabled=memory_enabled,
            source='online',
            history_frames=memory_bank_size,
            max_queries_per_frame=256,
            write_threshold=memory_confidence_threshold,
            embed_dims=memory_embed_dims,
            num_heads=memory_num_heads,
            spatial_radius=12.0,
            topk=32,
            max_age=3.0,
            lambda_position=memory_lambda_pos,
            lambda_time=memory_lambda_time,
            lambda_confidence=memory_lambda_conf,
            dropout=memory_dropout,
            max_time_gap=2.0,
            log_diagnostics=False,
            freeze_base_model=False,
        )
        self.query_memory_cfg = self._build_query_memory_cfg(
            query_memory_cfg, legacy_memory_cfg)
        if future_memory_adapter_version is not None:
            self.query_memory_cfg['future_memory_adapter_version'] = str(
                future_memory_adapter_version)
        memory_ablation = self.query_memory_cfg.get(
            'memory_ablation_memory_enabled', None)
        refiner_ablation = self.query_memory_cfg.get(
            'memory_ablation_refiner_enabled', None)
        self.memory_ablation_memory_enabled = memory_ablation
        self.memory_ablation_refiner_enabled = refiner_ablation
        configured_memory_enabled = bool(
            self.query_memory_cfg.get('enabled', False))
        self.query_memory_enabled = (
            configured_memory_enabled if memory_ablation is None else
            bool(memory_ablation))
        # Keep the module topology (and therefore deterministic initialization
        # order) identical across M=OFF/M=ON cells.  M=OFF is an execution
        # bypass; its STAC-QM parameters remain frozen and outside optimizer.
        self.query_memory_module_enabled = bool(
            configured_memory_enabled or memory_ablation is not None)
        self.memory_enabled = self.query_memory_enabled
        self.memory_finetune_mode = bool(
            self.query_memory_cfg.get('memory_finetune_mode', False))
        self.memory_joint_finetune_mode = bool(
            self.query_memory_cfg.get('memory_joint_finetune_mode', False))
        self.memory_phase2_finetune_mode = bool(
            self.query_memory_cfg.get('memory_phase2_finetune_mode', False))
        self.memory_phase3_finetune_mode = bool(
            self.query_memory_cfg.get('memory_phase3_finetune_mode', False))
        future_adapter_cfg = self.query_memory_cfg.get(
            'future_memory_adapter', None)
        future_adapter_v2_cfg = self.query_memory_cfg.get(
            'future_memory_adapter_v2', None)
        self.future_memory_adapter_version = str(
            self.query_memory_cfg.get(
                'future_memory_adapter_version',
                'v2' if future_adapter_v2_cfg is not None else 'v1')).lower()
        selected_future_cfg = (future_adapter_v2_cfg
                               if self.future_memory_adapter_version == 'v2'
                               and future_adapter_v2_cfg is not None
                               else future_adapter_cfg)
        self.future_memory_adapter_enabled = bool(
            self.query_memory_cfg.get('future_memory_adapter_enabled', False)
            if selected_future_cfg is None else
            selected_future_cfg.get('enabled', False))
        self.future_memory_adapter_finetune_mode = bool(
            self.query_memory_cfg.get('future_memory_adapter_finetune_mode',
                                      self.future_memory_adapter_enabled))
        self.future_memory_target_horizons = tuple(
            float(v) for v in self.query_memory_cfg.get(
                'future_memory_target_horizons', (1.0, 2.0, 3.0)))
        if self.future_memory_target_horizons != (1.0, 2.0, 3.0):
            raise ValueError(
                'FutureMemoryAdapter target horizons must be exactly '
                '(1.0, 2.0, 3.0) seconds')
        refiner_cfg = self.query_memory_cfg.get(
            'memory_conditioned_refiner', None)
        configured_refiner_enabled = (
            self.memory_phase3_finetune_mode
            if refiner_cfg is None else bool(refiner_cfg))
        self.memory_conditioned_refiner_enabled = (
            configured_refiner_enabled if refiner_ablation is None else
            bool(refiner_ablation))
        self.memory_phase3_base_aux_weight = float(
            self.query_memory_cfg.get('memory_phase3_base_aux_weight', 0.25))
        # Keep the raw OPUS observation Query for 0s while allowing STAC-QM
        # and the Memory-conditioned refiner to affect only scheduled future
        # Query groups.  This isolates future benefit from the harmful 0s
        # route observed in the first formal Phase 3 run.
        self.memory_phase3_future_only = bool(
            self.query_memory_cfg.get('memory_phase3_future_only', False))
        if sum((self.future_memory_adapter_finetune_mode,
                self.memory_finetune_mode,
                self.memory_joint_finetune_mode,
                self.memory_phase2_finetune_mode,
                self.memory_phase3_finetune_mode)) > 1:
            raise ValueError(
                'future_memory_adapter_finetune_mode, memory_finetune_mode, '
                'memory_joint_finetune_mode, '
                'memory_phase2_finetune_mode, and '
                'memory_phase3_finetune_mode are mutually exclusive')
        self.query_memory_source = self.query_memory_cfg.get('source', 'cache')
        self.query_memory_frame_interval = float(
            self.query_memory_cfg.get('frame_interval', 0.5))
        self.query_memory_log_diagnostics = bool(
            self.query_memory_cfg.get('log_diagnostics', False))
        self.query_memory_diagnostics = []
        self.memory_refiner_diagnostics = []
        self._last_query_memory_valid_slots = 0.0
        self.query_memory = None
        self.query_memory_bank = None
        self.frozen_num_stamps_all = None
        self.frozen_ind_stamps_all = None
        self._frozen_rap_masks = None
        self.future_memory_adapter = None
        self.future_memory_adapter_v2 = None
        self.future_memory_v2_loss_cfg = {}
        query_embed_dims = int(self.pts_bbox_head.transformer.embed_dims)
        if self.query_memory_module_enabled:
            if self.query_memory_source not in ('cache', 'online'):
                raise ValueError(
                    'query_memory_cfg.source must be "cache" or "online", '
                    f'got {self.query_memory_source!r}')
            self.query_memory = STACQueryMemory(
                enabled=True,
                embed_dims=self.query_memory_cfg['embed_dims'],
                num_heads=self.query_memory_cfg['num_heads'],
                spatial_radius=self.query_memory_cfg['spatial_radius'],
                topk=self.query_memory_cfg['topk'],
                max_age=self.query_memory_cfg['max_age'],
                lambda_position=self.query_memory_cfg['lambda_position'],
                lambda_time=self.query_memory_cfg['lambda_time'],
                lambda_reliability=self.query_memory_cfg.get(
                    'lambda_reliability'),
                lambda_confidence=self.query_memory_cfg['lambda_confidence'],
                dropout=self.query_memory_cfg['dropout'],
                motion_compensation=self.query_memory_cfg.get(
                    'motion_compensation', True),
                max_velocity=self.query_memory_cfg.get('max_velocity', 20.0),
                pc_range=self.pc_range,
                fusion_gate_bias=self.query_memory_cfg.get(
                    'fusion_gate_bias', -4.0),
                fusion_alpha_init=self.query_memory_cfg.get(
                    'fusion_alpha_init', 0.0),
                fusion_out_proj_gain=self.query_memory_cfg.get(
                    'fusion_out_proj_gain', 0.0))
            if self.query_memory_source == 'online':
                self.query_memory_bank = QueryMemoryBank(
                    history_frames=self.query_memory_cfg['history_frames'],
                    max_queries_per_frame=self.query_memory_cfg[
                        'max_queries_per_frame'],
                    write_threshold=self.query_memory_cfg['write_threshold'],
                    max_time_gap=self.query_memory_cfg.get('max_time_gap'),
                    history_selection_mode=self.query_memory_cfg.get(
                        'history_selection_mode', 'recent'),
                    history_target_ages=self.query_memory_cfg.get(
                        'history_target_ages', [2.5, 3.5, 4.5]),
                    history_age_tolerance=self.query_memory_cfg.get(
                        'history_age_tolerance', 0.35),
                    visual_history_window=self.query_memory_cfg.get(
                        'visual_history_window', 2.0),
                    retention_seconds=self.query_memory_cfg.get(
                        'retention_seconds', 5.0),
                    max_bank_entries=self.query_memory_cfg.get(
                        'max_bank_entries', 16),
                    min_reliability=self.query_memory_cfg.get(
                        'min_reliability', 0.0),
                    spatial_cell_size=self.query_memory_cfg.get(
                        'spatial_cell_size', 4.0),
                    max_per_spatial_cell=self.query_memory_cfg.get(
                        'max_per_spatial_cell', 16),
                    max_per_class=self.query_memory_cfg.get(
                        'max_per_class', 64))

        # Do not route the repaired adapter through STACQueryMemory or the
        # legacy MemoryConditionedRefiner.  It is an independent output-only
        # correction stream and is instantiated only by the dedicated config.
        if getattr(self, 'future_memory_adapter_enabled', False):
            if self.future_memory_adapter_version == 'v2':
                adapter_cfg = dict(future_adapter_v2_cfg or future_adapter_cfg or {})
                adapter_cfg.pop('enabled', None)
                self.future_memory_v2_loss_cfg = {
                    key: adapter_cfg.get(key) for key in (
                        'loss_occ_weight', 'loss_sem_weight',
                        'loss_threshold_weight', 'loss_soft_voxel_weight',
                        'loss_base_cls_weight', 'loss_pts_weight', 'threshold_margin',
                        'positive_radius') if key in adapter_cfg}
                adapter_cfg.setdefault('embed_dims', query_embed_dims)
                adapter_cfg.setdefault('num_classes', 17)
                adapter_cfg.setdefault('num_points', self.num_refines)
                adapter_cfg.setdefault('pc_range', self.pc_range)
                self.future_memory_adapter_v2 = FutureMemoryAdapterV2(**adapter_cfg)
            else:
                adapter_cfg = dict(future_adapter_cfg or {})
                adapter_cfg.pop('enabled', None)
                adapter_cfg.setdefault('embed_dims', query_embed_dims)
                adapter_cfg.setdefault('num_classes', 17)
                adapter_cfg.setdefault('num_points', self.num_refines)
                adapter_cfg.setdefault('pc_range', self.pc_range)
                self.future_memory_adapter = FutureMemoryAdapter(**adapter_cfg)

        self.memory_refiner = MemoryConditionedRefiner(query_embed_dims)
        self.memory_horizon_embedding = nn.Embedding(
            self.num_fu_frames + 1, query_embed_dims)
        nn.init.normal_(self.memory_horizon_embedding.weight, mean=0.0,
                        std=0.02)
        self.memory_cls_branch = nn.Sequential(
            nn.Linear(query_embed_dims, query_embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(query_embed_dims, self.num_refines * 17))
        self.memory_reg_branch = nn.Sequential(
            nn.Linear(query_embed_dims, query_embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(query_embed_dims, self.num_refines * 3))
        self.memory_vel_branch = nn.Sequential(
            nn.Linear(query_embed_dims, query_embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(query_embed_dims, self.num_refines * 2))
        self._configure_query_memory_trainability()

    def init_weights(self):
        self.pts_bbox_head.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        if self._memory_tuning_active():
            self.pretrain = False
            self.pts_bbox_head.pretrain = False
            self._assert_memory_finetune_temporal_state()
            return
        if epoch<self.finetune_epoch:
            self.pretrain = True
            self.pts_bbox_head.pretrain = True
            if getattr(self.pts_bbox_head, 'num_stamps_all', None) is not None:
                self.pts_bbox_head.num_stamps_all[:] = 1  # avoid diving 0
        else:
            self.pretrain = False
            self.pts_bbox_head.pretrain = False
            num_stamps = self.pts_bbox_head.num_stamps_all / torch.sum(self.pts_bbox_head.num_stamps_all, dim=-1,
                                                                       keepdim=True)
            self.pts_bbox_head.ind_stamps_all = get_matched_inds(num_stamps, [self.num_query] + self.num_fu_query)

            self.pts_bbox_head.reset_mask()


    def trans_points(self, points_proposal, points_delta, trans_matrix):
        trans_matrix = trans_matrix.to(device=points_proposal.device,
                                       dtype=points_proposal.dtype)
        inv_trans_matrix = torch.linalg.inv(trans_matrix)
        points_proposal = decode_points(points_proposal, self.pc_range)
        # points_proposal = points_proposal.mean(dim=2, keepdim=True) fengze
        new_points = torch.matmul(points_proposal, trans_matrix[..., :3, :3].transpose(1, 2)) + trans_matrix[..., None,
                                                                                                :3, 3]
        new_points = new_points + points_delta
        new_points = torch.matmul(new_points, inv_trans_matrix[..., :3, :3].transpose(1, 2)) + inv_trans_matrix[...,
                                                                                               None, :3, 3]

        return encode_points(new_points, self.pc_range)

    def _forward_backbone_future_memory_adapter(
            self, img_metas, kwargs, B, ego_feat, outs, query_feat,
            query_pos, query_cls, ind_stamps_all, memory_context):
        """Run the repaired output-only FutureMemoryAdapter topology.

        ``state_*`` tensors below are the unmodified OPUS baseline stream.  A
        target-horizon adapter is evaluated from a snapshot of that stream and
        its corrected outputs are appended to the forecast lists only; no
        corrected feature, position, or logit is ever written back to state.
        """
        current_mask = ind_stamps_all == 0
        state_feat = query_feat[:, current_mask]
        state_pos = query_pos[:, current_mask]
        state_timestamp = query_pos.new_zeros(
            B, state_feat.shape[1], self.num_refines, 1)
        base_current_cls = query_cls[:, current_mask]
        outputs = dict(
            cls_score=base_current_cls,
            refine_pts=state_pos,
            outs=outs,
            _raw_query_feat=state_feat,
            _raw_query_pos=state_pos,
            _raw_query_cls=base_current_cls)
        forecast_points_list = []
        forecast_semantics_list = []
        pred_trajs_list = []
        forecast_points_mask_list = []
        adapter_diagnostics = []
        base_cls_list = []
        base_points_list = []
        forecast_semantic_logits_list = []
        forecast_occupancy_logits_list = []

        target_ego2global = None
        if memory_context is not None and \
                memory_context.get('memory_source_ego2global') is not None:
            target_ego2global = self._current_ego2global_tensor(
                img_metas, state_feat.device)

        # The six OPUS internal states are 0.5, 1.0, ..., 3.0 s.  Only state
        # indices 1, 3, and 5 (1/2/3 s) receive the new adapter.
        target_horizons = {1: (0, '1'), 3: (1, '2'), 5: (2, '3')}
        adapter_is_v2 = getattr(self, 'future_memory_adapter_version', 'v1') == 'v2'
        adapter_module = (self.future_memory_adapter_v2
                          if adapter_is_v2 else self.future_memory_adapter)
        for interval in range(self.num_fu_frames):
            # Baseline trajectory/state update.  Inputs are intentionally
            # detached exactly as in the original OPUS recurrence.
            fused_ego_feat, _ = self.ego_cross_attn(
                ego_feat.new_ones(B, 1, 3) * 0.5, ego_feat,
                state_pos.detach(), state_feat.detach())
            pred_traj = self.traj_head(fused_ego_feat)
            pred_trajs_list.append(pred_traj)

            scheduled_mask = ind_stamps_all == interval + 1
            scheduled_feat = query_feat[:, scheduled_mask]
            scheduled_pos = query_pos[:, scheduled_mask]
            if scheduled_feat.shape[1] > 0:
                state_feat = torch.cat([state_feat, scheduled_feat], dim=1)
                state_pos = torch.cat([state_pos, scheduled_pos], dim=1)
                state_timestamp = torch.cat([
                    state_timestamp,
                    state_pos.new_ones(
                        B, scheduled_feat.shape[1], self.num_refines, 1) * 0.5
                ], dim=1)

            pos_embedding = self.position_encoder(
                torch.cat([state_pos, state_timestamp], dim=-1).flatten(2, 3))
            state_feat = state_feat + fused_ego_feat + pos_embedding
            base_cls = self.cls_branch(state_feat).unflatten(-1, (-1, 17))
            vel_offset = self.vel_branch(state_feat).unflatten(-1, (-1, 2))
            reg_offset = self.reg_branch(state_feat).unflatten(-1, (-1, 3)) * 0.5
            pred_labels = base_cls.argmax(-1)
            pred_moving_mask = torch.logical_and(
                pred_labels >= 2, pred_labels <= 10).unsqueeze(-1)
            reg_offset = torch.cat([
                reg_offset[..., :2] + vel_offset * pred_moving_mask,
                reg_offset[..., 2:]], dim=-1).flatten(2, 3)
            state_pos = self.refine_points(state_pos, reg_offset)

            # Snapshot baseline outputs before any target-horizon correction.
            base_cls_snapshot = base_cls
            base_pos_snapshot = state_pos
            output_cls = base_cls_snapshot
            output_pos = base_pos_snapshot
            diag = dict(enabled=False, internal_step=int(interval + 1),
                        horizon_id=None, query_count=int(state_feat.shape[1]))
            if interval in target_horizons:
                horizon_id, horizon_name = target_horizons[interval]
                if memory_context is None:
                    result = adapter_module(
                        state_feat, decode_points_metric(state_pos, self.pc_range),
                        base_cls_snapshot, None, horizon_id)
                else:
                    result = adapter_module(
                        state_feat, decode_points_metric(state_pos, self.pc_range),
                        base_cls_snapshot, memory_context, horizon_id,
                        target_ego2global=target_ego2global)
                gate = result['gate']
                if adapter_is_v2:
                    # Keep semantic ordering and occupancy thresholding as
                    # separate quantities, then adapt to the legacy decoder.
                    sem_logits, occ_logits, output_cls = \
                        decode_semantic_occupancy_logits(
                            base_cls_snapshot, result['delta_s'],
                            result['delta_o'], gate)
                else:
                    sem_logits = base_cls_snapshot + gate[..., None] * result['delta_s']
                    occ_logits = base_cls_snapshot.max(-1, keepdim=True).values + gate[..., None] * result['delta_o']
                    output_cls = base_cls_snapshot + gate[..., None] * (
                        result['delta_s'] + result['delta_o'])
                # The historical ``refine_points`` helper intentionally
                # averages proposal points before applying a Baseline
                # regression update.  Reusing that averaging operation for a
                # zero-initialized Memory residual would change a Baseline
                # output even when ``DeltaP_memory == 0``.  The adapter path
                # therefore uses the coordinate-safe residual variant below:
                # decode the already-refined Baseline points, add only the
                # gated Memory delta, then encode back without another mean.
                output_pos = self._refine_future_memory_points(
                    base_pos_snapshot,
                    (gate * result['delta_p'] if adapter_is_v2 else
                     gate[..., None] * result['delta_p']).flatten(2, 3))
                diag = dict(result.get('diagnostics', {}))
                diag.update(enabled=True, internal_step=int(interval + 1),
                            horizon_id=int(horizon_id),
                            horizon_name=horizon_name,
                            query_count=int(state_feat.shape[1]))
                if adapter_is_v2:
                    diag['semantic_logits'] = sem_logits.detach()
                    diag['occupancy_logits'] = occ_logits.detach()
            forecast_semantics_list.append(output_cls)
            if adapter_is_v2:
                # The lists are consumed by V2 losses; non-target steps use
                # the unmodified baseline values.
                if interval not in target_horizons:
                    forecast_semantic_logits_list.append(base_cls_snapshot)
                    forecast_occupancy_logits_list.append(
                        base_cls_snapshot.max(-1, keepdim=True).values)
                else:
                    forecast_semantic_logits_list.append(sem_logits)
                    forecast_occupancy_logits_list.append(occ_logits)
            forecast_points_list.append(output_pos)
            base_cls_list.append(base_cls_snapshot)
            base_points_list.append(base_pos_snapshot)

            if self.training:
                if 'temporal_trajs' not in kwargs:
                    raise KeyError(
                        'temporal_trajs is required for future trajectory loss')
                ego2lidar = torch.as_tensor(
                    np.stack([meta['ego2lidar'] for meta in img_metas]),
                    device=state_pos.device, dtype=torch.float32)
                gt_traj = kwargs['temporal_trajs'][:, interval:interval + 1, :]
                pred_traj_expand = torch.cat(
                    [-gt_traj, torch.zeros_like(pred_traj[:, :, :1])], dim=-1)
                gt_points = self.trans_points(
                    output_pos.flatten(1, 2), pred_traj_expand,
                    ego2lidar).reshape(output_pos.shape)
                forecast_points_mask_list.append(gt_points[..., 0] >= 0)
            adapter_diagnostics.append(diag)

        outputs.update(
            forecast_semantics_list=forecast_semantics_list,
            forecast_points_list=forecast_points_list,
            pred_trajs_list=pred_trajs_list,
            forecast_points_mask_list=forecast_points_mask_list,
            memory_adapter_diagnostics=adapter_diagnostics,
            # Raw baseline snapshots are retained for audit/tests and are not
            # used as recurrent state after this point.
            base_cls_score=base_current_cls,
            base_refine_pts=query_pos[:, current_mask],
            baseline_forecast_semantics_list=base_cls_list,
            baseline_forecast_points_list=base_points_list)
        if adapter_is_v2:
            outputs['forecast_semantic_logits_list'] = forecast_semantic_logits_list
            outputs['forecast_occupancy_logits_list'] = forecast_occupancy_logits_list
        return outputs

    def _future_memory_v2_losses(self, semantic_logits, occupancy_logits,
                                  points, points_mask, gt_voxel_semantics):
        """Compute point occupancy/semantic and sparse voxel losses."""
        B, Q, R, C = semantic_logits.shape
        device = semantic_logits.device
        points_mask = (torch.ones(B, Q, R, dtype=torch.bool, device=device)
                       if points_mask is None else points_mask.to(device).bool())
        metric = decode_points_metric(points, self.pc_range).float()
        origin = self.pc_range[:3].to(device=device, dtype=torch.float32)
        voxel_size = getattr(self.pts_bbox_head, 'voxel_size',
                             torch.tensor([0.4, 0.4, 0.4], device=device))
        voxel_size = voxel_size.to(device=device, dtype=torch.float32)
        grid = torch.as_tensor(gt_voxel_semantics.shape[1:4], device=device)
        index = torch.floor((metric - origin) / voxel_size).long()
        inside = ((index >= 0) & (index < grid)).all(-1)
        safe = index.clamp_min(0)
        for dim in range(3):
            safe[..., dim] = safe[..., dim].clamp_max(grid[dim] - 1)
        batch = torch.arange(B, device=device)[:, None, None].expand(B, Q, R)
        gt_tensor = gt_voxel_semantics.to(device=device).long()
        labels = gt_tensor[batch, safe[..., 0], safe[..., 1], safe[..., 2]]
        centres = (safe.float() + 0.5) * voxel_size + origin
        loss_cfg = getattr(self, 'future_memory_v2_loss_cfg', {})
        radius = float(loss_cfg.get(
            'positive_radius', float(voxel_size.max().item()) * 0.5))
        valid = points_mask & inside & torch.isfinite(metric).all(-1)
        # Occupancy target follows the evaluator notion: nearest non-empty GT
        # voxel centre, rather than merely the GT label at the predicted voxel.
        nearest_dist = torch.full((B, Q, R), float('inf'), device=device)
        nearest_label = torch.zeros((B, Q, R), dtype=torch.long, device=device)
        empty_label = int(getattr(self, 'empty_idx', 17))
        for b in range(B):
            gt_idx = torch.nonzero(gt_tensor[b] != empty_label, as_tuple=False)
            if gt_idx.numel() == 0:
                continue
            gt_centres = (gt_idx.float() + 0.5) * voxel_size + origin
            gt_labels = gt_tensor[b][gt_idx[:, 0], gt_idx[:, 1], gt_idx[:, 2]]
            pred_flat = metric[b].reshape(-1, 3)
            near = torch.full((pred_flat.shape[0],), float('inf'), device=device)
            near_idx = torch.zeros(pred_flat.shape[0], dtype=torch.long, device=device)
            # Chunk both axes: no point×all-GT distance matrix is retained.
            for ps in range(0, pred_flat.shape[0], 2048):
                pe = min(pred_flat.shape[0], ps + 2048)
                local_dist = torch.full((pe - ps,), float('inf'), device=device)
                local_idx = torch.zeros(pe - ps, dtype=torch.long, device=device)
                for gs in range(0, gt_centres.shape[0], 8192):
                    ge = min(gt_centres.shape[0], gs + 8192)
                    d = torch.cdist(pred_flat[ps:pe], gt_centres[gs:ge])
                    dmin, didx = d.min(-1)
                    update = dmin < local_dist
                    local_dist = torch.where(update, dmin, local_dist)
                    local_idx = torch.where(update, didx + gs, local_idx)
                near[ps:pe] = local_dist
                near_idx[ps:pe] = local_idx
            nearest_dist[b] = near.reshape(Q, R)
            nearest_label[b] = gt_labels[near_idx].reshape(Q, R)
        positive = valid & (nearest_dist <= radius)
        labels = torch.where(positive, nearest_label, labels)
        y_occ = positive.float()
        occ = occupancy_logits.squeeze(-1)
        valid_count = valid.float().sum().clamp_min(1.)
        loss_occ = (F.binary_cross_entropy_with_logits(
            occ, y_occ, reduction='none') * valid.float()).sum() / valid_count
        configured_weights = getattr(self.pts_bbox_head, 'train_cfg', {}).get(
            'cls_weights', None)
        class_weights = (configured_weights if configured_weights is not None
                         else getattr(self, 'class_weights', None))
        if class_weights is not None:
            class_weights = class_weights.to(device=device,
                                              dtype=semantic_logits.dtype)
        if positive.any():
            loss_sem = F.cross_entropy(semantic_logits.reshape(-1, C)[positive.reshape(-1)],
                                       labels.reshape(-1)[positive.reshape(-1)].clamp(0, C - 1),
                                       weight=class_weights)
        else:
            loss_sem = semantic_logits.sum() * 0.
        score_thr = self.pts_bbox_head.test_cfg.get('score_thr', 0.1)
        score_thr = torch.as_tensor(score_thr, device=device,
                                    dtype=semantic_logits.dtype).flatten()
        if score_thr.numel() == 1:
            score_thr = score_thr.expand(C)
        if score_thr.numel() < C:
            score_thr = F.pad(score_thr, (0, C - score_thr.numel()),
                              value=float(score_thr[-1]))
        score_thr = score_thr[:C].clamp(1e-4, 1 - 1e-4)
        tau = torch.log(score_thr / (1. - score_thr))
        pred_class = semantic_logits.argmax(-1)
        target_class = torch.where(positive, labels, pred_class).clamp(0, C - 1)
        margin = float(loss_cfg.get('threshold_margin', .1))
        boundary = tau[target_class]
        per_point = torch.where(positive, F.softplus(boundary + margin - occ),
                                F.softplus(occ - boundary + margin))
        loss_threshold = (per_point * valid.float()).sum() / valid_count
        loss_voxel = sparse_soft_voxel_iou_loss(
            semantic_logits, occupancy_logits, points, gt_voxel_semantics,
            points_mask=points_mask, pc_range=self.pc_range,
            voxel_size=voxel_size, num_classes=C, class_weights=class_weights)
        return dict(loss_sem=loss_sem, loss_occ=loss_occ,
                    loss_threshold=loss_threshold,
                    loss_soft_voxel=loss_voxel)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def _refine_future_memory_points(self, points_proposal, points_delta):
        """Apply only a Memory point residual in metric coordinates.

        ``refine_points`` is the original OPUS recurrence helper and reduces
        the proposal to its mean before adding a regression delta.  A
        correction stream must be identity when its residual is zero, so it
        keeps every Baseline proposal point and applies only the adapter
        delta.  Both conversions use the same project coordinate helpers.
        """
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)
        points_metric = decode_points(points_proposal, self.pc_range)
        return encode_points(points_metric + points_delta, self.pc_range)

    def loss_traj(self, pred_traj, gt_traj, ego_interval):
        loss_dict = dict()
        loss_dict[f'loss_traj_{str(ego_interval)}s'] = self.l2_loss(pred_traj, gt_traj)

        return loss_dict

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        img = [img] if img is None else img

        result = self.simple_test(img_metas[0], img[0], **kwargs)

        

        return result


    # legacy query_memory_cfg keys -> canonical STAC-QM keys. Mapping is applied
    # with a single explicit warning and NEVER silently changes meaning.
    _QUERY_MEMORY_LEGACY_KEYS = {
        'lambda_confidence': 'lambda_reliability',
        'memory_conf': 'min_reliability',
        'confidence_threshold': 'write_threshold',
        'bank_size': 'history_frames',
    }

    def _build_query_memory_cfg(self, query_memory_cfg, legacy_cfg):
        cfg = dict(
            enabled=False,
            source='cache',
            # --- history selection (Problem 6) ---
            history_selection_mode='target_age',
            history_frames=3,
            history_target_ages=[2.5, 3.5, 4.5],
            history_age_tolerance=0.35,
            visual_history_window=2.0,
            retention_seconds=5.0,
            max_bank_entries=16,
            frame_interval=0.5,
            # --- diversity selection (Problem 5) ---
            max_queries_per_frame=256,
            min_reliability=0.0,
            spatial_cell_size=4.0,
            max_per_spatial_cell=16,
            max_per_class=64,
            write_threshold=0.35,
            # --- attention / fusion ---
            embed_dims=256,
            num_heads=8,
            spatial_radius=12.0,
            topk=32,
            max_age=8.0,
            lambda_position=1.0,
            lambda_time=1.0,
            lambda_reliability=1.0,
            # kept for the CausalQueryMemoryAttention legacy alias
            lambda_confidence=1.0,
            dropout=0.0,
            # --- motion compensation (Problem 3) ---
            motion_compensation=True,
            max_velocity=20.0,
            # --- misc ---
            max_time_gap=None,
            schema_version=2,
            log_diagnostics=False,
            freeze_base_model=False,
            memory_finetune_mode=False,
            memory_joint_finetune_mode=False,
            memory_phase2_finetune_mode=False,
            memory_phase3_finetune_mode=False,
            future_memory_adapter_enabled=False,
            future_memory_adapter_finetune_mode=False,
            future_memory_target_horizons=(1.0, 2.0, 3.0),
            future_memory_adapter=None,
            memory_conditioned_refiner=None,
            memory_phase3_base_aux_weight=0.25,
            memory_phase3_future_only=False,
            # Explicit 2x2 ablation switches. ``None`` preserves the legacy
            # mode-derived behavior; bool values override Memory/refiner
            # independently while keeping the same future-only routing.
            memory_ablation_memory_enabled=None,
            memory_ablation_refiner_enabled=None,
            fusion_gate_bias=-4.0,
            fusion_alpha_init=0.0,
            fusion_out_proj_gain=0.0,
        )
        user_cfg = query_memory_cfg
        if user_cfg is None:
            # legacy constructor-arg path (memory_* kwargs)
            if legacy_cfg.get('enabled', False):
                user_cfg = legacy_cfg
            else:
                return cfg
        cfg.update(self._map_legacy_query_memory_keys(dict(user_cfg)))
        return cfg

    def _map_legacy_query_memory_keys(self, user_cfg):
        legacy_present = [
            k for k in self._QUERY_MEMORY_LEGACY_KEYS if k in user_cfg]
        # lambda_confidence is still a live kwarg of CausalQueryMemoryAttention;
        # only *remap* it when the canonical lambda_reliability is absent.
        for legacy_key in legacy_present:
            canonical = self._QUERY_MEMORY_LEGACY_KEYS[legacy_key]
            if canonical in user_cfg:
                continue
            user_cfg[canonical] = user_cfg[legacy_key]
        if legacy_present:
            warnings.warn(
                'STAC-QM query_memory_cfg received legacy keys '
                f'{legacy_present}; mapped to '
                f'{[self._QUERY_MEMORY_LEGACY_KEYS[k] for k in legacy_present]}'
                '. These legacy keys are deprecated and will be removed; '
                'update the config to the canonical keys.',
                DeprecationWarning)
        return user_cfg

    def _memory_tuning_active(self):
        return bool(
            getattr(self, 'future_memory_adapter_finetune_mode', False) or
            getattr(self, 'memory_finetune_mode', False) or
            getattr(self, 'memory_joint_finetune_mode', False) or
            getattr(self, 'memory_phase2_finetune_mode', False) or
            getattr(self, 'memory_phase3_finetune_mode', False))

    def memory_tuning_trainable_prefixes(self):
        if getattr(self, 'future_memory_adapter_finetune_mode', False):
            if getattr(self, 'future_memory_adapter_version', 'v1') == 'v2':
                return self._FUTURE_MEMORY_V2_TRAINABLE_PREFIXES
            return self._FUTURE_MEMORY_TRAINABLE_PREFIXES
        if getattr(self, 'memory_phase3_finetune_mode', False):
            prefixes = []
            if getattr(self, 'query_memory_enabled', False):
                prefixes.append('query_memory.')
            if getattr(self, 'memory_conditioned_refiner_enabled', False):
                prefixes.extend((
                    'memory_refiner.',
                    'memory_cls_branch.',
                    'memory_reg_branch.',
                    'memory_vel_branch.',
                    'memory_horizon_embedding.',
                ))
            return tuple(prefixes)
        if getattr(self, 'memory_phase2_finetune_mode', False):
            return self._MEMORY_PHASE2_TRAINABLE_PREFIXES
        if getattr(self, 'memory_joint_finetune_mode', False):
            return self._MEMORY_JOINT_TRAINABLE_PREFIXES
        if getattr(self, 'memory_finetune_mode', False) or \
                self.query_memory_cfg.get('freeze_base_model', False):
            return self._MEMORY_ONLY_TRAINABLE_PREFIXES
        return tuple()

    def memory_tuning_train_modules(self):
        if getattr(self, 'future_memory_adapter_finetune_mode', False):
            if getattr(self, 'future_memory_adapter_version', 'v1') == 'v2':
                return self._FUTURE_MEMORY_V2_TRAIN_MODULES
            return self._FUTURE_MEMORY_TRAIN_MODULES
        if getattr(self, 'memory_phase3_finetune_mode', False):
            modules = []
            if getattr(self, 'query_memory_enabled', False):
                modules.append('query_memory')
            if getattr(self, 'memory_conditioned_refiner_enabled', False):
                modules.extend((
                    'memory_refiner',
                    'memory_cls_branch',
                    'memory_reg_branch',
                    'memory_vel_branch',
                    'memory_horizon_embedding',
                ))
            return tuple(modules)
        if getattr(self, 'memory_phase2_finetune_mode', False):
            return self._MEMORY_PHASE2_TRAIN_MODULES
        if getattr(self, 'memory_joint_finetune_mode', False):
            return self._MEMORY_JOINT_TRAIN_MODULES
        if getattr(self, 'memory_finetune_mode', False):
            return ('query_memory',)
        return tuple()

    def is_memory_tuning_parameter(self, name):
        return any(
            name.startswith(prefix)
            for prefix in self.memory_tuning_trainable_prefixes())

    def _configure_query_memory_trainability(self):
        """Apply one deterministic trainability policy after module creation."""
        freeze_base = bool(
            self.query_memory_cfg.get('freeze_base_model', False))
        tuning_modes = (
            getattr(self, 'future_memory_adapter_finetune_mode', False),
            getattr(self, 'memory_finetune_mode', False),
            getattr(self, 'memory_joint_finetune_mode', False),
            getattr(self, 'memory_phase2_finetune_mode', False),
            getattr(self, 'memory_phase3_finetune_mode', False),
        )
        if sum(tuning_modes) > 1:
            raise ValueError(
                'future_memory_adapter_finetune_mode, memory_finetune_mode, '
                'memory_joint_finetune_mode, '
                'memory_phase2_finetune_mode, and '
                'memory_phase3_finetune_mode are mutually exclusive')
        if self._memory_tuning_active():
            if getattr(self, 'future_memory_adapter_finetune_mode', False):
                mode_name = 'future_memory_adapter_finetune_mode'
            elif getattr(self, 'memory_phase3_finetune_mode', False):
                mode_name = 'memory_phase3_finetune_mode'
            elif getattr(self, 'memory_phase2_finetune_mode', False):
                mode_name = 'memory_phase2_finetune_mode'
            elif getattr(self, 'memory_joint_finetune_mode', False):
                mode_name = 'memory_joint_finetune_mode'
            else:
                mode_name = 'memory_finetune_mode'
            if (getattr(self, 'future_memory_adapter_finetune_mode', False)):
                adapter_present = (
                    self.future_memory_adapter_v2 is not None
                    if getattr(self, 'future_memory_adapter_version', 'v1') == 'v2'
                    else self.future_memory_adapter is not None)
                if not self.future_memory_adapter_enabled or not adapter_present:
                    raise ValueError(
                        'future_memory_adapter_finetune_mode=True requires '
                        'an enabled FutureMemoryAdapter')
                if self.query_memory_source != 'cache':
                    raise ValueError(
                        'future_memory_adapter_finetune_mode=True requires '
                        'source="cache"')
            elif (getattr(self, 'memory_phase3_finetune_mode', False) and
                    not self.query_memory_enabled and
                    not self.memory_conditioned_refiner_enabled):
                raise ValueError(
                    'memory_phase3_finetune_mode requires at least one of '
                    'Memory or refiner to be enabled')
            if (not getattr(self, 'future_memory_adapter_finetune_mode', False) and
                    not getattr(self, 'memory_phase3_finetune_mode', False) and
                    (not self.query_memory_enabled or
                     self.query_memory is None)):
                raise ValueError(
                    f'{mode_name}=True requires enabled STAC-QM')
            if (not getattr(self, 'future_memory_adapter_finetune_mode', False) and
                    self.query_memory_enabled and self.query_memory is None):
                raise ValueError(
                    f'{mode_name}=True requires enabled STAC-QM')
            if (not getattr(self, 'future_memory_adapter_finetune_mode', False) and
                    self.query_memory_enabled and self.query_memory_source != 'cache'):
                raise ValueError(
                    f'{mode_name}=True requires source="cache"')
            if not freeze_base:
                raise ValueError(
                    f'{mode_name}=True requires freeze_base_model=True')

        if freeze_base:
            prefixes = self.memory_tuning_trainable_prefixes()
            for name, param in self.named_parameters():
                param.requires_grad = any(
                    name.startswith(prefix) for prefix in prefixes)
        elif self.query_memory is not None:
            # Preserve the legacy non-finetune policy: query memory is inert unless
            # the base-freezing path is explicitly requested.
            for param in self.query_memory.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self._memory_tuning_active() and mode:
            # Keep the root in training mode so MMDetection executes forward_train
            # and computes losses. Only the explicit tuning modules may change
            # runtime state; every frozen child remains in eval mode so BN buffers
            # and dropout behavior stay fixed.
            train_modules = set(self.memory_tuning_train_modules())
            for name, module in self.named_children():
                module.train(name in train_modules)
        return self

    def _current_rap_masks(self):
        masks = []
        transformer = getattr(self.pts_bbox_head, 'transformer', None)
        decoder = getattr(transformer, 'decoder', None)
        layers = getattr(decoder, 'decoder_layers', [])
        for layer in layers:
            self_attn = getattr(layer, 'self_attn', None)
            mask = getattr(self_attn, 'ind_mask', None)
            if mask is not None:
                masks.append(mask.detach().clone())
        return masks

    def _freeze_memory_finetune_temporal_state(self):
        if not self._memory_tuning_active():
            return
        if self.frozen_num_stamps_all is not None:
            self._assert_memory_finetune_temporal_state()
            return
        head = self.pts_bbox_head
        if getattr(head, 'num_stamps_all', None) is None:
            raise RuntimeError(
                'Memory tuning requires pts_bbox_head.num_stamps_all')
        num_stamps = head.num_stamps_all.float()
        denominator = num_stamps.sum(dim=-1, keepdim=True)
        if (denominator <= 0).any():
            raise RuntimeError('num_stamps_all contains an empty timestamp row')
        normalized = num_stamps / denominator
        head.ind_stamps_all = get_matched_inds(
            normalized, [self.num_query] + self.num_fu_query)
        head.reset_mask()
        head.freeze_tass_state = True
        self.pretrain = False
        head.pretrain = False
        self.frozen_num_stamps_all = head.num_stamps_all.detach().clone()
        self.frozen_ind_stamps_all = head.ind_stamps_all.detach().clone()
        self._frozen_rap_masks = self._current_rap_masks()

    def _assert_memory_finetune_temporal_state(self):
        if not self._memory_tuning_active():
            return
        if self.frozen_num_stamps_all is None or \
                self.frozen_ind_stamps_all is None:
            raise RuntimeError(
                'Memory tuning temporal state was not finalized after checkpoint '
                'loading. Call validate_query_memory_training_setup first.')
        head = self.pts_bbox_head
        if not torch.equal(head.num_stamps_all, self.frozen_num_stamps_all):
            raise RuntimeError('num_stamps_all changed during Memory tuning')
        if not torch.equal(head.ind_stamps_all, self.frozen_ind_stamps_all):
            raise RuntimeError('ind_stamps_all changed during Memory tuning')
        current_masks = self._current_rap_masks()
        frozen_masks = self._frozen_rap_masks or []
        if len(current_masks) != len(frozen_masks):
            raise RuntimeError('RAP causal mask count changed during Memory tuning')
        for current, frozen in zip(current_masks, frozen_masks):
            if not torch.equal(current, frozen):
                raise RuntimeError('RAP causal mask changed during Memory tuning')

    def validate_query_memory_training_setup(
            self, optimizer=None, optimizer_cfg=None, logger=None):
        if (not self.query_memory_enabled and
                not self.memory_conditioned_refiner_enabled and
                not self._memory_tuning_active()):
            return dict(trainable_names=[], trainable_count=0)
        if self._memory_tuning_active():
            self._freeze_memory_finetune_temporal_state()
        if self.query_memory_source == 'online' and self.training:
            return dict(trainable_names=[], trainable_count=0)

        named_parameters = list(self.named_parameters())
        trainable = [
            (name, param) for name, param in named_parameters
            if param.requires_grad
        ]
        trainable_names = [name for name, _ in trainable]
        if self.query_memory_cfg.get('freeze_base_model', False):
            prefixes = self.memory_tuning_trainable_prefixes()
            expected = [
                (name, param) for name, param in named_parameters
                if any(name.startswith(prefix) for prefix in prefixes)
            ]
            bad = [
                name for name in trainable_names
                if not any(name.startswith(prefix) for prefix in prefixes)
            ]
            missing_trainable = [
                name for name, param in expected if not param.requires_grad
            ]
            if bad:
                raise RuntimeError(
                    'freeze_base_model=True found parameters outside the '
                    f'allowed tuning scope: {bad[:20]}')
            if missing_trainable:
                raise RuntimeError(
                    'Expected tuning parameters are frozen: '
                    f'{missing_trainable[:20]}')
            if not trainable:
                raise RuntimeError(
                    'freeze_base_model=True left no trainable parameters.')

        if self._memory_tuning_active() and optimizer is not None:
            trainable_by_id = {id(param): name for name, param in trainable}
            all_by_id = {id(param): name for name, param in named_parameters}
            optimizer_params = [
                param
                for group in optimizer.param_groups
                for param in group['params']
            ]
            optimizer_ids = [id(param) for param in optimizer_params]
            optimizer_id_counts = {}
            for param_id in optimizer_ids:
                optimizer_id_counts[param_id] = (
                    optimizer_id_counts.get(param_id, 0) + 1)
            optimizer_id_set = set(optimizer_id_counts)
            duplicate_ids = {
                param_id for param_id, count in optimizer_id_counts.items()
                if count != 1
            }
            unexpected_ids = optimizer_id_set - set(trainable_by_id)
            missing_ids = set(trainable_by_id) - optimizer_id_set
            if duplicate_ids:
                duplicate_names = [
                    all_by_id.get(param_id, f'<unknown:{param_id}>')
                    for param_id in duplicate_ids
                ]
                raise RuntimeError(
                    'Memory tuning optimizer contains duplicate parameters: '
                    f'{duplicate_names[:20]}')
            if unexpected_ids:
                unexpected_names = [
                    all_by_id.get(param_id, f'<unknown:{param_id}>')
                    for param_id in unexpected_ids
                ]
                raise RuntimeError(
                    'Memory tuning optimizer contains frozen or unexpected '
                    f'parameters: {unexpected_names[:20]}')
            if missing_ids:
                missing_names = [
                    trainable_by_id[param_id] for param_id in missing_ids
                ]
                raise RuntimeError(
                    'Memory tuning optimizer is missing trainable parameters: '
                    f'{missing_names[:20]}')

            if optimizer_cfg is not None:
                base_lr = optimizer_cfg.get('lr', None)
                base_wd = optimizer_cfg.get('weight_decay', None)
                paramwise_cfg = optimizer_cfg.get('paramwise_cfg', {}) or {}
                custom_keys = paramwise_cfg.get('custom_keys', {})
                sorted_keys = sorted(
                    sorted(custom_keys), key=len, reverse=True)
                optimizer_groups_by_id = {}
                for group in optimizer.param_groups:
                    for param in group['params']:
                        optimizer_groups_by_id[id(param)] = group
                lr_scales = []
                for name, param in trainable:
                    expected_lr = base_lr
                    expected_wd = base_wd
                    for key in sorted_keys:
                        if key not in name:
                            continue
                        if base_lr is not None:
                            expected_lr = base_lr * custom_keys[key].get(
                                'lr_mult', 1.0)
                        if base_wd is not None:
                            expected_wd = base_wd * custom_keys[key].get(
                                'decay_mult', 1.0)
                        break
                    group = optimizer_groups_by_id[id(param)]
                    actual_lr = group.get('lr', optimizer.defaults.get('lr'))
                    actual_wd = group.get(
                        'weight_decay', optimizer.defaults.get('weight_decay'))
                    if expected_lr is not None:
                        # MMCV restores the scheduler-adjusted current LR when
                        # resuming.  ``initial_lr`` remains the configured LR
                        # and is therefore the correct absolute value to
                        # validate.  The current values are checked below via
                        # their common scheduler scale.
                        initial_lr = group.get('initial_lr', None)
                        if initial_lr is not None and not np.isclose(
                                initial_lr, expected_lr,
                                rtol=1e-9, atol=1e-12):
                            raise RuntimeError(
                                f'Unexpected optimizer initial lr for {name}: '
                                f'{initial_lr} != {expected_lr}')
                        if expected_lr == 0:
                            if not np.isclose(
                                    actual_lr, 0.0, rtol=0.0, atol=1e-12):
                                raise RuntimeError(
                                    f'Unexpected optimizer lr for {name}: '
                                    f'{actual_lr} != 0')
                        else:
                            lr_scales.append((name, actual_lr / expected_lr))
                    if expected_wd is not None and not np.isclose(
                            actual_wd, expected_wd, rtol=1e-9, atol=1e-12):
                        raise RuntimeError(
                            f'Unexpected optimizer weight decay for {name}: '
                            f'{actual_wd} != {expected_wd}')
                if lr_scales:
                    reference_name, reference_scale = lr_scales[0]
                    for name, lr_scale in lr_scales[1:]:
                        if not np.isclose(
                                lr_scale, reference_scale,
                                rtol=1e-9, atol=1e-12):
                            raise RuntimeError(
                                'Optimizer parameter groups have inconsistent '
                                'lr scheduler scales: '
                                f'{name}={lr_scale} vs '
                                f'{reference_name}={reference_scale}')

        trainable_count = sum(param.numel() for _, param in trainable)
        message = (
            'STAC-QM trainable parameters '
            f'({trainable_count} scalars): {trainable_names}')
        if logger is not None:
            logger.info(message)
        elif self._memory_tuning_active():
            print(message, flush=True)
        return dict(
            trainable_names=trainable_names,
            trainable_count=trainable_count)

    def _meta_scene_id(self, meta):
        scene_id = meta.get('scene_token', None)
        if scene_id is None:
            scene_id = meta.get('scene_name', meta.get('scene_id', None))
        return None if scene_id is None else str(scene_id)

    def _meta_sample_idx(self, meta):
        sample_idx = meta.get('sample_idx', meta.get('sample_token', None))
        return None if sample_idx is None else str(sample_idx)

    def _meta_frame_idx(self, meta):
        frame_idx = meta.get('frame_idx', None)
        if frame_idx is None and 'curr' in meta:
            frame_idx = meta['curr'].get('frame_idx', None)
        return None if frame_idx is None else int(frame_idx)

    def _meta_timestamp(self, meta):
        timestamp = meta.get('timestamp', None)
        if timestamp is None:
            timestamp = meta.get('img_timestamp', None)
        if isinstance(timestamp, (list, tuple)):
            timestamp = timestamp[0]
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.detach().cpu().reshape(-1)[0].item()
        if timestamp is None:
            raise KeyError('timestamp or img_timestamp is required for STAC-QM')
        return float(timestamp)

    def _current_ego2global_tensor(self, img_metas, device):
        ego2globals = []
        for meta in img_metas:
            if 'ego2global' not in meta:
                raise KeyError('ego2global is required for STAC-QM alignment')
            value = meta['ego2global']
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            ego2globals.append(np.asarray(value, dtype=np.float32))
        return torch.tensor(
            np.stack(ego2globals), device=device, dtype=torch.float32)

    def _tensor_from_memory_kwargs(self, kwargs, key, device, dtype=None):
        value = kwargs[key]
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.to(device=device)
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def _query_memory_context(self, kwargs, img_metas, device, dtype):
        if not getattr(self, 'query_memory_enabled', False) and \
                not getattr(self, 'future_memory_adapter_enabled', False):
            return None
        if self.query_memory_source == 'cache':
            required = [
                'memory_query_feat', 'memory_points_metric', 'memory_conf',
                'memory_valid', 'memory_source_ego2global', 'memory_age'
            ]
            if getattr(self, 'future_memory_adapter_enabled', False):
                required.extend((
                    'memory_semantic_distribution', 'memory_label',
                    'memory_reliability'))
            missing = [key for key in required if key not in kwargs]
            if missing:
                if self._memory_tuning_active():
                    raise KeyError(
                        'Memory tuning requires cache tensors from the strict '
                        f'loader; missing: {missing}')
                return None
            context = dict(
                memory_query_feat=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_query_feat', device, dtype),
                memory_points_metric=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_points_metric', device, dtype),
                memory_conf=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_conf', device, torch.float32),
                memory_valid=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_valid', device).bool(),
                memory_source_ego2global=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_source_ego2global', device, torch.float32),
                memory_age=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_age', device, torch.float32))
            # schema-v2 optional fields (Problem 4). Absent for v1 caches, in
            # which case STAC-QM falls back to memory_conf as reliability.
            if 'memory_reliability' in kwargs:
                context['memory_reliability'] = self._tensor_from_memory_kwargs(
                    kwargs, 'memory_reliability', device, torch.float32)
            if 'memory_label' in kwargs:
                context['memory_label'] = self._tensor_from_memory_kwargs(
                    kwargs, 'memory_label', device, torch.long)
            if 'memory_semantic_distribution' in kwargs:
                context['memory_semantic_distribution'] = \
                    self._tensor_from_memory_kwargs(
                        kwargs, 'memory_semantic_distribution', device,
                        torch.float32)
            return context

        if self.training:
            return None  # STAC-QM params frozen, memory skipped during training
        _, world_size = get_dist_info()
        if world_size != 1:
            raise RuntimeError(
                'STAC-QM online bank requires single-GPU sequential eval; '
                f'got world_size={world_size}')
        if len(img_metas) != 1:
            raise RuntimeError(
                'STAC-QM online bank only supports batch_size=1; got '
                f'{len(img_metas)} samples')
        meta = img_metas[0]
        return self.query_memory_bank.read(
            scene_id=self._meta_scene_id(meta),
            sample_idx=self._meta_sample_idx(meta),
            frame_idx=self._meta_frame_idx(meta),
            timestamp=self._meta_timestamp(meta),
            device=device,
            dtype=dtype)

    def _apply_query_memory_once(self, query_feat, query_pos, query_cls,
                                 img_metas, memory_context, future_offset=0.0):
        if not self.query_memory_enabled:
            return query_feat
        if memory_context is None:
            return query_feat
        query_points_metric = decode_points_metric(query_pos, self.pc_range)
        query_conf = logits_to_query_confidence(query_cls.detach())
        target_ego2global = self._current_ego2global_tensor(
            img_metas, query_feat.device)
        fused, diagnostics = self.query_memory(
            query_feat=query_feat,
            query_points_metric=query_points_metric,
            current_confidence=query_conf,
            memory=memory_context,
            target_ego2global=target_ego2global,
            future_offset=future_offset)
        if self.query_memory_log_diagnostics:
            call_diagnostics = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor)
                else value
                for key, value in diagnostics.items()
            }
            call_diagnostics['query_count'] = int(query_feat.shape[1])
            call_diagnostics['future_offset'] = float(future_offset)
            self.query_memory_diagnostics.append(call_diagnostics)
        return fused

    def _memory_refiner_active(self):
        return bool(getattr(self, 'memory_conditioned_refiner_enabled', False))

    def _apply_memory_conditioned_refiner(self,
                                           query_feat,
                                           query_pos,
                                           base_cls,
                                           horizon_index):
        """Refine one Query group for both current and future routing.

        ``query_pos`` stays in OPUS encoded space at the API boundary.  The
        refiner consumes metric-space centers, while geometry is returned via
        the existing ``refine_points`` decode/refine/encode implementation.
        """
        if not self._memory_refiner_active():
            zero_vel = query_feat.new_zeros(
                query_feat.shape[0], query_feat.shape[1], self.num_refines, 2)
            return query_feat, base_cls, query_pos, zero_vel, dict(
                enabled=False, residual_ratio=0.0)

        query_pos_metric = decode_points(
            query_pos, self.pc_range).mean(dim=2)
        horizon = self.memory_horizon_embedding(
            query_feat.new_tensor(int(horizon_index), dtype=torch.long))
        refined_feat = self.memory_refiner(
            query_feat, query_pos_metric, horizon)
        cls_delta = self.memory_cls_branch(refined_feat).reshape(
            query_feat.shape[0], query_feat.shape[1], self.num_refines, 17)
        refined_cls = base_cls + cls_delta
        reg_delta = self.memory_reg_branch(refined_feat) * 0.5
        refined_pts = self.refine_points(query_pos, reg_delta)
        vel_delta = self.memory_vel_branch(refined_feat).reshape(
            query_feat.shape[0], query_feat.shape[1], self.num_refines, 2)
        residual_ratio = (
            (refined_feat - query_feat).float().norm(dim=-1) /
            (query_feat.float().norm(dim=-1) + 1e-6))
        return refined_feat, refined_cls, refined_pts, vel_delta, dict(
            enabled=True,
            horizon_index=int(horizon_index),
            residual_ratio=residual_ratio.detach(),
            cls_delta_norm=cls_delta.detach().float().norm(dim=-1),
            reg_delta_norm=reg_delta.detach().float().norm(dim=-1),
            vel_delta_norm=vel_delta.detach().float().norm(dim=-1))

    def _prepare_online_query_memory(self, img_metas):
        if not self.query_memory_enabled or self.query_memory_source != 'online':
            return
        if len(img_metas) != 1:
            raise RuntimeError(
                'STAC-QM online bank only supports batch_size=1; got '
                f'{len(img_metas)} samples')
        meta = img_metas[0]
        scene_id = self._meta_scene_id(meta)
        frame_idx = self._meta_frame_idx(meta)
        timestamp = self._meta_timestamp(meta)
        bank = self.query_memory_bank
        if bank._last_scene_id is not None and scene_id != bank._last_scene_id:
            bank.clear()
            return
        if frame_idx is not None and bank._last_frame_idx is not None:
            if frame_idx < bank._last_frame_idx:
                bank.clear()
                return
        if timestamp is not None and bank._last_timestamp is not None:
            if timestamp < bank._last_timestamp:
                bank.clear()
                return
            max_time_gap = self.query_memory_cfg.get('max_time_gap', None)
            if max_time_gap is not None:
                if timestamp - bank._last_timestamp > float(max_time_gap):
                    bank.clear()

    def _memory_write(self, curr_query_feat, curr_query_pos, curr_query_cls,
                      img_metas):
        if not self.query_memory_enabled or self.query_memory_source != 'online':
            return
        if curr_query_feat.shape[0] != 1:
            raise RuntimeError(
                'STAC-QM online bank only supports batch_size=1; got '
                f'B={curr_query_feat.shape[0]}')
        meta = img_metas[0]
        ego2global = self._current_ego2global_tensor(
            img_metas, curr_query_feat.device)
        points_metric = decode_points_metric(curr_query_pos, self.pc_range)
        self.query_memory_bank.write(
            curr_query_feat,
            points_metric,
            cls_scores=curr_query_cls,
            ego2global=ego2global,
            timestamp=self._meta_timestamp(meta),
            scene_id=self._meta_scene_id(meta),
            sample_idx=self._meta_sample_idx(meta),
            frame_idx=self._meta_frame_idx(meta))

    def _check_scene_change(self, img_metas):
        self._prepare_online_query_memory(img_metas)
        return False

    def forward_backbone(self,img,img_metas,**kwargs):

        B = img.shape[0]
        self._last_query_memory_valid_slots = 0.0
        if self.query_memory_log_diagnostics:
            self.query_memory_diagnostics.clear()
        if not hasattr(self, 'memory_refiner_diagnostics'):
            self.memory_refiner_diagnostics = []
        else:
            self.memory_refiner_diagnostics.clear()
        ego_states = kwargs['temporal_ego_states'][0]
        bs, _, dim_ = ego_states.shape
        ego_states = ego_states.view((bs, 1, dim_))
        ego_feat = self.plan_head(ego_states)
        points_scale = self.points_scale_branch(ego_feat)
        points_scale = torch.tanh(points_scale)
        self.pts_bbox_head.points_scale = (points_scale + 1) / 2 * (1.5 - 0.8) + 0.8

        if self.training:
            img_feats = self.extract_feat(img, img_metas)
            outs = self.pts_bbox_head(img_feats, img_metas)
        else:
            outs = self.simple_test_online(img_metas,img)

        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        query_feat = outs['query_feat']
        query_pos = outs['all_refine_pts'][-1]
        query_cls = outs['all_cls_scores'][-1]

        curr_query_feat = query_feat[:, ind_stamps_all == 0]
        curr_query_pos = query_pos[:, ind_stamps_all == 0].detach()
        curr_query_timestamp = query_pos.new_zeros(
            B, curr_query_feat.shape[1], self.num_refines, 1)
        curr_query_cls = query_cls[:, ind_stamps_all == 0]
        curr_query_cls_for_memory = curr_query_cls

        if getattr(self, 'future_memory_adapter_enabled', False):
            # The repaired adapter has its own cache read and output-only
            # correction stream.  It never enters the legacy STAC/refiner
            # path below, so baseline recurrence remains byte-for-byte in
            # terms of state topology.
            memory_context = self._query_memory_context(
                kwargs, img_metas, curr_query_feat.device,
                curr_query_feat.dtype)
            if memory_context is None:
                raise RuntimeError(
                    'FutureMemoryAdapter requires a cache memory context')
            memory_valid = memory_context.get('memory_valid')
            if isinstance(memory_valid, torch.Tensor) and memory_valid.numel():
                if memory_valid.ndim >= 3:
                    valid_slots = memory_valid.any(dim=-1).float().sum(dim=-1)
                else:
                    valid_slots = memory_valid.float().sum(dim=-1)
                self._last_query_memory_valid_slots = float(
                    valid_slots.mean().item())
            return self._forward_backbone_future_memory_adapter(
                img_metas, kwargs, B, ego_feat, outs, query_feat, query_pos,
                query_cls, ind_stamps_all, memory_context)

        memory_refiner_diagnostics = []
        outputs = dict(cls_score=curr_query_cls,
                       refine_pts=curr_query_pos,
                       outs=outs)

        if self.query_memory_enabled:
            outputs['_raw_query_feat'] = curr_query_feat.clone()
            outputs['_raw_query_pos'] = curr_query_pos.clone()
            outputs['_raw_query_cls'] = curr_query_cls.clone()
            memory_context = self._query_memory_context(
                kwargs, img_metas, curr_query_feat.device,
                curr_query_feat.dtype)
            if memory_context is not None and hasattr(memory_context, 'get'):
                memory_valid = memory_context.get('memory_valid')
                if isinstance(memory_valid, torch.Tensor) and \
                        memory_valid.numel():
                    if memory_valid.ndim >= 3:
                        valid_slots = memory_valid.any(dim=-1).float().sum(dim=-1)
                    else:
                        valid_slots = memory_valid.float().sum(dim=-1)
                    self._last_query_memory_valid_slots = float(
                        valid_slots.mean().item())
        else:
            memory_context = None

        # Problem 1 (single-read) + Problem 2 (future-aware age):
        # the observation queries read history memory EXACTLY ONCE, before the
        # SCF recursion, at effective_age = base_age + 0. Already-fused active
        # queries are never re-read inside the loop.  The future-only Phase 4
        # variant intentionally skips this observation read/refiner so that
        # the 0s output remains the raw OPUS baseline path.
        if getattr(self, 'memory_phase3_future_only', False):
            curr_query_memory_vel_offset = curr_query_feat.new_zeros(
                B, curr_query_feat.shape[1], self.num_refines, 2)
            refiner_diag = dict(
                enabled=False,
                future_only_skipped=True,
                residual_ratio=0.0)
        else:
            curr_query_feat = self._apply_query_memory_once(
                curr_query_feat, curr_query_pos, curr_query_cls_for_memory,
                img_metas, memory_context, future_offset=0.0)
            curr_query_feat, curr_query_cls, curr_query_pos, \
                curr_query_memory_vel_offset, refiner_diag = \
                self._apply_memory_conditioned_refiner(
                    curr_query_feat, curr_query_pos, curr_query_cls, 0)
        memory_refiner_diagnostics.append(refiner_diag)
        self.memory_refiner_diagnostics.append(refiner_diag)
        curr_query_cls_state = curr_query_cls
        curr_query_cls_for_memory = curr_query_cls_state
        outputs['cls_score'] = curr_query_cls
        outputs['refine_pts'] = curr_query_pos

        forecast_points_list = list()
        forecast_semantics_list = list()
        pred_trajs_list = list()
        forecast_points_mask_list = list()
        if self.training and not self._memory_tuning_active():
            num_fu_frames = max(
                1, min(
                    self.curr_epoch - self.finetune_epoch + 1,
                    self.num_fu_frames))
        else:
            # Both Memory-only and joint tuning train every configured future
            # horizon from their first iteration. The original progressive
            # schedule remains unchanged outside explicit Memory tuning.
            num_fu_frames = self.num_fu_frames

        for interval in range(num_fu_frames):
            # NOTE: the observation queries were already fused ONCE before this
            # loop and are NOT re-read here (Problem 1). Only the scheduled
            # future-query group added this step reads memory, and it does so
            # exactly once (below), with its own future_offset (Problem 2).
            fused_ego_feat,_ = self.ego_cross_attn(ego_feat.new_ones(B, 1, 3)*0.5, ego_feat, curr_query_pos.detach(),
                                                    curr_query_feat.detach(), )
            pred_traj = self.traj_head(fused_ego_feat)
            pred_trajs_list.append(pred_traj)

            scheduled_mask = ind_stamps_all == interval + 1
            scheduled_feat = query_feat[:, scheduled_mask]
            scheduled_pos = query_pos[:, scheduled_mask].detach()
            scheduled_cls = query_cls[:, scheduled_mask]
            if scheduled_feat.shape[1] > 0:
                # scheduled future-query group `interval` reads memory ONCE, at
                # effective_age = base_age + (interval + 1) * frame_interval.
                scheduled_future_offset = (
                    interval + 1) * self.query_memory_frame_interval
                scheduled_feat = self._apply_query_memory_once(
                    scheduled_feat, scheduled_pos, scheduled_cls, img_metas,
                    memory_context, future_offset=scheduled_future_offset)
                scheduled_feat, scheduled_cls, scheduled_pos, \
                    scheduled_memory_vel_offset, refiner_diag = \
                    self._apply_memory_conditioned_refiner(
                        scheduled_feat, scheduled_pos, scheduled_cls,
                        interval + 1)
                memory_refiner_diagnostics.append(refiner_diag)
                self.memory_refiner_diagnostics.append(refiner_diag)
                curr_query_feat = torch.cat(
                    [curr_query_feat, scheduled_feat], dim=1)
                curr_query_pos = torch.cat(
                    [curr_query_pos, scheduled_pos], dim=1)
                if not self._memory_refiner_active():
                    curr_query_pos = curr_query_pos.detach()
                curr_query_cls_for_memory = torch.cat(
                    [curr_query_cls_for_memory, scheduled_cls], dim=1)
                curr_query_cls_state = curr_query_cls_for_memory
                curr_query_memory_vel_offset = torch.cat(
                    [curr_query_memory_vel_offset,
                     scheduled_memory_vel_offset], dim=1)
                curr_query_timestamp = torch.cat([
                    curr_query_timestamp,
                    curr_query_pos.new_ones(
                        B, scheduled_feat.shape[1], self.num_refines, 1) * 0.5
                ], dim=1)

            pos_embedding = self.position_encoder(torch.cat([curr_query_pos,curr_query_timestamp],dim=-1).flatten(2,3))
            curr_query_feat = curr_query_feat + fused_ego_feat + pos_embedding

            reg_offset = self.reg_branch(curr_query_feat).unflatten(-1, (-1, 3)) * 0.5
            cls_delta = self.cls_branch(curr_query_feat).unflatten(-1, (-1, 17))
            if self._memory_refiner_active():
                cls_score = curr_query_cls_state + cls_delta
            else:
                cls_score = cls_delta
            curr_query_cls_for_memory = cls_score
            vel_offset = self.vel_branch(curr_query_feat).unflatten(-1, (-1, 2))
            if self._memory_refiner_active():
                vel_offset = vel_offset + curr_query_memory_vel_offset
            #
            pred_labels = cls_score.argmax(-1)
            pred_moving_mask = torch.logical_and(pred_labels >= 2, pred_labels <= 10).unsqueeze(-1)
            reg_offset = torch.cat(
                [reg_offset[..., :2] + vel_offset * pred_moving_mask, reg_offset[..., 2:]], dim=-1)
            reg_offset = reg_offset.flatten(2, 3)

            # reg_offset = self.reg_branch(curr_query_feat) * 0.5
            curr_query_pos = self.refine_points(curr_query_pos, reg_offset)
            forecast_semantics_list.append(cls_score)
            forecast_points_list.append(curr_query_pos)
            if self.training:
                ego2lidar =torch.tensor(np.stack([meta['ego2lidar'] for meta in img_metas]) ,device=device,dtype=torch.float32)
                gt_traj = kwargs['temporal_trajs'][:, interval:interval + 1, :]
                pred_traj_expand = torch.cat([-gt_traj, torch.zeros_like(pred_traj[:, :, :1])], dim=-1)

                gt_points = self.trans_points(curr_query_pos.flatten(1, 2), pred_traj_expand, ego2lidar).reshape(
                    curr_query_pos.shape)
                forecast_points_mask_list.append(gt_points[..., 0] >= 0)

        if not self.pretrain and len(pred_trajs_list)<self.num_fu_frames :
            fused_ego_feat,_ = self.ego_cross_attn(ego_feat.new_zeros(B, 1, 3), ego_feat, curr_query_pos,
                                                 curr_query_feat)
            pred_traj = self.traj_head(fused_ego_feat)
            pred_trajs_list.append(pred_traj)

        outputs.update(
                        dict(forecast_semantics_list = forecast_semantics_list,
                       forecast_points_list = forecast_points_list,
                       pred_trajs_list = pred_trajs_list,
                       forecast_points_mask_list = forecast_points_mask_list,
                       memory_refiner_diagnostics=memory_refiner_diagnostics,
                       base_cls_score=query_cls[:, ind_stamps_all == 0],
                       base_refine_pts=query_pos[:, ind_stamps_all == 0].detach()))
        return outputs

    def simple_test(self,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        for key in kwargs.keys():
            kwargs[key] = kwargs[key][0]

        if self.query_memory_enabled and self.query_memory_source == 'online':
            self._prepare_online_query_memory(img_metas)

        outputs = self.forward_backbone(img, img_metas, **kwargs)
        cls_score, curr_query_pos, outs = outputs['cls_score'],outputs['refine_pts'],outputs['outs']

        if self.query_memory_enabled and self.query_memory_source == 'online':
            raw_feat = outputs.get('_raw_query_feat')
            raw_pos = outputs.get('_raw_query_pos')
            raw_cls = outputs.get('_raw_query_cls')
            if raw_feat is None or raw_pos is None or raw_cls is None:
                raw_cls = outs['all_cls_scores'][-1][:, self.pts_bbox_head.ind_stamps_all == 0]
                raw_pos = outs['all_refine_pts'][-1][:, self.pts_bbox_head.ind_stamps_all == 0]
                raw_feat = outs['query_feat'][:, self.pts_bbox_head.ind_stamps_all == 0]
            self._memory_write(raw_feat, raw_pos, raw_cls, img_metas)

        # 0s occupancy must consume the Memory-conditioned refined state.  The
        # online write above intentionally uses raw OPUS observation Queries;
        # fixed-cache training never performs post-Memory recursive writes.
        pred_dict = dict(cls_scores=cls_score, refine_pts=curr_query_pos)
        occ_pred = self.pts_bbox_head.get_occ(pred_dict)[0]
        # self.pred_num += torch.bincount(occ_pred.flatten())
        geo_pred = torch.ones_like(occ_pred) * 17
        geo_pred[occ_pred != 17] = 0
        res_dict = {f'semantic_occ_0s': [occ_pred.cpu().numpy()],
                    f'geo_occ_0s': [geo_pred.cpu().numpy()]}

        forecast_points_list, forecast_semantics_list, pred_trajs_list = \
            outputs['forecast_points_list'],outputs['forecast_semantics_list'],outputs['pred_trajs_list']

        for interval in range(self.num_fu_frames):
            input_dict = dict(cls_scores=forecast_semantics_list[interval],
                              refine_pts=forecast_points_list[interval])
            occ_forecast = self.pts_bbox_head.get_occ(input_dict)[0]  # eval for single batch
            geo_forecast = torch.ones_like(occ_forecast) * 17
            geo_forecast[occ_forecast != 17] = 0
            # pred_traj_list.append(pred_traj)
            res_dict.update({
                f'semantic_occ_{int(interval + 1)}s': [occ_forecast.cpu().numpy()],
                f'geo_occ_{int(interval + 1)}s': [geo_forecast.cpu().numpy()],
            })

        res_dict['pred_traj'] = torch.cat(pred_trajs_list, 1)
        return res_dict

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **kwargs):

        temporal_semantics = kwargs['temporal_semantics']
        B = img.shape[0]
        temporal2ego = kwargs['temporal2ego']
        outputs = self.forward_backbone(img,img_metas,**kwargs)
        cls_score,refine_pts,outs = outputs['cls_score'],outputs['refine_pts'],outputs['outs']

        losses = dict()
        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        if self.future_memory_adapter_enabled:
            # Keep the 0 s objective exactly on the raw OPUS observation
            # queries.  The adapter is not present in this path.
            current_outs = dict(outs)
            current_outs['init_points'] = None
            current_outs['all_cls_scores'] = [
                score[:, ind_stamps_all == 0]
                for score in outs['all_cls_scores']]
            current_outs['all_refine_pts'] = [
                points[:, ind_stamps_all == 0]
                for points in outs['all_refine_pts']]
            losses.update(self.pts_bbox_head.loss(
                voxel_semantics, current_outs))
        elif self.pretrain:
            loss_inputs = [voxel_semantics, temporal_semantics, temporal2ego, outs]
            losses.update(self.pts_bbox_head.loss_pretrain(*loss_inputs))
        else:
            # Keep the original OPUS path as a scaled auxiliary objective in
            # Phase 3.  The primary current occupancy loss below is routed
            # through the Memory-conditioned refined outputs.
            loss_inputs = [voxel_semantics, temporal_semantics, temporal2ego, outs]
            raw_aux_losses = self.pts_bbox_head.loss_pretrain(*loss_inputs)
            if self._memory_refiner_active():
                raw_aux_losses = {
                    key: value * self.memory_phase3_base_aux_weight
                    for key, value in raw_aux_losses.items()}
            losses.update(raw_aux_losses)
            if self._memory_refiner_active():
                refined_outs = dict(
                    init_points=None,
                    all_cls_scores=[cls_score],
                    all_refine_pts=[refine_pts])
                losses.update(self.pts_bbox_head.loss(
                    voxel_semantics, refined_outs))
            else:
                outs['init_points'] = None
                for i in range(len(outs['all_cls_scores'])):
                    outs['all_cls_scores'][i] = outs['all_cls_scores'][i][:,ind_stamps_all==0]
                    outs['all_refine_pts'][i] = outs['all_refine_pts'][i][:,ind_stamps_all==0]
                loss_inputs = [voxel_semantics,outs,]
                losses.update(self.pts_bbox_head.loss(*loss_inputs))

        forecast_points_list = outputs['forecast_points_list']
        forecast_semantics_list = outputs['forecast_semantics_list']
        pred_trajs_list = outputs['pred_trajs_list']
        forecast_points_mask_list = outputs['forecast_points_mask_list']

        voxel_semantics_temporal = [sem['voxel_semantics'] for sem in kwargs['temporal_semantics'].values()]

        num_fu_frames = len(forecast_semantics_list)
        if self.future_memory_adapter_enabled:
            # Only target horizons receive corrected-output supervision.  The
            # existing validated matching loss is reused one horizon at a
            # time, then names are made explicit to avoid confusing internal
            # 0.5 s step numbers with evaluation horizons.
            for step_index, horizon_name in ((1, '1s'), (3, '2s'),
                                             (5, '3s')):
                if step_index >= num_fu_frames or \
                        step_index >= len(voxel_semantics_temporal):
                    raise RuntimeError(
                        f'FutureMemoryAdapter requires temporal step '
                        f'{step_index + 1} for mem_{horizon_name} loss')
                raw_semantic = (outputs['forecast_semantic_logits_list'][step_index]
                                if self.future_memory_adapter_version == 'v2'
                                else forecast_semantics_list[step_index])
                raw = self.pts_bbox_head.loss_future(
                    [voxel_semantics_temporal[step_index]],
                    [forecast_points_list[step_index]],
                    [raw_semantic],
                    [forecast_points_mask_list[step_index]])
                if self.future_memory_adapter_version == 'v2':
                    v2 = self._future_memory_v2_losses(
                        outputs['forecast_semantic_logits_list'][step_index],
                        outputs['forecast_occupancy_logits_list'][step_index],
                        forecast_points_list[step_index],
                        forecast_points_mask_list[step_index],
                        voxel_semantics_temporal[step_index])
                    cfg = self.future_memory_v2_loss_cfg
                    losses[f'mem_{horizon_name}.loss_base_cls'] = raw['fu1.loss_cls'] * float(cfg.get('loss_base_cls_weight', 1.0))
                    losses[f'mem_{horizon_name}.loss_sem'] = v2['loss_sem'] * float(cfg.get('loss_sem_weight', 0.25))
                    losses[f'mem_{horizon_name}.loss_occ'] = v2['loss_occ'] * float(cfg.get('loss_occ_weight', 0.25))
                    losses[f'mem_{horizon_name}.loss_threshold'] = v2['loss_threshold'] * float(cfg.get('loss_threshold_weight', 0.1))
                    losses[f'mem_{horizon_name}.loss_soft_voxel'] = v2['loss_soft_voxel'] * float(cfg.get('loss_soft_voxel_weight', 0.1))
                    losses[f'mem_{horizon_name}.loss_pts'] = raw['fu1.loss_pts'] * float(cfg.get('loss_pts_weight', 0.5))
                else:
                    losses[f'mem_{horizon_name}.loss_cls'] = raw['fu1.loss_cls']
                    losses[f'mem_{horizon_name}.loss_pts'] = raw['fu1.loss_pts']
        else:
            losses.update(
                self.pts_bbox_head.loss_future(
                    voxel_semantics_temporal[:num_fu_frames],
                    forecast_points_list, forecast_semantics_list,
                    forecast_points_mask_list))
        for interval,pred_traj in enumerate(pred_trajs_list):

            loss_traj = self.loss_traj(pred_traj.squeeze(1), kwargs['temporal_trajs'][:, interval, :], interval + 1)
            losses.update(loss_traj)

        return losses
