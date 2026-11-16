"""Frozen T1 perception + trainable Memory planning fusion, opt-in detector."""

import torch
from mmdet.models.builder import DETECTORS

from .memory_planning_fusion import MemoryPlanningFusion
from .sparseworld_4d_traj import SparseWorld4DTraj


@DETECTORS.register_module()
class SparseWorld4DTrajMemoryPlanning(SparseWorld4DTraj):
    def __init__(self, memory_planning_cfg=None,
                 memory_planning_source_checkpoint=None, **kwargs):
        super().__init__(**kwargs)
        if (not self.future_memory_adapter_enabled or self.future_memory_adapter_version != 'v2'
                or self.query_memory_source != 'cache' or
                not self.query_memory_cfg.get('freeze_base_model', False)):
            raise ValueError('Planning fusion requires frozen, cache-based T1/V2')
        if getattr(self.future_memory_adapter_v2, 'ablation_mode', 'full') != 'full':
            raise ValueError('Planning fusion uses the full T1 Memory model')
        cfg = dict(memory_planning_cfg or {})
        cfg.setdefault('embed_dims', self.out_dim)
        cfg.setdefault('num_classes', self.future_memory_adapter_v2.num_classes)
        cfg.setdefault('voxel_size', self.future_memory_adapter_v2.voxel_size)
        self.memory_planning_fusion = MemoryPlanningFusion(**cfg)
        self.memory_planning_source_checkpoint = memory_planning_source_checkpoint
        self.planning_fusion_only = True
        # The Memory model is frozen: retain its actual diagnostics, not the
        # training checkpoint wrapper's residual-only return values.
        self.future_memory_adapter_v2_activation_checkpoint = False
        self.future_memory_adapter_v2_save_on_cpu = False
        self._configure_query_memory_trainability()
        self.train(self.training)

    def init_weights(self):
        super().init_weights()
        if self.memory_planning_source_checkpoint:
            source = torch.load(self.memory_planning_source_checkpoint, map_location='cpu')
            self.load_frozen_t1_state(source['state_dict'])

    def load_frozen_t1_state(self, state):
        """Only the new fusion may be absent; reject missing frozen weights."""
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        expected = self.state_dict()
        missing = set(expected) - set(state)
        extra = set(state) - set(expected)
        wrong = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
        if extra or wrong or any(not k.startswith('memory_planning_fusion.') for k in missing):
            raise RuntimeError(dict(missing=sorted(missing), unexpected=sorted(extra), wrong_shape=wrong))
        self.load_state_dict({**expected, **state}, strict=True)
        return dict(missing_new_fusion=sorted(missing), loaded_frozen_keys=len(state))

    def memory_tuning_trainable_prefixes(self):
        if hasattr(self, 'memory_planning_fusion'):
            return ('memory_planning_fusion.',)
        return super().memory_tuning_trainable_prefixes()

    def memory_tuning_train_modules(self):
        if hasattr(self, 'memory_planning_fusion'):
            return ('memory_planning_fusion',)
        return super().memory_tuning_train_modules()

    def _make_memory_planning_snapshot(self, step_index, horizon_id, ego_feat,
                                       native_increment, query_feat, base_points,
                                       memory_points, base_logits, semantic_logits,
                                       occupancy_logits, adapter_result):
        tokens, valid = self.memory_planning_fusion.pack_scene(
            query_feat, base_points, memory_points, base_logits, semantic_logits,
            occupancy_logits, adapter_result, self.pc_range)
        return dict(step_index=step_index, horizon_id=horizon_id,
                    ego_feat=ego_feat.detach(), native_increment=native_increment.detach(),
                    tokens=tokens, query_valid=valid)

    def forward_backbone(self, img, img_metas, **kwargs):
        with torch.no_grad():
            outputs = super().forward_backbone(img, img_metas, **kwargs)
        snapshots = outputs.pop('memory_planning_snapshots')
        native = outputs['pred_trajs_list']
        planned = list(native)
        deltas = []
        for snapshot in snapshots:
            step = snapshot['step_index']
            delta = self.memory_planning_fusion(
                snapshot['ego_feat'], snapshot['tokens'], snapshot['query_valid'],
                snapshot['horizon_id'], snapshot['native_increment'])
            active = snapshot['query_valid'].any(-1)[:, None, None]
            planned[step] = torch.where(active, native[step] + delta, native[step])
            deltas.append(delta)
        outputs.update(pred_trajs_list=planned, baseline_pred_trajs_list=native,
                       memory_planning_deltas=deltas)
        return outputs

    def planning_losses(self, outputs, temporal_trajs):
        """Exactly reuse existing per-step loss_traj with original GT increments."""
        predictions = outputs['pred_trajs_list']
        if temporal_trajs.shape != (predictions[0].shape[0], len(predictions), 2):
            raise ValueError('Expected temporal_trajs [B,6,2] in original increment coordinates')
        losses = {}
        for index, prediction in enumerate(predictions):
            losses.update(self.loss_traj(prediction.squeeze(1), temporal_trajs[:, index], index + 1))
        return losses

    def forward_train(self, points=None, img_metas=None, img=None,
                      voxel_semantics=None, mask_camera=None, **kwargs):
        if 'temporal_trajs' not in kwargs:
            raise KeyError('temporal_trajs is required for planning supervision')
        # Labels enter the loss ONLY. Occupancy/future masks are unnecessary for
        # the frozen T1 forward; no GT trajectory enters the fusion inputs.
        model_inputs = {k: v for k, v in kwargs.items() if k != 'temporal_trajs'}
        outputs = self.forward_backbone(img, img_metas, **model_inputs)
        return self.planning_losses(outputs, kwargs['temporal_trajs'])
