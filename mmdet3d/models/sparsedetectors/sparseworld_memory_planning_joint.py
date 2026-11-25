"""Joint occupancy-adapter and planning-fusion fitting from one fixed best T1.

The existing stop-gradient at scene-token packing is retained: occupancy losses
train V2; trajectory losses train the planning fusion. Baseline remains frozen.
"""
import torch
from mmdet.models.builder import DETECTORS
from .sparseworld_4d_traj import SparseWorld4DTraj
from .sparseworld_memory_planning import SparseWorld4DTrajMemoryPlanning


@DETECTORS.register_module()
class SparseWorld4DTrajMemoryPlanningJoint(SparseWorld4DTrajMemoryPlanning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planning_joint_training = True
        self.future_memory_adapter_v2_activation_checkpoint = True
        self.future_memory_adapter_v2_save_on_cpu = False
        self._configure_query_memory_trainability()
        self.train(self.training)

    def memory_tuning_trainable_prefixes(self):
        return ('future_memory_adapter_v2.', 'memory_planning_fusion.')

    def memory_tuning_train_modules(self):
        return ('future_memory_adapter_v2', 'memory_planning_fusion')

    def forward_backbone(self, img, img_metas, **kwargs):
        # Baseline tensors have no trainable producers; V2 is deliberately NOT
        # enclosed in the planning-only parent's global no_grad scope.
        outputs = SparseWorld4DTraj.forward_backbone(self, img, img_metas, **kwargs)
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

    def forward_train(self, *args, **kwargs):
        # One shared forward; preserve every T1 occupancy objective and the
        # six original increment-space trajectory losses. No extra surrogate.
        return SparseWorld4DTraj.forward_train(self, *args, **kwargs)
