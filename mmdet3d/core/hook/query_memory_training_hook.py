# Copyright (c) Phigent Robotics. All rights reserved.
import hashlib

import torch
from mmcv.runner import HOOKS, OptimizerHook


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@HOOKS.register_module()
class QueryMemoryConnectivityOptimizerHook(OptimizerHook):
    """Optimizer hook for guarded STAC-QM Memory-only smoke training.

    It performs the normal zero-grad/backward/clip/step sequence while checking
    that only query-memory parameters receive gradients, the restored TASS state
    never changes, expected memory reads actually occur, and the zero-initialized
    fusion path becomes connected after a short warm-up.
    """

    _GRAD_GROUPS = {
        'fusion_out': ('query_memory.fusion.out_proj.weight',),
        'fusion_gate': ('query_memory.fusion.gate_mlp.',),
        'attention_q': ('query_memory.attention.q_proj.',),
        'attention_k': ('query_memory.attention.k_proj.',),
        'attention_v': ('query_memory.attention.v_proj.',),
        'motion_last': ('query_memory.motion_compensator.mlp.2.',),
    }

    def __init__(self,
                 grad_clip=None,
                 connectivity_check_iter=100,
                 log_interval=20,
                 expected_memory_reads=7,
                 expected_fused_queries=1040):
        super().__init__(grad_clip=grad_clip)
        self.connectivity_check_iter = int(connectivity_check_iter)
        self.log_interval = int(log_interval)
        self.expected_memory_reads = int(expected_memory_reads)
        self.expected_fused_queries = int(expected_fused_queries)
        self._ever_nonzero = {key: False for key in self._GRAD_GROUPS}
        self._base_parameter_digest = None
        self._base_buffer_digest = None
        self._connectivity_checked = False

    def before_run(self, runner):
        model = _unwrap_model(runner.model)
        if not getattr(model, 'memory_finetune_mode', False):
            raise RuntimeError(
                'QueryMemoryConnectivityOptimizerHook requires '
                'memory_finetune_mode=True')
        model.validate_query_memory_training_setup(
            optimizer=runner.optimizer, logger=runner.logger)
        self._base_parameter_digest = _tensor_digest([
            (name, param) for name, param in model.named_parameters()
            if not name.startswith('query_memory.')
        ])
        self._base_buffer_digest = _tensor_digest([
            (name, buffer) for name, buffer in model.named_buffers()
            if not name.startswith('query_memory.')
        ])

    def _gradient_metrics(self, model):
        metrics = {key: 0.0 for key in self._GRAD_GROUPS}
        for name, param in model.named_parameters():
            grad = param.grad
            if not name.startswith('query_memory.'):
                if grad is not None and torch.count_nonzero(grad).item() != 0:
                    raise RuntimeError(
                        f'Frozen base parameter received gradient: {name}')
                continue
            if grad is None:
                continue
            grad_norm = float(grad.detach().float().norm().item())
            for key, prefixes in self._GRAD_GROUPS.items():
                if any(name == prefix or name.startswith(prefix)
                       for prefix in prefixes):
                    metrics[key] += grad_norm
        for key, value in metrics.items():
            if value > 0.0:
                self._ever_nonzero[key] = True
        return metrics

    def _memory_metrics(self, model):
        diagnostics = list(getattr(model, 'query_memory_diagnostics', []))
        metrics = dict(
            memory_read_count=float(len(diagnostics)),
            memory_fused_queries=float(sum(
                int(item.get('query_count', 0)) for item in diagnostics)),
            memory_empty_forward=float(not diagnostics))
        tensor_keys = {
            'has_candidate': 'memory_has_candidate_ratio',
            'candidate_count': 'memory_candidate_count',
            'avg_gate': 'memory_gate_mean',
            'conditional_gate': 'memory_conditional_gate',
            'candidate_ratio': 'memory_candidate_ratio',
            'residual_norm': 'memory_residual_norm',
            'residual_ratio': 'memory_residual_ratio',
            'fusion_alpha': 'memory_fusion_alpha',
            'effective_age': 'memory_effective_age',
            'motion_residual_mean': 'memory_motion_residual',
        }
        for source_key, metric_key in tensor_keys.items():
            values = []
            for item in diagnostics:
                value = item.get(source_key)
                if isinstance(value, torch.Tensor) and value.numel():
                    # Diagnostics can mix CUDA tensors from active Query
                    # groups with Python scalars from skipped groups (for
                    # example 0s in the future-only route).  Aggregate on CPU
                    # so logging never introduces a cross-device torch.cat.
                    values.append(
                        value.detach().float().reshape(-1).cpu())
                elif isinstance(value, (float, int)):
                    values.append(torch.tensor([float(value)]))
            if values:
                metrics[metric_key] = float(torch.cat(values).mean().item())
        refiner_diagnostics = list(
            getattr(model, 'memory_refiner_diagnostics', []))
        refiner_keys = {
            'residual_ratio': 'memory_refiner_residual_ratio',
            'cls_delta_norm': 'memory_refiner_cls_delta_norm',
            'reg_delta_norm': 'memory_refiner_reg_delta_norm',
            'vel_delta_norm': 'memory_refiner_vel_delta_norm',
        }
        for source_key, metric_key in refiner_keys.items():
            values = []
            for item in refiner_diagnostics:
                value = item.get(source_key)
                if isinstance(value, torch.Tensor) and value.numel():
                    values.append(
                        value.detach().float().reshape(-1).cpu())
                elif isinstance(value, (float, int)):
                    values.append(torch.tensor([float(value)]))
            if values:
                metrics[metric_key] = float(torch.cat(values).mean().item())
        valid_slots = getattr(model, '_last_query_memory_valid_slots', None)
        if valid_slots is not None:
            metrics['memory_valid_slots'] = float(valid_slots)
        return metrics

    def _check_connectivity(self, runner):
        missing = [
            key for key in (
                'fusion_out', 'fusion_gate', 'attention_q', 'attention_k',
                'attention_v')
            if not self._ever_nonzero[key]
        ]
        if missing:
            raise RuntimeError(
                'STAC-QM connectivity check found no nonzero gradient for: '
                f'{missing}')
        if not self._ever_nonzero['motion_last']:
            runner.logger.warning(
                'STAC-QM motion compensator final layer still has zero gradient '
                'at connectivity check; inspect motion candidates before the '
                'formal run.')
        self._connectivity_checked = True

    def after_train_iter(self, runner):
        model = _unwrap_model(runner.model)
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        model._assert_memory_finetune_temporal_state()
        gradient_metrics = self._gradient_metrics(model)
        memory_metrics = self._memory_metrics(model)
        if int(memory_metrics['memory_read_count']) != self.expected_memory_reads:
            raise RuntimeError(
                'Unexpected STAC-QM read count: '
                f'{int(memory_metrics["memory_read_count"])} != '
                f'{self.expected_memory_reads}')
        if int(memory_metrics['memory_fused_queries']) != \
                self.expected_fused_queries:
            raise RuntimeError(
                'Unexpected STAC-QM fused-query count: '
                f'{int(memory_metrics["memory_fused_queries"])} != '
                f'{self.expected_fused_queries}')

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                gradient_metrics['total'] = float(grad_norm)
        runner.optimizer.step()

        iteration = int(runner.iter) + 1
        if iteration % self.log_interval == 0 or iteration == 1:
            log_values = {
                f'qm_grad_{key}': value
                for key, value in gradient_metrics.items()
            }
            log_values.update(memory_metrics)
            runner.log_buffer.update(
                log_values, runner.outputs.get('num_samples', 1))
        if not self._connectivity_checked and \
                iteration >= self.connectivity_check_iter:
            self._check_connectivity(runner)

    def after_run(self, runner):
        model = _unwrap_model(runner.model)
        model._assert_memory_finetune_temporal_state()
        parameter_digest = _tensor_digest([
            (name, param) for name, param in model.named_parameters()
            if not name.startswith('query_memory.')
        ])
        buffer_digest = _tensor_digest([
            (name, buffer) for name, buffer in model.named_buffers()
            if not name.startswith('query_memory.')
        ])
        if parameter_digest != self._base_parameter_digest:
            raise RuntimeError('Frozen base parameters changed during smoke run')
        if buffer_digest != self._base_buffer_digest:
            raise RuntimeError(
                'Frozen base buffers/BN statistics changed during smoke run')
        if not self._connectivity_checked:
            self._check_connectivity(runner)


@HOOKS.register_module()
class QueryMemoryJointConnectivityOptimizerHook(
        QueryMemoryConnectivityOptimizerHook):
    """Connectivity and frozen-state guard for joint STAC-QM tuning."""

    _MODE_ATTRIBUTE = 'memory_joint_finetune_mode'
    _MODE_NAME = 'memory_joint_finetune_mode'
    _CONNECTIVITY_LABEL = 'Joint STAC-QM'
    _RUN_LABEL = 'joint'

    _GRAD_GROUPS = {
        'fusion_out': ('query_memory.fusion.out_proj.weight',),
        'fusion_gate': ('query_memory.fusion.gate_mlp.',),
        'attention_q': ('query_memory.attention.q_proj.',),
        'attention_k': ('query_memory.attention.k_proj.',),
        'attention_v': ('query_memory.attention.v_proj.',),
        'motion_last': ('query_memory.motion_compensator.mlp.2.',),
        'position_encoder': ('position_encoder.',),
        'reg_branch': ('reg_branch.',),
        'vel_branch': ('vel_branch.',),
        'cls_branch': ('cls_branch.',),
        'ego_cross_attn': ('ego_cross_attn.',),
    }
    _REQUIRED_GRAD_GROUPS = (
        'fusion_out', 'fusion_gate', 'attention_q', 'attention_k',
        'attention_v', 'position_encoder', 'reg_branch', 'vel_branch',
        'cls_branch', 'ego_cross_attn')

    def before_run(self, runner):
        model = _unwrap_model(runner.model)
        if not getattr(model, self._MODE_ATTRIBUTE, False):
            raise RuntimeError(
                f'{type(self).__name__} requires {self._MODE_NAME}=True')
        model.validate_query_memory_training_setup(
            optimizer=runner.optimizer, logger=runner.logger)
        self._base_parameter_digest = _tensor_digest([
            (name, param) for name, param in model.named_parameters()
            if not model.is_memory_tuning_parameter(name)
        ])
        self._base_buffer_digest = _tensor_digest([
            (name, buffer) for name, buffer in model.named_buffers()
            if not model.is_memory_tuning_parameter(name)
        ])

    def _gradient_metrics(self, model):
        metrics = {key: 0.0 for key in self._GRAD_GROUPS}
        for name, param in model.named_parameters():
            grad = param.grad
            if not model.is_memory_tuning_parameter(name):
                if grad is not None and torch.count_nonzero(grad).item() != 0:
                    raise RuntimeError(
                        f'Frozen parameter received gradient: {name}')
                continue
            if grad is None:
                continue
            grad_norm = float(grad.detach().float().norm().item())
            for key, prefixes in self._GRAD_GROUPS.items():
                if any(name == prefix or name.startswith(prefix)
                       for prefix in prefixes):
                    metrics[key] += grad_norm
        for key, value in metrics.items():
            if value > 0.0:
                self._ever_nonzero[key] = True
        return metrics

    def _check_connectivity(self, runner):
        missing = [
            key for key in self._REQUIRED_GRAD_GROUPS
            if not self._ever_nonzero[key]
        ]
        if missing:
            raise RuntimeError(
                f'{self._CONNECTIVITY_LABEL} connectivity check found no '
                'nonzero gradient '
                f'for: {missing}')
        if not self._ever_nonzero['motion_last']:
            runner.logger.warning(
                'STAC-QM motion compensator final layer still has zero gradient '
                f'at {self._RUN_LABEL} connectivity check; inspect motion '
                'candidates before the formal run.')
        self._connectivity_checked = True

    def after_run(self, runner):
        model = _unwrap_model(runner.model)
        model._assert_memory_finetune_temporal_state()
        parameter_digest = _tensor_digest([
            (name, param) for name, param in model.named_parameters()
            if not model.is_memory_tuning_parameter(name)
        ])
        buffer_digest = _tensor_digest([
            (name, buffer) for name, buffer in model.named_buffers()
            if not model.is_memory_tuning_parameter(name)
        ])
        if parameter_digest != self._base_parameter_digest:
            raise RuntimeError(
                f'Frozen parameters changed during {self._RUN_LABEL} smoke '
                'run')
        if buffer_digest != self._base_buffer_digest:
            raise RuntimeError(
                'Frozen buffers/BN statistics changed during '
                f'{self._RUN_LABEL} smoke run')
        if not self._connectivity_checked:
            self._check_connectivity(runner)


@HOOKS.register_module()
class QueryMemoryPhase2ConnectivityOptimizerHook(
        QueryMemoryJointConnectivityOptimizerHook):
    """Connectivity and frozen-state guard for phase2 STAC-QM tuning."""

    _MODE_ATTRIBUTE = 'memory_phase2_finetune_mode'
    _MODE_NAME = 'memory_phase2_finetune_mode'
    _CONNECTIVITY_LABEL = 'Phase2 STAC-QM'
    _RUN_LABEL = 'phase2'
    _GRAD_GROUPS = {
        **QueryMemoryJointConnectivityOptimizerHook._GRAD_GROUPS,
        'pts_bbox_head': ('pts_bbox_head.',),
    }
    _REQUIRED_GRAD_GROUPS = (
        *QueryMemoryJointConnectivityOptimizerHook._REQUIRED_GRAD_GROUPS,
        'pts_bbox_head',
    )


@HOOKS.register_module()
class QueryMemoryPhase3ConnectivityOptimizerHook(
        QueryMemoryJointConnectivityOptimizerHook):
    """Connectivity guard for the Memory-conditioned occupancy refiner."""

    _MODE_ATTRIBUTE = 'memory_phase3_finetune_mode'
    _MODE_NAME = 'memory_phase3_finetune_mode'
    _CONNECTIVITY_LABEL = 'Phase3 STAC-QM'
    _RUN_LABEL = 'phase3'
    _GRAD_GROUPS = {
        'fusion_out': ('query_memory.fusion.out_proj.',),
        'fusion_gate': ('query_memory.fusion.gate_mlp.',),
        'fusion_alpha': ('query_memory.fusion.alpha',),
        'attention_q': ('query_memory.attention.q_proj.',),
        'attention_k': ('query_memory.attention.k_proj.',),
        'attention_v': ('query_memory.attention.v_proj.',),
        'motion_last': ('query_memory.motion_compensator.mlp.2.',),
        'memory_refiner': ('memory_refiner.',),
        'memory_cls_branch': ('memory_cls_branch.',),
        'memory_reg_branch': ('memory_reg_branch.',),
        'memory_vel_branch': ('memory_vel_branch.',),
        'memory_horizon_embedding': ('memory_horizon_embedding.',),
    }
    _REQUIRED_GRAD_GROUPS = (
        'fusion_out', 'fusion_gate', 'fusion_alpha', 'attention_q',
        'attention_k', 'attention_v', 'memory_refiner',
        'memory_cls_branch', 'memory_reg_branch', 'memory_vel_branch',
        'memory_horizon_embedding')

    def _required_grad_groups(self, model):
        """Return the gradient groups required by the active 2x2 cell.

        Phase 3 historically required both STAC-QM and the refiner.  Strict
        Memory/refiner ablations intentionally enable either side alone, so
        the connectivity assertion must follow the effective trainability
        policy instead of requiring gradients from disabled modules.
        """
        required = []
        if getattr(model, 'query_memory_enabled', False):
            required.extend((
                'fusion_out', 'fusion_gate', 'fusion_alpha',
                'attention_q', 'attention_k', 'attention_v'))
        if getattr(model, 'memory_conditioned_refiner_enabled', False):
            required.extend((
                'memory_refiner', 'memory_cls_branch',
                'memory_reg_branch', 'memory_vel_branch',
                'memory_horizon_embedding'))
        return tuple(required)

    def _check_connectivity(self, runner):
        model = _unwrap_model(runner.model)
        required = self._required_grad_groups(model)
        missing = [key for key in required if not self._ever_nonzero[key]]
        if missing:
            raise RuntimeError(
                f'{self._CONNECTIVITY_LABEL} connectivity check found no '
                'nonzero gradient for: '
                f'{missing}')
        if not self._ever_nonzero['motion_last'] and \
                getattr(model, 'query_memory_enabled', False):
            runner.logger.warning(
                'STAC-QM motion compensator final layer still has zero gradient '
                f'at {self._RUN_LABEL} connectivity check; inspect motion '
                'candidates before the formal run.')
        self._connectivity_checked = True
