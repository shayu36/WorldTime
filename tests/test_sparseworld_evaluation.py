from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmcv import Config
from mmcv.runner import HOOKS

from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook

from mmdet3d.apis.sparseworld_test import (
    _interleave_result_parts, _parse_sparseworld_result)
from mmdet3d.apis.train import (
    _select_detector_eval_hook, _select_detector_optimizer_target)
from mmdet3d.core.evaluation import (
    SparseWorldDistEvalHook, SparseWorldEvalHook)
from mmdet3d.core.evaluation.sparseworld_eval_hooks import (
    _flatten_eval_results)
from mmdet3d.core.hook.query_memory_training_hook import (
    QueryMemoryPhase3ConnectivityOptimizerHook)


def _temporal_result():
    return {
        'semantic_occ_0s': [np.full((2, 3), 1, dtype=np.int64)],
        'semantic_occ_2s': [np.full((2, 3), 2, dtype=np.int64)],
        'semantic_occ_4s': [np.full((2, 3), 3, dtype=np.int64)],
        'semantic_occ_6s': [np.full((2, 3), 4, dtype=np.int64)],
        'pred_traj': torch.ones(1, 6, 2),
    }


def test_sparseworld_dictionary_result_is_parsed_without_sequence_indexing():
    occ, pred_traj = _parse_sparseworld_result(_temporal_result())

    assert occ.shape == (4, 2, 3)
    assert occ.dtype == np.uint8
    assert occ[:, 0, 0].tolist() == [1, 2, 3, 4]
    assert pred_traj.device.type == 'cpu'
    assert pred_traj.shape == (1, 6, 2)


@pytest.mark.parametrize('invalid_result', [[], tuple(), np.zeros(1)])
def test_sparseworld_result_rejects_generic_sequence_outputs(invalid_result):
    with pytest.raises(TypeError, match='expects a dictionary'):
        _parse_sparseworld_result(invalid_result)


def test_distributed_result_interleave_trims_4219_sampler_padding():
    dataset_size = 4219
    rank_zero = list(range(0, dataset_size, 2))
    rank_one = list(range(1, dataset_size, 2)) + [0]

    ordered = _interleave_result_parts(
        [rank_zero, rank_one], size=dataset_size)

    assert ordered == list(range(dataset_size))


def test_temporal_metric_lists_are_flattened_for_tensorboard():
    flattened = _flatten_eval_results({
        'IoU': [25.68, 23.15, 22.27, 21.21],
        'mIoU': np.array([18.20, 14.96, 13.18, 11.53]),
        'classes': 17,
    })

    assert flattened['IoU_0s'] == pytest.approx(25.68)
    assert flattened['IoU_3s'] == pytest.approx(21.21)
    assert flattened['mIoU_future_mean'] == pytest.approx(
        (14.96 + 13.18 + 11.53) / 3)
    assert flattened['classes'] == 17
    assert all(not isinstance(value, (list, tuple, np.ndarray))
               for value in flattened.values())


def test_eval_hook_places_only_scalars_in_log_buffer():
    class Dataset:
        def evaluate(self, results, **kwargs):
            del results, kwargs
            return {
                'IoU': [25.68, 23.15, 22.27, 21.21],
                'mIoU': [18.20, 14.96, 13.18, 11.53],
                'classes': 17,
            }

    hook = SparseWorldEvalHook.__new__(SparseWorldEvalHook)
    hook.dataloader = SimpleNamespace(dataset=Dataset())
    hook.eval_kwargs = {}
    hook.save_best = None
    runner = SimpleNamespace(
        logger=None,
        log_buffer=SimpleNamespace(output={}, ready=False))

    assert hook.evaluate(runner, results=[]) is None
    assert runner.log_buffer.ready is True
    assert 'IoU' not in runner.log_buffer.output
    assert 'mIoU' not in runner.log_buffer.output
    assert runner.log_buffer.output['mIoU_future_mean'] == pytest.approx(
        (14.96 + 13.18 + 11.53) / 3)
    assert all(not isinstance(value, list)
               for value in runner.log_buffer.output.values())


def test_phase3_diagnostics_aggregate_mixed_scalar_and_tensor_devices():
    # Future-only routing records a Python scalar for the skipped 0s group and
    # tensors for active future groups.  On GPU these values must be moved to a
    # common device before aggregation.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNamespace(
        query_memory_diagnostics=[
            dict(query_count=1, residual_ratio=0.0),
            dict(query_count=2,
                 residual_ratio=torch.tensor([1.0, 3.0], device=device)),
        ],
        memory_refiner_diagnostics=[
            dict(residual_ratio=0.0),
            dict(residual_ratio=torch.tensor([2.0, 4.0], device=device)),
        ],
        _last_query_memory_valid_slots=0.0)

    metrics = QueryMemoryPhase3ConnectivityOptimizerHook()._memory_metrics(
        model)

    assert metrics['memory_read_count'] == pytest.approx(2.0)
    assert metrics['memory_fused_queries'] == pytest.approx(3.0)
    assert metrics['memory_residual_ratio'] == pytest.approx(4.0 / 3.0)
    assert metrics['memory_refiner_residual_ratio'] == pytest.approx(2.0)


def test_sparseworld_model_selects_dictionary_aware_eval_hooks():
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import (
        SparseWorld4DTraj)

    assert SparseWorld4DTraj.uses_sparseworld_eval_api is True

    class SparseWorldModel:
        uses_sparseworld_eval_api = True

    model = SparseWorldModel()
    assert _select_detector_eval_hook(model, distributed=False) is \
        SparseWorldEvalHook
    assert _select_detector_eval_hook(model, distributed=True) is \
        SparseWorldDistEvalHook


def test_standard_model_keeps_generic_mmdet_eval_hooks():
    model = object()
    assert _select_detector_eval_hook(model, distributed=False) is \
        MMDET_EvalHook
    assert _select_detector_eval_hook(model, distributed=True) is \
        MMDET_DistEvalHook


def test_phase2_optimizer_target_requires_trainable_only_constructor():
    model = SimpleNamespace(
        memory_finetune_mode=False,
        memory_joint_finetune_mode=False,
        memory_phase2_finetune_mode=True)

    with pytest.raises(RuntimeError, match=(
            'memory_phase2_finetune_mode requires optimizer.constructor')):
        _select_detector_optimizer_target(model, dict(type='AdamW'))

    unwrapped, target = _select_detector_optimizer_target(
        model,
        dict(type='AdamW',
             constructor='TrainableOnlyOptimizerConstructor'))
    assert unwrapped is model
    assert target is model


def test_phase2_configs_load_with_strict_cache_routes_and_registered_hook():
    config_root = (Path(__file__).resolve().parents[1] / 'configs' /
                   'sparseworld' / 'nuscenes-temporal')
    formal = Config.fromfile(
        str(config_root / 'sparseworld-traj-memory-phase2.py'))
    smoke = Config.fromfile(
        str(config_root / 'sparseworld-traj-memory-phase2-smoke.py'))

    memory_cfg = formal.model.query_memory_cfg
    assert memory_cfg.memory_finetune_mode is False
    assert memory_cfg.memory_joint_finetune_mode is False
    assert memory_cfg.memory_phase2_finetune_mode is True
    assert formal.optimizer.constructor == \
        'TrainableOnlyOptimizerConstructor'
    custom_keys = formal.optimizer.paramwise_cfg.custom_keys
    assert custom_keys.query_memory.lr_mult == pytest.approx(5.0)
    assert custom_keys.ego_cross_attn.lr_mult == pytest.approx(0.5)
    assert custom_keys.pts_bbox_head.lr_mult == pytest.approx(0.5)
    assert formal.runner.max_epochs == 12
    assert formal.evaluation.interval == 1
    assert formal.load_from == 'ckpts/epoch_56.pth'
    assert formal.resume_from is None

    expected_roots = {
        'train': './data/query_memory/sparseworld_epoch56_schema2_train',
        'val': './data/query_memory/sparseworld_epoch56_schema2_val',
        'test': './data/query_memory/sparseworld_epoch56_schema2_val',
    }
    for split, expected_root in expected_roots.items():
        dataset_cfg = formal.data[split]
        assert dataset_cfg.query_memory_cache_root == expected_root
        assert dataset_cfg.query_memory_strict is True
        loaders = [
            transform for transform in dataset_cfg.pipeline
            if transform.type == 'LoadQueryMemoryFromFiles'
        ]
        assert len(loaders) == 1
        assert loaders[0].cache_root == expected_root
        assert loaders[0].strict is True

    assert smoke.runner.type == 'IterBasedRunner'
    assert smoke.runner.max_iters == 200
    assert smoke.model.query_memory_cfg.log_diagnostics is True
    assert smoke.optimizer_config.type == \
        'QueryMemoryPhase2ConnectivityOptimizerHook'
    assert HOOKS.get('QueryMemoryPhase2ConnectivityOptimizerHook') is not None


def test_future_only_ablation_smoke_counts_match_scheduled_query_groups():
    config_root = (Path(__file__).resolve().parents[1] / 'configs' /
                   'sparseworld' / 'nuscenes-temporal')
    expected = {
        'sparseworld-traj-memory-ablation-M0R1-smoke.py': (0, 0),
        'sparseworld-traj-memory-ablation-M1R0-smoke.py': (6, 320),
        'sparseworld-traj-memory-ablation-M1R1-smoke.py': (6, 320),
    }
    for filename, (read_count, query_count) in expected.items():
        cfg = Config.fromfile(str(config_root / filename))
        assert cfg.model.query_memory_cfg.memory_phase3_future_only is True
        assert cfg.optimizer_config.expected_memory_reads == read_count
        assert cfg.optimizer_config.expected_fused_queries == query_count
