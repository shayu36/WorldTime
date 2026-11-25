"""CPU-only protocol/initialization checks; no training or gradient computation."""
import ast
import copy
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = load('planning_seeds', ROOT / 'tools/run_memory_planning_five_seeds.py')
FUSION = load('planning_seed_fusion', ROOT / 'mmdet3d/models/sparsedetectors/memory_planning_fusion.py')


def test_five_configs_only_differ_in_seed_and_output(tmp_path):
    normalized = None
    for seed in RUN.SEEDS:
        cfg = RUN.configuration(tmp_path, seed)
        RUN.assert_configuration(cfg, seed)
        values = copy.deepcopy(cfg._cfg_dict)
        values.pop('seed')
        values.pop('work_dir')
        if normalized is not None:
            assert values == normalized
        normalized = values


def test_training_seeds_set_before_model_build_and_no_optimizer_resume(tmp_path):
    tree = ast.parse((ROOT / 'tools/train.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    seed_line = min(n.lineno for n in calls if n.func.id == 'set_random_seed')
    build_line = min(n.lineno for n in calls if n.func.id == 'build_model')
    assert seed_line < build_line
    for seed in RUN.SEEDS:
        argv = RUN.train_command(tmp_path / 'config.py', seed, 29681)
        assert argv[argv.index('--seed') + 1] == str(seed)
        assert '--resume-from' not in argv and '--auto-resume' not in argv


def test_fresh_fusion_varies_by_seed_but_zero_output_is_identical():
    states = []
    with torch.no_grad():
        for seed in (*RUN.SEEDS, RUN.SEEDS[0]):
            torch.manual_seed(seed)
            module = FUSION.MemoryPlanningFusion()
            assert sum(p.numel() for p in module.parameters()) == 224002
            assert len(list(module.parameters())) == 19
            states.append(module.token_encoder[0].weight.clone())
            ego = torch.ones(2, 1, 256)
            tokens = torch.ones(2, 3, module.token_dims)
            native = torch.tensor([[[.12345, .56789]], [[-.33, .77]]])
            valid = torch.tensor([[True, True, False], [False, False, False]])
            delta = module(ego, tokens, valid, 0, native)
            assert torch.equal(delta, torch.zeros_like(delta))
            assert torch.equal(native + delta, native)
    assert torch.equal(states[0], states[-1])
    assert all(not torch.equal(states[0], x) for x in states[1:-1])


def test_occupied_gpu_guard_never_launches_job(monkeypatch):
    monkeypatch.setattr(RUN.subprocess, 'check_output', lambda *a, **k: '21059\n22422\n')
    with pytest.raises(RuntimeError, match='Existing jobs left running'):
        RUN.assert_free_gpus('0,1')


def test_aggregate_does_not_fill_pending_seeds_with_zero():
    assert RUN.aggregate([]) == dict(n=0, mean=None, sample_std=None)
    assert RUN.aggregate([2.]) == dict(n=1, mean=2., sample_std=None)
    assert RUN.aggregate([1., 3.])['sample_std'] == pytest.approx(2. ** .5)
