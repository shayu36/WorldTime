"""Shared provenance and strict initialization for the RecallOcc 2x2 study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/sparseworld/recallocc'
MODES = {'E1': ('global', 'none'), 'E2': ('point', 'shared'),
         'E3': ('query', 'task_horizon'), 'E4': ('point', 'task_horizon')}
PREFIX = 'future_memory_adapter_v2.'
# These are dormant legacy modules constructed by SparseWorld, absent from
# epoch56. No original backbone/TASS keys may be missing or unexpected.
NEW_BASE_PREFIXES = (PREFIX, 'query_memory.', 'memory_refiner.',
                     'memory_horizon_embedding.', 'memory_cls_branch.',
                     'memory_reg_branch.', 'memory_vel_branch.')
POINT_MODULES = ('point_query_proj', 'point_key_proj', 'point_value_proj',
                 'memory_point_pos', 'semantic_query_proj', 'semantic_memory_proj')
GATE_MODULES = ('gate_occ_head', 'gate_pos_head', 'task_horizon_bias')
GLOBAL_UNUSED_MODULES = ('query_norm', 'memory_norm', 'query_proj',
                         'key_proj', 'value_proj', 'out_proj', 'gate_head')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def state_dict(checkpoint):
    return {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}


def tensor_hashes(state):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in state.items()}


def validate_state(model, state):
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    wrong = [k for k in expected.keys() & state.keys()
             if expected[k].shape != state[k].shape]
    if missing or extra or wrong:
        raise RuntimeError(dict(missing=missing, unexpected=extra, wrong_shape=wrong))


def canonical_subset(state, experiment):
    read, gate = MODES[experiment]
    excluded = (() if read == 'point' else POINT_MODULES) + (
        GLOBAL_UNUSED_MODULES if read == 'global' else ()) + (
        () if gate == 'task_horizon' else GATE_MODULES)
    def allowed(k):
        return any(k == PREFIX + m or k.startswith(PREFIX + m + '.') for m in excluded)
    removed = sorted(k for k in state if allowed(k))
    return {k: v for k, v in state.items() if not allowed(k)}, removed


def source_files():
    # Includes untracked experiment code; ignores datasets, checkpoints/env.
    return sorted(p for directory in ('mmdet3d', 'configs', 'tools')
                  for p in (ROOT / directory).rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.sh')
                  if '__pycache__' not in p.parts)


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in source_files()}


def gpu_check(gpus):
    ids = gpus.split(',')
    if not ids or any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('GPUs must be distinct comma-separated physical indices')
    rows = command(['nvidia-smi', '--id=' + gpus,
                    '--query-gpu=index,name,uuid,memory.used',
                    '--format=csv,noheader,nounits']).splitlines()
    if len(rows) != len(ids) or any(int(r.split(',')[-1]) > 1500 for r in rows):
        raise RuntimeError(f'GPU unavailable/busy; no process stopped: {rows}')
    return rows


def runtime_env(gpus):
    return dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=gpus,
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                NCCL_ASYNC_ERROR_HANDLING='1', NCCL_P2P_DISABLE='1',
                PYTORCH_CUDA_ALLOC_CONF='backend:native,max_split_size_mb:128')


def create_initialization(root, seed):
    """One canonical *untrained* E4 per seed; strict subsets for each arm.

    This copies actual tensors, not just integer seeds. Frozen base weights
    and all common randomly initialized parameters are bitwise identical.
    """
    import torch
    from mmcv import Config
    from mmdet.apis import set_random_seed
    from mmdet3d.models import build_model
    path = Path(root) / 'initializations' / f'seed_{seed}'
    path.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    set_random_seed(seed, deterministic=True)
    cfg = Config.fromfile(str(CONFIGS / 'E4.py'))
    model = build_model(cfg.model)
    model.init_weights()
    base_path = ROOT / cfg.recallocc.base_checkpoint
    base = state_dict(torch.load(base_path, map_location='cpu'))
    expected = model.state_dict()
    extra = sorted(set(base) - set(expected))
    missing = sorted(set(expected) - set(base))
    forbidden = [k for k in missing if not k.startswith(NEW_BASE_PREFIXES)]
    wrong = [k for k in base.keys() & expected.keys() if base[k].shape != expected[k].shape]
    if extra or forbidden or wrong:
        raise RuntimeError(dict(unexpected=extra, forbidden_missing=forbidden, wrong_shape=wrong))
    # Populate the exact expected key set; strict load never hides errors.
    full = dict(expected)
    full.update(base)
    model.load_state_dict(full, strict=True)
    model.validate_query_memory_training_setup()
    full = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert all(torch.equal(full[k], v) for k, v in base.items())
    hashes = tensor_hashes(full)
    report = dict(seed=seed, base_checkpoint=str(base_path), base_sha256=digest(base_path),
                  allowed_new_keys=missing, arms={}, common_tensor_hashes=hashes,
                  source_hashes=source_hashes())
    del model, expected
    for experiment in MODES:
        subset, removed = canonical_subset(full, experiment)
        cfg = Config.fromfile(str(CONFIGS / f'{experiment}.py'))
        model = build_model(cfg.model)
        validate_state(model, subset)
        model.load_state_dict(subset, strict=True)
        model.validate_query_memory_training_setup()
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert all(torch.equal(subset[k], v) for k, v in base.items())
        out = path / f'{experiment}.pth'
        torch.save(dict(state_dict=subset, meta=dict(seed=seed,
            experiment=experiment, initialization='fresh_canonical_E4_subset',
            base_checkpoint_sha256=report['base_sha256'])), out)
        report['arms'][experiment] = dict(checkpoint=str(out), sha256=digest(out),
            removed_keys=removed, trainable_parameters=count,
            trainable_tensors=sum(p.requires_grad for p in model.parameters()),
            total_parameters=sum(p.numel() for p in model.parameters()),
            common_parameters_bitwise_equal=True)
        del model
    write_json(path / 'initialization.json', report)
    return report
