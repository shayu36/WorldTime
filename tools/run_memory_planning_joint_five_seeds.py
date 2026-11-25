"""Five independent joint V2 + planning-fusion fits from one fixed best T1.

prepare/check do not train or run GPU inference. run is explicitly invoked
later and refuses occupied GPUs. Completed seeds can be reused; interrupted
partial epochs are preserved and require a new output root (no fake resume).
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 23, 41, 73, 101)
CONFIG = ROOT / 'configs/sparseworld/nuscenes-temporal/memanchorworld-joint-reproduce.py'
SOURCE = ROOT / 'artifacts/initial_t1/epoch_3.pth'
SOURCE_SHA = 'bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c'
BASELINE = ROOT / 'artifacts/baseline/metrics.json'
DEFAULT_OUTPUT = ROOT / 'reproduce_outputs/five_seeds'
PREFIX = 'memory_planning_fusion.'
PREFIXES = ('future_memory_adapter_v2.', PREFIX)
PYTHON = Path(sys.executable)
TORCHRUN = [str(PYTHON), '-m', 'torch.distributed.run']
PROTOCOL = 'fixed_best_T1_joint_V2_fresh_planning_5seeds_1epoch_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def configuration(output, seed):
    from mmcv import Config
    cfg = Config.fromfile(str(CONFIG))
    cfg.work_dir = str(output / f'seed_{seed}' / 'train')
    cfg.seed = seed
    cfg.planning_source_checkpoint = str(SOURCE)
    cfg.model.memory_planning_source_checkpoint = str(SOURCE)
    cfg.load_from = None
    cfg.resume_from = None
    cfg.auto_resume = False
    cfg.runner.max_epochs = 1
    cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
    cfg.log_config.interval = 20
    return cfg


def assert_configuration(cfg, seed):
    assert cfg.seed == seed
    assert cfg.model.type == 'SparseWorld4DTrajMemoryPlanningJoint'
    assert cfg.model.memory_planning_source_checkpoint == str(SOURCE)
    assert not cfg.load_from and not cfg.resume_from and not cfg.auto_resume
    assert cfg.runner.max_epochs == 1
    assert cfg.data.samples_per_gpu == 1
    assert cfg.optimizer.constructor == 'TrainableOnlyOptimizerConstructor'
    assert cfg.optimizer.lr == 1e-4 and cfg.lr_config.policy == 'fixed'
    assert cfg.model.query_memory_cfg.freeze_base_model
    assert cfg.model.query_memory_cfg.future_memory_adapter_v2.activation_checkpoint
    assert cfg.optimizer.paramwise_cfg.custom_keys['future_memory_adapter_v2']['lr_mult'] == 0.2
    assert cfg.optimizer.paramwise_cfg.custom_keys['memory_planning_fusion']['lr_mult'] == 1.0


def train_command(config, seed, port):
    # --seed is read BEFORE build_model() by tools/train.py. No --resume-from:
    # T1 is loaded by init_weights(); the planning optimizer starts empty.
    return [*TORCHRUN, '--nproc_per_node=2', f'--master_port={port}',
            'deployment/joint_runtime.py', 'train', str(config), '--launcher', 'pytorch', '--seed', str(seed)]


def prepare(output):
    if output.exists():
        raise FileExistsError(f'{output} already exists; use check or run')
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError('Best T1 checkpoint SHA256 mismatch')
    baseline = read(BASELINE)
    assert digest(ROOT/'AD-MLP/deps/stp3/stp3/planning_metrics.py') == baseline['metric_source_sha256']
    for name, expected in baseline['gt_sha256'].items():
        assert digest(ROOT/'stp3_val'/name) == expected, 'Planning GT mismatch: '+name
    import torch
    torch.set_num_threads(1)
    cp = torch.load(SOURCE, map_location='cpu')
    assert not any(k.startswith(PREFIX) for k in cp['state_dict'])
    assert cp['meta']['epoch'] == 3
    source_info = dict(path=str(SOURCE), sha256=SOURCE_SHA,
                       source_tensor_count=len(cp['state_dict']),
                       frozen_tensor_count=sum(not k.startswith(PREFIXES) for k in cp['state_dict']),
                       epoch=cp['meta']['epoch'], iteration=cp['meta']['iter'])
    del cp
    output.mkdir(parents=True)
    (output / 'configs').mkdir()
    entries = []
    normalized = None
    for seed in SEEDS:
        cfg = configuration(output, seed)
        assert_configuration(cfg, seed)
        current = copy.deepcopy(cfg._cfg_dict)
        for key in ('seed', 'work_dir'):
            current.pop(key)
        if normalized is not None:
            assert current == normalized, 'Unexpected between-seed config change'
        normalized = current
        path = output / 'configs' / f'seed_{seed}.py'
        cfg.dump(str(path))
        entries.append(dict(seed=seed, config=str(path), config_sha256=digest(path),
                            work_dir=cfg.work_dir))
    # Freeze executable Python/config sources, including untracked planning code.
    # A later source edit causes check/run to fail instead of mixing versions.
    sources = {}
    for folder in ('mmdet3d', 'configs', 'tools', 'deployment'):
        for path in (ROOT / folder).rglob('*.py'):
            sources[str(path.relative_to(ROOT))] = digest(path)
    registry = dict(protocol=PROTOCOL, status='prepared_not_started', seeds=list(SEEDS),
        source=source_info, source_root=str(ROOT), source_sha256=sources,
        baseline_metrics=str(BASELINE), baseline_metrics_sha256=digest(BASELINE),
        entries=entries, epochs=1, iterations_per_seed=9865, train_samples=19730,
        validation_samples=4219, trainable_prefixes=list(PREFIXES),
        trainable_parameter_tensors=77, trainable_parameter_scalars=1722147,
        optimizer='fresh AdamW per seed; no T1 optimizer/scaler restored',
        random_scope='fresh planning initialization, sampler, image augmentation, V2 semantic dropout; frozen Baseline stays eval',
        evaluation_seed=0, statistics='sample standard deviation, ddof=1',
        scope='one-epoch joint adaptation conditional on one fixed best T1; not full-method training variance',
        gradients='occupancy losses -> V2; trajectory losses -> fusion; detached scene tokens',
        learning_rates=dict(V2=2e-5, planning_fusion=1e-4))
    write(output / 'registry.json', registry)
    summarize(output, registry)
    print('Prepared five configurations; no training or inference launched:', output)


def check(output):
    from mmcv import Config
    reg = read(output / 'registry.json')
    assert reg['protocol'] == PROTOCOL and reg['seeds'] == list(SEEDS)
    assert digest(SOURCE) == reg['source']['sha256'] == SOURCE_SHA
    assert digest(BASELINE) == reg['baseline_metrics_sha256']
    changed = [p for p, h in reg['source_sha256'].items()
               if not (ROOT / p).exists() or digest(ROOT / p) != h]
    if changed:
        raise RuntimeError('Prepared source changed: ' + repr(changed[:10]))
    for entry in reg['entries']:
        assert digest(entry['config']) == entry['config_sha256']
        assert_configuration(Config.fromfile(entry['config']), entry['seed'])
    return reg


def assert_free_gpus(gpus):
    if len(gpus.split(',')) != 2:
        raise ValueError('This protocol requires two GPUs, global batch size 2')
    text = subprocess.check_output(['nvidia-smi', '-i', gpus,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    used = [int(x.strip()) for x in text.splitlines()]
    # A conservative availability guard, not a model hyperparameter. A busy
    # controller exits without sending signals to any existing process.
    if len(used) != 2 or any(x > 1024 for x in used):
        raise RuntimeError(f'GPUs occupied ({used} MiB). Existing jobs left running.')


def audit_checkpoint(path, seed):
    import torch
    torch.set_num_threads(1)
    cp = torch.load(path, map_location='cpu')
    original = torch.load(SOURCE, map_location='cpu')['state_dict']
    assert cp['meta']['epoch'] == 1 and cp['meta']['iter'] == 9865
    assert cp['meta']['seed'] == seed
    state = cp['state_dict']
    frozen = {k:v for k,v in original.items() if not k.startswith(PREFIXES)}
    assert all(k in state and torch.equal(v, state[k]) for k,v in frozen.items())
    changed_v2 = [k for k,v in original.items() if k.startswith(PREFIXES[0]) and not torch.equal(v,state[k])]
    assert changed_v2, 'V2 did not update'
    fusion = {k:v for k,v in state.items() if k.startswith(PREFIX)}
    assert set(state) - set(original) == set(fusion)
    assert len(fusion) == 19 and sum(v.numel() for v in fusion.values()) == 224002
    assert all(torch.isfinite(v).all() for k,v in state.items() if k.startswith(PREFIXES))
    assert torch.count_nonzero(fusion[PREFIX+'residual_head.2.weight']) > 0
    assert len(cp['optimizer']['state']) == 77
    return dict(checkpoint=str(path), sha256=digest(path), seed=seed,
                epoch=1, iterations=9865, frozen_unchanged=True,
                frozen_scope='Baseline only, excluding V2 and planning fusion',
                V2_changed_tensors=changed_v2, fusion_tensors=19,
                trainable_parameter_tensors=77)



def aggregate(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def summarize(output, reg):
    rows = []
    for entry in reg['entries']:
        path = output / f"seed_{entry['seed']}" / 'eval_epoch_1' / 'all_metrics.json'
        row = dict(seed=entry['seed'], status='pending')
        if path.exists():
            result = read(path)
            audit = read(path.parent.parent / 'checkpoint_audit.json')
            assert result['checkpoint_sha256'] == audit['sha256']
            assert result['epoch'] == 1 and result['occupancy_samples'] == 4219
            assert result['planning_samples'] == 4219
            row.update(status='complete', results=str(path),
                       future_IoU=result['future_IoU'], future_mIoU=result['future_mIoU'])
            for h in ('1s', '2s', '3s'):
                values = result['planning']['prefix'][h]
                row[f'L2_{h}'] = values['L2']
                row[f'Obj_percent_{h}'] = 100 * values['obj_col']
                row[f'Box_percent_{h}'] = values['box_collision_percent']
            for metric in ('L2', 'Obj_percent', 'Box_percent'):
                row[f'{metric}_avg'] = statistics.mean(row[f'{metric}_{h}'] for h in ('1s','2s','3s'))
        rows.append(row)
    completed = [r for r in rows if r['status'] == 'complete']
    keys = [k for k in completed[0] if k not in ('seed', 'status', 'results')] if completed else []
    write(output / 'summary.json', dict(protocol=PROTOCOL, completed_seeds=len(completed),
          final=len(completed) == len(SEEDS), rows=rows,
          statistics={k:aggregate([r[k] for r in completed]) for k in keys},
          note='Independent one-epoch joint V2/fresh-fusion fits from fixed best T1; Baseline frozen.'))
    subprocess.run([str(PYTHON), 'deployment/export_planning_five_seeds_details.py',
                    '--output', str(output)], cwd=ROOT, check=True)


def run(output, gpus, port):
    reg = check(output)
    assert_free_gpus(gpus)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS='1',
               NCCL_ASYNC_ERROR_HANDLING='1', NCCL_P2P_DISABLE='1',
               PYTORCH_CUDA_ALLOC_CONF='backend:native')

    def invoke(command, log, cpu=False):
        runtime = dict(env)
        runtime['JOINT_TELEMETRY_DIR'] = str(log.parent / (log.stem + '_telemetry'))
        if cpu:
            runtime['CUDA_VISIBLE_DEVICES'] = ''
        write(output / 'state.json', dict(status='running', command=command, log=str(log)))
        print(json.dumps(command), flush=True)
        with log.open('xb') as stream:
            subprocess.run(command, cwd=ROOT, env=runtime, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for entry in reg['entries']:
                check(output)
                seed = entry['seed']
                work = Path(entry['work_dir'])
                base = work.parent
                base.mkdir(exist_ok=True)
                checkpoint = work / 'epoch_1.pth'
                if not checkpoint.exists():
                    if work.exists() or (base / 'train.log').exists():
                        raise RuntimeError(f'Partial epoch preserved in {base}; use a new output root to retry')
                    assert_free_gpus(gpus)
                    invoke(train_command(entry['config'], seed, port), base / 'train.log')
                audit = audit_checkpoint(checkpoint, seed)
                write(base / 'checkpoint_audit.json', audit)
                evaluation = base / 'eval_epoch_1'
                if not (evaluation / 'complete.json').exists():
                    if evaluation.exists():
                        raise RuntimeError(f'Partial evaluation preserved in {evaluation}; inspect before retry')
                    assert_free_gpus(gpus)
                    invoke([*TORCHRUN, '--nproc_per_node=2', f'--master_port={port+1}',
                        'tools/eval_memory_planning_joint.py', 'predict', '--config', entry['config'],
                        '--checkpoint', str(checkpoint), '--output', str(evaluation)], base / 'predict.log')
                    invoke([str(PYTHON), '-u', 'tools/eval_memory_planning_joint.py', 'score',
                        '--config', entry['config'], '--output', str(evaluation),
                        '--baseline-metrics', str(BASELINE)], base / 'metrics.log', cpu=True)
                summarize(output, reg)
            write(output / 'state.json', dict(status='complete', completed_seeds=5))
        except BaseException as exc:
            write(output / 'state.json', dict(status='failed', error=repr(exc)))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'check', 'run', 'summary'))
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--port', type=int, default=29741)
    args = p.parse_args()
    os.chdir(ROOT)
    output = args.output.resolve()
    if args.stage == 'prepare':
        prepare(output)
    elif args.stage == 'check':
        check(output)
        print('Five-seed source/config/checkpoint check: PASS; no GPU work started')
    elif args.stage == 'summary':
        summarize(output, read(output / 'registry.json'))
    else:
        run(output, args.gpus, args.port)


if __name__ == '__main__':
    main()
