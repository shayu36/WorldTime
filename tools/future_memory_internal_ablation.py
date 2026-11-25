"""Prepare, audit and run a local matched D5 continuation pair.

No cache/checkpoint is overwritten. Probe results are conditional on the
archived ego/cache protocol and are not a clean-causality validation result.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'configs/sparseworld/nuscenes-temporal'
PREFIX = 'sparseworld-traj-memory-future-adapter-v2-internal-d5-'
ARMS = ('3slots', '1slot')
SOURCE = (ROOT / 'work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-'
          'softvoxel-center-t1-from-center2/epoch_3.pth')
PYTHON = ROOT / 'env/bin/python'
TORCHRUN = ROOT / 'env/bin/torchrun'
TRACKED = [
    'mmdet3d/models/sparsedetectors/query_memory.py',
    'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py',
    'mmdet3d/models/sparsedetectors/opus_head.py',
    'mmdet3d/datasets/nuscenes_dataset_occ_trajectory.py',
    'mmdet3d/datasets/pipelines/loading_query_memory.py',
    'tools/train.py', 'mmdet3d/apis/train.py',
    'tools/train_future_memory_internal.py',
    'tools/future_memory_internal_ablation.py',
    'tools/eval_future_memory_internal.py',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Artifact output only; call sites create their run root exclusively.
    with Path(path).open('w') as f:
        json.dump(value, f, indent=2)


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def gpu_check():
    rows = command(['nvidia-smi', '--query-gpu=index,name,uuid,memory.used',
                    '--format=csv,noheader,nounits']).splitlines()
    if len(rows) != 2 or any('RTX 3090' not in row for row in rows):
        raise RuntimeError(f'Expected JXY local pair of RTX 3090s: {rows}')
    if any(int(row.split(',')[-1]) > 1500 for row in rows):
        raise RuntimeError(f'GPU busy; no process was stopped: {rows}')
    return rows


def check_sources(record):
    for path, digest in record['source_hashes'].items():
        if sha256(ROOT/path) != digest:
            raise RuntimeError(f'Source changed after prepare: {path}; prepare a new run')


def prepare(args):
    import torch
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.datasets.pipelines.loading_query_memory import LoadQueryMemoryFromFiles
    torch.set_num_threads(1)
    configs = {a: Config.fromfile(str(CONFIG_ROOT / (PREFIX + a + '.py')))
               for a in ARMS}
    reduced = copy.deepcopy(configs['1slot'].model)
    assert reduced.query_memory_cfg.future_memory_adapter_v2.pop('history_slot_limit') == 1
    assert reduced == configs['3slots'].model
    for key in ('data', 'optimizer', 'lr_config', 'fp16', 'runner', 'resume_from'):
        assert configs['1slot'][key] == configs['3slots'][key], key
    checkpoint = torch.load(SOURCE, map_location='cpu')
    assert checkpoint['meta']['epoch'] == 3
    assert checkpoint['optimizer']['state']
    start_iter = checkpoint['meta']['iter']
    state = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
    base = torch.load(ROOT / 'ckpts/epoch_56.pth', map_location='cpu')['state_dict']
    base = {k.removeprefix('module.'): v for k, v in base.items()}
    differences = [k for k in base if k not in state or not torch.equal(base[k], state[k])]
    if differences:
        raise RuntimeError(f'Frozen Baseline differs: {differences[:20]}')
    del checkpoint, state, base

    run = Path(args.run_dir).resolve()
    if ROOT / 'work_dirs' not in run.parents:
        raise ValueError('New runs must be under this repository work_dirs')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'configs').mkdir()
    git_status = command(['git', 'status', '--short', '--branch'])
    (run / 'initial_git_status.txt').write_text(git_status + '\n')
    (run / 'initial_processes.txt').write_text(command([
        'ps', '-eo', 'pid,ppid,stat,etime,comm']) + '\n')
    # Keep the user's dirty documentation out of the recorded source patch.
    patch = subprocess.check_output(['git', 'diff', '--', *TRACKED,
                                     'tests/test_future_memory_adapter_v2.py'], cwd=ROOT)
    (run / 'source_changes.patch').write_bytes(patch)
    for relative in TRACKED:
        dst = run / 'source_snapshot' / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / relative).read_bytes())
    audit = {}
    train_tokens = set()
    for split in ('train', 'test'):
        cfg = configs['3slots'].data[split]
        dataset = build_dataset(copy.deepcopy(cfg))
        tokens = {dataset.data_infos[i]['token'] for i in dataset.temp2nusc_map}
        if split == 'train':
            train_tokens = tokens
        else:
            assert not (train_tokens & tokens), 'train/validation token overlap'
        loader_cfg = dict(next(p for p in cfg.pipeline
                              if p['type'] == 'LoadQueryMemoryFromFiles'))
        loader_cfg.pop('type')
        assert loader_cfg['history_target_ages'] == [2.5, 3.5, 4.5]
        loader = LoadQueryMemoryFromFiles(**loader_cfg)
        counts = [0, 0, 0]
        checked_paths = {}
        for index, ni in enumerate(dataset.temp2nusc_map):
            info = dataset.data_infos[ni]
            histories = dataset._build_query_memory_history_infos(ni)
            assert len({h['sample_idx'] for h in histories}) == len(histories)
            current_ts = dataset._query_memory_timestamp(info)
            for h in histories:
                assert h['scene_id'] == dataset._query_memory_scene_id(info)
                assert h['timestamp'] < current_ts and h['frame_idx'] < info['frame_idx']
                path = loader._find_cache_path(loader.cache_root, h)
                assert path.is_file(), path
                counts[h['slot_index']] += 1
                # Bounded feature validation, complete metadata/path check.
                if index % max(1, len(dataset)//32) == 0 and str(path) not in checked_paths:
                    cache = torch.load(path, map_location='cpu')
                    loader._validate_cache(cache, path, hist=h, current_ts=current_ts,
                                           current_frame=info['frame_idx'])
                    assert cache['source_checkpoint'] == 'ckpts/epoch_56.pth'
                    checked_paths[str(path)] = dict(sha256=sha256(path),
                        queries=cache['query_feat'].shape[0], slot=h['slot_index'])
        audit[split] = dict(samples=len(dataset), available_slot_samples=counts,
                            feature_record_samples=checked_paths,
                            annotations_sha256=sha256(cfg.ann_file),
                            ego_input_path=cfg.get('ego_gt_path') or 'admlp/stp3_val/data_nuscene.pkl',
                            ego_input_sha256=sha256(cfg.get('ego_gt_path') or
                                                    'admlp/stp3_val/data_nuscene.pkl'))
        if split == 'test':
            # Evenly distributed deterministic indices; no load_interval
            # change, which would change future GT/history indexing.
            import numpy as np
            indices = np.unique(np.linspace(0, len(dataset)-1, args.eval_samples, dtype=int))
            manifest = [dict(index=int(i), token=dataset.data_infos[dataset.temp2nusc_map[i]]['token'],
                             scene=dataset.data_infos[dataset.temp2nusc_map[i]]['scene_name'])
                        for i in indices]
            write_json(run / 'eval_manifest.json', manifest)
        del dataset
    gpu = gpu_check()
    for arm, cfg in configs.items():
        folder = run / arm
        folder.mkdir()
        cfg.work_dir = str(folder)
        cfg.resume_from = str(SOURCE)
        cfg.load_from = None
        cfg.seed = 17
        cfg.auto_resume = False
        if args.mode == 'probe':
            cfg.runner = dict(type='IterBasedRunner', max_iters=start_iter+args.steps)
            cfg.checkpoint_config = dict(by_epoch=False, interval=args.steps,
                                          max_keep_ckpts=2, save_last=True)
        cfg.log_config.interval = 5 if args.mode == 'probe' else 50
        cfg.custom_hooks = list(cfg.get('custom_hooks', [])) + [dict(
            type='InternalAblationAuditHook', source_checkpoint=str(SOURCE),
            probe=args.mode == 'probe', priority='HIGH')]
        # tools/train.py writes its effective config into work_dir. Keep our
        # immutable launch config outside that directory for resumable checks.
        cfg.dump(str(run / 'configs' / (arm + '.py')))
    record = dict(created=now(), host=socket.gethostname(), repository=str(ROOT),
                  git_head=command(['git', 'rev-parse', 'HEAD']), gpu=gpu,
                  git_remotes=command(['git', 'remote', '-v']),
                  seed=17, mode=args.mode, steps=args.steps if args.mode == 'probe' else 9865,
                  source_checkpoint=str(SOURCE), source_sha256=sha256(SOURCE),
                  source_epoch=3, source_iter=start_iter,
                  source_hashes={p: sha256(ROOT/p) for p in TRACKED},
                  config_hashes={a: sha256(run/'configs'/(a+'.py')) for a in ARMS},
                  dataset_audit=audit,
                  protocol='archived T1 ego/cache; known ego future-dependency audit; conditional comparison only',
                  purpose='D5 matched continuation, not from-scratch architecture ranking',
                  status='prepared')
    provenance = ROOT / 'work_dirs/future_memory_t1_ego_provenance_20261020_verified.json'
    if provenance.is_file():
        record['known_ego_audit'] = dict(path=str(provenance), sha256=sha256(provenance))
    write_json(run / 'manifest.json', record)
    print(json.dumps(dict(run_dir=str(run), source_iter=start_iter,
                         samples={k: v['samples'] for k, v in audit.items()},
                         protocol=record['protocol']), indent=2), flush=True)


def run_logged(argv, log, env):
    print(shlex.join(map(str, argv)), flush=True)
    with Path(log).open('a') as out:
        out.write('\nCOMMAND ' + shlex.join(map(str, argv)) + '\n')
        out.flush()
        subprocess.run(list(map(str, argv)), cwd=ROOT, env=env,
                       stdout=out, stderr=subprocess.STDOUT, check=True)


def run_pair(args):
    import fcntl
    run = Path(args.run_dir).resolve()
    record = json.loads((run/'manifest.json').read_text())
    if socket.gethostname() != record['host']:
        raise RuntimeError('This prepared run belongs to another host')
    lock = (run/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    global_lock = (ROOT/'work_dirs/.future-memory-internal.lock').open('a')
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_sources(record)
    assert sha256(SOURCE) == record['source_sha256']
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='0,1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NCCL_ASYNC_ERROR_HANDLING='1', NCCL_P2P_DISABLE='1',
               PYTORCH_CUDA_ALLOC_CONF='backend:native,max_split_size_mb:128')
    status = {'started': now(), 'pid': os.getpid(), 'stages': {}}
    if (run/'status.json').exists():
        if not args.resume:
            raise RuntimeError('Run already has status; use --resume to continue completed stages')
        status = json.loads((run/'status.json').read_text())
        status['resumed_at'] = now()
        status['pid'] = os.getpid()
    write_json(run/'status.json', status)
    for arm in ARMS:
        folder = run / arm
        config = run / 'configs' / (arm + '.py')
        assert sha256(config) == record['config_hashes'][arm]
        stage = status['stages'].setdefault(arm, {})
        if stage.get('train') != 'complete':
            existing = sorted(folder.glob('*.pth'))
            source = str(SOURCE)
            if existing:
                # MMCV does not restore the exact sampler/RNG position for
                # partial-epoch resumes. Do not silently break paired inputs.
                raise RuntimeError(
                    f'{folder} has incomplete-run checkpoints. Start a new '
                    'matched pair from the original T1 source; files are retained.')
            check_sources(record)
            gpu_check()
            train = [TORCHRUN, '--nproc_per_node=2', '--master_port=29625',
                     'tools/train_future_memory_internal.py', config,
                     '--work-dir', folder, '--resume-from', source,
                     '--launcher', 'pytorch', '--seed', '17']
            stage.update(train='running', train_started=now(), train_command=shlex.join(map(str, train)))
            write_json(run/'status.json', status)
            try:
                run_logged(train, folder/'train_console.log', env)
            except Exception as exc:
                stage.update(train='failed', error=str(exc))
                write_json(run/'status.json', status)
                raise
            stage.update(train='complete', train_finished=now())
            write_json(run/'status.json', status)
        if stage.get('eval') != 'complete':
            check_sources(record)
            gpu_check()
            out = folder / 'evaluation'
            if out.exists():
                out = folder / ('evaluation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
            evaluate = [PYTHON, 'tools/eval_future_memory_internal.py', config,
                        folder/'latest.pth', '--manifest', run/'eval_manifest.json',
                        '--output-dir', out]
            stage.update(eval='running', evaluation_dir=str(out), eval_started=now(),
                         eval_command=shlex.join(map(str, evaluate)))
            write_json(run/'status.json', status)
            try:
                run_logged(evaluate, folder/'eval_console.log', env)
            except Exception as exc:
                stage.update(eval='failed', error=str(exc))
                write_json(run/'status.json', status)
                raise
            stage.update(eval='complete', eval_finished=now())
            write_json(run/'status.json', status)
    compare = [PYTHON, 'tools/eval_future_memory_internal.py', '--compare',
               *[status['stages'][a]['evaluation_dir'] for a in ARMS],
               '--output-dir', run/'comparison']
    if not (run/'comparison/paired_summary.json').exists():
        run_logged(compare, run/'comparison_console.log', env)
    status['completed'] = now()
    write_json(run/'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--mode', choices=('probe', 'epoch'), default='probe')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--eval-samples', type=int, default=32)
    p = sub.add_parser('run')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'prepare':
        if args.steps < 1 or not 1 <= args.eval_samples <= 4219:
            parser.error('steps must be positive; eval-samples must be 1..4219')
        prepare(args)
    else:
        run_pair(args)


if __name__ == '__main__':
    main()
