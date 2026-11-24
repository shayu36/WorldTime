"""Resume three planning epochs; score every new checkpoint before continuing."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def check_checkpoint(path, source):
    import torch
    ckpt = torch.load(path, map_location='cpu')
    original = torch.load(source, map_location='cpu')['state_dict']
    state = ckpt['state_dict']
    changed = [k for k, v in original.items() if k not in state or not torch.equal(v, state[k])]
    if changed:
        raise RuntimeError('Frozen T1 state changed: '+repr(changed[:10]))
    fusion = {k:v for k,v in state.items() if k.startswith('memory_planning_fusion.')}
    if not fusion or not all(torch.isfinite(v).all() for v in fusion.values()):
        raise RuntimeError('Missing/nonfinite planning fusion')
    if 'optimizer' not in ckpt or len(ckpt['optimizer']['state']) != len(fusion):
        raise RuntimeError('Cannot resume complete fusion optimizer')
    return dict(checkpoint=str(path), sha256=digest(path), epoch=ckpt['meta']['epoch'],
                iteration=ckpt['meta']['iter'], frozen_keys=len(original), frozen_unchanged=True,
                optimizer_states=len(ckpt['optimizer']['state']),
                fp16_scaler=ckpt['meta'].get('fp16'))


def summary(work, epochs):
    rows=[]
    for epoch in epochs:
        p=work/f'eval_epoch_{epoch}_all'/'all_metrics.json'
        if p.exists():
            d=json.loads(p.read_text())
            rows.append(dict(epoch=epoch, IoU=d['occupancy']['IoU'], mIoU=d['occupancy']['mIoU'],
                             Future_IoU=d['future_IoU'], Future_mIoU=d['future_mIoU'],
                             prefix_L2=[d['planning']['prefix'][h]['L2'] for h in ('1s','2s','3s')],
                             prefix_Obj_percent=[100*d['planning']['prefix'][h]['obj_col'] for h in ('1s','2s','3s')],
                             prefix_Box_percent=[d['planning']['prefix'][h]['box_collision_percent'] for h in ('1s','2s','3s')],
                             mean_L2=d['mean_prefix_L2'], mean_L2_delta=d['mean_prefix_L2_delta'],
                             checkpoint_sha256=d['checkpoint_sha256'], results=str(p)))
    write(work/'planning_all_epochs_summary.json', rows)
    lines=['# Planning fusion: native occupancy and Baseline-protocol planning', '',
           '| Epoch | Future IoU | Future mIoU | Prefix L2 1/2/3s (m) | Obj 1/2/3s (%) | Box 1/2/3s (%) |',
           '| --- | ---: | ---: | --- | --- | --- |']
    for r in rows:
        fmt=lambda values:' / '.join(f'{x:.6f}' for x in values)
        lines.append(f"| {r['epoch']} | {r['Future_IoU']:.4f} | {r['Future_mIoU']:.4f} | "
                     f"{fmt(r['prefix_L2'])} | {fmt(r['prefix_Obj_percent'])} | {fmt(r['prefix_Box_percent'])} |")
    lines+=['','Each row is read from its checkpoint-specific all_metrics.json. '
            'Missing rows are pending, not zero results. Raw six-step and endpoint metrics remain in each directory.']
    (work/'planning_all_epochs_summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--job-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--resume', required=True)
    p.add_argument('--additional-epochs', type=int, default=3)
    args=p.parse_args()
    root=Path.cwd();work=Path(args.work_dir).resolve();job=Path(args.job_dir).resolve()
    job.mkdir(parents=True,exist_ok=True)
    if (job/'state.json').exists():
        raise FileExistsError('Job state exists; inspect before a new invocation')
    env=os.environ.copy();env.update(PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES='0,1',
        NCCL_ASYNC_ERROR_HANDLING='1', NCCL_P2P_DISABLE='1', OMP_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='backend:native')
    python=str(root/'env/bin/python');torchrun=str(root/'env/bin/torchrun')
    from mmcv import Config
    cfg=Config.fromfile(args.config)
    source=Path(cfg.planning_source_checkpoint)
    initial=check_checkpoint(Path(args.resume),source)
    start=initial['epoch'];target=start+args.additional_epochs
    if args.additional_epochs != 3:
        raise ValueError('This authorized job is exactly three additional epochs')
    state=dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               resume=initial, target_epoch=target, additional_epochs=args.additional_epochs,
               phase='preflight', commands=[], evaluations=[])
    tracked=json.loads((job/'source_hashes.json').read_text())

    def verify_sources():
        changed=[name for name,h in tracked.items() if digest(root/name)!=h]
        if changed:
            raise RuntimeError('Runtime source changed during job: '+repr(changed))

    def run(command, phase, log, gpu=True):
        verify_sources();state['phase']=phase
        state['commands'].append(dict(phase=phase,argv=command,log=str(log)))
        write(job/'state.json',state)
        print(phase, json.dumps(command),flush=True)
        runtime=dict(env)
        if not gpu:runtime['CUDA_VISIBLE_DEVICES']=''
        with log.open('xb') as f:
            result=subprocess.run(command, env=runtime, stdout=f, stderr=subprocess.STDOUT)
        state['commands'][-1]['exit_code']=result.returncode
        write(job/'state.json',state)
        if result.returncode:
            raise RuntimeError(f'{phase} failed: {result.returncode}, see {log}')

    def evaluate(epoch):
        checkpoint=work/f'epoch_{epoch}.pth';output=work/f'eval_epoch_{epoch}_all'
        if output.exists():
            raise FileExistsError('Refusing to overwrite evaluation '+str(output))
        run([torchrun,'--nproc_per_node=2','--master_port=29548',
             'tools/eval_memory_planning_joint.py','predict','--config',args.config,
             '--checkpoint',str(checkpoint),'--output',str(output)],
            f'eval_epoch_{epoch}_predict',job/f'eval_epoch_{epoch}_predict.log')
        run([python,'-u','tools/eval_memory_planning_joint.py','score','--config',args.config,
             '--output',str(output)],f'eval_epoch_{epoch}_metrics',job/f'eval_epoch_{epoch}_metrics.log',gpu=False)
        state['evaluations'].append(epoch);write(job/'state.json',state)
        summary(work,range(start,target+1))

    try:
        write(job/'state.json',state)
        resume=Path(args.resume).resolve()
        for epoch in range(start+1,target+1):
            if (work/f'epoch_{epoch}.pth').exists():
                raise FileExistsError('Refusing to overwrite checkpoint')
            run([torchrun,'--nproc_per_node=2','--master_port=29547','tools/train.py',args.config,
                 '--work-dir',str(work),'--resume-from',str(resume),'--launcher','pytorch',
                 '--seed','17','--cfg-options',f'runner.max_epochs={epoch}',
                 'data.samples_per_gpu=1','data.workers_per_gpu=1','log_config.interval=20',
                 'checkpoint_config.max_keep_ckpts=-1'],f'train_epoch_{epoch}',job/f'train_epoch_{epoch}.log')
            resume=work/f'epoch_{epoch}.pth';check=check_checkpoint(resume,source)
            if check['epoch']!=epoch:
                raise RuntimeError('Wrong completed epoch')
            write(job/f'checkpoint_epoch_{epoch}_check.json',check)
            evaluate(epoch)
        # Epoch1 already has planning-only results: now fill its occupancy and
        # bind all metrics to a new joint forward without overwriting that record.
        evaluate(start)
        state.update(status='complete',phase='complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        write(job/'state.json',state)


if __name__=='__main__':
    main()
