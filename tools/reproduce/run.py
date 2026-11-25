"""Portable train/eval launcher. No heredocs or machine-specific paths required."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/sparseworld/nuscenes-temporal/memanchorworld-joint-reproduce.py'
CHECKPOINT = ROOT/'artifacts/seed17/epoch_1.pth'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','predict','score','evaluate','five-seeds'])
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--gpus',default='0,1')
    p.add_argument('--port',type=int,default=29741)
    p.add_argument('--output',type=Path,default=ROOT/'reproduce_outputs/seed_17')
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--limit',type=int,default=0,help='Predict-only sanity check; cannot be scored as full validation')
    a=p.parse_args();os.chdir(ROOT)
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=a.gpus,
             OMP_NUM_THREADS='1',NCCL_P2P_DISABLE='1',NCCL_ASYNC_ERROR_HANDLING='1',
             PYTORCH_CUDA_ALLOC_CONF='backend:native')
    py=sys.executable
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    ddp=[py,'-m','torch.distributed.run','--nproc_per_node=2',f'--master_port={a.port}']
    def call(cmd,log,cpu=False):
        print(' '.join(map(str,cmd)),flush=True)
        runenv=dict(env)
        if cpu:runenv['CUDA_VISIBLE_DEVICES']=''
        with log.open('x') as stream:
            proc=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=runenv,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:sys.stdout.write(line);stream.write(line)
            if proc.wait():raise subprocess.CalledProcessError(proc.returncode,cmd)
    if a.limit and a.stage!='predict':p.error('--limit is only valid for predict')
    if a.stage=='train':
        train=output/'train'
        if train.exists():raise FileExistsError(train)
        env['JOINT_TELEMETRY_DIR']=str(output/'train_telemetry')
        call(ddp+['deployment/joint_runtime.py','train',str(CONFIG),'--launcher','pytorch',
                  '--seed',str(a.seed),'--work-dir',str(train),'--cfg-options','log_config.interval=20'],output/'train.log')
    elif a.stage=='five-seeds':
        # Preparation captures an immutable source/config registry. Reuse is explicit.
        target=output/'five_seeds'
        command=[py,'tools/run_memory_planning_joint_five_seeds.py']
        subprocess.run(command+['prepare','--output',str(target)],cwd=ROOT,env=env,check=True)
        call(command+['run','--output',str(target),'--gpus',a.gpus,'--port',str(a.port)],output/'controller.log')
    else:
        evaluation=output/'eval_epoch_1'
        if a.stage in ('predict','evaluate'):
            if evaluation.exists():raise FileExistsError(evaluation)
            command=ddp+['tools/eval_memory_planning_joint.py','predict','--config',str(CONFIG),
                         '--checkpoint',str(a.checkpoint.resolve()),'--output',str(evaluation)]
            if a.limit:command+=['--limit',str(a.limit)]
            call(command,output/'predict.log')
        if a.stage in ('score','evaluate'):
            call([py,'tools/eval_memory_planning_joint.py','score','--config',str(CONFIG),
                  '--output',str(evaluation),'--baseline-metrics','artifacts/baseline/metrics.json'],output/'metrics.log',cpu=True)


if __name__=='__main__':main()
