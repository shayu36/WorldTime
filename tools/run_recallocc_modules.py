"""Prepare/run new E2/E3 module controls serially, with explicit checked resume."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from recallocc_modules_common import (ROOT, CONFIGS, MODES, command, create_initialization,
    digest, gpu_check, read_json, runtime_env, source_hashes, write_json)

PYTHON = ROOT/'env/bin/python'
TORCHRUN = ROOT/'env/bin/torchrun'


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def run_logged(argv, log, env):
    print(shlex.join(map(str, argv)), flush=True)
    with Path(log).open('a') as stream:
        stream.write('\nCOMMAND '+shlex.join(map(str,argv))+'\n')
        stream.flush()
        subprocess.run(list(map(str,argv)), cwd=ROOT,env=env,stdout=stream,
                       stderr=subprocess.STDOUT,check=True)


def data_manifest(root, smoke_samples, memory_disabled=False):
    import copy
    import numpy as np
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.datasets.pipelines.loading_query_memory import LoadQueryMemoryFromFiles
    cfg = Config.fromfile(str(CONFIGS/('E3_current_only_task_time.py' if memory_disabled else 'E2_full_retrieval_concat.py')))
    audit = {}
    train_tokens = set()
    for split in ('train','test'):
        dataset = build_dataset(copy.deepcopy(cfg.data[split]))
        tokens = {dataset.data_infos[i]['token'] for i in dataset.temp2nusc_map}
        if split == 'train':
            train_tokens = tokens
        else:
            assert not train_tokens & tokens
        if memory_disabled:
            audit[split] = dict(samples=len(dataset), memory_disabled=True,
                annotation=str((ROOT/cfg.data[split].ann_file).resolve()),
                annotation_sha256=digest(cfg.data[split].ann_file))
        else:
            lc = dict(next(p for p in cfg.data[split].pipeline if p['type']=='LoadQueryMemoryFromFiles'))
            lc.pop('type')
            cache_loader = LoadQueryMemoryFromFiles(**lc)
            counts=[0,0,0]
            for index,ni in enumerate(dataset.temp2nusc_map):
                for hist in dataset._build_query_memory_history_infos(ni):
                    path = cache_loader._find_cache_path(cache_loader.cache_root,hist)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    counts[hist['slot_index']]+=1
            audit[split] = dict(samples=len(dataset), annotation=str((ROOT/cfg.data[split].ann_file).resolve()),
                annotation_sha256=digest(cfg.data[split].ann_file), cache_root=str((ROOT/lc['cache_root']).resolve()),
                history_slots=lc['history_target_ages'], capacity=lc['max_queries_per_frame'],
                cache_references_by_slot=counts,
                ego_path=str((ROOT/(cfg.data[split].get('ego_gt_path') or 'admlp/stp3_val/data_nuscene.pkl')).resolve()))
        if split == 'test':
            indices = (np.unique(np.linspace(30,len(dataset)-31,smoke_samples,dtype=int)).tolist()
                       if smoke_samples else list(range(len(dataset))))
            manifest = [dict(index=i,token=dataset.data_infos[dataset.temp2nusc_map[i]]['token'],
                             scene=dataset.data_infos[dataset.temp2nusc_map[i]]['scene_name']) for i in indices]
            write_json(root/'eval_manifest.json',manifest)
    write_json(root/'data_audit.json',audit)
    return audit


def ensure_study(args):
    from mmcv import Config
    root = Path(args.work_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    sources = source_hashes()
    planned_seeds = args.seeds if getattr(args, 'action', 'batch') == 'batch' else [args.seed]
    spec = dict(smoke_steps=args.smoke_steps, eval_samples=args.eval_samples if args.smoke_steps else 0,
                epochs=args.epochs, effective_batch_size=2, world_size=len(args.gpus.split(',')),
                seeds=planned_seeds, experiment_modes=MODES)
    # JSON roundtrip normalizes tuple/list comparisons.
    spec = json.loads(json.dumps(spec))
    study_path = root/'study.json'
    if study_path.exists():
        study=read_json(study_path)
        if study['spec'] != spec:
            raise RuntimeError('Study budget/world-size differs: choose another --work-root')
        if study['source_hashes'] != sources:
            changed=[k for k in set(sources)|set(study['source_hashes'])
                     if sources.get(k)!=study['source_hashes'].get(k)]
            raise RuntimeError(f'Source differs from study snapshot: {changed}')
        if digest(ROOT/'ckpts/epoch_56.pth') != study['base_sha256']:
            raise RuntimeError('Base checkpoint changed')
        return root,study
    if any(root.iterdir()):
        raise FileExistsError(f'Unrecognized nonempty study root {root}')
    configs={e:Config.fromfile(str(CONFIGS/f'{e}.py')) for e in MODES}
    reference=Config.fromfile(str(CONFIGS/'E4.py')).to_dict()
    for e,c in configs.items():
        for key in ('optimizer','lr_config','optimizer_config','fp16','runner','test_cfg'):
            assert c.get(key)==reference.get(key), (e,key)
        model=c.to_dict()['model']
        model['query_memory_cfg']['future_memory_adapter_v2'].pop('ablation_mode')
        model['query_memory_cfg']['future_memory_adapter_v2'].pop('current_point_support_fraction', None)
        assert model==reference['model'],e
    (root/'configs').mkdir()
    (root/'source_snapshot').mkdir()
    for path in sources:
        out=root/'source_snapshot'/path
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes((ROOT/path).read_bytes())
    (root/'git-status.txt').write_text(command(['git','status','--short','--branch'])+'\n')
    (root/'workspace.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
    requested = args.experiments if args.action == 'batch' else [args.experiment]
    audit=data_manifest(root,spec['eval_samples'], all(e.startswith('E3_') for e in requested))
    study=dict(created=stamp(),repository=str(ROOT),git_branch=command(['git','branch','--show-current']),
        git_commit=command(['git','rev-parse','HEAD']),spec=spec,source_hashes=sources,
        source_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        base_checkpoint=str(ROOT/'ckpts/epoch_56.pth'),base_sha256=digest(ROOT/'ckpts/epoch_56.pth'),
        dataset=audit,eval_manifest_sha256=digest(root/'eval_manifest.json'),
        checkpoint_selection='final_iteration' if args.smoke_steps else 'final_epoch')
    write_json(study_path,study)
    return root,study


def initialization(root,seed,env):
    path=root/'initializations'/f'seed_{seed}'/'initialization.json'
    if not path.exists():
        run_logged([PYTHON,'tools/run_recallocc_modules.py','initialize','--work-root',root,'--seed',seed],
                   root/f'initialization_seed{seed}.log',env)
    report=read_json(path)
    if report['source_hashes']!=source_hashes():
        raise RuntimeError('Initialization code changed')
    for arm in report['arms'].values():
        if digest(arm['checkpoint'])!=arm['sha256']:
            raise RuntimeError('Initialization checkpoint changed')
    return report


def check_resume(checkpoint,record,study):
    import torch
    payload=torch.load(checkpoint,map_location='cpu')
    meta=payload.get('meta',{}).get('recallocc',{})
    for key in ('experiment','seed','config_sha256','source_sha256','base_sha256'):
        if meta.get(key)!=record[key]:
            raise RuntimeError(f'Resume mismatch {key}: {meta.get(key)} != {record[key]}')
    if not payload.get('optimizer',{}).get('state'):
        raise RuntimeError('Resume requires optimizer state')
    iteration=payload['meta']['iter']
    if not study['spec']['smoke_steps']:
        per_epoch=(study['dataset']['train']['samples']+1)//2
        if iteration % per_epoch:
            raise RuntimeError('Only epoch-boundary resume preserves the matched sampling sequence')
    return payload['meta']


def run_arm(args,root,study,experiment,seed,env,checkpoint=None):
    from mmcv import Config
    folder=root/experiment/f'seed_{seed}'
    record_path=folder/'run.json'
    if record_path.exists():
        if not args.resume:
            raise FileExistsError(f'{folder} exists; explicit --resume CHECKPOINT (or batch --resume auto) required')
        record=read_json(record_path)
        if digest(record['config'])!=record['config_sha256']:
            raise RuntimeError('Launch configuration changed')
        if record['source_sha256']!=study['source_sha256'] or record['base_sha256']!=study['base_sha256']:
            raise RuntimeError('Study identity mismatch')
    else:
        if args.resume and args.resume!='auto':
            raise RuntimeError('Cannot resume into an unrecognized run directory')
        init=initialization(root,seed,env)
        folder.mkdir(parents=True,exist_ok=False)
        cfg=Config.fromfile(str(CONFIGS/f'{experiment}.py'))
        cfg.work_dir=str(folder)
        cfg.seed=seed
        cfg.load_from=init['arms'][experiment]['checkpoint']
        cfg.resume_from=None
        cfg.auto_resume=False
        cfg.data.samples_per_gpu=2//len(args.gpus.split(','))
        cfg.data.workers_per_gpu=1
        cfg.recallocc.checkpoint_selection=study['checkpoint_selection']
        if args.smoke_steps:
            cfg.runner=dict(type='IterBasedRunner',max_iters=args.smoke_steps)
            cfg.checkpoint_config=dict(by_epoch=False,interval=1,max_keep_ckpts=args.smoke_steps,save_last=True)
            cfg.log_config.interval=1
        else:
            cfg.runner.max_epochs=args.epochs
        cfg.custom_hooks=list(cfg.custom_hooks)+[dict(type='RecallOccModulesAuditHook',source_checkpoint=cfg.load_from,
            seed=seed,run_manifest=str(record_path),priority=45)]
        config=root/'configs'/f'{experiment}_seed_{seed}.py'
        cfg.dump(str(config))
        read,gate=MODES[experiment]
        record=dict(experiment=experiment,seed=seed,read_mode=read,gate_mode=gate,
            config=str(config),config_sha256=digest(config),source_sha256=study['source_sha256'],
            source_snapshot=str(root/'source_snapshot'),git_commit=study['git_commit'],
            workspace_patch=str(root/'workspace.patch'),base_sha256=study['base_sha256'],
            base_checkpoint=study['base_checkpoint'],
            variant=cfg.recallocc.variant, design='modules_v2', budget=study['spec'],
            memory='memory_disabled' if experiment.startswith('E3_') else study['dataset'],
            initialization=init['arms'][experiment],checkpoint_selection=study['checkpoint_selection'],
            launch_history=[],train='prepared',evaluation=None)
        write_json(record_path,record)
    if record['train']!='complete':
        gpu_check(args.gpus)
        source=record['initialization']['checkpoint']
        resume=None
        if args.resume:
            if args.resume=='auto':
                candidate=folder/'latest.pth'
                if candidate.exists():
                    resume=candidate.resolve()
                elif list(folder.glob('*.pth')):
                    raise RuntimeError('Checkpoints exist without latest.pth; choose one explicitly')
            else:
                resume=Path(args.resume).resolve()
            if resume:
                check_resume(resume,record,study)
                source=str(resume)
        elif list(folder.glob('*.pth')):
            raise RuntimeError('Existing checkpoints require explicit --resume')
        argv=[TORCHRUN,'--standalone',f'--nproc_per_node={len(args.gpus.split(","))}',
              'tools/train_recallocc_modules.py',record['config'],'--launcher','pytorch',
              '--seed',seed,'--deterministic']
        if resume:
            argv+=['--resume-from',resume]
        log=folder/f'train_{stamp()}.log'
        record['launch_history'].append(dict(command=shlex.join(map(str,argv)),gpus=args.gpus,
            source=source,log=str(log),time=stamp()))
        record['train']='running'
        write_json(record_path,record)
        try:
            run_logged(argv,log,dict(env,RECALLOCC_LOAD_SOURCE=source))
        except Exception as exc:
            record.update(train='failed', failure=str(exc))
            write_json(record_path,record)
            raise
        selected=folder/(f'iter_{args.smoke_steps}.pth' if args.smoke_steps else f'epoch_{args.epochs}.pth')
        if not selected.is_file():
            raise FileNotFoundError(selected)
        check_resume(selected,record,study)
        record.update(train='complete',selected_checkpoint=str(selected),
                      selected_checkpoint_sha256=digest(selected))
        write_json(record_path,record)
    if record['evaluation'] is None:
        evaluate_run(args,root,record,record_path,env)
    return read_json(record_path)


def evaluate_run(args,root,record,record_path,env):
    gpu_check(args.gpus)
    folder=record_path.parent
    output=folder/'evaluations'/stamp()
    argv=[PYTHON,'tools/eval_recallocc_modules.py',record['config'],record['selected_checkpoint'],
          '--manifest',root/'eval_manifest.json','--output-dir',output]
    log=folder/f'eval_{stamp()}.log'
    record['launch_history'].append(dict(command=shlex.join(map(str,argv)),log=str(log),time=stamp()))
    write_json(record_path,record)
    run_logged(argv,log,dict(env,CUDA_VISIBLE_DEVICES=args.gpus.split(',')[0]))
    record['evaluation']=str(output/'metrics.json')
    write_json(record_path,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','batch','initialize','evaluate'])
    parser.add_argument('--work-root',required=True)
    parser.add_argument('--experiment',choices=list(MODES),default='E2_full_retrieval_concat')
    parser.add_argument('--experiments',nargs='+',choices=list(MODES),default=list(MODES))
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0])
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--eval-samples',type=int,default=2)
    parser.add_argument('--resume',help='Exact checkpoint or auto to inspect the latest verified checkpoint')
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.action=='initialize':
        create_initialization(args.work_root,args.seed)
        return
    if len(args.gpus.split(',')) not in (1,2):
        parser.error('Effective batch 2 supports one or two GPUs')
    if args.epochs<1 or args.smoke_steps<0 or args.eval_samples<1:
        parser.error('epochs/eval-samples must be positive; smoke-steps nonnegative')
    if args.resume=='auto' and args.action not in ('run','batch'):
        parser.error('--resume auto is supported for run and batch continuation')
    if args.action=='batch' and args.resume not in (None,'auto'):
        parser.error('batch accepts --resume auto; run accepts an exact checkpoint')
    # Process-wide lock; protects against overlapping our own launchers. Also
    # inspect actual selected GPUs before every new training/evaluation stage.
    lock=(ROOT/'work_dirs/.recallocc.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=runtime_env(args.gpus)
    gpu_check(args.gpus)
    root,study=ensure_study(args)
    if args.action in ('run','batch'):
        experiments=args.experiments if args.action=='batch' else [args.experiment]
        seeds=args.seeds if args.action=='batch' else [args.seed]
        for seed in seeds:
            for experiment in experiments:
                run_arm(args,root,study,experiment,seed,env)
                run_logged([PYTHON,'tools/summarize_recallocc_modules.py','--work-root',root],root/'summary.log',env)
    elif args.action=='evaluate':
        path=root/args.experiment/f'seed_{args.seed}'/'run.json'
        record=read_json(path)
        if record['train']!='complete':
            raise RuntimeError('Training has not completed the fixed selection budget')
        evaluate_run(args,root,record,path,env)
    run_logged([PYTHON,'tools/summarize_recallocc_modules.py','--work-root',root],root/'summary.log',env)

if __name__=='__main__':
    main()
