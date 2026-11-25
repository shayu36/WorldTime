"""Real TRAIN sample; isolated gradient routes, two volatile optimizer updates."""
import argparse,hashlib,json,time,types
from pathlib import Path
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel,collate
from mmcv.runner import build_optimizer,wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.utils import compat_cfg,setup_multi_processes
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import patch_config

PREFIXES=('future_memory_adapter_v2.','memory_planning_fusion.')
def fingerprint(state):
    h=hashlib.sha256()
    for k,v in state:
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def grads(model,scale=1.):
    groups={p:0. for p in PREFIXES}; bad=[];missing=[]
    for name,p in model.named_parameters():
        prefix=next((q for q in PREFIXES if name.startswith(q)),None)
        if prefix is None:
            if p.requires_grad or p.grad is not None:bad.append(name)
        elif p.grad is None:missing.append(name)
        else:
            if not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite gradient '+name)
            groups[prefix]+=float(p.grad.float().norm())/scale
    assert not bad,bad
    return dict(groups=groups,missing=missing,frozen_grad_none=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--index',type=int,default=64)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(1);torch.cuda.set_device(0)
    set_random_seed(17,deterministic=True)
    cfg=patch_config(compat_cfg(Config.fromfile('configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-planning-joint.py')))
    setup_multi_processes(cfg);cfg.dump(str(out/'config.py'))
    model=build_model(cfg.model);model.init_weights();wrap_fp16_model(model);model.cuda().train()
    optimizer=build_optimizer(model,cfg.optimizer)
    setup=model.validate_query_memory_training_setup(optimizer,cfg.optimizer)
    print('Strict source load / two trainable prefixes:',setup['trainable_count'],flush=True)
    before={p:fingerprint((n,v) for n,v in model.state_dict().items() if n.startswith(p)) for p in PREFIXES}
    frozen=fingerprint((n,v) for n,v in model.state_dict().items() if not n.startswith(PREFIXES))
    assert all(not child.training for n,child in model.named_children() if n not in ('future_memory_adapter_v2','memory_planning_fusion'))
    ds=build_dataset(cfg.data.train);sample=ds[args.index];batch=collate([sample],samples_per_gpu=1)
    snapshots=[];native=[];targets={};identity=[]
    hook=model.memory_planning_fusion.register_forward_pre_hook(lambda module,values:snapshots.append(tuple(v.detach().clone() if torch.is_tensor(v) else v for v in values)))
    original_backbone=model.forward_backbone
    def capture(self,*a,**k):
        output=original_backbone(*a,**k)
        native[:]=[x.detach().clone() for x in output['baseline_pred_trajs_list']]
        identity.append(all(torch.equal(a,b) for a,b in zip(output['pred_trajs_list'],output['baseline_pred_trajs_list'])))
        return output
    model.forward_backbone=types.MethodType(capture,model)
    original_loss=model.loss_traj
    def capture_loss(self,pred,gt,index):
        targets[index]=gt.detach().clone();return original_loss(pred,gt,index)
    model.loss_traj=types.MethodType(capture_loss,model)
    parallel=MMDataParallel(model,device_ids=[0]);scaler=torch.cuda.amp.GradScaler(init_scale=128.)
    torch.cuda.reset_peak_memory_stats()
    losses=parallel(return_loss=True,**batch)
    hook.remove();model.forward_backbone=original_backbone;model.loss_traj=original_loss
    assert identity==[True] and len(snapshots)==3 and all(bool(s[2].any()) for s in snapshots)
    expected={f'mem_{h}.loss_{k}' for h in ('1s','2s','3s') for k in ('base_cls','sem','occ','occ_anchor','threshold','soft_voxel','pts')}
    assert expected.issubset(losses),sorted(set(losses))
    occ=sum(v for n,v in losses.items() if n.startswith('mem_') and 'loss' in n)+losses['loss_future_memory_v2_graph_anchor']
    scaler.scale(occ).backward();occ_grad=grads(model,scaler.get_scale())
    assert occ_grad['groups'][PREFIXES[0]]>0 and occ_grad['groups'][PREFIXES[1]]==0
    print('Occupancy-only gradient reaches V2 and not fusion: PASS',flush=True)
    del losses,occ;optimizer.zero_grad(set_to_none=True)
    planned=list(native)
    for step,snapshot in zip((1,3,5),snapshots):planned[step]=planned[step]+model.memory_planning_fusion(*snapshot)
    plan=sum(model.loss_traj(p.squeeze(1),targets[i+1],i+1)[f'loss_traj_{i+1}s'] for i,p in enumerate(planned))
    scaler.scale(plan).backward();plan_grad=grads(model,scaler.get_scale())
    assert plan_grad['groups'][PREFIXES[1]]>0 and plan_grad['groups'][PREFIXES[0]]==0
    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    del plan,planned,snapshots
    print('Planning-only gradient reaches fusion; first volatile update: PASS',flush=True)
    losses=parallel(return_loss=True,**batch)
    loss=sum(v for n,v in losses.items() if 'loss' in n)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.unscale_(optimizer);joint=grads(model)
    assert not joint['missing'],joint['missing']
    assert all(v>0 for v in joint['groups'].values())
    parts={}
    for prefix in PREFIXES:
        for name,p in model.named_parameters():
            if name.startswith(prefix):
                group=name[len(prefix):].split('.')[0]
                parts[prefix+group]=parts.get(prefix+group,0.)+float(p.grad.float().norm())
    assert all(v>0 for k,v in parts.items() if k.startswith(PREFIXES[1])),parts
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
    scaler.step(optimizer);scaler.update()
    after={p:fingerprint((n,v) for n,v in model.state_dict().items() if n.startswith(p)) for p in PREFIXES}
    assert all(before[p]!=after[p] for p in PREFIXES)
    assert frozen==fingerprint((n,v) for n,v in model.state_dict().items() if not n.startswith(PREFIXES))
    result=dict(status='PASS',train_index=args.index,train_samples=len(ds),initial_T1_identity=True,
        trainable_prefixes=PREFIXES,trainable_count=setup['trainable_count'],loss_keys=sorted(losses),
        occupancy_only=occ_grad,planning_only=plan_grad,joint=joint,joint_groups=parts,
        frozen_state_unchanged=True,both_trainable_modules_changed=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.monotonic()-start,
        volatile_optimizer_steps=2,backwards=3,checkpoint_saved=False)
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('joint_groups','loss_keys')},indent=2),flush=True)
if __name__=='__main__':main()
