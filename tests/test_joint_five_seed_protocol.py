"""Independent starts, actual training scope, and checkpoint audit regression."""
import copy
import importlib.util
from pathlib import Path
import torch
import pytest
from mmcv import Config
ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('joint_protocol', ROOT/'tools/run_memory_planning_joint_five_seeds.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_same_source_fresh_optimizer_full_epoch_every_seed(tmp_path):
    normalized=[]
    for seed in m.SEEDS:
        cfg=m.configuration(tmp_path, seed)
        m.assert_configuration(cfg, seed)
        d=copy.deepcopy(cfg._cfg_dict)
        d.pop('seed');d.pop('work_dir');normalized.append(d)
    assert all(d==normalized[0] for d in normalized)
    cfg.model.type='SparseWorld4DTrajMemoryPlanning'
    with pytest.raises(AssertionError):m.assert_configuration(cfg,seed)


def test_joint_build_optimizer_scope_and_rates():
    from mmdet3d.models import build_model
    from mmcv.runner import build_optimizer
    cfg=Config.fromfile(str(m.CONFIG))
    model=build_model(cfg.model).train()
    opt=build_optimizer(model,cfg.optimizer)
    names={id(p):n for n,p in model.named_parameters()}
    selected=[names[id(p)] for g in opt.param_groups for p in g['params']]
    assert len(selected)==77 and all(n.startswith(m.PREFIXES) for n in selected)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1722147
    for g in opt.param_groups:
        n=names[id(g['params'][0])]
        assert g['lr']==pytest.approx(2e-5 if n.startswith(m.PREFIXES[0]) else 1e-4)
    assert not opt.state
    assert all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith(m.PREFIXES))
    assert all(not child.training for n,child in model.named_children() if n not in ('future_memory_adapter_v2','memory_planning_fusion'))
    assert model.future_memory_adapter_v2_activation_checkpoint


def test_checkpoint_audit_requires_v2_update_and_keeps_baseline(tmp_path,monkeypatch):
    original={'backbone.weight':torch.tensor([1.]), 'future_memory_adapter_v2.weight':torch.tensor([2.])}
    src=tmp_path/'source.pth';torch.save(dict(state_dict=original),src);monkeypatch.setattr(m,'SOURCE',src)
    fusion={m.PREFIX+'residual_head.2.weight':torch.ones(223984)}
    fusion.update({m.PREFIX+f'p{i}':torch.ones(1) for i in range(18)})
    state={**original,**fusion};state['future_memory_adapter_v2.weight']=torch.tensor([3.])
    cp=dict(meta=dict(epoch=1,iter=9865,seed=17),state_dict=state,optimizer=dict(state={i:{} for i in range(77)}))
    path=tmp_path/'epoch_1.pth';torch.save(cp,path)
    audit=m.audit_checkpoint(path,17)
    assert audit['frozen_unchanged'] and audit['V2_changed_tensors']==['future_memory_adapter_v2.weight']
    cp['state_dict']['backbone.weight']=torch.tensor([2.]);torch.save(cp,path)
    with pytest.raises(AssertionError):m.audit_checkpoint(path,17)
    cp['state_dict']['backbone.weight']=original['backbone.weight']
    cp['state_dict']['future_memory_adapter_v2.weight']=original['future_memory_adapter_v2.weight'];torch.save(cp,path)
    with pytest.raises(AssertionError,match='V2 did not update'):m.audit_checkpoint(path,17)
