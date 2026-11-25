"""Actual module inputs, retrieval parity, history independence and gradients."""
from unittest.mock import patch
import copy
import pytest
import torch
from test_future_memory_adapter_v2 import QM, ROOT, _adapter, _memory


def inputs():
    torch.manual_seed(91)
    return (torch.randn(2,3,8), torch.randn(2,3,2,3)*.1,
            torch.randn(2,3,2,3), _memory(batch=2,count=5))


def nonzero(model):
    with torch.no_grad():
        for head in (model.delta_s_head, model.delta_o_head, model.delta_p_head):
            head[-1].weight.normal_(0,.1)


def test_concat_exact_input_retrieval_masks_no_gate():
    full = _adapter(memory_gate_mode='task_horizon').eval()
    model = _adapter(ablation_mode='full_retrieval_concat').eval()
    state=model.state_dict()
    for k,v in full.state_dict().items():
        if k in state and state[k].shape==v.shape: state[k]=v
    model.load_state_dict(state,strict=True)
    nonzero(model)
    assert not any('gate' in k for k,_ in model.named_parameters())
    q,p,l,m=inputs(); m['memory_valid'][1]=False
    seen=[]
    hook=model.adapter.register_forward_pre_hook(lambda _,a:seen.append(a[0]))
    with patch.object(model,'task_gates',side_effect=AssertionError('no gate calls')):
        out=model(q,p,l,m,1)
    hook.remove()
    ref=full(q,p,l,m,1)
    for k in ('context','attention_query'):
        assert torch.equal(out[k],ref[k])
    for k in ('coarse_indices','coarse_valid','point_semantic_memory'):
        assert torch.equal(out['diagnostics'][k],ref['diagnostics'][k])
    expected=torch.cat([q[:,:,None].expand(-1,-1,2,-1),model.current_point_pos(p),
        out['context'],model.horizon_embedding.weight[1].view(1,1,1,8).expand_as(out['context']),
        l.softmax(-1),out['diagnostics']['point_semantic_memory']],-1)
    torch.testing.assert_close(torch.cat(seen,1),expected,rtol=0,atol=0)
    has=out['diagnostics']['has_candidate'][...,None]
    for k in ('gate_sem','gate_occ','gate_pos'): assert torch.equal(out[k],has.float())
    sem,occ,combined=QM.decode_semantic_occupancy_logits(l,out['delta_s'],out['delta_o'],out['gate_sem'],out['gate_occ'])
    assert torch.equal(sem,l+has*out['delta_s'])
    assert torch.equal(occ,l.max(-1,keepdim=True).values+has*out['delta_o'])
    assert torch.equal(combined[1],l[1])
    sum(out[k].square().mean() for k in ('delta_s','delta_o','delta_p')).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    assert all(head[-1].weight.grad.abs().sum()>0 for head in (model.delta_s_head,model.delta_o_head,model.delta_p_head))
    # Stats remain retrieval diagnostics, and cannot influence fusion via a side input.
    with patch.object(model,'aggregate_support_statistics',side_effect=lambda w,*a: tuple(torch.ones_like(w[...,0,0])*123 for _ in range(3))):
        altered=model(q,p,l,m,1)
    for k in ('delta_s','delta_o','delta_p'): assert torch.equal(out[k],altered[k])


def test_current_only_nonzero_and_independent_history_horizon_gates():
    model=_adapter(ablation_mode='current_only_task_time').eval(); nonzero(model)
    q,p,l,m=inputs()
    assert not hasattr(model,'aligner') and not hasattr(model,'query_proj')
    with patch.object(model,'_reshape_memory',side_effect=AssertionError('history')), \
         patch.object(model,'_coarse',side_effect=AssertionError('retrieval')):
        for h in range(3):
            seen=[]
            hook=model.adapter.register_forward_pre_hook(lambda _,a:seen.append(a[0]))
            out=model(q+h,p+h,l+h,None,h); hook.remove()
            other=model(q+h,p+h,l+h,{'memory_query_feat':object()},h)
            for k in out:
                if torch.is_tensor(out[k]): assert torch.equal(out[k],other[k])
            expected=torch.cat([(q+h)[:,:,None].expand(-1,-1,2,-1),model.current_point_pos(p+h),
                (l+h).softmax(-1),model.horizon_embedding.weight[h].view(1,1,1,8).expand(2,3,2,8)],-1)
            torch.testing.assert_close(torch.cat(seen,1),expected,rtol=0,atol=0)
            assert out['gate'].min()>0 and out['delta_s'].abs().sum()>0
            sem,_,_=QM.decode_semantic_occupancy_logits(l,out['delta_s'],out['delta_o'],out['gate_sem'],out['gate_occ'])
            assert not torch.equal(sem,l)
            sum((out[g]*out[d]).square().mean() for g,d in [('gate_sem','delta_s'),('gate_occ','delta_o'),('gate_pos','delta_p')]).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    assert (model.task_horizon_bias.grad.abs().sum(-1)>0).all()
    assert model.gate_head.weight.grad.abs().sum()>0


def test_partial_current_support_is_fixed_memory_free_and_reverts_masked_points():
    model = _adapter(ablation_mode='current_only_task_time',
                     current_point_support_fraction=.70).eval()
    nonzero(model)
    q,p,l,m = inputs()
    for horizon in range(3):
        with patch.object(model, '_reshape_memory', side_effect=AssertionError('history')), \
             patch.object(model, '_coarse', side_effect=AssertionError('retrieval')):
            out = model(q+horizon,p+horizon,l+horizon,None,horizon)
            fake = model(q+horizon,p+horizon,l+horizon,
                         {'memory_query_feat':object()},horizon)
        support = out['diagnostics']['current_point_support']
        assert support.shape == (2,3,2)
        assert support.sum(-1).sum(-1).tolist() == [4,4]
        assert torch.equal(out['gate_sem'],fake['gate_sem'])
        for name in ('gate_sem','gate_occ','gate_pos'):
            assert torch.all(out[name][~support] == 0)
            assert torch.all(out[name][support] > 0)
        sem,occ,_ = QM.decode_semantic_occupancy_logits(
            l+horizon,out['delta_s'],out['delta_o'],out['gate_sem'],out['gate_occ'])
        assert torch.equal(sem[~support],(l+horizon)[~support])
        assert torch.equal(occ[~support],(l+horizon).max(-1,keepdim=True).values[~support])
        assert not torch.equal(sem[support],(l+horizon)[support])
    sum((out[g]*out[d]).square().mean() for g,d in
        [('gate_sem','delta_s'),('gate_occ','delta_o'),('gate_pos','delta_p')]).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all()
               for v in model.parameters())


def test_partial_current_config_keeps_original_e3_data_and_training():
    from mmcv import Config
    original=Config.fromfile(str(ROOT/'configs/sparseworld/recallocc/E3_current_only_task_time.py'))
    partial=Config.fromfile(str(ROOT/'configs/sparseworld/recallocc/E3_current_only_70pct.py'))
    assert partial.model.query_memory_cfg.future_memory_adapter_v2.current_point_support_fraction == .70
    for split in ('train','val','test'):
        assert original.data[split] == partial.data[split]
    for key in ('optimizer','lr_config','fp16','runner','optimizer_config'):
        assert original[key] == partial[key]


@pytest.mark.parametrize('fraction,expected',[(.30,3),(.20,2)])
def test_lower_coverage_current_only_forward(fraction,expected):
    model=_adapter(ablation_mode='current_only_task_time',
                   current_point_support_fraction=fraction).eval()
    nonzero(model)
    torch.manual_seed(92)
    q,p,l=torch.randn(2,5,8),torch.randn(2,5,2,3),torch.randn(2,5,2,3)
    with patch.object(model,'_reshape_memory',side_effect=AssertionError('history')), \
         patch.object(model,'_coarse',side_effect=AssertionError('retrieval')):
        out=model(q,p,l,None,0)
        fake=model(q,p,l,{'memory_query_feat':object()},0)
    support=out['diagnostics']['current_point_support']
    assert support.sum((1,2)).tolist()==[expected,expected]
    assert torch.equal(out['gate_sem'],fake['gate_sem'])
    for key in ('gate_sem','gate_occ','gate_pos'):
        assert torch.all(out[key][~support]==0)
        assert torch.all(out[key][support]>0)
    sem,occ,_=QM.decode_semantic_occupancy_logits(
        l,out['delta_s'],out['delta_o'],out['gate_sem'],out['gate_occ'])
    assert torch.equal(sem[~support],l[~support])
    assert torch.equal(occ[~support],l.max(-1,keepdim=True).values[~support])
    assert not torch.equal(sem[support],l[support])


@pytest.mark.parametrize('percent', [30,20])
def test_lower_coverage_configs_match_original_e3(percent):
    from mmcv import Config
    original=Config.fromfile(str(ROOT/'configs/sparseworld/recallocc/E3_current_only_task_time.py'))
    cfg=Config.fromfile(str(ROOT/f'configs/sparseworld/recallocc/E3_current_only_{percent}pct.py'))
    assert cfg.model.query_memory_cfg.future_memory_adapter_v2.current_point_support_fraction==percent/100
    for split in ('train','val','test'):
        assert cfg.data[split]==original.data[split]
    for key in ('optimizer','lr_config','fp16','runner','optimizer_config'):
        assert cfg[key]==original[key]


def test_current_config_all_splits():
    from mmcv import Config
    cfg=Config.fromfile(str(ROOT/'configs/sparseworld/recallocc/E3_current_only_task_time.py'))
    def inspect(x):
        if isinstance(x,dict):
            assert x.get('type')!='LoadQueryMemoryFromFiles'
            assert not any(k.startswith('memory_') for k in x.get('keys',[]))
            for v in x.values(): inspect(v)
        elif isinstance(x,list):
            for v in x: inspect(v)
    for split in ('train','val','test'):
        inspect(cfg.data[split].pipeline)
        assert cfg.data[split].query_memory_cache_root is None
        assert cfg.data[split].query_memory_history_frames==0


def test_current_visual_pipeline_and_public_supervision_match_full():
    from mmcv import Config
    full=Config.fromfile(str(ROOT/'configs/sparseworld/recallocc/E4.py'))
    current=Config.fromfile(str(ROOT/'configs/sparseworld/recallocc/E3_current_only_task_time.py'))
    def strip(x):
        if isinstance(x,list): return [strip(v) for v in x if not isinstance(v,dict) or v.get('type')!='LoadQueryMemoryFromFiles']
        if isinstance(x,dict): return {k:([v for v in value if not v.startswith('memory_')] if k=='keys' else strip(value)) for k,value in x.items()}
        return x
    for split in ('train','val','test'):
        assert strip(full.data[split].pipeline)==current.data[split].pipeline
    for key in ('optimizer','lr_config','fp16','runner','optimizer_config'):
        assert current[key]==full[key]
    for key in ('loss_base_cls_weight','loss_pts_weight','loss_sem_weight','loss_occ_weight',
                'loss_occ_anchor_weight','loss_threshold_weight','loss_soft_voxel_weight'):
        assert current.model.query_memory_cfg.future_memory_adapter_v2[key]==full.model.query_memory_cfg.future_memory_adapter_v2[key]


def test_new_summary_seed_statistics_and_reject_legacy(tmp_path):
    import sys,json
    sys.path.insert(0,str(ROOT/'tools'))
    from recallocc_modules_common import MODES,digest,write_json
    from summarize_recallocc_modules import summarize
    write_json(tmp_path/'study.json',dict(spec=dict(seeds=[0,1],smoke_steps=0),eval_manifest_sha256='manifest'))
    e=next(iter(MODES))
    for seed in (0,1):
        run=tmp_path/e/f'seed_{seed}';run.mkdir(parents=True)
        ckpt=run/'epoch_3.pth';ckpt.write_bytes(b'test checkpoint')
        ev=run/'metrics.json'
        write_json(ev,dict(model=e,seed=seed,manifest_sha256='manifest',checkpoint_sha256=digest(ckpt),
            full_validation=True,units='percent',horizons=[0,1,2,3],
            metrics=[dict(miou=x+seed*2,iou=x*2+seed*2) for x in (0,1,2,6)]))
        write_json(run/'run.json',dict(design='modules_v2',experiment=e,seed=seed,
            evaluation=str(ev),selected_checkpoint=str(ckpt),selected_checkpoint_sha256=digest(ckpt)))
    out=summarize(tmp_path)
    assert out['aggregates'][e]['Mean mIoU']['mean']==4
    assert out['aggregates'][e]['Mean mIoU']['sample_std']==pytest.approx(2**.5)
    assert all(r['Mean mIoU'] is None for r in out['rows'] if r['Experiment'] in ('E1','E4'))
    assert out['comparisons'] is None
    path=tmp_path/e/'seed_0/run.json';record=json.loads(path.read_text());record['design']='legacy';write_json(path,record)
    with pytest.raises(ValueError,match='identity'): summarize(tmp_path)
