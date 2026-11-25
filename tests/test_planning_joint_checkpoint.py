import importlib.util
from pathlib import Path
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('helper',Path(__file__).parents[1]/'mmdet3d/models/sparsedetectors/planning_checkpoint.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Adapter(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(4,4);self.drop=nn.Dropout(.1)
    def forward(self,q,p,s,memory,horizon,**kwargs):
        x=self.drop(self.proj(q))[:,:,None].expand(-1,-1,3,-1)
        valid=torch.ones(x.shape[:3],device=x.device,dtype=torch.bool)
        if memory is None:valid.zero_()
        x=x*valid[...,None];g=torch.sigmoid(x[...,:1])*valid[...,None]
        return dict(delta_s=x,delta_o=x[...,:1],delta_p=x[...,:3],gate=g,context=x,diagnostics=dict(has_candidate=valid))

def test_checkpoint_retains_same_values_and_gradients_with_frozen_inputs():
    torch.manual_seed(4);a=Adapter().train();args=(torch.randn(2,5,4),torch.zeros(2,5,3,3),torch.zeros(2,5,3,4),{},1)
    torch.manual_seed(9);x=a(*args,return_diagnostics=True)
    sum(x[k].sum() for k in ('delta_s','delta_o','delta_p','gate')).backward()
    grads=[p.grad.clone() for p in a.parameters()];a.zero_grad(set_to_none=True)
    torch.manual_seed(9);y=m.checkpoint_adapter_with_context(a,args)
    for k in ('delta_s','delta_o','delta_p','gate','context'):assert torch.equal(x[k],y[k])
    assert not y['context'].requires_grad
    sum(y[k].sum() for k in ('delta_s','delta_o','delta_p','gate')).backward()
    assert all(torch.equal(g,p.grad) for g,p in zip(grads,a.parameters()))

def test_checkpoint_empty_memory_keeps_zero_context_and_live_residual_graph():
    a=Adapter();args=(torch.randn(1,2,4),None,None,None,0)
    x=m.checkpoint_adapter_with_context(a,args)
    assert not x['diagnostics']['has_candidate'].any()
    assert torch.count_nonzero(x['context'])==0 and torch.count_nonzero(x['gate'])==0
    sum(x[k].sum() for k in ('delta_s','delta_o','delta_p','gate')).backward()
    assert all(p.grad is not None and not p.grad.any() for p in a.parameters())
