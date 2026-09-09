import torch
from optim.adamw_reference import ReferenceAdamW

def test_formula_grad_none_zero_and_steps():
    p=torch.nn.Parameter(torch.tensor([1.0],dtype=torch.float64)); o=ReferenceAdamW([p],lr=.1,betas=(.5,.5),eps=.01,weight_decay=.2); o.step(); assert p.item()==1 and not o.state[p]; p.grad=torch.zeros_like(p); o.step(); assert o.state[p]["step"]==1 and torch.allclose(p,torch.tensor([.98],dtype=torch.float64)); p.grad=None; o.step(); assert o.state[p]["step"]==1

def test_reference_tracks_torch_multistep():
    a=torch.nn.Parameter(torch.tensor([1.,-2.])); b=torch.nn.Parameter(a.detach().clone()); oa=ReferenceAdamW([a],lr=1e-3,betas=(.9,.95),eps=1e-8,weight_decay=.1); ob=torch.optim.AdamW([b],lr=1e-3,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False,fused=False)
    for g in ([.1,-.2],[0.,3.],[-1.,1e-7]): a.grad=torch.tensor(g); b.grad=torch.tensor(g); oa.step(); ob.step()
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-4) and torch.allclose(oa.state[a]["exp_avg"],ob.state[b]["exp_avg"],atol=1e-7)
