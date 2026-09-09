import torch
from optim.adamw_reference import ReferenceAdamW

def test_rounding_after_update_and_fp32_storage():
    a=torch.nn.Parameter(torch.tensor([1.])); b=torch.nn.Parameter(torch.tensor([1.])); oa=ReferenceAdamW([a],lr=.1,betas=(0.,0.),eps=1e-8); ob=ReferenceAdamW([b],lr=.1,betas=(0.,0.),eps=1e-8,state_simulation="bf16_roundtrip"); g=torch.tensor([.1234567]); a.grad=g; b.grad=g; oa.step(); ob.step()
    assert torch.equal(a,b) and ob.state[b]["exp_avg"].dtype==torch.float32 and ob.state[b]["exp_avg"].item()==float(g.to(torch.bfloat16))

def test_disabled_simulation_exact():
    a=torch.nn.Parameter(torch.tensor([1.])); b=torch.nn.Parameter(torch.tensor([1.])); oa=ReferenceAdamW([a]); ob=ReferenceAdamW([b],state_simulation="none")
    for g in [0.,1e-20,1.,-10.]: a.grad=torch.tensor([g]); b.grad=torch.tensor([g]); oa.step(); ob.step()
    assert torch.equal(a,b) and torch.equal(oa.state[a]["exp_avg_sq"],ob.state[b]["exp_avg_sq"])
