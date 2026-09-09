import torch
from config.recipe import load_recipe
from train_platform import build_model, parameter_groups

def execute(model,batches):
    groups,_=parameter_groups(model,0.0); opt=torch.optim.SGD(groups,lr=.01); opt.zero_grad(set_to_none=True)
    for x,y in batches: (model(x,y)["loss"]/len(batches)).backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); return norm

def test_batch_and_micro_accumulation_match():
    cfg=load_recipe("recipes/diagnostic_cpu.json"); a=build_model(cfg,torch.device("cpu")); b=build_model(cfg,torch.device("cpu")); b.load_state_dict(a.state_dict()); x=torch.randint(0,256,(4,16)); y=torch.randint(0,256,(4,16)); na=execute(a,[(x,y)]); nb=execute(b,[(x[i:i+1],y[i:i+1]) for i in range(4)])
    assert torch.allclose(na,nb,atol=1e-6,rtol=1e-4)
    for pa,pb in zip(a.parameters(),b.parameters()): assert torch.allclose(pa,pb,atol=1e-6,rtol=1e-4)
