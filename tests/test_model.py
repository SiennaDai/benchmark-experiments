import torch
from config.recipe import load_recipe
from train_platform import build_model, parameter_groups

def test_causality_ce_and_tied_group_dedup():
    cfg=load_recipe("recipes/diagnostic_cpu.json"); model=build_model(cfg,torch.device("cpu")); model.eval(); x=torch.randint(0,256,(1,8)); changed=x.clone(); changed[0,6:]=torch.randint(0,256,(2,)); a=model(x,get_logits=True)["logits"]; b=model(changed,get_logits=True)["logits"]
    assert torch.equal(a[:,:6],b[:,:6])
    y=torch.randint(0,256,(1,8)); out=model(x,y,get_logits=True); expected=torch.nn.functional.cross_entropy(out["logits"].reshape(-1,256),y.reshape(-1)); assert torch.allclose(out["loss"],expected)
    groups,records=parameter_groups(model,.1); params=[p for g in groups for p in g["params"]]; assert len(params)==len({id(p) for p in params})==len(records)
