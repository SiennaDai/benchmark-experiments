import torch
from config.recipe import load_recipe
from train_platform import build_model
def test_local_initialization_and_fixed_state_forward_repeatable():
    cfg=load_recipe("recipes/diagnostic_cpu.json");a=build_model(cfg,torch.device("cpu"));b=build_model(cfg,torch.device("cpu"));x=torch.arange(16).view(1,-1)%256;y=(x+1)%256;oa=a(x,y,get_logits=True);ob=b(x,y,get_logits=True);assert torch.equal(oa["logits"],ob["logits"]) and torch.equal(oa["loss"],ob["loss"])
