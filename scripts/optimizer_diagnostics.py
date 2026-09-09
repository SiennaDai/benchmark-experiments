#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import torch
from optim.adamw_reference import ReferenceAdamW
def main():
 p=argparse.ArgumentParser();p.add_argument("--recipe",required=True);p.add_argument("--output",required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 gradients=[0.0,1e-12,0.25,-0.5,10.0,-10.0,0.0,0.125];theta=torch.tensor([1.0,-2.0,0.0,3.0],dtype=torch.float32);p0=torch.nn.Parameter(theta.clone());ps=torch.nn.Parameter(theta.clone());o0=ReferenceAdamW([p0],lr=1e-3,betas=(.9,.95),eps=1e-8,weight_decay=.1);os=ReferenceAdamW([ps],lr=1e-3,betas=(.9,.95),eps=1e-8,weight_decay=.1,state_simulation="bf16_roundtrip");trace=[]
 for i,g in enumerate(gradients,1):
  grad=torch.tensor([g,-g,g*1e-3,0.0]);p0.grad=grad.clone();ps.grad=grad.clone();o0.step();os.step();dm=(os.state[ps]["exp_avg"]-o0.state[p0]["exp_avg"]).abs();dv=(os.state[ps]["exp_avg_sq"]-o0.state[p0]["exp_avg_sq"]).abs();direction=ps.detach()-theta;ref=p0.detach()-theta;cos=None if direction.norm()==0 or ref.norm()==0 else float(torch.nn.functional.cosine_similarity(direction,ref,dim=0));trace.append({"step":i,"max_parameter_abs_error":float((ps-p0).abs().max()),"max_m_abs_error":float(dm.max()),"max_v_abs_error":float(dv.max()),"update_direction_cosine":cos,"v_nonnegative":bool((os.state[ps]["exp_avg_sq"]>=0).all())})
 result={"schema_version":1,"simulation":"bf16_roundtrip","semantics":"update uses unrounded m/v; BF16 roundtrip is persisted for the next step","trace":trace};(out/"diagnostics.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
