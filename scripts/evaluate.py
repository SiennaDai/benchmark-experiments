#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import torch
from data.frozen_tokens import FrozenWindows,load_manifest
from train_platform import build_model,evaluate,resolve_device
p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--split",choices=["validation","test"],default="validation");p.add_argument("--to-device","--to_device",default="auto",dest="to_device");a=p.parse_args();device=resolve_device(a.to_device);ck=torch.load(a.checkpoint,map_location=device,weights_only=False);run=Path(a.checkpoint).resolve().parents[1];cfg=json.loads((run/"resolved_config.json").read_text());manifest=load_manifest(cfg["data"]["manifest"]);model=build_model(cfg,device);model.load_state_dict(ck["model"]);windows=FrozenWindows(manifest,a.split,cfg["model"]["sequence_length"]);result=evaluate(model,windows,cfg["eval"]["max_target_tokens"],cfg["eval"]["batch_size"],cfg,device);result.update({"to_device":a.to_device,"device":str(device)});print(json.dumps(result,indent=2))
