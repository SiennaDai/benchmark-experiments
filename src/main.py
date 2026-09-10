#!/usr/bin/env python3
"""Controlled recipe entry point for the low-precision optimizer platform."""
import argparse,importlib.util,json,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from config.recipe import load_recipe
def main():
 p=argparse.ArgumentParser();p.add_argument("--recipe",required=True);p.add_argument("--dry-run",action="store_true");p.add_argument("--run-dir");p.add_argument("--resume");p.add_argument("--max-wall-seconds",type=float);p.add_argument("--data-root",type=Path,default=ROOT);p.add_argument("--to-device","--to_device",default="auto",dest="to_device");a=p.parse_args();cfg=load_recipe(a.recipe)
 data_root=a.data_root.expanduser().resolve();manifest=Path(cfg["data"]["manifest"]);manifest=manifest.expanduser().resolve() if manifest.is_absolute() else (data_root/manifest).resolve();cfg["data"]["manifest"]=str(manifest)
 import torch
 from data.frozen_tokens import load_manifest, validate_data_capacity
 from train_platform import resolve_device
 resource_reason=None;device=None
 try: device=resolve_device(a.to_device)
 except (ValueError,RuntimeError) as exc: resource_reason=str(exc)
 if resource_reason is None and cfg["precision"]["compute"] == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
  resource_reason="BF16 compute requires a CUDA device with BF16 support"
 if resource_reason is None and cfg["optimizer"]["name"].startswith("bnb_") and device.type != "cuda":
  resource_reason=f"{cfg['optimizer']['name']} requires a CUDA device"
 if resource_reason is None and cfg["optimizer"]["name"].startswith("bnb_") and importlib.util.find_spec("bitsandbytes") is None:
  resource_reason=f"{cfg['optimizer']['name']} requires the bitsandbytes package"
 capacity_error = None
 if manifest.exists():
  try:
   data_manifest = load_manifest(manifest)
   validate_data_capacity(data_manifest, sequence_length=cfg["model"]["sequence_length"], train_split=cfg["data"]["train_split"],
    validation_split=cfg["data"]["validation_split"], eval_target_tokens=cfg["eval"]["max_target_tokens"],
    train_target_tokens=cfg["train"]["target_tokens"], micro_batch_size=cfg["train"]["micro_batch_size"],
    accumulation_steps=cfg["train"]["accumulation_steps"], total_updates=cfg["derived"]["total_updates"],
    allow_repeated_epochs=cfg["data"]["allow_repeated_epochs"])
  except (OSError, KeyError, ValueError) as exc:
   capacity_error = str(exc)
 plan={"recipe":str(Path(a.recipe).resolve()),"fingerprint":cfg["fingerprint"],"total_updates":cfg["derived"]["total_updates"],"schedule_total_updates":cfg["derived"]["schedule_total_updates"],"tokens_per_update":cfg["derived"]["tokens_per_update"],"data_root":str(data_root),"manifest":str(manifest),"manifest_exists":manifest.exists(),"optimizer":cfg["optimizer"]["name"],"compute":cfg["precision"]["compute"],"to_device":a.to_device,"resolved_device":str(device) if device is not None else None}
 if resource_reason: plan["unsupported_reason"]=resource_reason
 if capacity_error: plan["capacity_error"] = capacity_error
 if a.dry_run: print(json.dumps(plan,indent=2));return 0 if manifest.exists() and resource_reason is None and capacity_error is None else 2
 if resource_reason:p.error(resource_reason)
 if not manifest.exists():p.error(f"data manifest not found: {manifest}")
 if capacity_error:p.error(capacity_error)
 from train_platform import run
 if a.resume: resume=Path(a.resume).resolve();run_dir=Path(a.run_dir).resolve() if a.run_dir else resume.parents[1]
 else:
  resume=None;run_dir=Path(a.run_dir).resolve() if a.run_dir else ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+cfg["fingerprint"][:8]+"-"+uuid.uuid4().hex[:6])
  if run_dir.exists() and any(run_dir.iterdir()):p.error(f"new run directory is not empty: {run_dir}")
 summary=run(cfg,run_dir,resume,a.max_wall_seconds,a.to_device);return 0 if summary["status"] in {"completed","paused_budget"} else 1
if __name__=="__main__":raise SystemExit(main())
