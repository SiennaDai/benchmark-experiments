"""Single-process fixed-token pretraining loop for optimizer experiments."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from config.recipe import dump_resolved
from data.frozen_tokens import DeterministicSampler, FrozenWindows, load_manifest, validate_data_capacity
from experiment_io import EventWriter, atomic_torch_save, environment_snapshot, restore_rng, rng_state, source_snapshot, write_json
from models.llama import Llama
from optim.lowp_adapter import make_optimizer, optimizer_state_summary


def resolve_device(to_device: str) -> torch.device:
    """Resolve an explicit runtime device without silently changing its type."""
    if to_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if to_device == "cpu":
        return torch.device("cpu")
    if to_device == "cuda" or to_device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested device {to_device!r}, but CUDA is unavailable")
        device = torch.device("cuda:0" if to_device == "cuda" else to_device)
        if device.index is None or device.index < 0 or device.index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device index is unavailable: {device}")
        return device
    raise ValueError("to_device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")


def learning_rate(k: int, n: int, peak: float, warmup: int, ratio: float, name: str) -> float:
    if name == "constant" or n == 1: return peak
    if warmup and k <= warmup: return peak * k / warmup
    u = (k - warmup) / (n - warmup) if warmup else (k - 1) / max(n - 1, 1)
    return peak * (ratio + (1-ratio) * (1+math.cos(math.pi*u))/2)


def model_namespace(cfg: dict) -> SimpleNamespace:
    m, p = cfg["model"], cfg["precision"]
    return SimpleNamespace(**m, untied_embeds=not m["tie_embeddings"], attention_backend=p["attention_backend"],
        moe=False, parallel_block=False, moe_routing="standard_gating")


def build_model(cfg: dict, device: torch.device) -> Llama:
    torch.manual_seed(cfg["experiment"]["seed"])
    model = Llama(model_namespace(cfg)).to(device=device, dtype=torch.float32)
    m = cfg["model"]
    assert len(model.transformer.h) == m["n_layer"] and model.head_dim == m["n_embd"] // m["n_head"]
    assert model.transformer.h[0].mlp.w1.out_features == m["ffn_dim"]
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    return model


def parameter_groups(model, weight_decay: float):
    seen, decay, no_decay, records = set(), [], [], []
    for name, p in model.named_parameters():
        if id(p) in seen: continue
        seen.add(id(p)); use_decay = p.ndim >= 2
        (decay if use_decay else no_decay).append(p)
        records.append({"name": name, "group": "decay" if use_decay else "no_decay", "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype)})
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], records


def autocast_context(cfg: dict, device: torch.device):
    compute = cfg["precision"]["compute"]
    if compute == "fp32": return contextlib.nullcontext()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 compute is unsupported on this device")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(model, windows, target_tokens: int, batch_size: int, cfg: dict, device: torch.device):
    was_training = model.training; model.eval(); started = time.monotonic()
    n_windows = target_tokens // windows.sequence_length
    if n_windows > windows.num_windows:
        raise ValueError(f"validation capacity insufficient: available_windows={windows.num_windows}, requested_windows={n_windows}")
    total_nll = 0.0
    for start in range(0, n_windows, batch_size):
        pairs = [windows.window(i) for i in range(start, min(start+batch_size, n_windows))]
        x = torch.stack([z[0] for z in pairs]).to(device); y = torch.stack([z[1] for z in pairs]).to(device)
        out = model(x, y, get_logits=False)
        total_nll += float(out["loss"].double()) * y.numel()
    model.train(was_training)
    mean = total_nll / target_tokens
    return {"evaluated_tokens": target_tokens, "nll": mean, "ppl": math.exp(mean), "elapsed_seconds": time.monotonic()-started}


def _checkpoint(cfg, model, optimizer, sampler, completed, processed, run_id, segment, elapsed, manifest, records):
    return {"schema_version": 1, "scientific_fingerprint": cfg["fingerprint"], "data_fingerprint": manifest["fingerprint"],
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "parameter_records": records,
        "completed_updates": completed, "processed_target_tokens": processed, "sampler": sampler.state_dict(),
        "rng": rng_state(torch.cuda.is_available()), "elapsed_seconds": elapsed, "run_id": run_id, "next_segment_id": segment+1}


def run(cfg: dict, run_dir: Path, resume: Path | None = None, max_wall_seconds: float | None = None, to_device: str = "auto"):
    root = Path(__file__).resolve().parents[1]; run_dir = run_dir.resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(to_device)
    if cfg["precision"]["compute"] == "bf16" and device.type != "cuda": raise RuntimeError("BF16 compute is unsupported without CUDA")
    torch.use_deterministic_algorithms(cfg["precision"]["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = cfg["precision"]["tf32"]
    random.seed(cfg["experiment"]["seed"]); np.random.seed(cfg["experiment"]["seed"]); torch.manual_seed(cfg["experiment"]["seed"])
    manifest = load_manifest(cfg["data"]["manifest"]); train = FrozenWindows(manifest, cfg["data"]["train_split"], cfg["model"]["sequence_length"]); val = FrozenWindows(manifest, cfg["data"]["validation_split"], cfg["model"]["sequence_length"])
    if manifest["max_token_id"] >= cfg["model"]["vocab_size"]: raise ValueError("data token id exceeds model vocabulary")
    needed = cfg["derived"]["total_updates"] * cfg["train"]["micro_batch_size"] * cfg["train"]["accumulation_steps"]
    validate_data_capacity(manifest, sequence_length=cfg["model"]["sequence_length"], train_split=cfg["data"]["train_split"],
        validation_split=cfg["data"]["validation_split"], eval_target_tokens=cfg["eval"]["max_target_tokens"],
        train_target_tokens=cfg["train"]["target_tokens"], micro_batch_size=cfg["train"]["micro_batch_size"],
        accumulation_steps=cfg["train"]["accumulation_steps"], total_updates=cfg["derived"]["total_updates"],
        allow_repeated_epochs=cfg["data"]["allow_repeated_epochs"])
    model = build_model(cfg, device); groups, records = parameter_groups(model, cfg["optimizer"]["weight_decay"]); optimizer = make_optimizer(cfg["optimizer"]["name"], groups, cfg["optimizer"])
    sampler = DeterministicSampler(train.num_windows, cfg["experiment"]["data_seed"], cfg["data"]["allow_repeated_epochs"])
    completed = processed = segment = 0; prior_elapsed = 0.0
    run_id = run_dir.name
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if ckpt["scientific_fingerprint"] != cfg["fingerprint"] or ckpt["data_fingerprint"] != manifest["fingerprint"]: raise ValueError("resume configuration or data fingerprint mismatch")
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"]); sampler.load_state_dict(ckpt["sampler"])
        completed, processed, segment, prior_elapsed = ckpt["completed_updates"], ckpt["processed_target_tokens"], ckpt["next_segment_id"], ckpt["elapsed_seconds"]
        restore_rng(ckpt["rng"]); run_id = ckpt["run_id"]
    else:
        dump_resolved(cfg, run_dir/"resolved_config.json"); write_json(run_dir/"environment.json", environment_snapshot()); write_json(run_dir/"source.json", source_snapshot(root)); write_json(run_dir/"data_manifest.json", {k:v for k,v in manifest.items() if k != "_path"}); write_json(run_dir/"parameters.json", records); write_json(run_dir/"precision.json", {**cfg["precision"], "to_device": to_device, "device": str(device)})
    writer = EventWriter(run_dir/"metrics.jsonl", run_id, segment); started = time.monotonic(); status, reason = "completed", None
    print(f"[run] {run_id} | {cfg['optimizer']['name']} | {device} | {cfg['precision']['compute']} | {cfg['derived']['total_updates']} updates | {cfg['train']['target_tokens']} tokens")
    writer.write("lifecycle", completed, processed, phase="resume" if resume else "start", wall_clock_elapsed_seconds=prior_elapsed)
    if completed == 0:
        ev = evaluate(model, val, cfg["eval"]["max_target_tokens"], cfg["eval"]["batch_size"], cfg, device); writer.write("eval", 0, 0, split=cfg["data"]["validation_split"], wall_clock_elapsed_seconds=prior_elapsed+time.monotonic()-started, **ev); print(f"[eval] step 0 | val NLL {ev['nll']:.4f} | ppl {ev['ppl']:.2f} | {ev['evaluated_tokens']} tokens")
        if cfg["checkpoint"]["save_initial"]: atomic_torch_save(_checkpoint(cfg,model,optimizer,sampler,0,0,run_id,segment,prior_elapsed,manifest,records), run_dir/"checkpoints/initial.pt")
    try:
        while completed < cfg["derived"]["total_updates"]:
            if max_wall_seconds is not None and time.monotonic()-started >= max_wall_seconds: status, reason = "paused_budget", "max_wall_seconds reached at update boundary"; print(f"[pause] wall-clock budget reached @ step {completed}; resumable checkpoint saved"); break
            update = completed + 1; optimizer.zero_grad(set_to_none=True); ids = sampler.take(cfg["train"]["micro_batch_size"]*cfg["train"]["accumulation_steps"]); nll_sum = 0.0; n_tokens = 0; step_started = time.monotonic()
            for a in range(cfg["train"]["accumulation_steps"]):
                batch_ids = ids[a*cfg["train"]["micro_batch_size"]:(a+1)*cfg["train"]["micro_batch_size"]]
                pairs = [train.window(i) for i in batch_ids]; x=torch.stack([z[0] for z in pairs]).to(device); y=torch.stack([z[1] for z in pairs]).to(device)
                with autocast_context(cfg, device): out=model(x,y); loss=out["loss"]
                if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
                nll_sum += float(loss.detach().double())*y.numel(); n_tokens += y.numel(); (loss/cfg["train"]["accumulation_steps"]).backward()
            for name,p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError(f"non-finite gradient: {name}")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"]); clipped = float(grad_norm) > cfg["train"]["grad_clip_norm"]
            lr = learning_rate(update,cfg["derived"]["total_updates"],cfg["optimizer"]["lr"],cfg["schedule"]["warmup_updates"],cfg["schedule"]["final_lr_ratio"],cfg["schedule"]["name"])
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for name,p in model.named_parameters():
                if not torch.isfinite(p).all(): raise FloatingPointError(f"non-finite parameter: {name}")
            completed, processed = update, processed+n_tokens
            update_elapsed=time.monotonic()-step_started
            writer.write("train",completed,processed,lr=lr,train_nll=nll_sum/n_tokens,pre_clip_grad_norm=float(grad_norm),clipped=clipped,elapsed_seconds=update_elapsed,update_elapsed_seconds=update_elapsed,tokens_this_update=n_tokens,tokens_per_second=n_tokens/update_elapsed,wall_clock_elapsed_seconds=prior_elapsed+time.monotonic()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated() if device.type=="cuda" else None,cuda_peak_reserved=torch.cuda.max_memory_reserved() if device.type=="cuda" else None)
            if completed % cfg["logging"]["every_updates"] == 0 or completed == cfg["derived"]["total_updates"]: print(f"[train] {completed}/{cfg['derived']['total_updates']} | {processed} tok | loss {nll_sum/n_tokens:.4f} | lr {lr:.3e} | {update_elapsed*1000:.1f} ms/update | {n_tokens/update_elapsed:.0f} tok/s")
            do_eval = cfg["eval"]["every_updates"] > 0 and completed % cfg["eval"]["every_updates"] == 0
            if do_eval or completed == cfg["derived"]["total_updates"]:
                ev=evaluate(model,val,cfg["eval"]["max_target_tokens"],cfg["eval"]["batch_size"],cfg,device); writer.write("eval",completed,processed,split=cfg["data"]["validation_split"],wall_clock_elapsed_seconds=prior_elapsed+time.monotonic()-started,**ev); print(f"[eval] step {completed} | val NLL {ev['nll']:.4f} | ppl {ev['ppl']:.2f} | {ev['evaluated_tokens']} tokens")
            if completed % cfg["checkpoint"]["every_updates"] == 0 or completed == cfg["derived"]["total_updates"]:
                ck=_checkpoint(cfg,model,optimizer,sampler,completed,processed,run_id,segment,prior_elapsed+time.monotonic()-started,manifest,records); atomic_torch_save(ck,run_dir/"checkpoints/latest.pt"); writer.write("checkpoint",completed,processed,path="checkpoints/latest.pt",wall_clock_elapsed_seconds=prior_elapsed+time.monotonic()-started); print(f"[ckpt] latest.pt @ step {completed}")
    except KeyboardInterrupt:
        status, reason = "interrupted", "KeyboardInterrupt"
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"; write_json(run_dir/"summary.json", {"status":status,"reason":reason,"completed_updates":completed,"processed_target_tokens":processed}); raise
    elapsed=prior_elapsed+time.monotonic()-started
    if completed > 0 or status == "paused_budget":
        ck=_checkpoint(cfg,model,optimizer,sampler,completed,processed,run_id,segment,elapsed,manifest,records); atomic_torch_save(ck,run_dir/"checkpoints/latest.pt")
        if status=="completed" and cfg["checkpoint"]["save_final"]: atomic_torch_save(ck,run_dir/"checkpoints/final.pt")
    state=optimizer_state_summary(optimizer); writer.write("resource",completed,processed,optimizer_state=state,diagnostics=cfg["logging"]["diagnostics"],cuda_peak_allocated=torch.cuda.max_memory_allocated() if device.type=="cuda" else None,cuda_peak_reserved=torch.cuda.max_memory_reserved() if device.type=="cuda" else None)
    writer.write("lifecycle",completed,processed,phase=status,reason=reason,wall_clock_elapsed_seconds=elapsed)
    events=[json.loads(line) for line in (run_dir/"metrics.jsonl").read_text().splitlines()]; evals=[e for e in events if e["event_type"]=="eval" and e.get("split")==cfg["data"]["validation_split"]]; train_events=[e for e in events if e["event_type"]=="train"]
    summary={"run_id":run_id,"status":status,"reason":reason,"recipe_name":cfg["experiment"]["name"],"recipe_fingerprint":cfg["fingerprint"],"protocol_id":cfg["experiment"]["protocol_id"],"data_fingerprint":manifest["fingerprint"],"git_commit":source_snapshot(root)["upstream_commit"],"seed":cfg["experiment"]["seed"],"data_seed":cfg["experiment"]["data_seed"],"algorithm_seed":cfg["experiment"]["algorithm_seed"],"optimizer_name":cfg["optimizer"]["name"],"sequence_length":cfg["model"]["sequence_length"],"target_tokens":cfg["train"]["target_tokens"],"compute_precision":cfg["precision"]["compute"],"completed_updates":completed,"processed_target_tokens":processed,"total_updates":cfg["derived"]["total_updates"],"elapsed_seconds":elapsed,"total_elapsed_seconds":elapsed,"initial_validation_nll":evals[0]["nll"] if evals else None,"final_validation_nll":evals[-1]["nll"] if evals else None,"best_validation_nll":min((e["nll"] for e in evals),default=None),"best_validation_update":min(evals,key=lambda e:e["nll"])["completed_updates"] if evals else None,"train_elapsed_seconds":sum(e["elapsed_seconds"] for e in train_events),"eval_elapsed_seconds":sum(e["elapsed_seconds"] for e in evals),"mean_update_seconds":sum(e["elapsed_seconds"] for e in train_events)/len(train_events) if train_events else None,"effective_tokens_per_second":processed/sum(e["elapsed_seconds"] for e in train_events) if train_events else None,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,"cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved() if device.type=="cuda" else None,"optimizer_state_bytes":state["unique_storage_bytes"],"optimizer_state_tensor_count":len(state["tensors"]),"max_grad_norm":max((e["pre_clip_grad_norm"] for e in train_events),default=None),"grad_clip_event_count":sum(e["clipped"] for e in train_events),"optimizer_state":state}; write_json(run_dir/"summary.json",summary); print(f"[done] {status} | {completed}/{cfg['derived']['total_updates']} updates | {processed} tokens | {elapsed:.1f}s"); return summary
