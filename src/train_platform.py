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
from optim.state_simulation import persistence_metadata
from optim.adamw_state_diagnostics import AdamWStateDiagnostics, MuonStateDiagnostics
from optim.muon_update_fidelity import MuonUpdateFidelityObserver, save_snapshot


class NonFiniteMetricError(RuntimeError):
    """A scientific metric cannot be represented in the strict event schema."""

    def __init__(self, field: str, value_kind: str):
        super().__init__(f"non-finite metric {field}: {value_kind}")
        self.field = field
        self.value_kind = value_kind


def _nonfinite_kind(value) -> str | None:
    """Return a JSON-safe classification for a scalar non-finite value."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("metric finiteness checks require a scalar tensor")
        value = value.detach().item()
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return None


def require_finite_metric(field: str, value) -> float:
    """Validate an event metric before serialization, returning its float value."""
    result = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    kind = _nonfinite_kind(result)
    if kind is not None:
        raise NonFiniteMetricError(field, kind)
    return result


def _first_nonfinite_tensor_metric(field: str, value: torch.Tensor) -> None:
    """Raise with the first non-finite scalar class in an otherwise tensor metric."""
    bad = value.detach()[~torch.isfinite(value.detach())]
    if bad.numel():
        require_finite_metric(field, bad.reshape(-1)[0])


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


def parameter_groups(model, weight_decay: float, optimizer_name="reference_adamw"):
    seen, decay, no_decay, muon, auxiliary, records = set(), [], [], [], [], []
    for name, p in model.named_parameters():
        if id(p) in seen: continue
        seen.add(id(p)); use_decay = p.ndim >= 2
        # Muon is deliberately limited to internal hidden Linear weights.  The
        # tied embedding/lm_head is recognized by module path and kept auxiliary.
        eligible = optimizer_name in {"reference_muon", "recursive_muon"} and p.ndim == 2 and name.startswith("transformer.h.")
        if optimizer_name in {"reference_muon", "recursive_muon"}:
            (muon if eligible else auxiliary).append(p)
            group = "muon" if eligible else "auxiliary_adamw"
        else:
            (decay if use_decay else no_decay).append(p); group = "decay" if use_decay else "no_decay"
        records.append({"name": name, "group": group, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype)})
    # Both the historical FP32 reference and the recursive compressed
    # prototype use the same Muon/auxiliary parameter partition.  The
    # recursive optimizer dispatches on ``optimizer_group``; falling through
    # to generic decay/no_decay groups would silently run every parameter
    # through AdamW and bypass the codec entirely.
    if optimizer_name in {"reference_muon", "recursive_muon"}:
        return [{"params": muon, "weight_decay": weight_decay, "optimizer_group": "muon"},
                {"params": auxiliary, "weight_decay": weight_decay, "optimizer_group": "auxiliary_adamw"}], records
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
        total_nll += require_finite_metric("validation_nll", out["loss"]) * y.numel()
    model.train(was_training)
    mean = require_finite_metric("validation_nll", total_nll / target_tokens)
    try:
        ppl = math.exp(mean)
    except OverflowError:
        ppl = math.inf
    return {"evaluated_tokens": target_tokens, "nll": mean,
            "ppl": require_finite_metric("validation_ppl", ppl),
            "elapsed_seconds": require_finite_metric("validation_elapsed_seconds", time.monotonic()-started)}


def _checkpoint(cfg, model, optimizer, sampler, completed, processed, run_id, segment, elapsed, manifest, records):
    return {"schema_version": 1, "scientific_fingerprint": cfg["fingerprint"], "data_fingerprint": manifest["fingerprint"],
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "parameter_records": records,
        "completed_updates": completed, "processed_target_tokens": processed, "sampler": sampler.state_dict(),
        "rng": rng_state(torch.cuda.is_available()), "elapsed_seconds": elapsed, "run_id": run_id, "next_segment_id": segment+1}


def run(cfg: dict, run_dir: Path, resume: Path | None = None, max_wall_seconds: float | None = None, to_device: str = "auto", stop_at_update: int | None = None):
    root = Path(__file__).resolve().parents[1]; run_dir = run_dir.resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(to_device)
    if stop_at_update is not None and not 0 < int(stop_at_update) <= int(cfg["derived"]["total_updates"]):
        raise ValueError("stop_at_update must satisfy 0 < N <= recipe total_updates")
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
    model = build_model(cfg, device); groups, records = parameter_groups(model, cfg["optimizer"]["weight_decay"], cfg["optimizer"]["name"])
    recursive_codecs = None
    if cfg["optimizer"]["name"] == "recursive_muon":
        from optim.muon_recursive import build_recursive_codecs
        recursive_codecs = build_recursive_codecs(model, cfg["optimizer"], root)
    optimizer = make_optimizer(cfg["optimizer"]["name"], groups, cfg["optimizer"], codecs=recursive_codecs)
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
        state_policy = persistence_metadata(cfg["optimizer"]["name"], cfg["optimizer"]["state_simulation"],
            quantization_granularity=cfg["optimizer"].get("state_quantization_granularity", "per_state_tensor"),
            quantization_block_size=cfg["optimizer"].get("state_quantization_block_size", 2048))
        grouping = {"muon_parameters": sum(r["numel"] for r in records if r["group"] == "muon"), "auxiliary_adamw_parameters": sum(r["numel"] for r in records if r["group"] == "auxiliary_adamw"), "muon_tensor_count": sum(r["group"] == "muon" for r in records), "auxiliary_adamw_tensor_count": sum(r["group"] == "auxiliary_adamw" for r in records)}
        dump_resolved(cfg, run_dir/"resolved_config.json"); write_json(run_dir/"environment.json", environment_snapshot()); write_json(run_dir/"source.json", source_snapshot(root)); write_json(run_dir/"data_manifest.json", {k:v for k,v in manifest.items() if k != "_path"}); write_json(run_dir/"parameters.json", records); write_json(run_dir/"precision.json", {**cfg["precision"], "to_device": to_device, "device": str(device), "optimizer_state_simulation": cfg["optimizer"]["state_simulation"], "state_persistence": state_policy, "roundtrip_state_policy": state_policy["persistence_timing"], "parameter_grouping": grouping, "state_diagnostics": {"enabled": bool(cfg["logging"].get("state_diagnostics", False)), "stream": "state_diagnostics.jsonl", "timing": "detached read-only observation after FP32 moment/current update and after persistence simulation", "percentile_method": "bounded deterministic per-tensor evenly-spaced samples; exact counts/extrema/L2 reductions"}})
    writer = EventWriter(run_dir/"metrics.jsonl", run_id, segment)
    state_diagnostics_enabled = bool(cfg["logging"].get("state_diagnostics", False))
    state_writer = None
    collector = None
    if state_diagnostics_enabled:
        if cfg["optimizer"]["name"] not in {"reference_adamw", "reference_muon"}:
            raise ValueError("logging.state_diagnostics requires a reference optimizer")
        # Names are only used in the compact selected-landmark artifact.
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        bits = 4 if cfg["optimizer"]["state_simulation"].startswith("int4_") else 8
        if cfg["optimizer"]["name"] == "reference_adamw":
            collector = AdamWStateDiagnostics(names, tensor_landmarks=(1, 2, 3, 4, 5, 10, 20, 40, 60, 70, 75, 76),
                quantization_granularity=cfg["optimizer"].get("state_quantization_granularity", "per_state_tensor"),
                quantization_block_size=cfg["optimizer"].get("state_quantization_block_size", 2048),
                state_simulation=cfg["optimizer"]["state_simulation"], quantization_bits=bits)
        else:
            collector = MuonStateDiagnostics(quantization_granularity=cfg["optimizer"].get("state_quantization_granularity", "per_state_tensor"),
                quantization_block_size=cfg["optimizer"].get("state_quantization_block_size", 2048), quantization_bits=bits,
                state_simulation=cfg["optimizer"]["state_simulation"])
        optimizer.set_diagnostic_observer(collector.observe)
        state_writer = EventWriter(run_dir/"state_diagnostics.jsonl", run_id, segment)
    fidelity_enabled = bool(cfg["logging"].get("muon_update_fidelity", False))
    snapshot_updates = cfg["logging"].get("muon_momentum_snapshot_updates", [])
    fidelity_observer = None; fidelity_writer = None
    if fidelity_enabled or snapshot_updates:
        if cfg["optimizer"]["name"] != "reference_muon":
            raise ValueError("Muon update fidelity/snapshots require reference_muon")
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        fidelity_observer = MuonUpdateFidelityObserver(
            names, snapshot_updates=snapshot_updates,
            online_fidelity_enabled=fidelity_enabled)
        muon_bytes = sum(record["numel"] for record in records if record["group"] == "muon") * 4
        write_json(run_dir/"muon_update_fidelity_metadata.json", {
            "enabled": fidelity_enabled, "stream": "muon_update_fidelity.jsonl" if fidelity_enabled else None,
            "snapshot_format": "torch.save {format=muon_momentum_snapshot, version=1, metadata, tensors}; FP32 CPU tensors only; no model checkpoint",
            "snapshot_updates": snapshot_updates, "snapshot_selection": "full Muon momentum state (no sampling)",
            "expected_bytes_per_snapshot": muon_bytes,
            "expected_total_snapshot_bytes": muon_bytes * len(snapshot_updates),
            "post_muon_exclusion": "non-2D tensors; ReferenceMuon only orthogonalizes explicitly selected 2D Muon matrices",
            "quantizers": ["int8-linear-b2048", "int4-linear-b2048", "int4-dynamic-b2048"]})
        # Preserve the existing state-diagnostics callback rather than changing
        # optimizer behavior or its callback API.
        previous_observer = collector.observe if collector is not None else None
        def combined_observer(**kwargs):
            if previous_observer is not None: previous_observer(**kwargs)
            fidelity_observer.observe(**kwargs)
        optimizer.set_diagnostic_observer(combined_observer)
        if fidelity_enabled:
            fidelity_writer = EventWriter(run_dir/"muon_update_fidelity.jsonl", run_id, segment)
    started = time.monotonic(); status, reason = "completed", None
    divergence = None; last_finite_train_metric = None; active_update = completed; attempted_processed = processed
    print(f"[run] {run_id} | {cfg['optimizer']['name']} | {device} | {cfg['precision']['compute']} | {cfg['derived']['total_updates']} updates | schedule horizon {cfg['derived']['schedule_total_updates']} | {cfg['train']['target_tokens']} tokens")
    writer.write("lifecycle", completed, processed, phase="resume" if resume else "start", wall_clock_elapsed_seconds=prior_elapsed)
    try:
        if completed == 0:
            ev = evaluate(model, val, cfg["eval"]["max_target_tokens"], cfg["eval"]["batch_size"], cfg, device); writer.write("eval", 0, 0, split=cfg["data"]["validation_split"], wall_clock_elapsed_seconds=require_finite_metric("wall_clock_elapsed_seconds", prior_elapsed+time.monotonic()-started), **ev); print(f"[eval] step 0 | val NLL {ev['nll']:.4f} | ppl {ev['ppl']:.2f} | {ev['evaluated_tokens']} tokens")
            if cfg["checkpoint"]["save_initial"]:
                initial_ck = _checkpoint(cfg,model,optimizer,sampler,0,0,run_id,segment,prior_elapsed,manifest,records)
                atomic_torch_save(initial_ck, run_dir/"checkpoints/initial.pt"); atomic_torch_save(initial_ck, run_dir/"checkpoints/update_000000.pt")
        while completed < cfg["derived"]["total_updates"]:
            if max_wall_seconds is not None and time.monotonic()-started >= max_wall_seconds: status, reason = "paused_budget", "max_wall_seconds reached at update boundary"; print(f"[pause] wall-clock budget reached @ step {completed}; resumable checkpoint saved"); break
            update = completed + 1; active_update = update; attempted_processed = processed; optimizer.zero_grad(set_to_none=True); ids = sampler.take(cfg["train"]["micro_batch_size"]*cfg["train"]["accumulation_steps"]); nll_sum = 0.0; n_tokens = 0; step_started = time.monotonic()
            for a in range(cfg["train"]["accumulation_steps"]):
                batch_ids = ids[a*cfg["train"]["micro_batch_size"]:(a+1)*cfg["train"]["micro_batch_size"]]
                pairs = [train.window(i) for i in batch_ids]; x=torch.stack([z[0] for z in pairs]).to(device); y=torch.stack([z[1] for z in pairs]).to(device)
                with autocast_context(cfg, device): out=model(x,y); loss=out["loss"]
                n_tokens += y.numel(); attempted_processed = processed + n_tokens
                loss_value = require_finite_metric("train_nll", loss)
                nll_sum += loss_value*y.numel(); (loss/cfg["train"]["accumulation_steps"]).backward()
            for name,p in model.named_parameters():
                if p.grad is not None: _first_nonfinite_tensor_metric(f"gradient:{name}", p.grad)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"]); grad_norm_value = require_finite_metric("pre_clip_grad_norm", grad_norm); clipped = grad_norm_value > cfg["train"]["grad_clip_norm"]
            lr = require_finite_metric("learning_rate", learning_rate(update,cfg["derived"]["schedule_total_updates"],cfg["optimizer"]["lr"],cfg["schedule"]["warmup_updates"],cfg["schedule"]["final_lr_ratio"],cfg["schedule"]["name"]))
            for group in optimizer.param_groups: group["lr"] = lr
            if fidelity_observer is not None: fidelity_observer.begin_update(update)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            # The collector observes detached values created in ReferenceAdamW:
            # v before persistence and its next-step persisted counterpart.
            # It runs before event serialization and never mutates scientific state.
            train_nll = require_finite_metric("train_nll", nll_sum/n_tokens)
            if collector is not None:
                diagnostic = collector.finish_update(update=update, processed_target_tokens=processed+n_tokens,
                    train_nll=train_nll, pre_clip_grad_norm=grad_norm_value, learning_rate=lr)
                tensor_rows = diagnostic.pop("state_tensor_diagnostics", [])
                state_writer.write("state_diagnostics", update, processed+n_tokens,
                    **{key: value for key, value in diagnostic.items() if key not in {"update", "processed_target_tokens"}})
                for row in tensor_rows:
                    state_writer.write("state_tensor_diagnostics_tensor", update, processed+n_tokens, **row)
            if fidelity_observer is not None:
                fidelity_rows, snapshot_tensors = fidelity_observer.finish_update(update)
                if fidelity_writer is not None:
                    for row in fidelity_rows:
                        fidelity_writer.write("muon_update_fidelity", update, processed+n_tokens,
                                              **{key: value for key, value in row.items() if key != "update"})
                if snapshot_tensors:
                    source = source_snapshot(root)
                    metadata = {"update": update, "source_commit": source["upstream_commit"],
                                "recipe_fingerprint": cfg["fingerprint"], "data_fingerprint": manifest["fingerprint"],
                                "seeds": {key: cfg["experiment"][key] for key in ("seed", "data_seed", "algorithm_seed")},
                                "muon_transform": {"steps": cfg["optimizer"].get("muon_ns_steps", 5),
                                                   "coefficients": cfg["optimizer"].get("muon_ns_coefficients", [3.4445, -4.7750, 2.0315]),
                                                   "eps": cfg["optimizer"].get("muon_eps", 1e-7)}}
                    snapshot_path = run_dir/"muon_momentum_snapshots"/f"update_{update:06d}.pt"
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    save_snapshot(snapshot_path, metadata=metadata, tensors=snapshot_tensors)
            for name,p in model.named_parameters():
                _first_nonfinite_tensor_metric(f"parameter:{name}", p)
            completed, processed = update, processed+n_tokens
            update_elapsed=require_finite_metric("update_elapsed_seconds", time.monotonic()-step_started); tokens_per_second=require_finite_metric("tokens_per_second", n_tokens/update_elapsed); wall_clock_elapsed=require_finite_metric("wall_clock_elapsed_seconds", prior_elapsed+time.monotonic()-started)
            writer.write("train",completed,processed,lr=lr,train_nll=train_nll,pre_clip_grad_norm=grad_norm_value,clipped=clipped,elapsed_seconds=update_elapsed,update_elapsed_seconds=update_elapsed,tokens_this_update=n_tokens,tokens_per_second=tokens_per_second,wall_clock_elapsed_seconds=wall_clock_elapsed,cuda_peak_allocated=torch.cuda.max_memory_allocated() if device.type=="cuda" else None,cuda_peak_reserved=torch.cuda.max_memory_reserved() if device.type=="cuda" else None)
            last_finite_train_metric = {"update": completed, "train_nll": train_nll}
            if completed % cfg["logging"]["every_updates"] == 0 or completed == cfg["derived"]["total_updates"]: print(f"[train] {completed}/{cfg['derived']['total_updates']} | {processed} tok | loss {nll_sum/n_tokens:.4f} | lr {lr:.3e} | {update_elapsed*1000:.1f} ms/update | {n_tokens/update_elapsed:.0f} tok/s")
            do_eval = cfg["eval"]["every_updates"] > 0 and (completed % cfg["eval"]["every_updates"] == 0 or (stop_at_update is not None and completed == stop_at_update))
            if do_eval or completed == cfg["derived"]["total_updates"]:
                ev=evaluate(model,val,cfg["eval"]["max_target_tokens"],cfg["eval"]["batch_size"],cfg,device); writer.write("eval",completed,processed,split=cfg["data"]["validation_split"],wall_clock_elapsed_seconds=require_finite_metric("wall_clock_elapsed_seconds", prior_elapsed+time.monotonic()-started),**ev); print(f"[eval] step {completed} | val NLL {ev['nll']:.4f} | ppl {ev['ppl']:.2f} | {ev['evaluated_tokens']} tokens")
            if completed % cfg["checkpoint"]["every_updates"] == 0 or completed == cfg["derived"]["total_updates"] or (stop_at_update is not None and completed == stop_at_update):
                ck=_checkpoint(cfg,model,optimizer,sampler,completed,processed,run_id,segment,prior_elapsed+time.monotonic()-started,manifest,records); atomic_torch_save(ck,run_dir/"checkpoints/latest.pt"); atomic_torch_save(ck,run_dir/"checkpoints"/f"update_{completed:06d}.pt"); writer.write("checkpoint",completed,processed,path="checkpoints/latest.pt",landmark_path=f"checkpoints/update_{completed:06d}.pt",wall_clock_elapsed_seconds=prior_elapsed+time.monotonic()-started); print(f"[ckpt] latest.pt @ step {completed}")
            if stop_at_update is not None and completed == stop_at_update:
                status, reason = "paused_staged", f"explicit stage gate at update {stop_at_update}"
                print(f"[pause] staged gate reached at update {completed}; resume explicitly from checkpoints/latest.pt")
                break
    except KeyboardInterrupt:
        status, reason = "interrupted", "KeyboardInterrupt"
    except NonFiniteMetricError as exc:
        status, reason = "diverged_nonfinite", str(exc)
        divergence = {"status": status, "update": active_update, "processed_target_tokens": attempted_processed,
                      "offending_field": exc.field, "nonfinite_value": exc.value_kind,
                      "last_finite_train_metric": last_finite_train_metric,
                      "wall_clock_elapsed_seconds": require_finite_metric("wall_clock_elapsed_seconds", prior_elapsed+time.monotonic()-started),
                      "diagnostic_checkpoint_saved": False,
                      "diagnostic_checkpoint_policy": "not_saved_non_resumable"}
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"; write_json(run_dir/"summary.json", {"status":status,"reason":reason,"completed_updates":completed,"processed_target_tokens":processed}); raise
    elapsed=prior_elapsed+time.monotonic()-started
    # A divergence artifact is terminal and non-resumable.  Do not overwrite a
    # prior finite checkpoint with potentially invalid model/optimizer state.
    if status != "diverged_nonfinite" and (completed > 0 or status == "paused_budget"):
        ck=_checkpoint(cfg,model,optimizer,sampler,completed,processed,run_id,segment,elapsed,manifest,records); atomic_torch_save(ck,run_dir/"checkpoints/latest.pt"); atomic_torch_save(ck,run_dir/"checkpoints"/f"update_{completed:06d}.pt")
        if status=="completed" and cfg["checkpoint"]["save_final"]: atomic_torch_save(ck,run_dir/"checkpoints/final.pt")
    state = optimizer.recursive_state_summary() if hasattr(optimizer, "recursive_state_summary") else optimizer_state_summary(optimizer)
    writer.write("resource",completed,processed,optimizer_state=state,diagnostics=cfg["logging"]["diagnostics"],cuda_peak_allocated=torch.cuda.max_memory_allocated() if device.type=="cuda" else None,cuda_peak_reserved=torch.cuda.max_memory_reserved() if device.type=="cuda" else None)
    if divergence is not None:
        # The event envelope owns processed_target_tokens; retain the same
        # value in the nested summary schema without passing it twice.
        writer.write("divergence", completed, attempted_processed,
                     **{key: value for key, value in divergence.items() if key != "processed_target_tokens"})
    writer.write("lifecycle",completed,processed,phase=status,reason=reason,wall_clock_elapsed_seconds=elapsed,divergence=divergence)
    events=[json.loads(line) for line in (run_dir/"metrics.jsonl").read_text().splitlines()]; evals=[e for e in events if e["event_type"]=="eval" and e.get("split")==cfg["data"]["validation_split"]]; train_events=[e for e in events if e["event_type"]=="train"]
    summary={"run_id":run_id,"status":status,"reason":reason,"recipe_name":cfg["experiment"]["name"],"recipe_fingerprint":cfg["fingerprint"],"protocol_id":cfg["experiment"]["protocol_id"],"data_fingerprint":manifest["fingerprint"],"git_commit":source_snapshot(root)["upstream_commit"],"seed":cfg["experiment"]["seed"],"data_seed":cfg["experiment"]["data_seed"],"algorithm_seed":cfg["experiment"]["algorithm_seed"],"optimizer_name":cfg["optimizer"]["name"],"sequence_length":cfg["model"]["sequence_length"],"target_tokens":cfg["train"]["target_tokens"],"compute_precision":cfg["precision"]["compute"],"completed_updates":completed,"processed_target_tokens":processed,"total_updates":cfg["derived"]["total_updates"],"schedule_total_updates":cfg["derived"]["schedule_total_updates"],"elapsed_seconds":elapsed,"total_elapsed_seconds":elapsed,"initial_validation_nll":evals[0]["nll"] if evals else None,"final_validation_nll":evals[-1]["nll"] if evals else None,"best_validation_nll":min((e["nll"] for e in evals),default=None),"best_validation_update":min(evals,key=lambda e:e["nll"])["completed_updates"] if evals else None,"train_elapsed_seconds":sum(e["elapsed_seconds"] for e in train_events),"eval_elapsed_seconds":sum(e["elapsed_seconds"] for e in evals),"mean_update_seconds":sum(e["elapsed_seconds"] for e in train_events)/len(train_events) if train_events else None,"effective_tokens_per_second":processed/sum(e["elapsed_seconds"] for e in train_events) if train_events else None,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,"cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved() if device.type=="cuda" else None,"optimizer_state_bytes":state["unique_storage_bytes"],"optimizer_state_tensor_count":len(state["tensors"]),"max_grad_norm":max((e["pre_clip_grad_norm"] for e in train_events),default=None),"grad_clip_event_count":sum(e["clipped"] for e in train_events),"optimizer_state":state,"divergence":divergence}; write_json(run_dir/"summary.json",summary); print(f"[done] {status} | {completed}/{cfg['derived']['total_updates']} updates | {processed} tokens | {elapsed:.1f}s"); return summary
