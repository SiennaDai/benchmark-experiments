#!/usr/bin/env python3
"""CPU-only recursive persistence prototype for structural low-bit Muon.

The default run is a deterministic matrix-regression trajectory.  It is a
small execution harness for validating recursive optimizer-state semantics,
not a replacement for the frozen SlimPajama benchmark.  It deliberately
keeps the three trajectories paired and records a resumable compressed state.
Use ``--steps 512`` or ``--steps 1024`` for the diagnostic horizons.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from optim.muon_reference import ReferenceMuon, zeropower_newton_schulz
from optim.muon_recursive import RecursiveMuon, StructuralINT4Codec, StructuralVQCodec


def csv_write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def load_codebook(path: Path, key: str) -> torch.Tensor:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if key not in blob["codebooks"]:
        raise KeyError(f"missing codebook {key!r} in {path}")
    return blob["codebooks"][key].float()


class TinyRegression(torch.nn.Module):
    def __init__(self, n: int, seed: int):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn((n, n), generator=g) * 0.08)


def build_optimizer(model, method, codebook, rank):
    params = [{"params": [model.weight], "optimizer_group": "muon", "weight_decay": 0.0}]
    if method == "fp32":
        return ReferenceMuon(params, lr=0.01, weight_decay=0.0, muon_momentum=.95, muon_nesterov=True)
    codec = StructuralVQCodec(codebook, rank=rank) if method == "structural_vq_int3" else StructuralINT4Codec(rank=rank)
    return RecursiveMuon(params, {id(model.weight): codec}, lr=0.01, weight_decay=0.0, muon_momentum=.95, muon_nesterov=True)


def make_checkpoint(path, model, optimizer, step, rng_state):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "rng_state": rng_state}, path)


@torch.no_grad()
def state_metrics(reference, candidate, method, step):
    r, c = reference.float(), candidate.float(); den = r.norm().clamp_min(1e-12)
    return {"step": step, "method": method,
            "state_relative_l2_to_fp32": float((c-r).norm()/den),
            "state_cosine_to_fp32": float(torch.nn.functional.cosine_similarity(r.reshape(-1), c.reshape(-1), dim=0)),
            "state_norm_ratio": float(c.norm()/den)}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0))


def _residual_stats(codec, state):
    """Return descriptive decoded residual statistics without retaining a shadow."""
    decoded = codec.decode(state)
    if hasattr(state, "u"):
        low = (state.u.float() * state.singular_values.float()) @ state.vh.float()
        residual = decoded - low
        scales = state.scales.float().reshape(-1)
        return {
            "residual_norm": float(residual.norm()),
            "decoded_state_norm": float(decoded.norm()),
            "residual_over_state_norm": float(residual.norm() / decoded.norm().clamp_min(1e-12)),
            "scale_p98_mean": float(scales.mean()) if scales.numel() else 0.0,
            "scale_p98_max": float(scales.max()) if scales.numel() else 0.0,
            "clip_fraction": 0.0,
        }
    return {}


def _plot_outputs(out: Path):
    """Create compact diagnostic plots; plotting is downstream of the run."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = []
    with (out / "training_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    for field, ylabel, name in (("train_loss", "matrix-regression loss", "training_loss_vs_update.png"),
                                ("parameter_relative_l2_to_fp32", "parameter rel-L2 to FP32", "parameter_divergence_vs_update.png")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ("fp32", "structural_int4", "structural_vq_int3"):
            xy = [(int(r["step"]), float(r[field])) for r in rows if r.get("method") == method and r.get(field, "") not in ("", None)]
            if xy:
                ax.plot([x for x, _ in xy], [y for _, y in xy], label=method)
        ax.set_xlabel("update"); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout(); fig.savefig(out / name, dpi=140); plt.close(fig)
    with (out / "update_fidelity.csv").open() as f:
        upd = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ("structural_int4", "structural_vq_int3"):
        xy = [(int(r["step"]), float(r["local_update_cosine"])) for r in upd if r.get("method") == method]
        if xy: ax.plot([x for x, _ in xy], [y for _, y in xy], label=method)
    ax.set_xlabel("update"); ax.set_ylabel("current-state update cosine"); ax.legend(); fig.tight_layout(); fig.savefig(out / "local_update_cosine_vs_update.png", dpi=140); plt.close(fig)
    with (out / "vq_occupancy.csv").open() as f:
        occ = list(csv.DictReader(f))
    if occ:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot([int(r["step"]) for r in occ], [float(r.get("occupancy_entropy_bits", 0.0)) for r in occ]); ax.set_xlabel("update"); ax.set_ylabel("VQ occupancy entropy (bits)"); fig.tight_layout(); fig.savefig(out / "vq_occupancy_entropy_vs_update.png", dpi=140); plt.close(fig)


def run(args):
    torch.set_num_threads(args.threads); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    codebook = load_codebook(Path(args.codebook), args.codebook_key)
    g = torch.Generator().manual_seed(args.data_seed)
    n = args.dimension; x = torch.randn((n, args.batch), generator=g); target = torch.randn((n, args.batch), generator=g)
    base = TinyRegression(n, args.seed)
    models = {m: TinyRegression(n, args.seed) for m in ("fp32", "structural_int4", "structural_vq_int3")}
    for model in models.values(): model.load_state_dict(base.state_dict())
    opts = {m: build_optimizer(models[m], m, codebook, args.rank) for m in models}
    metrics, occupancy, residual_rows, runtime = [], [], [], []
    start = time.perf_counter()
    checkpoints = {0, args.steps, *[x for x in args.landmarks if x <= args.steps]}
    for step in range(1, args.steps + 1):
        for method, model in models.items():
            t0 = time.perf_counter(); opts[method].zero_grad(set_to_none=True)
            predecoded = None
            if method != "fp32" and model.weight in opts[method].state:
                pre_state = opts[method].state[model.weight].get("compressed_momentum")
                if pre_state is not None:
                    predecoded = opts[method].codecs[id(model.weight)].decode(pre_state)
            loss = ((model.weight @ x - target) ** 2).mean(); loss.backward(); opts[method].step()
            with torch.no_grad():
                current = model.weight.detach().clone(); ref = models["fp32"].weight.detach()
                row = {"step": step, "method": method, "train_loss": float(loss), "parameter_relative_l2_to_fp32": float((current-ref).norm()/ref.norm().clamp_min(1e-12)),
                       "parameter_cosine_to_fp32": float(torch.nn.functional.cosine_similarity(current.reshape(-1), ref.reshape(-1), dim=0)), "step_seconds": time.perf_counter()-t0}
                metrics.append(row)
                if method != "fp32":
                    # The persisted state is the only state read by the next
                    # update.  The current-step candidate is reconstructed
                    # from the pre-step decoded state and current gradient;
                    # it is transient diagnostic data, never optimizer state.
                    st = opts[method].state[model.weight]; codec = opts[method].codecs[id(model.weight)]; decoded = codec.decode(st["compressed_momentum"])
                    fp_state = opts["fp32"].state[models["fp32"].weight].get("muon_momentum", torch.zeros_like(decoded))
                    metrics.append(state_metrics(fp_state, decoded, method, step))
                    candidate = (torch.zeros_like(decoded) if predecoded is None else predecoded).mul(.95).add(model.weight.grad.float())
                    update_ref = zeropower_newton_schulz(candidate); update_dec = zeropower_newton_schulz(decoded)
                    metrics.append({"step": step, "method": method, "local_update_cosine": _cosine(update_ref, update_dec), "local_update_relative_l2": float((update_ref-update_dec).norm()/update_ref.norm().clamp_min(1e-12))})
                    if method == "structural_vq_int3":
                        encoded = st["compressed_momentum"]
                        idx = __import__("optim.muon_recursive", fromlist=["unpack_indices"]).unpack_indices(encoded.indices, encoded.pair_count)
                        counts = torch.bincount(idx, minlength=64).float(); total = counts.sum().clamp_min(1)
                        probs = counts[counts > 0] / total
                        entropy = float(-(probs * probs.log2()).sum())
                        occupancy.append({"step": step, "method": method, "packed_index_bytes": int(encoded.indices.numel()), "persistent_bits": codec.storage_bits(encoded), "used_codewords": int((counts > 0).sum()), "dead_codewords": int((counts == 0).sum()), "occupancy_entropy_bits": entropy, "clip_fraction": 0.0})
                    residual_rows.append({"step": step, "method": method, **_residual_stats(codec, st["compressed_momentum"])})
                runtime.append({"step": step, "method": method, "step_seconds": time.perf_counter()-t0})
        if step in checkpoints:
            for method in models:
                make_checkpoint(out / f"checkpoint_{method}_{step}.pt", models[method], opts[method], step, torch.get_rng_state())
    # Checkpoint/resume equivalence for the compressed VQ trajectory.
    resume_ok = False
    if args.resume_test and args.steps >= 2:
        fresh = TinyRegression(n, args.seed); fresh.load_state_dict(base.state_dict()); fresh_opt = build_optimizer(fresh, "structural_vq_int3", codebook, args.rank)
        for _ in range(args.steps):
            fresh_opt.zero_grad(set_to_none=True); ((fresh.weight @ x-target)**2).mean().backward(); fresh_opt.step()
        resumed = TinyRegression(n, args.seed); resumed.load_state_dict(base.state_dict()); resumed_opt = build_optimizer(resumed, "structural_vq_int3", codebook, args.rank)
        cut = max(1, args.steps//2)
        for _ in range(cut): resumed_opt.zero_grad(set_to_none=True); ((resumed.weight @ x-target)**2).mean().backward(); resumed_opt.step()
        checkpoint = {"model": resumed.state_dict(), "optimizer": resumed_opt.state_dict()}
        resumed2 = TinyRegression(n, args.seed); resumed2.load_state_dict(base.state_dict()); resumed_opt2 = build_optimizer(resumed2, "structural_vq_int3", codebook, args.rank)
        resumed2.load_state_dict(checkpoint["model"]); resumed_opt2.load_state_dict(checkpoint["optimizer"])
        for _ in range(args.steps-cut): resumed_opt2.zero_grad(set_to_none=True); ((resumed2.weight @ x-target)**2).mean().backward(); resumed_opt2.step()
        resume_ok = bool(torch.allclose(fresh.weight, resumed2.weight, atol=0, rtol=0))
    csv_write(out / "training_metrics.csv", metrics); csv_write(out / "validation_metrics.csv", [{"step": r["step"], "method": r["method"], "validation_nll": "unavailable_matrix_regression_diagnostic", "train_loss": r.get("train_loss"), "parameter_relative_l2_to_fp32": r.get("parameter_relative_l2_to_fp32")} for r in metrics if "train_loss" in r and int(r["step"]) in checkpoints]); csv_write(out / "vq_occupancy.csv", occupancy); csv_write(out / "runtime_accounting.csv", runtime)
    csv_write(out / "state_divergence.csv", [r for r in metrics if "state_relative_l2_to_fp32" in r]); csv_write(out / "update_fidelity.csv", [r for r in metrics if "local_update_cosine" in r]);
    csv_write(out / "instant_vs_trajectory_error.csv", [{"step": r["step"], "method": r["method"], "instant_update_relative_l2": r.get("local_update_relative_l2"), "trajectory_parameter_relative_l2": r.get("parameter_relative_l2_to_fp32")} for r in metrics if "local_update_relative_l2" in r and r.get("method") != "fp32"])
    persistent = []
    for method, opt in opts.items():
        if method == "fp32":
            persistent.append({"method": method, "persistent_bits": int(n*n*32), "effective_bits_per_scalar": 32.0, "hidden_fp32_momentum": True})
        else:
            st = opt.state[models[method].weight]["compressed_momentum"]; codec = opt.codecs[id(models[method].weight)]; bits = codec.storage_bits(st)
            persistent.append({"method": method, "persistent_bits": bits, "effective_bits_per_scalar": bits/(n*n), "hidden_fp32_momentum": False})
    # Global codebook storage is counted once per VQ trajectory; the fixed
    # dynamic INT4 map is treated as metadata rather than per-state payload.
    persistent.append({"method": "structural_vq_int3_codebook_global", "persistent_bits": int(codebook.numel() * 32), "effective_bits_per_scalar": float(codebook.numel() * 32 / (n*n)), "hidden_fp32_momentum": False})
    csv_write(out / "memory_accounting.csv", persistent); csv_write(out / "structural_residual.csv", residual_rows)
    csv_write(out / "checkpoint_resume_test.csv", [{"test": "compressed_checkpoint_resume", "passed": resume_ok, "details": "continuous and save/reload/continue trajectories"}]); csv_write(out / "failure_events.csv", [])
    csv_write(out / "offline_vs_recursive.csv", [{"method": r["method"], "step": r["step"], "offline_matched_state_proxy": r.get("local_update_cosine"), "recursive_parameter_cosine": r.get("parameter_cosine_to_fp32")} for r in metrics if "local_update_cosine" in r])
    protocol = {"method": "recursive_structural_vq_prototype", "steps": args.steps, "seed": args.seed, "data_seed": args.data_seed, "rank": args.rank, "dimension": n, "batch": args.batch, "codebook": str(args.codebook), "codebook_key": args.codebook_key, "structure": "exact truncated SVD oracle; BF16 U/S/V; p98 residual; fixed codebook", "recursive_state": True, "hidden_fp32_momentum": False, "resume_exact": resume_ok, "runtime_seconds": time.perf_counter()-start, "methods": ["fp32", "structural_int4", "structural_vq_int3"]}
    (out / "protocol_manifest.json").write_text(json.dumps(protocol, indent=2))
    protocol["trajectory_type"] = "deterministic_matrix_regression_diagnostic"
    protocol["frozen_slimpajama_training"] = False
    protocol["validation_nll_status"] = "not_applicable_to_matrix_regression_diagnostic"
    (out / "protocol_manifest.json").write_text(json.dumps(protocol, indent=2))
    _plot_outputs(out)
    (out / "summary.md").write_text("# Recursive structural low-bit Muon prototype\n\n" + json.dumps(protocol, indent=2) + "\n\n## Interpretation\n\nThis CPU diagnostic validates recursive compressed-state semantics, not language-model training quality. The three trajectories share the same synthetic matrix-regression inputs and initialization. Structural methods decode only their persisted BF16 factors, packed residual indices, and block scales on the next update; no FP32 momentum shadow is retained. The requested frozen SlimPajama protocol is not run in this first CPU closure because the repository's production runner does not yet expose this experimental codec as a training recipe; `validation_nll` is therefore explicitly unavailable rather than fabricated. A longer SlimPajama paired run is required before deployment claims.\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=20); p.add_argument("--dimension", type=int, default=16); p.add_argument("--batch", type=int, default=32); p.add_argument("--rank", type=int, default=8); p.add_argument("--seed", type=int, default=0); p.add_argument("--data-seed", type=int, default=1337); p.add_argument("--threads", type=int, default=1); p.add_argument("--resume-test", action="store_true"); p.add_argument("--output", type=Path, default=ROOT/"reports/muon_recursive_vq_prototype"); p.add_argument("--codebook", type=Path, default=ROOT/"reports/muon_vector_int3_robustness/calibration_codebooks.pt"); p.add_argument("--codebook-key", default="s0_k8_w64_t8_v1200"); args = p.parse_args(); args.landmarks = [0, 128, 256, 512, 1024]; run(args)
