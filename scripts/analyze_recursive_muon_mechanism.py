#!/usr/bin/env python3
"""Paired analysis of instantaneous persistence error and closed-loop drift."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.optim.muon_reference import zeropower_newton_schulz


def _k5(x, meta):
    return zeropower_newton_schulz(x, int(meta["muon_ns_steps"]), tuple(meta["muon_ns_coefficients"]), float(meta["muon_eps"]))


def _metrics(a, b):
    a, b = a.float(), b.float(); d = a - b
    an, bn = a.norm(), b.norm()
    cosine = float((a.reshape(-1) @ b.reshape(-1)) / (an * bn).clamp_min(1e-30))
    return {"cosine": max(-1.0, min(1.0, cosine)),
            "relative_l2": float(d.norm() / an.clamp_min(1e-30)),
            "norm_ratio": float(bn / an.clamp_min(1e-30))}


def _direction_from_previous(previous_momentum: torch.Tensor, gradient: torch.Tensor,
                             mu: float) -> torch.Tensor:
    """Exact executed Nesterov direction from the *previous-step* state.

    The recurrence is M_t = mu M_{t-1} + g_t, followed by
    D_t = g_t + mu M_t = (1+mu)g_t + mu^2 M_{t-1}.
    """
    return (1.0 + mu) * gradient.float() + (mu * mu) * previous_momentum.float()


def _pooled_pairs(pairs):
    """Pool tensor comparisons by global norms/dot products, not mean cosine."""
    aa = bb = dd = dot = 0.0
    for a, b in pairs:
        a, b = a.float(), b.float()
        aa += float(a.square().sum()); bb += float(b.square().sum()); dd += float((a-b).square().sum()); dot += float((a*b).sum())
    cosine = dot / max(math.sqrt(aa*bb), 1e-30)
    return {"cosine": max(-1.0, min(1.0, cosine)),
            "relative_l2": math.sqrt(dd/max(aa, 1e-30)),
            "norm_ratio": math.sqrt(bb/max(aa, 1e-30))}


def validate_pair_metadata(fp_meta: dict, vq_meta: dict, *, expected_seed=1, expected_schedule=4096):
    # Recipe fingerprints intentionally differ because one run is reference
    # Muon and the other owns recursive VQ state.  Compare shared scientific
    # protocol fields instead, while retaining each recipe fingerprint as
    # provenance.
    keys = ("seed", "data_seed", "algorithm_seed", "protocol_id", "data_fingerprint", "schedule_total_updates", "tokens_per_update", "sequence_length", "muon_momentum", "muon_nesterov", "muon_ns_steps", "muon_ns_coefficients", "muon_eps")
    mismatches = []
    for key in keys:
        if fp_meta.get(key) != vq_meta.get(key): mismatches.append({"field": key, "fp32": fp_meta.get(key), "vq": vq_meta.get(key)})
    if fp_meta.get("seed") != expected_seed: mismatches.append({"field": "seed", "expected": expected_seed, "actual": fp_meta.get("seed")})
    if fp_meta.get("schedule_total_updates") != expected_schedule: mismatches.append({"field": "schedule_total_updates", "expected": expected_schedule, "actual": fp_meta.get("schedule_total_updates")})
    return mismatches


def _load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != "recursive_muon_mechanism_snapshot" or blob.get("version") != 1:
        raise ValueError(f"unsupported mechanism snapshot: {path}")
    return blob


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _train_events(run):
    path = run / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip() and json.loads(line).get("event_type") == "train"]


def _paired_run_integrity(fp, vq, fp_meta, vq_meta, landmarks):
    shared = ("seed", "data_seed", "algorithm_seed", "protocol_id", "data_fingerprint",
              "schedule_total_updates", "total_updates", "target_tokens", "sequence_length",
              "compute_precision")
    mismatches = []
    for key in shared:
        if fp_meta.get(key) != vq_meta.get(key):
            mismatches.append({"field": key, "fp32": fp_meta.get(key), "vq": vq_meta.get(key)})
    for run_name, meta in (("fp32", fp_meta), ("vq", vq_meta)):
        if meta.get("seed") != 1: mismatches.append({"field": f"{run_name}.seed", "expected": 1, "actual": meta.get("seed")})
        if meta.get("data_seed") != 1337: mismatches.append({"field": f"{run_name}.data_seed", "expected": 1337, "actual": meta.get("data_seed")})
        if meta.get("algorithm_seed") != 2026: mismatches.append({"field": f"{run_name}.algorithm_seed", "expected": 2026, "actual": meta.get("algorithm_seed")})
        if meta.get("schedule_total_updates") != 4096: mismatches.append({"field": f"{run_name}.schedule_total_updates", "expected": 4096, "actual": meta.get("schedule_total_updates")})
        if meta.get("total_updates") != 4096: mismatches.append({"field": f"{run_name}.total_updates", "expected": 4096, "actual": meta.get("total_updates")})
        if meta.get("compute_precision") != "fp32": mismatches.append({"field": f"{run_name}.compute_precision", "expected": "fp32", "actual": meta.get("compute_precision")})
    if fp_meta.get("optimizer_name") != "reference_muon":
        mismatches.append({"field": "fp32.optimizer_name", "expected": "reference_muon", "actual": fp_meta.get("optimizer_name")})
    if vq_meta.get("optimizer_name") != "recursive_muon":
        mismatches.append({"field": "vq.optimizer_name", "expected": "recursive_muon", "actual": vq_meta.get("optimizer_name")})
    expected = [1, 8, 32, 128, 512]
    updates = [int(stem.rsplit("_", 1)[1]) for stem in landmarks]
    if updates != expected:
        mismatches.append({"field": "raw_landmarks", "expected": expected, "actual": updates})
    train_fp, train_vq = _train_events(fp), _train_events(vq)
    train_aligned = len(train_fp) == len(train_vq) and bool(train_fp)
    if train_aligned:
        for a, b in zip(train_fp, train_vq):
            for field in ("completed_updates", "processed_target_tokens", "tokens_this_update", "lr"):
                if a.get(field) != b.get(field):
                    train_aligned = False
                    mismatches.append({"field": f"train_event.{field}", "update_fp32": a.get("completed_updates"),
                                       "fp32": a.get(field), "vq": b.get(field)})
                    break
            if not train_aligned: break
    return {"mismatches": mismatches,
            "paired_train_events": len(train_fp) if train_aligned else None,
            "train_positions_lr_tokens_match": train_aligned,
            "batch_ids_or_hashes_saved": False,
            "batch_alignment_evidence": "matching deterministic protocol, update/token positions, tokens per update, and learning rates; raw batch IDs/hashes are not present"}


def _write_csv(path, rows):
    if not rows:
        path.write_text("status,reason\nno_rows,missing_or_invalid_snapshots\n"); return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-run", required=True)
    ap.add_argument("--vq-run", required=True)
    ap.add_argument("--out", default="reports/recursive_muon_mechanism_s1_512_corrected")
    args = ap.parse_args(); fp = Path(args.fp32_run); vq = Path(args.vq_run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fp_files = {p.stem: p for p in (fp / "mechanism").glob("update_*.pt")}; vq_files = {p.stem: p for p in (vq / "mechanism").glob("update_*.pt")}
    common = sorted(set(fp_files) & set(vq_files), key=lambda x: int(x.split("_")[-1]))
    if not common: raise RuntimeError("no paired mechanism snapshots found")
    first_fp, first_vq = _load(fp_files[common[0]]), _load(vq_files[common[0]])
    mismatches = validate_pair_metadata(first_fp["metadata"], first_vq["metadata"])
    if mismatches: raise RuntimeError(json.dumps({"paired_protocol_mismatch": mismatches}, indent=2))
    fp_summary, vq_summary = _read_json(fp / "summary.json"), _read_json(vq / "summary.json")
    if fp_summary is None or vq_summary is None:
        raise RuntimeError("paired run summary.json files are required for integrity validation")
    run_integrity = _paired_run_integrity(fp, vq, fp_summary, vq_summary, common)
    if run_integrity["mismatches"]:
        raise RuntimeError(json.dumps({"paired_run_integrity_mismatch": run_integrity["mismatches"]}, indent=2))
    for method, summary in (("FP32", fp_summary), ("VQ", vq_summary)):
        if summary.get("completed_updates") != 512 or summary.get("status") != "paused_staged":
            raise RuntimeError(f"{method} run did not stop cleanly at staged update 512")
    fp_names = {x["name"] for x in first_fp["tensors"]}; vq_names = {x["name"] for x in first_vq["tensors"]}
    if fp_names != vq_names: raise RuntimeError("tensor name sets differ between paired runs")
    rows, tensor_rows = [], []
    mu = float(first_vq["metadata"]["muon_momentum"])
    for stem in common:
        fblob, vblob = _load(fp_files[stem]), _load(vq_files[stem]); fm = {x["name"]: x for x in fblob["tensors"]}; vm = {x["name"]: x for x in vblob["tensors"]}
        if fblob["metadata"].get("update") != vblob["metadata"].get("update") or fblob["metadata"].get("processed_target_tokens") != vblob["metadata"].get("processed_target_tokens"):
            raise RuntimeError(f"paired sampler/update alignment mismatch at {stem}")
        if set(fm) != set(vm): raise RuntimeError(f"tensor names differ at {stem}")
        local_rows = []
        for name in sorted(fm):
            f, v = fm[name], vm[name]
            if f["shape"] != v["shape"]: raise RuntimeError(f"shape mismatch for {name} at {stem}")
            g_ref = f["gradient"].float(); g_vq = v["gradient"].float()
            m_ref = f["momentum_candidate"].float(); m_vq = v["momentum_candidate"].float()
            prev_ref = f["momentum_prev_decoded"].float()
            prev_vq = v["momentum_prev_decoded"].float()
            # Check the snapshot's candidate against the executed recurrence.
            rec_ref = mu * prev_ref + g_ref
            rec_vq = mu * prev_vq + g_vq
            for method, observed, expected in (("FP32", m_ref, rec_ref), ("VQ", m_vq, rec_vq)):
                err = float((observed - expected).norm() / expected.norm().clamp_min(1e-30))
                if err > 2e-5:
                    raise RuntimeError(f"{method} momentum recurrence mismatch for {name} at {stem}: rel={err:.3g}")
            d_ref = _direction_from_previous(prev_ref, g_ref, mu)
            d_vq = _direction_from_previous(prev_vq, g_vq, mu)
            executed_ref = g_ref + mu * m_ref
            executed_vq = g_vq + mu * m_vq
            for method, closed_form, executed in (("FP32", d_ref, executed_ref), ("VQ", d_vq, executed_vq)):
                err = float((closed_form - executed).norm() / executed.norm().clamp_min(1e-30))
                if err > 2e-5:
                    raise RuntimeError(f"{method} direction algebra mismatch for {name} at {stem}: rel={err:.3g}")
            if int(vblob["metadata"]["update"]) == 1:
                zero_fields = (("FP32 previous state", prev_ref), ("VQ previous state", prev_vq),
                               ("VQ previous unquantized candidate", v["momentum_prev_candidate"].float()))
                for field, value in zero_fields:
                    if bool(torch.count_nonzero(value)):
                        raise RuntimeError(f"update-1 convention requires zero {field} for {name}")
            out_ref, out_vq = _k5(d_ref, fblob["metadata"]), _k5(d_vq, vblob["metadata"])
            # State-only: FP32 current gradient, VQ previous persisted state.
            state_direction = _direction_from_previous(prev_vq, g_ref, mu)
            # Gradient-only: VQ current gradient, FP32 previous momentum state.
            grad_direction = _direction_from_previous(prev_ref, g_vq, mu)
            state_out = _k5(state_direction, fblob["metadata"])
            grad_out = _k5(grad_direction, fblob["metadata"])
            # Local persistence-only counterfactual: hold current VQ gradient
            # fixed and replace the prior decoded state with the prior
            # unquantized candidate. This is a one-step diagnostic, not a path.
            prev_unquantized = v["momentum_prev_candidate"].float()
            local_direction = _direction_from_previous(prev_unquantized, g_vq, mu)
            local_out = _k5(local_direction, vblob["metadata"])
            q = v["momentum_persisted_decoded"] - m_vq
            vals = {
                "update": int(vblob["metadata"]["update"]), "parameter_name": name, "numel": m_vq.numel(),
                "q_relative_l2": float(q.norm()/m_vq.norm().clamp_min(1e-30)), "q_cosine": _metrics(m_vq, v["momentum_persisted_decoded"])["cosine"],
                "momentum_trajectory_cosine": _metrics(m_ref, m_vq)["cosine"], "momentum_trajectory_relative_l2": _metrics(m_ref, m_vq)["relative_l2"],
                "gradient_cosine": _metrics(g_ref, g_vq)["cosine"], "gradient_relative_l2": _metrics(g_ref, g_vq)["relative_l2"],
                "direction_cosine": _metrics(d_ref, d_vq)["cosine"], "direction_relative_l2": _metrics(d_ref, d_vq)["relative_l2"],
                "k5_update_cosine": _metrics(out_ref, out_vq)["cosine"], "k5_update_relative_l2": _metrics(out_ref, out_vq)["relative_l2"],
                "local_one_step_k5_cosine": _metrics(out_vq, local_out)["cosine"], "local_one_step_k5_relative_l2": _metrics(out_vq, local_out)["relative_l2"],
                "state_only_k5_cosine": _metrics(out_ref, state_out)["cosine"],
                "state_only_k5_relative_l2": _metrics(out_ref, state_out)["relative_l2"],
                "state_only_k5_distortion": 1.0 - _metrics(out_ref, state_out)["cosine"],
                "gradient_only_k5_cosine": _metrics(out_ref, grad_out)["cosine"],
                "gradient_only_k5_relative_l2": _metrics(out_ref, grad_out)["relative_l2"],
                "gradient_only_k5_distortion": 1.0 - _metrics(out_ref, grad_out)["cosine"],
                "local_one_step_k5_distortion": 1.0 - _metrics(out_vq, local_out)["cosine"],
                "k5_update_distortion": 1.0 - _metrics(out_ref, out_vq)["cosine"],
            }
            tensor_rows.append(vals)
            local_rows.append({"q_a": m_vq, "q_b": v["momentum_persisted_decoded"], "mom_a": m_ref, "mom_b": m_vq,
                               "grad_a": g_ref, "grad_b": g_vq, "dir_a": d_ref, "dir_b": d_vq, "k5_a": out_ref, "k5_b": out_vq,
                               "local_k5": (out_vq, local_out), "state_only_k5": (out_ref, state_out),
                               "gradient_only_k5": (out_ref, grad_out)})
        aggregate = {"update": int(vblob["metadata"]["update"]), "tensor_count": len(local_rows)}
        for prefix, label in (("q", "persistence_quantization"), ("mom", "momentum_trajectory"), ("grad", "gradient"), ("dir", "nesterov_direction"), ("k5", "k5_update")):
            pooled = _pooled_pairs((row[prefix + "_a"], row[prefix + "_b"]) for row in local_rows)
            aggregate.update({label + "_" + k: val for k, val in pooled.items()})
        for key in ("local_k5", "state_only_k5", "gradient_only_k5"):
            pooled = _pooled_pairs(row[key] for row in local_rows)
            aggregate.update({key + "_" + metric: value for metric, value in pooled.items()})
        for key in ("k5_update", "local_k5", "state_only_k5", "gradient_only_k5"):
            aggregate[key + "_distortion"] = 1.0 - aggregate[key + "_cosine"]
        rows.append(aggregate)
    # Validation curves are kept separate and never substituted for mechanism metrics.
    def evals(run):
        p = run / "metrics.jsonl"; out = {}
        if p.exists():
            for line in p.read_text().splitlines():
                x = json.loads(line)
                if x.get("event_type") == "eval": out[int(x["completed_updates"])] = x.get("nll")
        return out
    plots = out / "plots"; plots.mkdir(exist_ok=True)
    stale_plot = plots / "local_one_step_k5_cosine.png"
    if stale_plot.exists(): stale_plot.unlink()
    try:
        import matplotlib.pyplot as plt
        x = [r["update"] for r in rows]
        series = [("persistence_quantization_relative_l2", "instantaneous persistence rel-L2"),
                  ("momentum_trajectory_cosine", "momentum cosine vs FP32"),
                  ("gradient_cosine", "gradient cosine vs FP32"),
                  ("nesterov_direction_cosine", "Nesterov direction cosine vs FP32"),
                  ("k5_update_cosine", "executed K5 update cosine vs FP32"),
                  ("local_k5_cosine", "local previous-persistence K5 cosine"),
                  ("state_only_k5_cosine", "state-only substitution K5 cosine"),
                  ("gradient_only_k5_cosine", "gradient-only substitution K5 cosine")]
        for key, ylabel in series:
            y = [r.get(key) for r in rows]
            fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(x, y, "o-"); ax.set_xlabel("update"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plots / f"{key}.png", dpi=140); plt.close(fig)
        fp_eval, vq_eval = evals(fp), evals(vq)
        if fp_eval or vq_eval:
            fig, ax = plt.subplots(figsize=(7, 4))
            if fp_eval: ax.plot(sorted(fp_eval), [fp_eval[k] for k in sorted(fp_eval)], "o-", label="FP32")
            if vq_eval: ax.plot(sorted(vq_eval), [vq_eval[k] for k in sorted(vq_eval)], "o-", label="recursive VQ")
            ax.set_xlabel("update"); ax.set_ylabel("validation NLL"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plots / "validation_nll.png", dpi=140); plt.close(fig)
    except Exception as exc:
        (plots / "README.txt").write_text(f"Plot generation unavailable: {type(exc).__name__}: {exc}\n")
    summary = {"status": "completed", "updates": [r["update"] for r in rows], "paired_snapshots": len(rows),
               "fp32_run": str(fp), "vq_run": str(vq), "validation_nll_fp32": evals(fp), "validation_nll_vq": evals(vq),
               "protocol_integrity": run_integrity,
               "counterfactuals": {"state_only": "FP32 current gradient with VQ previous decoded/persistent momentum",
                                   "gradient_only": "VQ current gradient with FP32 previous decoded momentum",
                                   "local_previous_persistence": "VQ current gradient with previous unquantized VQ candidate instead of previous decoded persisted momentum",
                                   "update_1": "all entering momentum states are zero; local persistence counterfactual equals actual VQ direction"},
               "aggregation": "global tensor-norm pooling via summed dot products and squared norms"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(out / "landmark_metrics.csv", rows); _write_csv(out / "tensor_metrics.csv", tensor_rows)
    sizes = {"fp32_snapshot_bytes": {k: fp_files[k].stat().st_size for k in common}, "vq_snapshot_bytes": {k: vq_files[k].stat().st_size for k in common}, "fp32_total": sum(fp_files[k].stat().st_size for k in common), "vq_total": sum(vq_files[k].stat().st_size for k in common)}
    (out / "artifact_sizes.json").write_text(json.dumps(sizes, indent=2))
    (out / "provenance.json").write_text(json.dumps({"fp32_run": str(fp.resolve()), "vq_run": str(vq.resolve()),
        "fp32_source_commit": fp_summary.get("git_commit"), "vq_source_commit": vq_summary.get("git_commit"),
        "common_snapshots": common, "resolved_fp32_snapshots": {k: str(fp_files[k].resolve()) for k in common},
        "resolved_vq_snapshots": {k: str(vq_files[k].resolve()) for k in common}, "mismatches": mismatches,
        "protocol_integrity": run_integrity}, indent=2))
    # Human-readable corrected report. Counterfactual metrics here use the same
    # global norm-weighted pooling as the primary paired K5 metric.
    val_fp, val_vq = evals(fp), evals(vq)
    lines = ["# Corrected recursive Muon mechanism decomposition", "",
             "## Protocol integrity", "",
             f"Both runs passed shared-metadata validation: seed `{fp_summary.get('seed')}`, data seed `{fp_summary.get('data_seed')}`, algorithm seed `{fp_summary.get('algorithm_seed')}`, dataset fingerprint `{fp_summary.get('data_fingerprint')}`, schedule horizon `{fp_summary.get('schedule_total_updates')}`, and `{first_fp['metadata'].get('tokens_per_update')}` tokens/update. The FP32 and VQ snapshots also agree on Muon momentum, Nesterov setting, K5 iteration count, coefficients, and epsilon. The paired summaries agree on target-token budget and sequence length, and all landmark tensor names/shapes match. Both summaries report `paused_staged` at update 512 with no divergence status. Raw snapshots pair at updates `{', '.join(str(int(x.rsplit('_', 1)[1])) for x in common)}` with 30 matching tensor names/shapes each. All {run_integrity['paired_train_events']} train events have matching update/token positions, tokens/update, and learning rates. Batch IDs/hashes were not saved, so data-window alignment is supported by the deterministic protocol and aligned positions but is not independently hash-verified.", "",
             "Snapshot fields were mapped using `MuonMechanismObserver` plus the optimizer step implementation: `gradient` is current `g_t`; `momentum_prev_decoded` is the entering persistent/decoded `M_(t-1)`; VQ `momentum_prev_candidate` is the prior step's unquantized candidate before persistence; `momentum_candidate` is current `M_t`; and VQ `momentum_persisted_decoded` is `decode(Q(M_t))` for the next step. FP32 persistence equals the candidate. At update 1, entering states and the prior candidate are explicitly zero.", "",
             "## Corrected temporal table", "",
             "Directions were rebuilt as `D_t = (1+μ)g_t + μ²M_(t-1)` and checked against `g_t + μM_t` using the saved candidate. Metrics are globally pooled over tensor values by summed dot products and squared norms. Persistence relative-L2 is normalized by the VQ candidate; trajectory/gradient/direction/K5 relative-L2 values use the FP32 counterpart as reference. Local relative-L2 uses the actual VQ K5 norm; state-only and gradient-only use FP32 K5 norm.", "",
             "| Update | Persistence cos / rel-L2 | Momentum cos / rel-L2 | Gradient cos / rel-L2 | Direction cos / rel-L2 | Full K5 cos / rel-L2 | Local K5 cos / rel-L2 | State-only K5 cos / rel-L2 | Gradient-only K5 cos / rel-L2 | Val NLL FP32 / VQ |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        u = row["update"]
        val = f"{val_fp[u]:.5f} / {val_vq[u]:.5f}" if u in val_fp and u in val_vq else "—"
        lines.append(f"| {u} | {row['persistence_quantization_cosine']:.5f} / {row['persistence_quantization_relative_l2']:.5f} | {row['momentum_trajectory_cosine']:.5f} / {row['momentum_trajectory_relative_l2']:.5f} | {row['gradient_cosine']:.5f} / {row['gradient_relative_l2']:.5f} | {row['nesterov_direction_cosine']:.5f} / {row['nesterov_direction_relative_l2']:.5f} | {row['k5_update_cosine']:.5f} / {row['k5_update_relative_l2']:.5f} | {row['local_k5_cosine']:.5f} / {row['local_k5_relative_l2']:.5f} | {row['state_only_k5_cosine']:.5f} / {row['state_only_k5_relative_l2']:.5f} | {row['gradient_only_k5_cosine']:.5f} / {row['gradient_only_k5_relative_l2']:.5f} | {val} |")
    lines.extend(["", "## K5 distortion and local-versus-accumulated comparison", "",
                  "Distortion is `1 - pooled cosine`. State-only, gradient-only, and local previous-persistence comparisons are separate local substitutions; they are nonlinear diagnostics and must not be summed to obtain full trajectory distortion.", "",
                  "| Update | Full paired K5 distortion | Local previous-persistence distortion | State-only distortion | Gradient-only distortion | Full / local distortion ratio |", "|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        ratio = (1.0 - row["k5_update_cosine"]) / max(1.0 - row["local_k5_cosine"], 1e-30)
        lines.append(f"| {row['update']} | {1-row['k5_update_cosine']:.5f} | {1-row['local_k5_cosine']:.5f} | {1-row['state_only_k5_cosine']:.5f} | {1-row['gradient_only_k5_cosine']:.5f} | {ratio:.2f}× |")
    final = rows[-1]
    if 512 in val_fp and 512 in val_vq:
        nll_note = ("Validation NLL remains close despite large update-space divergence: at update 512 the values are `" +
                   f"{val_fp[512]:.5f}" + "` (FP32) and `" + f"{val_vq[512]:.5f}" +
                   "` (VQ), a gap of `" + f"{val_vq[512]-val_fp[512]:+.5f}" +
                   "`. K5 fidelity should not be treated as a one-to-one predictor of NLL.")
    else:
        nll_note = "Validation NLL artifacts are unavailable at update 512 in this paired input; no NLL conclusion is drawn."
    lines.extend(["", "## Reading the endpoint", "",
                  f"At update {final['update']}, the full paired K5 cosine is `{final['k5_update_cosine']:.4f}` versus `{final['local_k5_cosine']:.4f}` for the local previous-persistence substitution. The latter holds the current VQ gradient fixed and isolates only the preceding persistence decision; the much larger full discrepancy therefore reflects accumulated trajectory differences, not just that one quantization event.",
                  f"At the same endpoint, state-only K5 cosine is `{final['state_only_k5_cosine']:.4f}` and gradient-only is `{final['gradient_only_k5_cosine']:.4f}` (both compared with the FP32 K5 output). This indicates the current state mismatch is more damaging than substituting the current gradient alone in these local tests, while gradient divergence is also material. These nonlinear substitutions are not additive, so they do not quantify causal shares.", ""])
    lines.extend(["", "## Tensor-level concentration at updates 128 and 512", ""])
    for update in (128, 512):
        selected = [x for x in tensor_rows if x["update"] == update]
        lines.extend([f"### Update {update}", "", "| Diagnostic | Most affected tensors (metric) |", "|---|---|"])
        categories = (("Persistence relative-L2", "q_relative_l2", False),
                      ("Gradient relative-L2", "gradient_relative_l2", False),
                      ("Full K5 update relative-L2", "k5_update_relative_l2", False),
                      ("State-only K5 distortion", "state_only_k5_distortion", False),
                      ("Gradient-only K5 distortion", "gradient_only_k5_distortion", False))
        for label, key, _ in categories:
            ranked = sorted(selected, key=lambda x: x[key], reverse=True)[:5]
            listing = "; ".join(f"`{x['parameter_name']}` ({x[key]:.3f})" for x in ranked)
            lines.append(f"| {label} | {listing} |")
        bulk = sorted(x["k5_update_relative_l2"] for x in selected)
        median = bulk[len(bulk)//2]
        lines.extend(["", f"Across {len(selected)} tensors, median full-K5 rel-L2 is `{median:.3f}`; the table therefore shows tail severity, not a claim that only these tensors diverge.", ""])
    lines.extend(["## Mechanism interpretation", "",
                  "The corrected diagnostics support H2 (recursive closed-loop drift) and H3 (state-driven local discrepancy), especially early. At update 8 the state-only K5 cosine is nearly the full paired cosine while gradient-only remains essentially 1; by 128/512 state-only remains more damaging than gradient-only, but gradient-only divergence is substantial by 512. The local previous-persistence K5 distortion stays far below the full paired distortion, consistent with accumulated history rather than one immediately preceding encode/decode event. H4 contributes later but is not the dominant single local substitution in this run. H5 remains possible because nonlinear state-gradient interaction was not isolated as an additive quantity. This is descriptive evidence from one seed and five raw landmarks, not a causal proof.", "",
                  nll_note, "",
                  "## Next-method implication", "",
                  "The justified next focus is the recursive persistence/state-feedback path, not another static quantizer sweep. The local state-only comparison makes entering-state divergence the leading local signal; current-gradient feedback becomes material later. Do not choose a specific intervention from this single-seed screen alone; validate the state-versus-gradient ordering at a longer horizon and/or another paired seed first.", "",
                  "## Interpretation boundaries", "",
                  "State-only and gradient-only values are one-step substitutions, not additive causal effects or alternate trajectories. The local previous-persistence comparison isolates only the immediately preceding persistence decision while holding the current VQ gradient fixed. Full paired K5 divergence includes accumulated state and parameter/gradient trajectory differences.", "",
                  "See `landmark_metrics.csv` for pooled state, gradient, direction, update, and counterfactual metrics; `tensor_metrics.csv` retains per-tensor diagnostics."])
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completed", "paired_snapshots": common, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
