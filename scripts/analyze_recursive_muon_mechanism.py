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
    return {"cosine": float((a.reshape(-1) @ b.reshape(-1)) / (an * bn).clamp_min(1e-30)),
            "relative_l2": float(d.norm() / an.clamp_min(1e-30)),
            "norm_ratio": float(bn / an.clamp_min(1e-30))}


def _pooled(rows, prefix):
    # Rows contain raw tensors only transiently.  Aggregate by squared norms.
    aa = bb = dd = dot = 0.0
    for row in rows:
        a, b = row[prefix + "_a"], row[prefix + "_b"]
        aa += float(a.square().sum()); bb += float(b.square().sum()); dd += float((a-b).square().sum()); dot += float((a*b).sum())
    return {prefix + "_cosine": dot / max(math.sqrt(aa*bb), 1e-30), prefix + "_relative_l2": math.sqrt(dd/max(aa, 1e-30)), prefix + "_norm_ratio": math.sqrt(bb/max(aa, 1e-30))}


def validate_pair_metadata(fp_meta: dict, vq_meta: dict, *, expected_seed=1, expected_schedule=4096):
    # Recipe fingerprints intentionally differ because one run is reference
    # Muon and the other owns recursive VQ state.  Compare shared scientific
    # protocol fields instead, while retaining each recipe fingerprint as
    # provenance.
    keys = ("seed", "data_seed", "algorithm_seed", "protocol_id", "data_fingerprint", "schedule_total_updates", "tokens_per_update", "sequence_length")
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


def _write_csv(path, rows):
    if not rows:
        path.write_text("status,reason\nno_rows,missing_or_invalid_snapshots\n"); return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-run", required=True)
    ap.add_argument("--vq-run", required=True)
    ap.add_argument("--out", default="reports/recursive_muon_mechanism_s1_512")
    args = ap.parse_args(); fp = Path(args.fp32_run); vq = Path(args.vq_run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fp_files = {p.stem: p for p in (fp / "mechanism").glob("update_*.pt")}; vq_files = {p.stem: p for p in (vq / "mechanism").glob("update_*.pt")}
    common = sorted(set(fp_files) & set(vq_files), key=lambda x: int(x.split("_")[-1]))
    if not common: raise RuntimeError("no paired mechanism snapshots found")
    first_fp, first_vq = _load(fp_files[common[0]]), _load(vq_files[common[0]])
    mismatches = validate_pair_metadata(first_fp["metadata"], first_vq["metadata"])
    if mismatches: raise RuntimeError(json.dumps({"paired_protocol_mismatch": mismatches}, indent=2))
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
            g_ref = f["gradient"]; g_vq = v["gradient"]; m_ref = f["momentum_candidate"]; m_vq = v["momentum_candidate"]
            d_ref = g_ref + mu * m_ref; d_vq = g_vq + mu * m_vq
            out_ref, out_vq = _k5(d_ref, fblob["metadata"]), _k5(d_vq, vblob["metadata"])
            # Local previous-persistence counterfactual: same current VQ g,
            # replace only the decoded previous state by prior unquantized M.
            prev_c = v["momentum_prev_candidate"]; local_candidate = mu * prev_c + g_vq
            local_direction = g_vq + mu * local_candidate
            local_out = _k5(local_direction, vblob["metadata"])
            state_direction = g_ref + mu * m_vq; grad_direction = g_vq + mu * m_ref
            state_out, grad_out = _k5(state_direction, fblob["metadata"]), _k5(grad_direction, fblob["metadata"])
            q = v["momentum_persisted_decoded"] - m_vq
            vals = {
                "update": int(vblob["metadata"]["update"]), "parameter_name": name, "numel": m_vq.numel(),
                "q_relative_l2": float(q.norm()/m_vq.norm().clamp_min(1e-30)), "q_cosine": _metrics(m_vq, v["momentum_persisted_decoded"])["cosine"],
                "momentum_trajectory_cosine": _metrics(m_ref, m_vq)["cosine"], "momentum_trajectory_relative_l2": _metrics(m_ref, m_vq)["relative_l2"],
                "gradient_cosine": _metrics(g_ref, g_vq)["cosine"], "gradient_relative_l2": _metrics(g_ref, g_vq)["relative_l2"],
                "direction_cosine": _metrics(d_ref, d_vq)["cosine"], "direction_relative_l2": _metrics(d_ref, d_vq)["relative_l2"],
                "k5_update_cosine": _metrics(out_ref, out_vq)["cosine"], "k5_update_relative_l2": _metrics(out_ref, out_vq)["relative_l2"],
                "local_one_step_k5_cosine": _metrics(out_vq, local_out)["cosine"], "local_one_step_k5_relative_l2": _metrics(out_vq, local_out)["relative_l2"],
                "state_only_k5_cosine": _metrics(out_ref, state_out)["cosine"], "gradient_only_k5_cosine": _metrics(out_ref, grad_out)["cosine"],
            }
            tensor_rows.append(vals)
            local_rows.append({"q_a": m_vq, "q_b": v["momentum_persisted_decoded"], "mom_a": m_ref, "mom_b": m_vq,
                               "grad_a": g_ref, "grad_b": g_vq, "dir_a": d_ref, "dir_b": d_vq, "k5_a": out_ref, "k5_b": out_vq})
        aggregate = {"update": int(vblob["metadata"]["update"]), "tensor_count": len(local_rows)}
        for prefix, label in (("q", "persistence_quantization"), ("mom", "momentum_trajectory"), ("grad", "gradient"), ("dir", "nesterov_direction"), ("k5", "k5_update")):
            aggregate.update({label + "_" + k: val for k, val in _pooled(local_rows, prefix).items() if k.startswith(prefix + "_")})
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
    try:
        import matplotlib.pyplot as plt
        x = [r["update"] for r in rows]
        series = [("persistence_quantization_relative_l2", "instantaneous persistence rel-L2"),
                  ("momentum_trajectory_cosine", "momentum cosine vs FP32"),
                  ("gradient_cosine", "gradient cosine vs FP32"),
                  ("nesterov_direction_cosine", "Nesterov direction cosine vs FP32"),
                  ("k5_update_cosine", "executed K5 update cosine vs FP32"),
                  ("local_one_step_k5_cosine", "local previous-persistence K5 cosine")]
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
    (out / "summary.json").write_text(json.dumps({"status": "completed", "updates": [r["update"] for r in rows], "paired_snapshots": len(rows), "fp32_run": str(fp), "vq_run": str(vq), "validation_nll_fp32": evals(fp), "validation_nll_vq": evals(vq)}, indent=2))
    _write_csv(out / "landmark_metrics.csv", rows); _write_csv(out / "tensor_metrics.csv", tensor_rows)
    sizes = {"fp32_snapshot_bytes": {k: fp_files[k].stat().st_size for k in common}, "vq_snapshot_bytes": {k: vq_files[k].stat().st_size for k in common}, "fp32_total": sum(fp_files[k].stat().st_size for k in common), "vq_total": sum(vq_files[k].stat().st_size for k in common)}
    (out / "artifact_sizes.json").write_text(json.dumps(sizes, indent=2))
    (out / "provenance.json").write_text(json.dumps({"fp32_run": str(fp), "vq_run": str(vq), "common_snapshots": common, "mismatches": mismatches}, indent=2))
    (out / "comparison.md").write_text("""# Recursive Muon mechanism decomposition

This report separates the immediate persistence error `decode(Q(M_candidate))-M_candidate` from the paired trajectory metrics. The local K5 column changes only the immediately preceding persisted state while holding the current VQ gradient fixed; it is not a counterfactual trajectory. Full VQ-vs-FP32 direction/update metrics include historical state error and parameter/gradient feedback.

The report intentionally does not claim causality from endpoint disagreement alone. Interpret temporal ordering across `landmark_metrics.csv`, and compare local one-step K5 distortion against full paired K5 update distortion.
""")
    print(json.dumps({"status": "completed", "paired_snapshots": common, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
