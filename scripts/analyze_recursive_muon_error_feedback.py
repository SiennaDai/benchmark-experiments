#!/usr/bin/env python3
"""Compare existing FP32/alpha=0 mechanism runs with error-feedback VQ runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

try:
    from .analyze_recursive_muon_mechanism import (
        _k5, _direction_from_previous, _load, _metrics, _pooled_pairs, _read_json,
    )
except ImportError:  # direct execution as scripts/analyze_recursive_muon_error_feedback.py
    from analyze_recursive_muon_mechanism import (  # type: ignore
        _k5, _direction_from_previous, _load, _metrics, _pooled_pairs, _read_json,
    )

LANDMARKS = [1, 8, 32, 128, 512]
SHARED = ("seed", "data_seed", "algorithm_seed", "protocol_id", "data_fingerprint",
          "schedule_total_updates", "total_updates", "target_tokens", "sequence_length",
          "compute_precision")


def _events(run, event_type):
    path = run / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip() and json.loads(line).get("event_type") == event_type]


def _summary(run):
    value = _read_json(run / "summary.json")
    if value is None:
        raise ValueError(f"missing summary.json: {run}")
    return value


def _snapshots(run):
    found = {int(path.stem.rsplit("_", 1)[1]): path
             for path in (run / "mechanism").glob("update_*.pt")}
    if sorted(found) != LANDMARKS:
        raise ValueError(f"{run}: expected raw landmarks {LANDMARKS}; found {sorted(found)}")
    return {step: _load(found[step]) for step in LANDMARKS}


def _validate(runs):
    summaries = {key: _summary(path) for key, path in runs.items()}
    baseline = summaries["fp32"]
    configs = {}
    for key, summary in summaries.items():
        mismatch = {field: (baseline.get(field), summary.get(field)) for field in SHARED
                    if baseline.get(field) != summary.get(field)}
        if mismatch:
            raise ValueError(f"protocol mismatch {key}: {mismatch}")
        if summary.get("completed_updates") != 512 or summary.get("status") != "paused_staged":
            raise ValueError(f"{key} did not stop cleanly at update 512")
        if summary.get("seed") != 1 or summary.get("data_seed") != 1337 or summary.get("algorithm_seed") != 2026:
            raise ValueError(f"{key} seed protocol mismatch")
        if summary.get("schedule_total_updates") != 4096:
            raise ValueError(f"{key} does not retain the 4096 update schedule")
        cfg_path = runs[key] / "resolved_config.json"
        if not cfg_path.exists():
            raise ValueError(f"missing resolved_config.json for {key}")
        configs[key] = json.loads(cfg_path.read_text())
    for field in ("model", "data", "train", "schedule", "precision", "eval"):
        values = {key: cfg[field] for key, cfg in configs.items()}
        if any(value != values["fp32"] for value in values.values()):
            raise ValueError(f"resolved recipe mismatch in {field}: {values}")
    common_muon = ("lr", "betas", "eps", "weight_decay", "fused", "foreach", "state_simulation",
                   "muon_momentum", "muon_nesterov", "muon_ns_steps", "muon_ns_coefficients", "muon_eps")
    ref_optimizer = configs["fp32"]["optimizer"]
    alpha0_optimizer = configs["alpha0"]["optimizer"]
    if any(ref_optimizer.get(field) != alpha0_optimizer.get(field) for field in common_muon):
        raise ValueError("FP32 reference and alpha=0 Muon hyperparameters differ")
    for key in ("alpha0", "alpha05", "alpha1"):
        opt = configs[key]["optimizer"]
        if opt.get("name") != "recursive_muon":
            raise ValueError(f"{key} must use recursive_muon")
        if any(opt.get(field) != configs["alpha0"]["optimizer"].get(field) for field in common_muon):
            raise ValueError(f"Muon optimizer hyperparameters differ for {key}")
        expected_alpha = {"alpha0": 0.0, "alpha05": .5, "alpha1": 1.0}[key]
        if float(opt.get("recursive_error_feedback_alpha", 0.0)) != expected_alpha:
            raise ValueError(f"{key} configured alpha does not match {expected_alpha}")
        expected_vq = {"recursive_rank": 8, "recursive_block_size": 2048,
                       "recursive_factor_dtype": "bf16", "recursive_structure_mode": "exact_svd_oracle",
                       "recursive_representation": "vq_int3", "recursive_codebook_key": "s0_k8_w64_t8_v1200"}
        for field, value in expected_vq.items():
            if opt.get(field) != value:
                raise ValueError(f"{key} {field} mismatch: expected {value}, got {opt.get(field)}")
    snapshots = {key: _snapshots(path) for key, path in runs.items()}
    names = None
    for key, updates in snapshots.items():
        for step, blob in updates.items():
            meta = blob["metadata"]
            if meta.get("seed") != 1 or meta.get("schedule_total_updates") != 4096:
                raise ValueError(f"snapshot metadata mismatch for {key} at update {step}")
            expected_alpha = {"fp32": 0.0, "alpha0": 0.0, "alpha05": .5, "alpha1": 1.0}[key]
            if float(meta.get("error_feedback_alpha", 0.0)) != expected_alpha:
                raise ValueError(f"snapshot alpha mismatch for {key} at update {step}")
            current = {item["name"]: tuple(item["shape"]) for item in blob["tensors"]}
            if names is None and key == "fp32" and step == 1:
                names = current
            if current != names:
                raise ValueError(f"tensor identities/shapes mismatch for {key} at update {step}")
    # Verify shared training windows and scheduler positions from event streams.
    ref = _events(runs["fp32"], "train")
    for key, path in runs.items():
        ev = _events(path, "train")
        if len(ev) < 512 or len(ref) < 512:
            raise ValueError(f"{key}: fewer than 512 train events")
        for a, b in zip(ref[:512], ev[:512]):
            for field in ("completed_updates", "processed_target_tokens", "tokens_this_update", "lr"):
                if a.get(field) != b.get(field):
                    raise ValueError(f"unpaired train event {key}, update {a.get('completed_updates')}, field={field}")
    # Require the frozen VQ calibration identity to be shared across all VQ runs.
    codebook_keys = {}
    for key in ("alpha0", "alpha05", "alpha1"):
        cfg = configs[key]
        codebook_keys[key] = cfg["optimizer"].get("recursive_codebook_key")
    if len(set(codebook_keys.values())) != 1 or codebook_keys["alpha0"] != "s0_k8_w64_t8_v1200":
        raise ValueError(f"VQ codebook mismatch: {codebook_keys}")
    codebook_paths = {configs[key]["optimizer"].get("recursive_codebook_path")
                      for key in ("alpha0", "alpha05", "alpha1")}
    if len(codebook_paths) != 1:
        raise ValueError(f"VQ codebook paths differ: {codebook_paths}")
    return summaries, snapshots, {"status": "compatible", "shared_fields": list(SHARED),
                                  "paired_train_events": 512, "landmarks": LANDMARKS,
                                  "batch_identity": "same seed/data seed, deterministic protocol, update/token/LR event alignment; batch IDs are not persisted",
                                  "vq_codebook_keys": codebook_keys,
                                  "vq_codebook_paths": sorted(codebook_paths),
                                  "resolved_config_protocol_match": True}


def _write_csv(path, rows):
    if not rows:
        path.write_text("status\nno_rows\n")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def analyze(runs, out):
    summaries, snapshots, integrity = _validate(runs)
    out.mkdir(parents=True, exist_ok=True)
    alphas = {"alpha0": 0.0, "alpha05": .5, "alpha1": 1.0}
    all_rows, error_rows, tensor_rows = [], [], []
    for step in LANDMARKS:
        blobs = {key: snapshots[key][step] for key in runs}
        meta = blobs["fp32"]["metadata"]
        mu = float(meta["muon_momentum"])
        maps = {key: {item["name"]: item for item in blob["tensors"]} for key, blob in blobs.items()}
        names = sorted(maps["fp32"])
        per_method = {key: [] for key in runs if key != "fp32"}
        error_detail = {key: [] for key in alphas}
        for name in names:
            f = maps["fp32"][name]
            g_ref = f["gradient"].float(); prev_ref = f["momentum_prev_decoded"].float()
            d_ref = _direction_from_previous(prev_ref, g_ref, mu)
            out_ref = _k5(d_ref, meta)
            for key in alphas:
                v = maps[key][name]
                if v["shape"] != f["shape"]:
                    raise ValueError(f"shape mismatch {name} {key} update {step}")
                grad = v["gradient"].float(); prev = v["momentum_prev_decoded"].float()
                candidate = v["momentum_candidate"].float()
                persisted = v["momentum_persisted_decoded"].float()
                prev_candidate = v["momentum_prev_candidate"].float()
                recurrence = mu * prev + grad
                if alphas[key]:
                    # e_(t-1) is exactly the previous unquantized candidate
                    # minus the state that was decoded at this step.
                    e_prev = prev_candidate - prev
                    recurrence = recurrence + (alphas[key] * mu) * e_prev
                elif key == "alpha0":
                    e_prev = prev_candidate - prev
                else:
                    e_prev = torch.zeros_like(prev)
                previous_effective = prev + alphas[key] * e_prev
                rec_err = float((candidate - recurrence).norm() / recurrence.norm().clamp_min(1e-30))
                if rec_err > 3e-5:
                    raise ValueError(f"momentum recurrence mismatch {key}/{name}/{step}: {rec_err}")
                d = _direction_from_previous(previous_effective, grad, mu)
                d_from_candidate = grad + mu * candidate
                if float((d - d_from_candidate).norm() / d_from_candidate.norm().clamp_min(1e-30)) > 3e-5:
                    raise ValueError(f"Nesterov algebra mismatch {key}/{name}/{step}")
                k5_out = _k5(d, blobs[key]["metadata"])
                state_dir = _direction_from_previous(previous_effective, g_ref, mu)
                grad_dir = _direction_from_previous(prev_ref, grad, mu)
                state_out = _k5(state_dir, meta)
                grad_out = _k5(grad_dir, meta)
                # Local counterfactual removes only the previous residual
                # injection and holds this step's gradient fixed.
                # The full-state run used prev + alpha*e_prev as the recurrence
                # input; this counterfactual restores prev only.
                local_no_error_dir = _direction_from_previous(prev, grad, mu)
                local_out = _k5(local_no_error_dir, blobs[key]["metadata"])
                q = candidate - persisted
                local_pairs = {
                    "momentum": (maps["fp32"][name]["momentum_candidate"].float(), candidate),
                    "gradient": (g_ref, grad), "direction": (d_ref, d), "k5": (out_ref, k5_out),
                    "state_only_k5": (out_ref, state_out),
                    "gradient_only_k5": (out_ref, grad_out),
                    "local_previous_persistence_k5": (k5_out, local_out)
                }
                metrics = {metric: _metrics(a, b) for metric, (a, b) in local_pairs.items()}
                tensor = {"update": step, "method": key, "alpha": alphas[key], "parameter_name": name,
                          "numel": candidate.numel(), "persistence_rel_l2": float(q.norm()/candidate.norm().clamp_min(1e-30)),
                          "error_buffer_prev_rel_l2": float(e_prev.norm()/prev_candidate.norm().clamp_min(1e-30)) if prev_candidate.numel() else 0.0,
                          "recurrence_relative_error": rec_err}
                for label, values in metrics.items():
                    tensor[f"{label}_cosine"] = values["cosine"]
                    tensor[f"{label}_relative_l2"] = values["relative_l2"]
                tensor_rows.append(tensor)
                per_method[key].append((name, candidate, grad, d, k5_out, state_out, grad_out, local_pairs, q, e_prev, prev_candidate))
                error_detail[key].append({"name": name, "candidate": candidate, "persisted": persisted,
                                          "q": q, "e_prev": e_prev, "prev_candidate": prev_candidate,
                                          "prev_decoded": prev, "gradient": grad})
        for key, entries in per_method.items():
            def pool(pair_index):
                return _pooled_pairs([(entry[7][pair_index][0], entry[7][pair_index][1]) for entry in entries])
            # Global-norm pooling across all matrix entries, matching the
            # canonical checkpoint analysis.
            base = {"update": step, "method": key, "alpha": alphas[key], "tensor_count": len(entries)}
            for metric in ("momentum", "gradient", "direction", "k5", "state_only_k5", "gradient_only_k5", "local_previous_persistence_k5"):
                values = pool(metric)
                for stat, val in values.items():
                    base[f"{metric}_{stat}"] = val
                if metric.endswith("k5"):
                    base[f"{metric}_distortion"] = 1.0 - values["cosine"]
            all_rows.append(base)
            detail = error_detail[key]
            cand_sq = sum(float(x["candidate"].square().sum()) for x in detail)
            q_sq = sum(float(x["q"].square().sum()) for x in detail)
            eprev_sq = sum(float(x["e_prev"].square().sum()) for x in detail)
            prev_c_sq = sum(float(x["prev_candidate"].square().sum()) for x in detail)
            reconstructed = sum(float((x["prev_decoded"] + x["e_prev"] - x["prev_candidate"]).square().sum()) for x in detail)
            observer_metrics = {}
            metrics_path = runs[key] / "mechanism" / "metrics.jsonl"
            if metrics_path.exists():
                observer_metrics = {int(event["update"]): event for event in
                    (json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip())}
            observed = observer_metrics.get(step, {})
            error_detail_row = {"update": step, "method": key, "alpha": alphas[key],
                                "instantaneous_persistence_rel_l2": math.sqrt(q_sq/max(cand_sq, 1e-30)),
                                "current_error_buffer_norm_ratio": math.sqrt(q_sq/max(cand_sq, 1e-30)) if alphas[key] > 0 else None,
                                "previous_persistence_residual_rel_l2": math.sqrt(eprev_sq/max(prev_c_sq, 1e-30)),
                                "residual_is_persisted_buffer": alphas[key] > 0,
                                "injected_correction_rel_l2": alphas[key] * float(meta["muon_momentum"]) * math.sqrt(eprev_sq/max(cand_sq, 1e-30)),
                                "error_buffer_reconstruction_rel_l2": observed.get("previous_candidate_reconstruction_relative_l2"),
                                "inferred_operand_identity_rel_l2": math.sqrt(reconstructed/max(prev_c_sq, 1e-30))}
            error_rows.append(error_detail_row)
    _write_csv(out / "landmark_metrics.csv", all_rows)
    _write_csv(out / "error_feedback_metrics.csv", error_rows)
    _write_csv(out / "tensor_metrics.csv", tensor_rows)
    storage_rows = []
    for key, summary in summaries.items():
        state = summary.get("optimizer_state", {})
        storage_rows.append({"method": key, "persistent_bytes": summary.get("optimizer_state_bytes", state.get("unique_storage_bytes")),
                             "compressed_state_bits": state.get("compressed_state_bits"),
                             "error_buffer_bits": state.get("error_buffer_bits", 0),
                             "error_buffer_bytes": state.get("error_buffer_bytes", 0),
                             "muon_scalar_count": state.get("muon_scalar_count"),
                             "muon_effective_bits_per_value": state.get("muon_effective_bits_per_value"),
                             "codebook_bits": state.get("codebook_bits"),
                             "has_fp32_momentum_shadow": any(x.get("has_fp32_momentum") for x in state.get("tensors", []))})
    _write_csv(out / "storage_breakdown.csv", storage_rows)
    evals = {key: {int(event["completed_updates"]): event["nll"] for event in _events(path, "eval")}
             for key, path in runs.items()}
    summary = {"status": "completed", "protocol_integrity": integrity,
               "runs": {key: str(path) for key, path in runs.items()},
               "validation_nll": evals,
               "aggregation": "global tensor-norm pooling over concatenated Muon tensor values",
               "recurrence": "M_t=mu*M_prev_decoded+g_t+alpha*mu*e_prev; D_t=g_t+mu*M_t; e_t=M_t-decode(Q(M_t))",
               "limitations": ["oracle uses a full FP32 persistence-error buffer and is not a storage-efficient optimizer",
                               "batch IDs are not persisted; pairing is validated by deterministic protocol and update/token/LR alignment",
                               "alpha=0 and FP32 trajectories are reused rather than rerun"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps({"source": "analyze_recursive_muon_error_feedback.py",
        "run_dirs": {key: str(value.resolve()) for key, value in runs.items()},
        "source_commits": {key: summaries[key].get("git_commit") for key in runs},
        "integrity": integrity}, indent=2) + "\n")
    lines = ["# Recursive Muon full-FP32 error-feedback oracle", "",
             "No training was run by this analysis. α=0 and FP32 are reused paired trajectories.", "",
             "## Protocol integrity", "", f"Compatible: {integrity['status']}; raw updates: {LANDMARKS}; paired training events: {integrity['paired_train_events']}.",
             "The event evidence matches update/token/LR positions under the deterministic data protocol; explicit batch IDs were not saved.", "",
             "## K5 trajectory fidelity vs FP32", "", "| update | method | K5 cosine | K5 rel-L2 | state-only cosine | gradient-only cosine | local feedback-only cosine |", "|---:|---|---:|---:|---:|---:|---:|"]
    for row in all_rows:
        lines.append(f"| {row['update']} | {row['method']} | {row['k5_cosine']:.6f} | {row['k5_relative_l2']:.6f} | {row['state_only_k5_cosine']:.6f} | {row['gradient_only_k5_cosine']:.6f} | {row['local_previous_persistence_k5_cosine']:.6f} |")
    lines += ["", "## Validation NLL", "", "| update | " + " | ".join(runs) + " |",
              "|---:|" + "---:|" * len(runs)]
    eval_steps = sorted(set().union(*(set(x) for x in evals.values())))
    for step in eval_steps:
        lines.append("| " + str(step) + " | " + " | ".join(f"{evals[k].get(step, float('nan')):.6f}" for k in runs) + " |")
    lines += ["", "## Interpretation boundary", "", "α=1 is an oracle because its full FP32 error buffer restores the information discarded by VQ at the previous persistence step. It is not evidence for a practical low-bit error-feedback state. Compare full trajectory metrics separately from the local one-step correction-only counterfactual.", ""]
    (out / "comparison.md").write_text("\n".join(lines))
    _make_plots(out, all_rows, evals, error_rows)
    return summary


def _make_plots(out, rows, evals, error_rows):
    plots = out / "plots"; plots.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        for metric, ylabel in (("momentum_cosine", "momentum cosine vs FP32"),
                               ("gradient_cosine", "gradient cosine vs FP32"),
                               ("direction_cosine", "Nesterov direction cosine vs FP32"),
                               ("k5_cosine", "actual K5 update cosine vs FP32")):
            fig, ax = plt.subplots(figsize=(7, 4))
            for method in ("alpha0", "alpha05", "alpha1"):
                selected = [x for x in rows if x["method"] == method]
                if selected: ax.plot([x["update"] for x in selected], [x[metric] for x in selected], "o-", label=method)
            ax.set_xlabel("update"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plots / f"{metric}.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, values in evals.items():
            if values: ax.plot(sorted(values), [values[i] for i in sorted(values)], "o-", label=key)
        ax.set_xlabel("update"); ax.set_ylabel("validation NLL"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plots / "validation_nll.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ("alpha0", "alpha05", "alpha1"):
            selected = [x for x in error_rows if x["method"] == method]
            if selected: ax.plot([x["update"] for x in selected], [x["previous_persistence_residual_rel_l2"] for x in selected], "o-", label=method)
        ax.set_xlabel("update"); ax.set_ylabel("previous persistence residual / previous candidate norm"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plots / "error_buffer_norm_ratio.png", dpi=140); plt.close(fig)
    except Exception as exc:
        (plots / "README.txt").write_text(f"Plot generation unavailable: {type(exc).__name__}: {exc}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-run", required=True); parser.add_argument("--alpha0-run", required=True)
    parser.add_argument("--alpha05-run", required=True); parser.add_argument("--alpha1-run", required=True)
    parser.add_argument("--out", default="reports/recursive_muon_error_feedback_oracle_s1_512")
    args = parser.parse_args()
    runs = {"fp32": Path(args.fp32_run), "alpha0": Path(args.alpha0_run),
            "alpha05": Path(args.alpha05_run), "alpha1": Path(args.alpha1_run)}
    analyze(runs, Path(args.out))


if __name__ == "__main__":
    main()
