#!/usr/bin/env python3
"""Compare periodic VQ residual feedback against reusable seed-1 baselines."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from analyze_recursive_muon_mechanism import _k5, _load

ROOT = Path(__file__).resolve().parents[1]
LANDMARKS = (1, 8, 32, 128, 512)
METHOD_INTERVALS = {"k4": 4, "k16": 16, "k64": 64}
COMMON_SUMMARY_FIELDS = ("seed", "data_seed", "algorithm_seed", "protocol_id", "data_fingerprint",
                         "schedule_total_updates", "total_updates", "target_tokens",
                         "sequence_length", "compute_precision")
VQ_FIELDS = {"recursive_rank": 8, "recursive_block_size": 2048,
             "recursive_factor_dtype": "bf16", "recursive_structure_mode": "exact_svd_oracle",
             "recursive_representation": "vq_int3", "recursive_codebook_key": "s0_k8_w64_t8_v1200"}


def _read_json(path: Path):
    return json.loads(path.read_text())


def _events(run: Path, kind: str):
    path = run / "metrics.jsonl"
    if not path.is_file():
        return []
    return [obj for line in path.read_text().splitlines() if line.strip()
            for obj in [json.loads(line)] if obj.get("event_type") == kind]


def _snapshot_paths(run: Path):
    found = {int(p.stem.rsplit("_", 1)[1]): p for p in (run / "mechanism").glob("update_*.pt")}
    if sorted(found) != list(LANDMARKS):
        raise ValueError(f"{run}: expected raw landmarks {list(LANDMARKS)}, found {sorted(found)}")
    return found


def _pooled(a_by_name: dict, b_by_name: dict):
    aa = bb = dd = dot = 0.0
    for name in sorted(a_by_name):
        a, b = a_by_name[name].float(), b_by_name[name].float()
        aa += float(a.square().sum()); bb += float(b.square().sum())
        dd += float((a-b).square().sum()); dot += float((a*b).sum())
    return {"cosine": max(-1.0, min(1.0, dot / max(math.sqrt(aa*bb), 1e-30))),
            "relative_l2": math.sqrt(dd / max(aa, 1e-30)),
            "norm_ratio": math.sqrt(bb / max(aa, 1e-30))}


def _validate(runs: dict[str, Path]):
    summaries, configs, snapshots, train_events, eval_events = {}, {}, {}, {}, {}
    for method, run in runs.items():
        if not run.is_dir():
            raise FileNotFoundError(f"run directory not found for {method}: {run}")
        summaries[method] = _read_json(run / "summary.json")
        configs[method] = _read_json(run / "resolved_config.json")
        snapshots[method] = _snapshot_paths(run)
        train_events[method] = _events(run, "train")
        eval_events[method] = _events(run, "eval")
        summary = summaries[method]
        if summary.get("completed_updates") != 512 or summary.get("status") != "paused_staged":
            raise ValueError(f"{method} must be a clean staged stop at 512; got {summary.get('status')} @ {summary.get('completed_updates')}")
        if any(summary.get(k) != value for k, value in (("seed", 1), ("data_seed", 1337),
               ("algorithm_seed", 2026), ("schedule_total_updates", 4096), ("total_updates", 4096),
               ("compute_precision", "fp32"))):
            raise ValueError(f"{method} summary does not match seed-1/4096/FP32 protocol")
        if summary.get("protocol_id") != "slimpajama-hash-split-v1" or summary.get("data_fingerprint") != "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18":
            raise ValueError(f"{method} frozen-data/protocol fingerprint mismatch")
    ref = summaries["fp32"]
    for method, summary in summaries.items():
        mismatches = {key: (ref.get(key), summary.get(key)) for key in COMMON_SUMMARY_FIELDS
                      if ref.get(key) != summary.get(key)}
        if mismatches:
            raise ValueError(f"protocol mismatch for {method}: {mismatches}")
        if len(train_events[method]) < 512:
            raise ValueError(f"{method}: missing train events")
        if sorted(int(x) for x in snapshots[method]) != list(LANDMARKS):
            raise ValueError(f"{method}: raw landmark set mismatch")
        for step in LANDMARKS:
            blob = _load(snapshots[method][step])
            meta = blob["metadata"]
            if int(meta.get("update", -1)) != step or int(meta.get("seed", -1)) != 1 or int(meta.get("schedule_total_updates", -1)) != 4096:
                raise ValueError(f"{method} snapshot metadata mismatch at update {step}")
    for field in ("model", "data", "train", "schedule", "precision", "eval"):
        baseline = configs["fp32"][field]
        if any(configs[method][field] != baseline for method in runs):
            raise ValueError(f"resolved protocol field {field} differs across runs")
    ref_opt = configs["fp32"]["optimizer"]
    common_optimizer = ("lr", "betas", "eps", "weight_decay", "fused", "foreach", "state_simulation",
                        "muon_momentum", "muon_nesterov", "muon_ns_steps", "muon_ns_coefficients", "muon_eps")
    for method in runs:
        cfg = configs[method]["optimizer"]
        for key in common_optimizer:
            if cfg.get(key) != ref_opt.get(key):
                raise ValueError(f"optimizer protocol mismatch {method}.{key}")
    fp_optimizer = configs["fp32"]["optimizer"]
    if fp_optimizer.get("name") != "reference_muon":
        raise ValueError("reused reference run is not reference_muon")
    for method, cfg in configs.items():
        if method == "fp32":
            continue
        opt = cfg["optimizer"]
        if opt.get("name") != "recursive_muon":
            raise ValueError(f"{method} is not recursive_muon")
        for field, expected in VQ_FIELDS.items():
            if opt.get(field) != expected:
                raise ValueError(f"{method} {field} expected {expected}, got {opt.get(field)}")
        if method in METHOD_INTERVALS:
            if opt.get("recursive_error_feedback_mode") != "periodic" or opt.get("recursive_error_feedback_interval") != METHOD_INTERVALS[method]:
                raise ValueError(f"{method} periodic configuration mismatch")
        else:
            expected_alpha = {"alpha0": 0.0, "alpha05": .5, "alpha1": 1.0}[method]
            alpha = float(opt.get("recursive_error_feedback_alpha", 0.0))
            if alpha != expected_alpha:
                raise ValueError(f"{method} alpha mismatch: {alpha} != {expected_alpha}")
        if opt.get("recursive_codebook_path") != configs["alpha0"]["optimizer"].get("recursive_codebook_path"):
            raise ValueError(f"{method}: VQ codebook path differs from alpha=0")
    baseline_codebook = configs["alpha0"]["optimizer"].get("recursive_codebook_key")
    if baseline_codebook != "s0_k8_w64_t8_v1200":
        raise ValueError("alpha=0 baseline does not use the fixed s0 codebook")
    ref_train = train_events["fp32"][:512]
    for method in runs:
        events = train_events[method][:512]
        for a, b in zip(ref_train, events):
            for field in ("completed_updates", "processed_target_tokens", "tokens_this_update", "lr"):
                if a.get(field) != b.get(field):
                    raise ValueError(f"unpaired data/schedule event {method} at update {a.get('completed_updates')}: {field}")
    event_alignment = "512 train events match update, token position/count and LR; batch IDs are not persisted"
    return summaries, configs, snapshots, train_events, eval_events, {"status": "compatible",
        "shared_protocol_fields": list(COMMON_SUMMARY_FIELDS), "event_alignment": event_alignment,
        "landmarks": list(LANDMARKS), "codebook_key": baseline_codebook}


def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def analyze(runs: dict[str, Path], output: Path):
    summaries, configs, paths, train_events, eval_events, integrity = _validate(runs)
    output.mkdir(parents=True, exist_ok=True)
    landmark_rows, tensor_rows = [], []
    for step in LANDMARKS:
        ref_blob = _load(paths["fp32"][step]); ref_meta = ref_blob["metadata"]
        ref_tensors = {item["name"]: item for item in ref_blob["tensors"]}
        ref_candidate = {name: item["momentum_candidate"].float() for name, item in ref_tensors.items()}
        ref_gradient = {name: item["gradient"].float() for name, item in ref_tensors.items()}
        ref_direction, ref_k5 = {}, {}
        mu = float(ref_meta["muon_momentum"])
        for name, item in ref_tensors.items():
            gradient, candidate = item["gradient"].float(), item["momentum_candidate"].float()
            direction = gradient + mu * candidate
            ref_direction[name] = direction
            ref_k5[name] = _k5(direction.clone(), ref_meta)
        method_keys = [key for key in runs if key != "fp32"]
        for method in method_keys:
            blob = _load(paths[method][step]); meta = blob["metadata"]
            tensors = {item["name"]: item for item in blob["tensors"]}
            if set(tensors) != set(ref_tensors):
                raise ValueError(f"tensor identities differ for {method} at {step}")
            actual_candidate, actual_gradient, actual_direction, actual_k5 = {}, {}, {}, {}
            for name, item in tensors.items():
                candidate, gradient = item["momentum_candidate"].float(), item["gradient"].float()
                direction = gradient + mu * candidate
                actual_candidate[name] = candidate; actual_gradient[name] = gradient
                actual_direction[name] = direction
                actual_k5[name] = _k5(direction.clone(), meta)
                for metric, a, b in (("momentum", ref_candidate[name], candidate),
                                     ("gradient", ref_gradient[name], gradient),
                                     ("direction", ref_direction[name], direction),
                                     ("actual_k5", ref_k5[name], actual_k5[name])):
                    aa = float(a.square().sum()); bb = float(b.square().sum())
                    cos = float((a.reshape(-1) @ b.reshape(-1)) / max(math.sqrt(aa*bb), 1e-30))
                    rel = float((a-b).norm()) / max(math.sqrt(aa), 1e-30)
                    tensor_rows.append({"update": step, "method": method, "tensor_name": name,
                        "metric": metric, "cosine": cos, "relative_l2": rel,
                        "reference_norm": math.sqrt(aa), "method_norm": math.sqrt(bb)})
            summary = {"update": step, "method": method,
                       "validation_nll": next((x.get("nll") for x in eval_events[method] if x.get("completed_updates") == step), None),
                       "train_loss": next((x.get("train_nll") for x in train_events[method] if x.get("completed_updates") == step), None)}
            for metric, a, b in (("momentum", ref_candidate, actual_candidate),
                                 ("gradient", ref_gradient, actual_gradient),
                                 ("direction", ref_direction, actual_direction),
                                 ("actual_k5", ref_k5, actual_k5)):
                pooled = _pooled(a, b)
                for stat, value in pooled.items():
                    summary[f"{metric}_{stat}"] = value
            landmark_rows.append(summary)
        # Include the FP32 validation/loss trace in each landmark only once.
        landmark_rows.append({"update": step, "method": "fp32",
            "validation_nll": next((x.get("nll") for x in eval_events["fp32"] if x.get("completed_updates") == step), None),
            "train_loss": next((x.get("train_nll") for x in train_events["fp32"] if x.get("completed_updates") == step), None),
            **{f"{metric}_{stat}": (1.0 if stat == "cosine" else 0.0 if stat == "relative_l2" else 1.0)
               for metric in ("momentum", "gradient", "direction", "actual_k5")
               for stat in ("cosine", "relative_l2", "norm_ratio")}})
    _csv(output / "landmark_metrics.csv", landmark_rows)
    _csv(output / "tensor_metrics.csv", tensor_rows)

    correction_rows = []
    for method, interval in METHOD_INTERVALS.items():
        event_path = runs[method] / "mechanism/metrics.jsonl"
        if not event_path.is_file():
            raise FileNotFoundError(f"missing mechanism scalar/event log: {event_path}")
        for line in event_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if int(event.get("update", -1)) % interval == 0 and int(event.get("update", -1)) > 0:
                correction_rows.append({"method": method, "interval": interval,
                    "update": event["update"], "correction_applied": event.get("correction_applied"),
                    "accumulator_norm_ratio_before_injection": event.get("periodic_accumulator_norm_ratio"),
                    "injected_correction_norm_ratio": event.get("periodic_injected_correction_norm_ratio"),
                    "accumulator_norm_ratio_after_update": event.get("periodic_accumulator_next_norm_ratio"),
                    "instantaneous_persistence_error_norm_ratio": event.get("error_buffer_norm_ratio"),
                    # Available only at the configured raw landmarks. This is
                    # the same-step local effect of injecting A_t, holding the
                    # current gradient fixed; it is not FP32 trajectory fidelity.
                    "local_no_injection_vs_injected_k5_cosine": event.get("local_one_step_k5_cosine"),
                    "local_no_injection_vs_injected_k5_relative_l2": event.get("local_one_step_k5_relative_l2"),
                    "train_nll": event.get("train_nll")})
    _csv(output / "correction_event_metrics.csv", correction_rows)

    storage_rows = []
    for method, summary in summaries.items():
        state = summary.get("optimizer_state", {})
        storage_rows.append({"method": method, "persistent_bytes": state.get("unique_storage_bytes", summary.get("optimizer_state_bytes")),
            "persistent_bits": state.get("persistent_bits"), "compressed_state_bits": state.get("compressed_state_bits"),
            "codebook_bits": state.get("codebook_bits"), "error_buffer_bits": state.get("error_buffer_bits", 0),
            "periodic_accumulator_bits": state.get("periodic_accumulator_bits", 0),
            "muon_scalar_count": state.get("muon_scalar_count"),
            "effective_bits_per_value": state.get("muon_effective_bits_per_value"),
            "has_fp32_momentum_shadow": any(x.get("has_fp32_momentum", False) for x in state.get("tensors", []))})
    _csv(output / "storage_breakdown.csv", storage_rows)

    # Plots use only pooled landmark observations; interval events are shown as
    # separate accumulator points so no dense trajectory is fabricated.
    plot_dir = output / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = (("momentum_cosine", "momentum cosine vs FP32"), ("gradient_cosine", "gradient cosine vs FP32"),
               ("direction_cosine", "Nesterov direction cosine vs FP32"), ("actual_k5_cosine", "actual K5 update cosine vs FP32"))
    for field, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ["alpha0", "k64", "k16", "k4", "alpha1"]:
            rows = sorted((r for r in landmark_rows if r["method"] == method and r.get(field) is not None), key=lambda x: x["update"])
            if rows: ax.plot([r["update"] for r in rows], [r[field] for r in rows], marker="o", label=method)
        ax.set_xscale("log"); ax.set_xticks(LANDMARKS, labels=[str(x) for x in LANDMARKS]); ax.set_ylim(-.05, 1.05)
        ax.set_xlabel("update"); ax.set_ylabel(field.replace("_", " ")); ax.set_title(title); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(plot_dir / f"{field}.png", dpi=145); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ["fp32", "alpha0", "k64", "k16", "k4", "alpha1"]:
        rows = sorted((r for r in landmark_rows if r["method"] == method and r.get("validation_nll") is not None), key=lambda x: x["update"])
        if rows: ax.plot([r["update"] for r in rows], [r["validation_nll"] for r in rows], marker="o", label=method)
    ax.set_xscale("log"); ax.set_xticks(LANDMARKS, labels=[str(x) for x in LANDMARKS]); ax.set_xlabel("update"); ax.set_ylabel("validation NLL"); ax.set_title("Training quality: validation NLL"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "validation_nll.png", dpi=145); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHOD_INTERVALS:
        rows = [r for r in correction_rows if r["method"] == method]
        if rows: ax.plot([r["update"] for r in rows], [r["accumulator_norm_ratio_before_injection"] for r in rows], marker=".", linewidth=.8, label=method)
    ax.set_xlabel("correction update"); ax.set_ylabel("unpaid accumulator / candidate norm"); ax.set_title("Periodic accumulator at correction events"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "accumulator_ratio.png", dpi=145); plt.close(fig)

    endpoint = [r for r in landmark_rows if r["update"] == 512]
    summary = {"status": "complete", "landmarks": list(LANDMARKS), "protocol_integrity": integrity,
        "methods": list(runs), "storage_rows": storage_rows, "correction_event_count": len(correction_rows),
        "endpoint_update_512": endpoint,
        "interpretation_boundary": "Trajectory fidelity and validation quality are separate outcomes. Periodic FP32 accumulator is an oracle costing approximately 32 additional bits/value; it is not a practical memory-saving method.",
        "source_commits": {method: summaries[method].get("git_commit") for method in runs}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    provenance = {"runs": {key: str(value.resolve()) for key, value in runs.items()},
        "summary_fingerprints": {key: summaries[key].get("recipe_fingerprint") for key in runs},
        "source_commits": summary["source_commits"], "protocol_integrity": integrity,
        "analysis_script": "scripts/analyze_recursive_muon_periodic_feedback.py"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    _write_markdown(output / "comparison.md", runs, configs, summaries, landmark_rows, correction_rows, storage_rows, integrity)
    return summary


def _write_markdown(path, runs, configs, summaries, landmark_rows, correction_rows, storage_rows, integrity):
    lines = ["# Recursive Muon periodic full-residual feedback: seed 1, 512 updates", "",
        "This is a causal correction-timescale experiment, not a practical low-memory method. Periodic runs retain compressed VQ momentum plus one persistent FP32 accumulator (about +32 bits per Muon value).", "",
        "## Protocol integrity", "", f"Validated: {integrity['status']}. {integrity['event_alignment']}. Formal schedule horizon is 4096; runtime ends at update 512.", "",
        "## Update-512 comparison", "", "| method | momentum cosine | gradient cosine | Nesterov cosine | actual K5 cosine | validation NLL |", "|---|---:|---:|---:|---:|---:|"]
    for method in ("alpha0", "k64", "k16", "k4", "alpha1", "fp32"):
        row = next(r for r in landmark_rows if r["update"] == 512 and r["method"] == method)
        def fmt(x): return "n/a" if x is None else f"{x:.6f}"
        lines.append(f"| {method} | {fmt(row.get('momentum_cosine'))} | {fmt(row.get('gradient_cosine'))} | {fmt(row.get('direction_cosine'))} | {fmt(row.get('actual_k5_cosine'))} | {fmt(row.get('validation_nll'))} |")
    lines += ["", "## Periodic correction behavior", "", f"The run logs contain {len(correction_rows)} correction-event observations across K=4/16/64. `correction_event_metrics.csv` records pre-injection accumulator norm, injected norm, next unpaid accumulator norm and current persistence error. Full raw snapshots are kept only at 1/8/32/128/512; therefore correction points without paired raw FP32 snapshots have no fabricated K5-fidelity values.", "",
        "## Persistent storage", "", "| method | persistent bytes | periodic accumulator bits | effective bits/value | FP32 momentum shadow |", "|---|---:|---:|---:|---|"]
    for row in storage_rows:
        lines.append(f"| {row['method']} | {row['persistent_bytes']} | {row['periodic_accumulator_bits']} | {row['effective_bits_per_value']} | {row['has_fp32_momentum_shadow']} |")
    lines += ["", "## Interpretation", "", "Read fidelity curves separately from validation NLL. A correction at time t cannot undo parameter updates already taken before t; an immediate state-fidelity jump therefore need not produce immediate gradient or task-loss recovery. This single seed and 512-update horizon cannot establish beneficial regularization.", "",
        "## Research-chain framing", "", "The current phase established that recursive structural 64-word 2D INT3 VQ can store Muon momentum at about 3.49 effective bits/value, but produces strong trajectory drift despite good static one-shot fidelity. Corrected mechanism analysis found early state-driven divergence followed by later gradient feedback. Full FP32 persistence-error feedback causally restored the FP32 trajectory; α=0.5 partially recovered it. The residual itself was not efficiently compressible at ≤1 added bit/value. This experiment tests correction frequency alone. Its conclusion must be based on the resulting fidelity and validation curves—not FP32 matching alone.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-run", type=Path, required=True)
    parser.add_argument("--alpha0-run", type=Path, required=True)
    parser.add_argument("--alpha1-run", type=Path, required=True)
    parser.add_argument("--alpha05-run", type=Path)
    parser.add_argument("--k4-run", type=Path, required=True)
    parser.add_argument("--k16-run", type=Path, required=True)
    parser.add_argument("--k64-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/recursive_muon_periodic_feedback_s1_512")
    args = parser.parse_args()
    runs = {"fp32": args.fp32_run, "alpha0": args.alpha0_run, "k4": args.k4_run,
            "k16": args.k16_run, "k64": args.k64_run, "alpha1": args.alpha1_run}
    if args.alpha05_run:
        runs["alpha05"] = args.alpha05_run
    result = analyze(runs, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
