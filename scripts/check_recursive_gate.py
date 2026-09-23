#!/usr/bin/env python3
"""Validate staged recursive recipes against the completed FP32 protocol.

This is a read-only gate checker.  It never starts training and never changes
the recipe scheduler horizon.  It emits JSON, CSV, and Markdown summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.recipe import load_recipe


FIELDS = {
    "data.manifest": lambda c: c["data"]["manifest"],
    "data.train_split": lambda c: c["data"]["train_split"],
    "data.validation_split": lambda c: c["data"]["validation_split"],
    "data.allow_repeated_epochs": lambda c: c["data"]["allow_repeated_epochs"],
    "model": lambda c: c["model"],
    "experiment.seed": lambda c: c["experiment"]["seed"],
    "experiment.data_seed": lambda c: c["experiment"]["data_seed"],
    "experiment.algorithm_seed": lambda c: c["experiment"]["algorithm_seed"],
    "train.tokens_per_update": lambda c: c["derived"]["tokens_per_update"],
    "train.micro_batch_size": lambda c: c["train"]["micro_batch_size"],
    "train.accumulation_steps": lambda c: c["train"]["accumulation_steps"],
    "train.grad_clip_norm": lambda c: c["train"]["grad_clip_norm"],
    "schedule.total_updates": lambda c: c["derived"]["schedule_total_updates"],
    "schedule.name": lambda c: c["schedule"]["name"],
    "schedule.warmup_updates": lambda c: c["schedule"]["warmup_updates"],
    "schedule.final_lr_ratio": lambda c: c["schedule"]["final_lr_ratio"],
    "optimizer.lr": lambda c: c["optimizer"]["lr"],
    "optimizer.weight_decay": lambda c: c["optimizer"]["weight_decay"],
    "optimizer.muon_momentum": lambda c: c["optimizer"].get("muon_momentum", .95),
    "optimizer.muon_nesterov": lambda c: c["optimizer"].get("muon_nesterov", True),
    "optimizer.muon_ns_steps": lambda c: c["optimizer"].get("muon_ns_steps", 5),
    "optimizer.muon_ns_coefficients": lambda c: c["optimizer"].get("muon_ns_coefficients", [3.4445, -4.775, 2.0315]),
    "optimizer.muon_eps": lambda c: c["optimizer"].get("muon_eps", 1e-7),
    "precision.compute": lambda c: c["precision"]["compute"],
    "eval.max_target_tokens": lambda c: c["eval"]["max_target_tokens"],
    "eval.batch_size": lambda c: c["eval"]["batch_size"],
    "eval.compute": lambda c: c["eval"]["compute"],
}


def manifest_fingerprint(recipe: dict, root: Path, data_root: Path | None = None) -> str | None:
    path = Path(recipe["data"]["manifest"])
    if not path.is_absolute(): path = (data_root or root) / path
    try:
        return json.loads(path.read_text())["fingerprint"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def compare(fp32_path: Path, candidates: list[Path], root: Path, data_root: Path | None = None) -> dict:
    fp = load_recipe(fp32_path)
    rows = []
    expected_dataset = "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18"
    for candidate_path in candidates:
        candidate = load_recipe(candidate_path)
        mismatches = []
        for name, getter in FIELDS.items():
            if getter(fp) != getter(candidate):
                mismatches.append({"field": name, "fp32": getter(fp), "candidate": getter(candidate)})
        fp_manifest = manifest_fingerprint(fp, root, data_root); candidate_manifest = manifest_fingerprint(candidate, root, data_root)
        if fp_manifest != expected_dataset or candidate_manifest != expected_dataset:
            mismatches.append({"field": "dataset.fingerprint", "fp32": fp_manifest, "candidate": candidate_manifest, "expected": expected_dataset})
        rows.append({"candidate": str(candidate_path), "compatible": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches})
    return {"status": "compatible" if all(r["compatible"] for r in rows) else "mismatch", "dataset_fingerprint": expected_dataset, "fp32_reference": str(fp32_path), "candidates": rows}


def _landmark_event(run_dir: Path, event_type: str, landmark: int):
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    found = None
    for line in path.read_text().splitlines():
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if event.get("event_type") == event_type and int(event.get("completed_updates", -1)) == landmark:
            found = event
    return found


def gate_snapshot(fp32_run: Path, candidate_run: Path, landmark: int) -> dict:
    fp_train = _landmark_event(fp32_run, "train", landmark); cand_train = _landmark_event(candidate_run, "train", landmark)
    fp_eval = _landmark_event(fp32_run, "eval", landmark); cand_eval = _landmark_event(candidate_run, "eval", landmark)
    out = {"landmark": landmark, "fp32_run": str(fp32_run), "candidate_run": str(candidate_run),
           "fp32_train_nll": fp_train.get("train_nll") if fp_train else None,
           "candidate_train_nll": cand_train.get("train_nll") if cand_train else None,
           "train_nll_delta": (cand_train.get("train_nll") - fp_train.get("train_nll")) if fp_train and cand_train else None,
           "fp32_validation_nll": fp_eval.get("nll") if fp_eval else None,
           "candidate_validation_nll": cand_eval.get("nll") if cand_eval else None,
           "validation_nll_delta": (cand_eval.get("nll") - fp_eval.get("nll")) if fp_eval and cand_eval else None,
           "checkpoint_present": (candidate_run / "checkpoints" / f"update_{landmark:06d}.pt").exists(),
           "optimizer_state_metrics": "available in checkpoint; detailed cross-run momentum comparison is method-specific and intentionally not inferred from loss"}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-recipe", type=Path, required=True)
    parser.add_argument("--candidate-recipe", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="root containing the recipe's relative data/ path")
    parser.add_argument("--fp32-run", type=Path)
    parser.add_argument("--candidate-run", type=Path)
    parser.add_argument("--landmark", type=int, choices=(128, 512, 1024, 2048, 4096))
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    report = compare(args.fp32_recipe, args.candidate_recipe, root, args.data_root.resolve() if args.data_root else None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fp32_protocol_compatibility.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_rows = []
    for item in report["candidates"]:
        csv_rows.append({"candidate": item["candidate"], "compatible": item["compatible"], "mismatch_count": item["mismatch_count"], "mismatches": json.dumps(item["mismatches"], sort_keys=True)})
    with (args.output / "fp32_protocol_compatibility.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate", "compatible", "mismatch_count", "mismatches"]); writer.writeheader(); writer.writerows(csv_rows)
    lines = ["# FP32 protocol compatibility", "", f"Status: **{report['status']}**", "", f"Dataset fingerprint: `{report['dataset_fingerprint']}`", ""]
    for item in report["candidates"]:
        lines.append(f"- `{item['candidate']}`: {'compatible' if item['compatible'] else 'MISMATCH'}")
        for mismatch in item["mismatches"]: lines.append(f"  - `{mismatch['field']}`: FP32={mismatch.get('fp32')!r}; candidate={mismatch.get('candidate')!r}")
    (args.output / "fp32_protocol_compatibility.md").write_text("\n".join(lines) + "\n")
    if args.fp32_run and args.candidate_run and args.landmark:
        gate = gate_snapshot(args.fp32_run, args.candidate_run, args.landmark)
        (args.output / f"gate_{args.landmark:04d}.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
        with (args.output / f"gate_{args.landmark:04d}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(gate)); writer.writeheader(); writer.writerow(gate)
        (args.output / f"gate_{args.landmark:04d}.md").write_text("# Recursive gate report\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in gate.items()) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["status"] == "compatible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
