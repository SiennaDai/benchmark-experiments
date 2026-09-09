"""Read-only aggregation of immutable run artifacts for reports and comparisons."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


MIB = 1024 * 1024


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _events(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _latest_by_update(events, event_type, split=None):
    """One logical point per update; later segments replace earlier duplicates."""
    points = {}
    for event in events:
        if event.get("event_type") != event_type or (split is not None and event.get("split") != split):
            continue
        key = event.get("completed_updates")
        previous = points.get(key)
        if previous is None or (event.get("segment_id", 0), event.get("event_index", 0)) >= (previous.get("segment_id", 0), previous.get("event_index", 0)):
            points[key] = event
    return [points[key] for key in sorted(points)]


def short_hash(value):
    return value[:12] if isinstance(value, str) else None


def summarize_run(run: str | Path) -> dict:
    run = Path(run).resolve()
    cfg = _read_json(run / "resolved_config.json") or {}
    stored = _read_json(run / "summary.json") or {}
    source = _read_json(run / "source.json") or {}
    manifest = _read_json(run / "data_manifest.json") or {}
    precision = _read_json(run / "precision.json") or {}
    events = _events(run / "metrics.jsonl")
    train = _latest_by_update(events, "train")
    evals = _latest_by_update(events, "eval", "validation")
    resources = _latest_by_update(events, "resource")
    update_seconds = [e["elapsed_seconds"] for e in train if isinstance(e.get("elapsed_seconds"), (int, float))]
    train_elapsed = sum(update_seconds) if update_seconds else None
    eval_seconds = [e["elapsed_seconds"] for e in evals if isinstance(e.get("elapsed_seconds"), (int, float))]
    eval_elapsed = sum(eval_seconds) if eval_seconds else None
    initial = evals[0].get("nll") if evals else None
    final = evals[-1].get("nll") if evals else None
    best = min(evals, key=lambda e: e.get("nll", float("inf"))) if evals else None
    resource = resources[-1] if resources else {}
    opt = resource.get("optimizer_state") or stored.get("optimizer_state") or {}
    processed = stored.get("processed_target_tokens")
    if processed is None and train:
        processed = train[-1].get("processed_target_tokens")
    state_bytes = opt.get("unique_storage_bytes")
    return {
        "run": str(run), "run_name": run.name, "run_id": stored.get("run_id") or (events[0].get("run_id") if events else run.name),
        "status": stored.get("status"), "reason": stored.get("reason"),
        "recipe_name": cfg.get("experiment", {}).get("name"), "recipe_fingerprint": cfg.get("fingerprint"),
        "protocol_id": cfg.get("experiment", {}).get("protocol_id"), "data_fingerprint": manifest.get("fingerprint"),
        "git_commit": source.get("upstream_commit"), "seed": cfg.get("experiment", {}).get("seed"),
        "data_seed": cfg.get("experiment", {}).get("data_seed"), "algorithm_seed": cfg.get("experiment", {}).get("algorithm_seed"),
        "optimizer_name": cfg.get("optimizer", {}).get("name"), "compute_precision": precision.get("compute") or cfg.get("precision", {}).get("compute"),
        "sequence_length": cfg.get("model", {}).get("sequence_length"), "target_tokens": cfg.get("train", {}).get("target_tokens"),
        "total_updates": stored.get("total_updates") or cfg.get("derived", {}).get("total_updates"),
        "completed_updates": stored.get("completed_updates"), "processed_target_tokens": processed,
        "initial_validation_nll": initial, "final_validation_nll": final,
        "best_validation_nll": best.get("nll") if best else None,
        "best_validation_update": best.get("completed_updates") if best else None,
        "total_elapsed_seconds": stored.get("elapsed_seconds"), "train_elapsed_seconds": train_elapsed,
        "eval_elapsed_seconds": eval_elapsed, "mean_update_seconds": statistics.mean(update_seconds) if update_seconds else None,
        "median_update_seconds": statistics.median(update_seconds) if update_seconds else None,
        "tokens_per_second": processed / train_elapsed if processed is not None and train_elapsed else None,
        "cuda_peak_allocated_bytes": resource.get("cuda_peak_allocated"), "cuda_peak_reserved_bytes": resource.get("cuda_peak_reserved"),
        "optimizer_state_bytes": state_bytes, "optimizer_state_tensor_count": len(opt.get("tensors", [])) if opt else None,
        "max_grad_norm": max((e.get("pre_clip_grad_norm", float("-inf")) for e in train), default=None),
        "grad_clip_event_count": sum(bool(e.get("clipped")) for e in train),
        "events": {"train": train, "eval": evals},
    }


def flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict): out.update(flatten(item, name))
        else: out[name] = item
    return out


def scientific_differences(configs, runs, vary):
    base = flatten(configs[0]); ignored = set(vary) | {"experiment.name", "fingerprint", "derived.recipe_path"}
    differences = []
    for run, cfg in zip(runs[1:], configs[1:]):
        other = flatten(cfg)
        for key in sorted(set(base) | set(other)):
            if key not in ignored and base.get(key) != other.get(key):
                differences.append({"run": str(run), "field": key, "base": base.get(key), "other": other.get(key)})
    return differences
