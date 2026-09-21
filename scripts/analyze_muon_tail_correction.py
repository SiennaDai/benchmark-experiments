#!/usr/bin/env python3
"""Offline INT4 Muon tail-residual correction headroom study.

Only existing FP32 snapshots are read.  Production ``persist_state`` and
``zeropower_newton_schulz`` are called on detached diagnostic copies.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optim import muon_reference  # noqa: E402
from optim.muon_spectral_sensitivity import quantize  # noqa: E402
from optim.muon_tail_correction import (  # noqa: E402
    BUDGETS, RANDOM_SEED, correction, decompose, mode_indices, row_for_budget,
)
from optim.muon_update_fidelity import load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
KINDS = ("diagonal", "full")
BANDS = ("tail", "head", "random")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def discover_snapshots(root: Path) -> list[tuple[int, int, Path]]:
    groups: dict[tuple[int, str], dict[int, Path]] = {}
    for path in sorted(root.rglob("update_*.pt")):
        if "muon_momentum_snapshots" not in path.parts:
            continue
        try:
            snapshot = load_snapshot(path)
            metadata = snapshot["metadata"]
            seed, update = int(metadata["seeds"]["seed"]), int(metadata["update"])
        except (KeyError, ValueError, RuntimeError, OSError):
            continue
        if seed in SEEDS and update in LANDMARKS:
            groups.setdefault((seed, str(path.parent)), {})[update] = path
    result: list[tuple[int, int, Path]] = []
    for seed in SEEDS:
        candidates = sorted((group, values) for (s, group), values in groups.items()
                            if s == seed and set(values) == set(LANDMARKS))
        if not candidates:
            continue
        values = candidates[0][1]
        result.extend((seed, update, values[update]) for update in LANDMARKS)
    return result


def transform_kwargs(metadata: dict) -> dict:
    cfg = metadata["muon_transform"]
    return {"steps": int(cfg["steps"]), "coefficients": tuple(float(x) for x in cfg["coefficients"]), "eps": float(cfg["eps"])}


def finite(value) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def identity(seed: int, update: int, item: dict) -> dict:
    return {"seed": seed, "update": update, "parameter_id": item["parameter_id"],
            "parameter_name": item.get("name", item["parameter_id"]), "shape": str(item["shape"])}


def select_representatives(rows: list[dict]) -> tuple[list[dict], dict[tuple, str]]:
    """Select before intervention: low/median/high baseline error across layers."""
    candidates = [r for r in rows if r.get("correction_type") == "diagonal" and r.get("band") == "tail" and r.get("budget") == 0]
    groups: dict[str, list[dict]] = {}
    for row in candidates:
        layer = row["parameter_name"].split(".")[0:3]
        groups.setdefault(".".join(layer), []).append(row)
    selected: list[dict] = []
    reasons: dict[tuple, str] = {}
    for layer in sorted(groups):
        ordered = sorted(groups[layer], key=lambda r: (float(r.get("update_relative_l2") or float("inf")), r["seed"], r["update"], r["parameter_id"]))
        for row, reason in ((ordered[0], "low_baseline_error"), (ordered[len(ordered)//2], "median_baseline_error"), (ordered[-1], "high_baseline_error")):
            key = (row["seed"], row["update"], row["parameter_id"])
            if key not in reasons:
                selected.append(row); reasons[key] = reason
    return selected, reasons


def correlations(rows: list[dict]) -> list[dict]:
    pairs = []
    features = ("update_cosine_gain", "update_l2_reduction_fraction", "correction_relative_residual_norm", "residual_energy_fraction_captured")
    target = "baseline_update_relative_l2"
    for feature in features:
        values = [(float(r[feature]), float(r[target])) for r in rows if finite(r.get(feature)) and finite(r.get(target))]
        if len(values) < 2:
            pairs.append({"feature": feature, "target": target, "sample_count": len(values), "pearson": None, "spearman": None}); continue
        x = torch.tensor([a for a, _ in values], dtype=torch.float64); y = torch.tensor([b for _, b in values], dtype=torch.float64)
        xc, yc = x-x.mean(), y-y.mean(); den = xc.norm()*yc.norm()
        orderx, ordery = torch.argsort(x, stable=True), torch.argsort(y, stable=True)
        rx = torch.empty_like(x); ry = torch.empty_like(y); rx[orderx] = torch.arange(len(x), dtype=torch.float64); ry[ordery] = torch.arange(len(y), dtype=torch.float64)
        rxc, ryc = rx-rx.mean(), ry-ry.mean(); rden = rxc.norm()*ryc.norm()
        pairs.append({"feature": feature, "target": target, "sample_count": len(values), "pearson": float((xc*yc).sum()/den) if den else None, "spearman": float((rxc*ryc).sum()/rden) if rden else None})
    return pairs


def make_plots(output: Path, rows: list[dict], representative: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (output / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs are complete.\n"); return
    summary = [r for r in rows if r["band"] in ("tail", "head", "random") and r["correction_type"] == "diagonal"]
    for metric, filename, ylabel in (("update_cosine", "rank_vs_update_cosine.png", "post-Muon update cosine"), ("update_relative_l2", "rank_vs_update_relative_l2.png", "post-Muon relative L2")):
        plt.figure(figsize=(8, 5))
        for band in BANDS:
            grouped = {}
            for row in summary:
                if row["band"] == band: grouped.setdefault(row["budget"], []).append(row.get(metric))
            xs = sorted(grouped); ys = [sum(v for v in (x for x in grouped[k] if finite(x))) / len([x for x in grouped[k] if finite(x)]) for k in xs]
            plt.plot(xs, ys, marker="o", label=band)
        plt.xlabel("correction rank"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(output / filename, dpi=140); plt.close()
    plt.figure(figsize=(8, 5))
    for kind in KINDS:
        subset = [r for r in summary if r["band"] == "tail" and r["correction_type"] == kind]
        grouped = {}
        for row in subset: grouped.setdefault(row["budget"], []).append(row.get("update_cosine_gain"))
        xs = sorted(grouped); ys = [sum(v for v in (x for x in grouped[k] if finite(x))) / len([x for x in grouped[k] if finite(x)]) for k in xs]
        plt.plot(xs, ys, marker="o", label=kind)
    plt.xlabel("tail correction rank"); plt.ylabel("update cosine gain"); plt.legend(); plt.tight_layout(); plt.savefig(output / "diagonal_vs_full_tail.png", dpi=140); plt.close()
    for rep in representative:
        key = (rep["seed"], rep["update"], rep["parameter_id"])
        subset = [r for r in summary if (r["seed"], r["update"], r["parameter_id"]) == key]
        if not subset: continue
        plt.figure(figsize=(8, 5))
        for band in BANDS:
            data = sorted((r for r in subset if r["band"] == band), key=lambda r: r["budget"])
            plt.plot([r["budget"] for r in data], [r["update_cosine"] for r in data], marker=".", label=band)
        plt.title(f"s{key[0]} u{key[1]} {key[2]}"); plt.xlabel("rank"); plt.ylabel("update cosine"); plt.legend(); plt.tight_layout(); plt.savefig(output / f"representative_{key[0]}_{key[1]}_{key[2].replace('.', '_')}.png", dpi=140); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/muon_tail_correction")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(); started = time.perf_counter()
    paths = discover_snapshots(args.reports_root)
    if len(paths) != 10: raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []; capture: list[dict] = []; cache = {}
    for seed, update, path in paths:
        snapshot = load_snapshot(path); kwargs = transform_kwargs(snapshot["metadata"])
        for item in snapshot["tensors"]:
            if len(item["shape"]) != 2: continue
            source = item["tensor"].float(); d = decompose(source); baseline = quantize(source, "int4-dynamic-b2048")
            ref_update = muon_reference.zeropower_newton_schulz(source.clone(), **kwargs)
            baseline_update = muon_reference.zeropower_newton_schulz(baseline.clone(), **kwargs)
            ident = identity(seed, update, item); key = (seed, update, item["parameter_id"]); cache[key] = (d, baseline, source, ref_update, baseline_update, kwargs)
            for band in BANDS:
                for kind in KINDS:
                    for budget in BUDGETS:
                        row = {**ident, **row_for_budget(d, baseline, band=band, kind=kind, budget=budget, transform_kwargs=kwargs, reference_update=ref_update, baseline_update=baseline_update), "random_seed": RANDOM_SEED}
                        # Capture residual energy of this orthogonal projection.
                        indices = mode_indices(d, budget, band); corr = correction(d, baseline, indices, kind); residual = source - baseline
                        total = residual.square().sum(); row["residual_energy_fraction_captured"] = float(corr.square().sum() / total) if total.item() else None
                        rows.append(row)
            # A direct baseline row is kept once per tensor for unambiguous reporting.
            capture.append({**ident, "residual_norm": float((source-baseline).norm()), "matrix_norm": float(source.norm()), "baseline_update_relative_l2": float(_safe_error(ref_update, baseline_update)), "effective_rank": d.effective_rank})
        print(f"processed seed={seed} update={update}", flush=True)
    representatives, reasons = select_representatives(rows)
    rep_keys = {(r["seed"], r["update"], r["parameter_id"]): reasons[(r["seed"], r["update"], r["parameter_id"])] for r in representatives}
    representative_rows = [r for r in rows if (r["seed"], r["update"], r["parameter_id"]) in rep_keys]
    for r in representative_rows: r["selection_reason"] = rep_keys[(r["seed"], r["update"], r["parameter_id"])]
    write_csv(args.output / "tensor_tail_correction_metrics.csv", rows)
    summary = []
    for band in BANDS:
        for kind in KINDS:
            for budget in BUDGETS:
                group = [r for r in rows if r["band"] == band and r["correction_type"] == kind and r["budget"] == budget]
                values = lambda key: [float(r[key]) for r in group if finite(r.get(key))]
                for metric in ("update_cosine_gain", "update_l2_reduction_fraction", "residual_energy_fraction_captured", "update_cosine", "update_relative_l2"):
                    vals = values(metric); summary.append({"band": band, "correction_type": kind, "budget": budget, "metric": metric, "mean": sum(vals)/len(vals) if vals else None, "median": float(torch.tensor(vals).median()) if vals else None, "count": len(vals)})
    write_csv(args.output / "budget_summary.csv", summary)
    write_csv(args.output / "head_tail_random_controls.csv", [r for r in rows if r["correction_type"] == "diagonal"])
    write_csv(args.output / "residual_capture.csv", capture)
    write_csv(args.output / "representative_curves.csv", representative_rows)
    write_csv(args.output / "sensitivity_correlations.csv", correlations([r for r in rows if r["band"] == "tail" and r["correction_type"] == "full" and r["budget"] == max(BUDGETS)]))
    if not args.skip_plots: make_plots(args.output, rows, representatives)
    elapsed = time.perf_counter() - started
    methodology = f"""# Methodology\n\nThis read-only CPU study used the ten existing formal FP32 Muon snapshots and {len(capture)} eligible 2D tensors. The baseline is exactly production `int4-dynamic-b2048` via `persist_state`; Muon metrics call production `zeropower_newton_schulz`.\n\nFor each cached reduced FP32 SVD `M=U diag(sigma) V^T`, active modes satisfy `sigma_i/sigma_max >= 1e-6`. Tail modes are the weakest `k` active modes; head modes are the strongest `k`; random controls use deterministic seed `{RANDOM_SEED}` and the same active rank. Budgets are `{BUDGETS}`. The residual is `R=M-Q(M)`. Diagonal correction is `U_T diag(diag(U_T R V)) V_T`; full correction is `U_T (U_T R V_T) V_T`. Rank zero exactly reproduces baseline.\n\nOracle scalar cost counts diagonal `k` or full `k^2` coefficients assuming U/V side information. Explicit storage is separately estimated as vectors plus coefficients. Residual capture is projected correction Frobenius energy divided by residual energy. Representatives are selected before intervention per deterministic layer prefix using low/median/high baseline update error; no success-based cherry-picking. Runtime: {elapsed:.2f}s CPU. This is structural headroom, not a deployable quantizer.\n"""
    (args.output / "methodology.md").write_text(methodology)
    (args.output / "summary.md").write_text(f"# Summary\n\nAnalyzed {len(capture)} tensors from {len(paths)} formal snapshots. Runtime: {elapsed:.2f}s CPU. Baseline and all correction rows are in `tensor_tail_correction_metrics.csv`; aggregate budget results are in `budget_summary.csv`. Tail/head/random controls use matched active-mode rank. Interpret gains as offline oracle headroom, not a deployable representation.\n")
    print(f"analyzed {len(capture)} tensors in {elapsed:.2f}s", flush=True)


def _safe_error(reference: torch.Tensor, observed: torch.Tensor) -> float:
    denom = reference.norm()
    return float((observed-reference).norm()/denom) if denom.item() else float("nan")


if __name__ == "__main__": main()
