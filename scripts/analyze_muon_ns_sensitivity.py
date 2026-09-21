#!/usr/bin/env python3
"""Offline finite-step Newton--Schulz sensitivity study for Muon snapshots."""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optim.muon_ns_sensitivity import (  # noqa: E402
    DEFAULT_K_GRID, PRODUCTION_COEFFICIENTS, PRODUCTION_EPS, PRODUCTION_STEPS,
    band_indices, controlled_metrics, decompose, exact_polar, quantize,
    rotation_perturbation, scalar_map_and_derivative, sv_only_perturbation,
    transform, transform_sweep,
)
from optim.muon_update_fidelity import _ratios, load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
QUANTIZER = "int4-dynamic-b2048"
EPSILONS = (0.0025, 0.005, 0.01, 0.02, 0.05)
THRESHOLD = 1e-6


def discover(root: Path) -> list[tuple[int, int, Path]]:
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
    selected = []
    for seed in SEEDS:
        candidates = sorted((name, rows) for (s, name), rows in groups.items()
                            if s == seed and set(rows) == set(LANDMARKS))
        if candidates:
            name, rows = candidates[0]
            selected.extend((seed, update, rows[update]) for update in LANDMARKS)
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        if not keys:
            return
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


TRANSFER_FIELDS = (
    "seed", "update", "parameter_id", "tensor_key", "steps", "mode",
    "normalized_sigma", "sigma", "f_k", "a_mag", "a_dir",
)


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def correlation(rows: list[dict], x_key: str, y_key: str) -> dict:
    values = [(float(r[x_key]), float(r[y_key])) for r in rows if finite(r.get(x_key)) and finite(r.get(y_key))]
    if len(values) < 2:
        return {"feature": x_key, "target": y_key, "sample_count": len(values), "pearson": None, "spearman": None}
    x = torch.tensor([a for a, _ in values], dtype=torch.float64); y = torch.tensor([b for _, b in values], dtype=torch.float64)
    xc, yc = x - x.mean(), y - y.mean(); den = xc.norm() * yc.norm()
    pearson = float((xc * yc).sum() / den) if den.item() else None
    def rank(v):
        order = torch.argsort(v, stable=True); out = torch.empty_like(v); out[order] = torch.arange(v.numel(), dtype=v.dtype)
        i = 0
        while i < len(v):
            j = i + 1
            while j < len(v) and v[order[j]] == v[order[i]]: j += 1
            if j - i > 1: out[order[i:j]] = (i + j - 1) / 2
            i = j
        return out
    xr, yr = rank(x), rank(y); xrc, yrc = xr - xr.mean(), yr - yr.mean(); den = xrc.norm() * yrc.norm()
    return {"feature": x_key, "target": y_key, "sample_count": len(values), "pearson": pearson,
            "spearman": float((xrc * yrc).sum() / den) if den.item() else None}


def selection(rows: list[dict]) -> tuple[set[tuple], dict[tuple, str]]:
    """Select before controlled outcomes using baseline condition/error strata."""
    keys = {(r["seed"], r["update"], r["parameter_id"]): r for r in rows}
    selected: dict[tuple, str] = {}
    by_cond = sorted(rows, key=lambda r: (float(r["effective_condition_number"]), r["seed"], r["update"], r["parameter_id"]))
    by_error = sorted(rows, key=lambda r: (float(r["update_direction_error"]), r["seed"], r["update"], r["parameter_id"]))
    candidates = [(by_cond[0], "low_condition"), (by_cond[len(by_cond)//2], "medium_condition"), (by_cond[-1], "high_condition"),
                  (by_error[0], "low_update_error"), (by_error[len(by_error)//2], "medium_update_error"), (by_error[-1], "high_update_error")]
    for row, reason in candidates:
        key = (row["seed"], row["update"], row["parameter_id"])
        if key in keys and key not in selected: selected[key] = reason
    return set(selected), selected


def plot_reports(output: Path, sweep: list[dict], controlled: list[dict], transfers: list[dict],
                 polar_rows: list[dict] | None = None, gap_rows: list[dict] | None = None,
                 prediction: list[dict] | None = None) -> None:
    try: import matplotlib.pyplot as plt
    except ImportError:
        (output / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV files remain complete.\n"); return
    def grouped(rows, x, y, filename, xlabel, ylabel):
        plt.figure(figsize=(7, 5))
        for label in sorted({r[x] for r in rows}):
            vals = [r for r in rows if r[x] == label]; vals.sort(key=lambda r: r["steps"])
            if vals: plt.plot([r["steps"] for r in vals], [r[y] for r in vals], alpha=.15, color="tab:blue")
        means = {}
        for r in rows: means.setdefault(r["steps"], []).append(r[y])
        ks = sorted(means); plt.plot(ks, [sum(means[k])/len(means[k]) for k in ks], marker="o", color="black", label="mean")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(output / filename, dpi=140); plt.close()
    grouped(sweep, "tensor_key", "update_cosine", "k_vs_update_cosine.png", "Newton--Schulz steps", "update cosine")
    grouped(sweep, "tensor_key", "update_relative_l2", "k_vs_update_relative_l2.png", "Newton--Schulz steps", "update relative L2")
    if gap_rows:
        plt.figure(figsize=(7, 5))
        x = [float(r["production_update_cosine"]) for r in gap_rows]
        y = [float(r["exact_polar_cosine"]) for r in gap_rows]
        plt.scatter(x, y, s=8, alpha=.35)
        lo, hi = min(x + y), max(x + y); plt.plot([lo, hi], [lo, hi], "k--", linewidth=.8)
        plt.xlabel("production finite-step update cosine"); plt.ylabel("exact polar update cosine")
        plt.tight_layout(); plt.savefig(output / "production_vs_exact_polar_cosine.png", dpi=140); plt.close()
    plt.figure(figsize=(7, 5));
    for kind, color in (("sv_only", "tab:blue"), ("rotation", "tab:orange")):
        vals = [r for r in controlled if r["kind"] == kind and r["band"] == "tail" and r["epsilon"] == .01 and r["transform"] != "exact_polar"]
        byk = {}
        for r in vals: byk.setdefault(r["steps"], []).append(r["update_relative_l2"])
        ks = sorted(byk); plt.plot(ks, [sum(byk[k])/len(byk[k]) for k in ks], marker="o", label=kind, color=color)
    plt.xlabel("Newton--Schulz steps"); plt.ylabel("tail perturbation update relative L2"); plt.legend(); plt.tight_layout(); plt.savefig(output / "controlled_channels_vs_k.png", dpi=140); plt.close()
    if transfers:
        plt.figure(figsize=(7, 5))
        for steps in sorted({r["steps"] for r in transfers}):
            vals = [r for r in transfers if r["steps"] == steps]
            vals.sort(key=lambda r: r["normalized_sigma"])
            plt.plot([r["normalized_sigma"] for r in vals], [r["f_k"] for r in vals], alpha=.35, label=f"K={steps}")
        plt.xlabel("normalized singular value"); plt.ylabel("f_K(sigma)"); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(output / "transfer_function_curves.png", dpi=140); plt.close()
        plt.figure(figsize=(7, 5))
        for steps in sorted({r["steps"] for r in transfers}):
            vals = [r for r in transfers if r["steps"] == steps]
            vals.sort(key=lambda r: r["normalized_sigma"]); plt.plot([r["normalized_sigma"] for r in vals], [abs(r["a_mag"]) for r in vals], alpha=.3, label=f"K={steps}")
        plt.xlabel("normalized singular value"); plt.ylabel("|f'_K(sigma)|"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(output / "transfer_derivative_curves.png", dpi=140); plt.close()
        # Overlay deterministic index bands (10% head/tail, middle centered)
        # on the sensitivity curve; this is a visual guide, not a new metric.
        plt.figure(figsize=(7, 5))
        vals = [r for r in transfers if r["steps"] == max(int(x["steps"]) for x in transfers)]
        vals.sort(key=lambda r: r["normalized_sigma"], reverse=True)
        n = len(vals); width = max(1, n // 10)
        for label, subset, color in (("head", vals[:width], "tab:blue"),
                                     ("middle", vals[n//2-width//2:n//2+width//2], "tab:green"),
                                     ("tail", vals[-width:], "tab:red")):
            plt.scatter([r["normalized_sigma"] for r in subset], [r["a_mag"] for r in subset], s=8, alpha=.5, label=label, color=color)
        plt.xlabel("normalized singular value"); plt.ylabel("|f'_K(sigma)| (largest K)"); plt.legend()
        plt.tight_layout(); plt.savefig(output / "real_spectrum_sensitivity_bands.png", dpi=140); plt.close()
    if prediction:
        plt.figure(figsize=(7, 5))
        x = [float(r["predicted_sv_output_error"]) for r in prediction]
        y = [float(r["actual_update_error"]) for r in prediction]
        plt.scatter(x, y, s=8, alpha=.25)
        plt.xlabel("predicted first-order singular-value output error"); plt.ylabel("actual finite-step update error")
        plt.tight_layout(); plt.savefig(output / "predicted_vs_actual_sv_error.png", dpi=140); plt.close()
    if polar_rows:
        plt.figure(figsize=(7, 5))
        plt.hist([float(r["update_cosine"]) for r in polar_rows], bins=30, alpha=.8)
        plt.xlabel("exact polar update cosine"); plt.ylabel("tensor count")
        plt.tight_layout(); plt.savefig(output / "exact_polar_ceiling_distribution.png", dpi=140); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    ap.add_argument("--output", type=Path, default=ROOT / "reports/muon_ns_sensitivity")
    ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args(); started = time.perf_counter(); args.output.mkdir(parents=True, exist_ok=True)
    paths = discover(args.reports_root)
    if len(paths) != 10: raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    sweep: list[dict] = []; polar_rows: list[dict] = []; gap_rows: list[dict] = []; controlled: list[dict] = []; prediction: list[dict] = []; source_rows: list[dict] = []
    transfer_path = args.output / "singular_transfer_function.csv"
    transfer_handle = transfer_path.open("w", newline="")
    transfer_writer = csv.DictWriter(transfer_handle, fieldnames=TRANSFER_FIELDS)
    transfer_writer.writeheader()
    # The complete transfer table is streamed to disk.  Plotting only needs a
    # deterministic sparse sample; retaining all per-mode rows caused a large
    # Python-object overhead for the 300 formal tensors.
    transfer_plot_rows: list[dict] = []
    transfer_row_number = 0
    # Keep only lightweight references during the first pass.  Retaining all
    # snapshot tensors/SVDs here would duplicate billions of FP32 elements
    # across the ten formal snapshots and make the offline study needlessly
    # memory-bound.  Representative tensors are reloaded in a second pass.
    source_refs: dict[tuple, tuple[Path, str]] = {}
    transform_cfg = None
    for seed, update, path in paths:
        snap = load_snapshot(path); transform_cfg = snap["metadata"]["muon_transform"]
        kwargs = {"coefficients": tuple(transform_cfg["coefficients"]), "eps": float(transform_cfg["eps"])}
        for item in snap["tensors"]:
            if len(item["shape"]) != 2: continue
            if args.max_tensors and len(source_rows) >= args.max_tensors: break
            source = item["tensor"].float(); d = decompose(source); q = quantize(source, QUANTIZER); qd = decompose(q)
            key = (seed, update, item["parameter_id"]); tensor_key = f"s{seed}/u{update}/{item['parameter_id']}"
            ref_sweep = transform_sweep(source, DEFAULT_K_GRID, **kwargs); q_sweep = transform_sweep(q, DEFAULT_K_GRID, **kwargs)
            ref_prod = transform(source, steps=int(transform_cfg["steps"]), **kwargs); q_prod = transform(q, steps=int(transform_cfg["steps"]), **kwargs)
            if int(transform_cfg["steps"]) in ref_sweep and not torch.equal(ref_prod, ref_sweep[int(transform_cfg["steps"])]):
                raise RuntimeError("batched Newton--Schulz path diverged from production implementation")
            prod = _ratios(ref_prod, q_prod, "update"); bands = band_indices(d.singular_values, threshold=THRESHOLD)
            cond = float(d.singular_values[0] / d.singular_values[bands["active"][-1]]) if bands["active"].numel() else None
            source_row = {"seed": seed, "update": update, "parameter_id": item["parameter_id"], "parameter_name": item.get("name", "<unknown>"), "tensor_key": tensor_key,
                          "shape": str(item["shape"]), "effective_rank": int(bands["active"].numel()), "effective_condition_number": cond,
                          "update_direction_error": 1 - float(prod["update_cosine"]), "production_update_cosine": prod["update_cosine"]}
            source_rows.append(source_row); source_refs[key] = (path, item["parameter_id"])
            for steps in DEFAULT_K_GRID:
                a = ref_sweep[steps]; b = q_sweep[steps]; m = _ratios(a, b, "update")
                sweep.append({"seed": seed, "update": update, "parameter_id": item["parameter_id"], "tensor_key": tensor_key, "steps": steps,
                              "is_production_steps": steps == int(transform_cfg["steps"]), **m})
            polar_m = _ratios(exact_polar(source), exact_polar(q), "update")
            polar_rows.append({"seed": seed, "update": update, "parameter_id": item["parameter_id"], "tensor_key": tensor_key, **polar_m})
            production_cos = float(prod["update_cosine"]); polar_cos = float(polar_m["update_cosine"])
            gap_rows.append({"seed": seed, "update": update, "parameter_id": item["parameter_id"], "tensor_key": tensor_key, "production_update_cosine": production_cos, "exact_polar_cosine": polar_cos,
                             "production_gap_to_polar": polar_cos - production_cos, "finite_step_recoverable_fraction": (1 - production_cos - (1 - polar_cos)) / (1 - production_cos) if production_cos != 1 else None})
            norm = float(source.norm()); s = d.singular_values
            for steps in DEFAULT_K_GRID:
                mapped, deriv = scalar_map_and_derivative(s, matrix_norm=norm, steps=steps, coefficients=kwargs["coefficients"], eps=kwargs["eps"])
                for mode in range(s.numel()):
                    transfer_row = {"seed": seed, "update": update, "parameter_id": item["parameter_id"], "tensor_key": tensor_key, "steps": steps, "mode": mode,
                                   "normalized_sigma": float(s[mode] / (norm + kwargs["eps"])), "sigma": float(s[mode]), "f_k": float(mapped[mode]),
                                   "a_mag": float(abs(deriv[mode])), "a_dir": float(abs(mapped[mode]) / s[mode].abs().clamp_min(1e-30))}
                    transfer_writer.writerow(transfer_row)
                    if not args.skip_plots and transfer_row_number % 100 == 0:
                        transfer_plot_rows.append(transfer_row)
                    transfer_row_number += 1
                delta_sigma = qd.singular_values - s; pred = torch.sqrt((deriv * delta_sigma).square().sum())
                actual = (b - a).norm()
                prediction.append({"seed": seed, "update": update, "parameter_id": item["parameter_id"], "tensor_key": tensor_key, "steps": steps,
                                   "predicted_sv_output_error": float(pred), "actual_update_error": float(actual), "update_cosine_error": 1 - float(m["update_cosine"]),
                                   "actual_singular_value_l2": float(delta_sigma.norm())})
    chosen, reasons = selection(source_rows)
    # Reload only the deterministic representative subset selected from
    # baseline condition/error strata; no intervention result is consulted.
    for key in sorted(chosen):
        path, parameter_id = source_refs[key]
        snap = load_snapshot(path)
        item = next(item for item in snap["tensors"] if item["parameter_id"] == parameter_id)
        source = item["tensor"].float(); d = decompose(source); q = quantize(source, QUANTIZER); qd = decompose(q)
        cfg = snap["metadata"]["muon_transform"]
        bands = band_indices(d.singular_values, threshold=THRESHOLD)
        base = next(row for row in source_rows if (row["seed"], row["update"], row["parameter_id"]) == key)
        kwargs = {"coefficients": tuple(cfg["coefficients"]), "eps": float(cfg["eps"])}; tensor_key = base["tensor_key"]
        reference_sweep = transform_sweep(source, DEFAULT_K_GRID, **kwargs)
        for epsilon in EPSILONS:
            for band_name, indices in (("head", bands["head"]), ("tail", bands["tail"])):
                for kind, builder in (("sv_only", sv_only_perturbation), ("rotation", rotation_perturbation)):
                    observed = builder(d.u, d.singular_values, d.vh, indices, epsilon, d.matrix.norm())
                    actual = float((observed - source).norm() / source.norm()) if source.norm().item() else None
                    observed_sweep = transform_sweep(observed, DEFAULT_K_GRID, **kwargs)
                    for steps in DEFAULT_K_GRID:
                        m = {"steps": steps, **_ratios(reference_sweep[steps], observed_sweep[steps], "update"),
                             **_ratios(source, observed, "raw")}
                        controlled.append({"seed": key[0], "update": key[1], "parameter_id": key[2], "tensor_key": tensor_key, "selection_reason": reasons[key], "kind": kind,
                                           "band": band_name, "epsilon": epsilon, "steps": steps, "transform": "finite_step", "actual_relative_frobenius": actual,
                                           "target_relative_frobenius": epsilon, **m})
                    pm = _ratios(exact_polar(source), exact_polar(observed), "update")
                    controlled.append({"seed": key[0], "update": key[1], "parameter_id": key[2], "tensor_key": tensor_key, "selection_reason": reasons[key], "kind": kind,
                                       "band": band_name, "epsilon": epsilon, "steps": "polar", "transform": "exact_polar", "actual_relative_frobenius": actual,
                                       "target_relative_frobenius": epsilon, **pm})
    transfer_handle.close()
    shutil.copyfile(transfer_path, args.output / "real_spectrum_sensitivity.csv")
    write_csv(args.output / "ns_iteration_sweep.csv", sweep); write_csv(args.output / "exact_polar_metrics.csv", polar_rows); write_csv(args.output / "finite_step_gap_summary.csv", gap_rows)
    write_csv(args.output / "first_order_prediction.csv", prediction)
    write_csv(args.output / "controlled_sv_perturbation_vs_k.csv", controlled); write_csv(args.output / "controlled_rotation_perturbation_vs_k.csv", [r for r in controlled if r["kind"] == "rotation"])
    write_csv(args.output / "representative_tensor_manifest.csv", [{**r, "selection_reason": reasons[(r["seed"], r["update"], r["parameter_id"])]} for r in source_rows if (r["seed"], r["update"], r["parameter_id"]) in chosen])
    correlations = [correlation(prediction, "predicted_sv_output_error", "actual_update_error"), correlation(prediction, "predicted_sv_output_error", "update_cosine_error")]
    write_csv(args.output / "spectral_update_correlations.csv", correlations)
    elapsed = time.perf_counter() - started
    production_k = int(transform_cfg["steps"])
    methodology = f"""# Methodology

This read-only CPU study consumed the ten existing formal FP32 Muon snapshots. Eligible 2D tensors only: {len(source_rows)}. The production transform is called directly from `optim.muon_reference.zeropower_newton_schulz`; its normalization is `X/(||X||_F+eps)`, with transpose for tall matrices, and coefficients `(3.4445,-4.7750,2.0315)` for `{production_k}` iterations. No optimizer code is changed.

The scalar transfer is `g(x)=a*x+b*x^3+c*x^5` applied after the same normalization; derivatives use the analytic recurrence `g'(x)=a+3*b*x^2+5*c*x^4`. `A_mag=|f'_K|` and `A_dir=|f_K|/max(|sigma|,1e-30)` are descriptive, not causal condition numbers. Effective modes satisfy `sigma/sigma_max >= {THRESHOLD}`. Exact polar uses reduced SVD `U@Vh`, with numerically inactive modes removed at the same threshold.

Controlled singular-value-only and left-singular-vector rotation perturbations use deterministic adjacent-mode generators and target relative Frobenius magnitudes `{EPSILONS}`. Representatives were selected before intervention outcomes from low/median/high effective condition and low/median/high production update error; deduplicated count is {len(chosen)}. First-order prediction is `sqrt(sum((f'_K * (sigma_q-sigma))^2))` and does not claim to explain subspace rotation. Runtime was {elapsed:.2f}s CPU.
"""
    (args.output / "methodology.md").write_text(methodology)
    def mean_metric(rows: list[dict], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "") and finite(row.get(key))]
        return statistics.fmean(values) if values else None
    summary_lines = [
        "# Summary", "",
        f"Analyzed {len(source_rows)} eligible 2D tensors from ten formal snapshots.",
        f"Production Newton--Schulz K={production_k}; K grid={DEFAULT_K_GRID}.",
        f"Runtime: {elapsed:.2f}s CPU.", "",
        "## Mean INT4 dynamic fidelity by iteration count", "",
        "| K | update cosine | update relative L2 | update norm ratio |",
        "|---:|---:|---:|---:|",
    ]
    for steps in DEFAULT_K_GRID:
        rows_k = [row for row in sweep if int(row["steps"]) == steps]
        summary_lines.append(f"| {steps} | {mean_metric(rows_k, 'update_cosine'):.6f} | {mean_metric(rows_k, 'update_relative_l2'):.6f} | {mean_metric(rows_k, 'update_norm_ratio'):.6f} |")
    summary_lines += [
        "", "## Exact-polar comparison", "",
        f"Mean production-K cosine: `{mean_metric(gap_rows, 'production_update_cosine'):.6f}`; mean exact-polar cosine: `{mean_metric(polar_rows, 'update_cosine'):.6f}`.",
        f"Mean descriptive gap (exact polar minus production): `{mean_metric(gap_rows, 'production_gap_to_polar'):.6f}`; this is not a causal decomposition.",
        "", "Controlled perturbation details and mode-level sensitivity are documented in `methodology.md`; raw tables are in the CSV outputs.", "",
    ]
    (args.output / "summary.md").write_text("\n".join(summary_lines))
    if not args.skip_plots: plot_reports(args.output, sweep, controlled, transfer_plot_rows, polar_rows, gap_rows, prediction)
    print(f"[done] {len(source_rows)} tensors; {elapsed:.1f}s CPU; output={args.output}")


if __name__ == "__main__": main()
