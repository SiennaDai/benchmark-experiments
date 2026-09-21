#!/usr/bin/env python3
"""Offline spectral-band error/contribution decomposition for INT4 Muon."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optim import muon_reference  # noqa: E402
from optim.muon_spectral_contribution import (  # noqa: E402
    ACCOUNTING_BANDS, BANDS, EPSILON, KINDS, block_energy, controlled_sensitivity, contribution_proxy,
    decompose, metrics, quantize, restoration,
)
from optim.muon_spectral_sensitivity import SPECTRAL_THRESHOLD  # noqa: E402
from optim.muon_update_fidelity import _ratios, load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
RESTORE_GROUPS = {
    "none": (), "head": ("head",), "middle": ("middle",), "tail": ("tail",),
    "head_middle": ("head", "middle"), "middle_tail": ("middle", "tail"),
    "head_tail": ("head", "tail"), "all": ("head", "middle", "tail"),
}
POLAR_GROUPS = frozenset(("none", "head", "middle", "tail", "all"))


def discover_snapshots(root: Path) -> list[tuple[int, int, Path]]:
    groups: dict[tuple[int, str], dict[int, Path]] = {}
    for path in sorted(root.rglob("update_*.pt")):
        if "muon_momentum_snapshots" not in path.parts:
            continue
        try:
            snap = load_snapshot(path)
            seed = int(snap["metadata"]["seeds"]["seed"])
            update = int(snap["metadata"]["update"])
        except (KeyError, ValueError, RuntimeError, OSError):
            continue
        if seed in SEEDS and update in LANDMARKS:
            groups.setdefault((seed, str(path.parent)), {})[update] = path
    selected: list[tuple[int, int, Path]] = []
    for seed in SEEDS:
        candidates = sorted((group, values) for (s, group), values in groups.items()
                            if s == seed and set(values) == set(LANDMARKS))
        if candidates:
            values = candidates[0][1]
            selected.extend((seed, update, values[update]) for update in LANDMARKS)
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def ident(seed: int, update: int, item: dict) -> dict:
    return {"seed": seed, "update": update,
            "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
            "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")),
            "shape": str(item["shape"])}


def transform_kwargs(snapshot: dict) -> dict:
    config = snapshot["metadata"]["muon_transform"]
    return {"steps": int(config["steps"]),
            "coefficients": tuple(float(x) for x in config["coefficients"]),
            "eps": float(config["eps"])}


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pearson_spearman(rows: list[dict], feature: str, target: str) -> dict:
    values = [(float(row[feature]), float(row[target])) for row in rows
              if finite(row.get(feature)) and finite(row.get(target))]
    if len(values) < 2:
        return {"feature": feature, "target": target, "sample_count": len(values), "pearson": None, "spearman": None}
    x = torch.tensor([a for a, _ in values], dtype=torch.float64)
    y = torch.tensor([b for _, b in values], dtype=torch.float64)

    def corr(a, b):
        ac, bc = a - a.mean(), b - b.mean(); denominator = ac.norm() * bc.norm()
        return float((ac * bc).sum() / denominator) if denominator.item() else None

    def rank(a):
        order = torch.argsort(a, stable=True); out = torch.empty_like(a)
        out[order] = torch.arange(a.numel(), dtype=a.dtype)
        i = 0
        while i < a.numel():
            j = i + 1
            while j < a.numel() and a[order[j]] == a[order[i]]:
                j += 1
            if j - i > 1:
                out[order[i:j]] = (i + j - 1) / 2
            i = j
        return out

    return {"feature": feature, "target": target, "sample_count": len(values),
            "pearson": corr(x, y), "spearman": corr(rank(x), rank(y))}


def select_representatives(rows: list[dict], max_per_stratum: int = 1) -> dict[tuple, str]:
    """Select before intervention, using baseline update distortion only."""
    ordered = sorted(rows, key=lambda row: (float(row.get("baseline_update_error", float("inf"))),
                                           row["seed"], row["update"], row["parameter_id"]))
    if not ordered:
        return {}
    picks = [(ordered[0], "low_baseline_error"),
             (ordered[(len(ordered) - 1) // 2], "median_baseline_error"),
             (ordered[-1], "high_baseline_error")]
    result: dict[tuple, str] = {}
    for row, reason in picks:
        key = (row["seed"], row["update"], row["parameter_id"])
        result.setdefault(key, reason)
    return result


def make_plots(out: Path, energy_rows: list[dict], sensitivity_rows: list[dict],
               restoration_rows: list[dict], efficiency_rows: list[dict], representative_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs are complete.\n")
        return

    def means(rows, field, group):
        values = defaultdict(list)
        for row in rows:
            if finite(row.get(field)):
                values[row[group]].append(float(row[field]))
        return [(key, sum(v) / len(v)) for key, v in sorted(values.items())]

    plt.figure(figsize=(8, 5))
    vals = means(energy_rows, "associated_fraction", "band")
    plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.ylabel("mean residual-energy fraction")
    plt.title("Actual INT4 residual by spectral band"); plt.tight_layout(); plt.savefig(out / "residual_energy_by_band.png", dpi=140); plt.close()

    plt.figure(figsize=(8, 5))
    for kind in KINDS:
        vals = means([r for r in sensitivity_rows if r["perturbation_kind"] == kind], "local_sensitivity", "band")
        plt.plot([x for x, _ in vals], [y for _, y in vals], marker="o", label=kind)
    plt.ylabel("local update relative-L2 / epsilon"); plt.title("Controlled sensitivity"); plt.legend(); plt.tight_layout(); plt.savefig(out / "controlled_sensitivity_by_band.png", dpi=140); plt.close()

    plt.figure(figsize=(8, 5))
    vals = means([r for r in restoration_rows if r["restore_group"] in BANDS], "update_cosine_gain", "restore_group")
    plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.ylabel("mean update cosine gain")
    plt.title("Direct update recovery by restoring one band"); plt.tight_layout(); plt.savefig(out / "restoration_gain_by_band.png", dpi=140); plt.close()

    plt.figure(figsize=(8, 5))
    vals = means([r for r in efficiency_rows if r["restore_group"] in BANDS], "cosine_gain_per_residual_fraction", "restore_group")
    plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.ylabel("cosine gain / removed residual fraction")
    plt.title("Recovery efficiency"); plt.tight_layout(); plt.savefig(out / "recovery_efficiency_by_band.png", dpi=140); plt.close()

    plt.figure(figsize=(8, 5))
    for band in BANDS:
        subset = [r for r in energy_rows if r["band"] == band]
        plt.scatter([r["associated_fraction"] for r in subset if finite(r.get("associated_fraction"))],
                    [r["baseline_update_error"] for r in subset if finite(r.get("associated_fraction"))], s=8, label=band)
    plt.xlabel("band residual-energy fraction"); plt.ylabel("baseline update direction error"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "sensitivity_vs_actual_contribution.png", dpi=140); plt.close()

    plt.figure(figsize=(8, 5))
    for key, label in (("K=5", "update_cosine"), ("exact_polar", "polar_update_cosine")):
        vals = means([r for r in restoration_rows if r["restore_group"] in BANDS], label, "restore_group")
        plt.plot([x for x, _ in vals], [y for _, y in vals], marker="o", label=key)
    plt.xlabel("restored band"); plt.ylabel("cosine"); plt.title("K=5 vs exact-polar restoration"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "k5_vs_exact_polar_restoration.png", dpi=140); plt.close()

    # Deterministic representative budget curves by update error strata.
    for key in sorted({(r["seed"], r["update"], r["parameter_id"]) for r in representative_rows}):
        subset = [r for r in representative_rows if (r["seed"], r["update"], r["parameter_id"]) == key]
        if not subset:
            continue
        plt.figure(figsize=(8, 5))
        for band in BANDS:
            values = [r for r in subset if r["restore_group"] == band]
            if values:
                plt.bar(band, values[0]["update_cosine_gain"])
        plt.title(f"s{key[0]} u{key[1]} {key[2]}"); plt.ylabel("update cosine gain"); plt.tight_layout()
        safe = str(key[2]).replace(".", "_")
        plt.savefig(out / f"representative_{key[0]}_{key[1]}_{safe}.png", dpi=140); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/muon_spectral_contribution")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(); started = time.perf_counter()
    paths = discover_snapshots(args.reports_root)
    if len(paths) != 10:
        raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    args.output.mkdir(parents=True, exist_ok=True)
    energy_rows: list[dict] = []; sensitivity_rows: list[dict] = []
    restoration_rows: list[dict] = []; efficiency_rows: list[dict] = []
    tensor_rows: list[dict] = []; proxy_rows: list[dict] = []
    cache: dict[tuple, tuple] = {}

    for seed, update, path in paths:
        snapshot = load_snapshot(path); kwargs = transform_kwargs(snapshot)
        for item in snapshot["tensors"]:
            if len(item["shape"]) != 2:
                continue
            d = decompose(item["tensor"]); source = d.matrix; q = quantize(source, "int4-dynamic-b2048")
            reference_update = muon_reference.zeropower_newton_schulz(source.clone(), **kwargs)
            baseline_update = muon_reference.zeropower_newton_schulz(q.clone(), **kwargs)
            raw = _ratios(source, q, "raw"); update_metrics = _ratios(reference_update, baseline_update, "update")
            base = ident(seed, update, item)
            base.update({"effective_rank": d.effective_rank,
                         "head_indices": str(d.bands["head"].tolist()),
                         "middle_indices": str(d.bands["middle"].tolist()),
                         "tail_indices": str(d.bands["tail"].tolist()),
                         "raw_relative_l2": raw.get("raw_relative_l2"), "raw_cosine": raw.get("raw_cosine"),
                         "baseline_update_relative_l2": update_metrics.get("update_relative_l2"),
                         "baseline_update_cosine": update_metrics.get("update_cosine"),
                         "baseline_update_error": (1.0 - update_metrics["update_cosine"] if isinstance(update_metrics.get("update_cosine"), float) else None)})
            blocks = block_energy(d, q)
            for band in ACCOUNTING_BANDS:
                energy_rows.append(base | {"band": band,
                    "associated_energy": blocks[f"{band}_associated_energy"],
                    "associated_fraction": blocks[f"{band}_associated_fraction"],
                    "total_residual_energy": blocks["total_residual_energy"],
                    "cross_band_mixing_fraction": blocks["cross_band_mixing_fraction"],
                    "effective_rank": d.effective_rank})
            block_row = base | blocks
            block_row["cross_band_mixing_energy_fraction"] = blocks["cross_band_mixing_fraction"]
            tensor_rows.append(block_row)

            key = (seed, update, base["parameter_id"])
            cache[key] = (d, q, source, reference_update, baseline_update, kwargs, base)
            for band in BANDS:
                for kind in KINDS:
                    row = controlled_sensitivity(d, band, kind, epsilon=EPSILON, transform_kwargs=kwargs)
                    sensitivity_rows.append(base | row)

            polar_source = None
            for group, selected in RESTORE_GROUPS.items():
                corrected = restoration(d, q, selected)
                m = metrics(source, corrected, reference_update=reference_update, transform_kwargs=kwargs)
                baseline_l2 = update_metrics.get("update_relative_l2")
                gain = (m.get("update_cosine") - update_metrics.get("update_cosine")
                        if finite(m.get("update_cosine")) and finite(update_metrics.get("update_cosine")) else None)
                reduction = ((baseline_l2 - m.get("update_relative_l2")) / baseline_l2
                             if finite(baseline_l2) and baseline_l2 != 0 and finite(m.get("update_relative_l2")) else None)
                # Exact-polar readouts are retained for the baseline, each
                # single-band ablation, and the all-band control.  Pairwise
                # combinations remain available at production K=5 without
                # multiplying the expensive SVD work by three.
                if group in POLAR_GROUPS:
                    if polar_source is None:
                        polar_ref = torch.linalg.svd(source, full_matrices=False)
                        polar_source = polar_ref[0] @ polar_ref[2]
                    polar_q = torch.linalg.svd(corrected, full_matrices=False)
                    polar_observed = polar_q[0] @ polar_q[2]
                    polar = _ratios(polar_source, polar_observed, "polar")
                else:
                    polar = {"polar_cosine": None, "polar_relative_l2": None, "polar_norm_ratio": None}
                removal = source - corrected
                residual = source - q
                removed_energy = removal.square().sum()
                total_energy = residual.square().sum()
                restoration_rows.append(base | {"restore_group": group, "restored_bands": ",".join(selected),
                    **m, "update_cosine_gain": gain, "update_l2_reduction_fraction": reduction,
                    "removed_residual_energy": float(removed_energy),
                    "removed_residual_fraction": float(removed_energy / total_energy) if total_energy.item() else None,
                    "polar_update_cosine": polar.get("polar_cosine"),
                    "polar_update_relative_l2": polar.get("polar_relative_l2"),
                    "polar_update_norm_ratio": polar.get("polar_norm_ratio")})
                efficiency_rows.append(base | {"restore_group": group,
                    "update_cosine_gain": gain, "update_l2_reduction_fraction": reduction,
                    "removed_residual_fraction": float(removed_energy / total_energy) if total_energy.item() else None,
                    "cosine_gain_per_residual_fraction": (gain / float(removed_energy / total_energy)
                                                           if gain is not None and total_energy.item() and removed_energy.item() else None),
                    "l2_reduction_per_residual_fraction": (reduction / float(removed_energy / total_energy)
                                                            if reduction is not None and total_energy.item() and removed_energy.item() else None)})
            # One simple, untuned error x sensitivity proxy per band.
            for band in BANDS:
                energy = next(row for row in energy_rows[::-1] if row["parameter_id"] == base["parameter_id"] and row["seed"] == seed and row["update"] == update and row["band"] == band)
                sens = next(row for row in sensitivity_rows[::-1] if row["parameter_id"] == base["parameter_id"] and row["seed"] == seed and row["update"] == update and row["band"] == band and row["perturbation_kind"] == "magnitude")
                proxy_rows.append(base | {"band": band, "sensitivity_kind": "magnitude",
                    "residual_energy": energy["associated_energy"], "residual_energy_fraction": energy["associated_fraction"],
                    "local_sensitivity": sens["local_sensitivity"],
                    "contribution_proxy": contribution_proxy(energy["associated_energy"], sens["local_sensitivity"])})
        del snapshot
        print(f"processed seed={seed} update={update}", flush=True)

    reps = select_representatives(tensor_rows)
    representative_rows = [row | {"selection_reason": reps[(row["seed"], row["update"], row["parameter_id"])]}
                           for row in restoration_rows
                           if (row["seed"], row["update"], row["parameter_id"]) in reps and row["restore_group"] in BANDS]
    correlations: list[dict] = []
    correlations.append(pearson_spearman(tensor_rows, "raw_relative_l2", "baseline_update_error"))
    for band in ACCOUNTING_BANDS:
        correlations.append(pearson_spearman(
            [row for row in energy_rows if row["band"] == band],
            "associated_fraction", "baseline_update_error") | {"band": band})
    for band in BANDS:
        correlations.append(pearson_spearman(
            [row for row in sensitivity_rows if row["band"] == band and row["perturbation_kind"] == "magnitude"],
            "local_sensitivity", "baseline_update_error") | {"band": band})
        correlations.append(pearson_spearman(
            [row for row in proxy_rows if row["band"] == band],
            "contribution_proxy", "baseline_update_error") | {"band": band})

    write_csv(args.output / "band_residual_energy.csv", energy_rows)
    write_csv(args.output / "band_controlled_sensitivity.csv", sensitivity_rows)
    write_csv(args.output / "band_restoration_metrics.csv", restoration_rows)
    write_csv(args.output / "band_efficiency.csv", efficiency_rows)
    write_csv(args.output / "band_contribution_proxy.csv", proxy_rows)
    write_csv(args.output / "tensor_contribution_summary.csv", tensor_rows)
    write_csv(args.output / "spectral_contribution_correlations.csv", correlations)
    manifest = [{"seed": key[0], "update": key[1], "parameter_id": key[2], "selection_reason": reason}
                for key, reason in sorted(reps.items())]
    write_csv(args.output / "representative_tensor_manifest.csv", manifest)
    if not args.skip_plots:
        make_plots(args.output, energy_rows, sensitivity_rows, restoration_rows, efficiency_rows, representative_rows)

    by_band = {}
    for band in BANDS:
        values = [float(row["associated_fraction"]) for row in energy_rows if row["band"] == band and finite(row.get("associated_fraction"))]
        sens = [float(row["local_sensitivity"]) for row in sensitivity_rows if row["band"] == band and row["perturbation_kind"] == "magnitude" and finite(row.get("local_sensitivity"))]
        gains = [float(row["update_cosine_gain"]) for row in restoration_rows if row["restore_group"] == band and finite(row.get("update_cosine_gain"))]
        by_band[band] = {"mean_residual_fraction": sum(values) / len(values) if values else None,
                         "median_residual_fraction": float(torch.tensor(values).median()) if values else None,
                         "mean_sensitivity": sum(sens) / len(sens) if sens else None,
                         "median_sensitivity": float(torch.tensor(sens).median()) if sens else None,
                         "mean_restoration_gain": sum(gains) / len(gains) if gains else None,
                         "median_restoration_gain": float(torch.tensor(gains).median()) if gains else None}
    seed_summary = []
    for seed in SEEDS:
        for band in BANDS:
            values = [float(row["associated_fraction"]) for row in energy_rows if row["band"] == band and row["seed"] == seed and finite(row.get("associated_fraction"))]
            sens = [float(row["local_sensitivity"]) for row in sensitivity_rows if row["band"] == band and row["perturbation_kind"] == "magnitude" and row["seed"] == seed and finite(row.get("local_sensitivity"))]
            gains = [float(row["update_cosine_gain"]) for row in restoration_rows if row["restore_group"] == band and row["seed"] == seed and finite(row.get("update_cosine_gain"))]
            seed_summary.append({"seed": seed, "band": band,
                "mean_residual_fraction": sum(values) / len(values) if values else None,
                "median_residual_fraction": float(torch.tensor(values).median()) if values else None,
                "mean_sensitivity": sum(sens) / len(sens) if sens else None,
                "median_sensitivity": float(torch.tensor(sens).median()) if sens else None,
                "mean_restoration_gain": sum(gains) / len(gains) if gains else None,
                "median_restoration_gain": float(torch.tensor(gains).median()) if gains else None})
    write_csv(args.output / "band_seed_summary.csv", seed_summary)
    elapsed = time.perf_counter() - started
    lines = ["# Offline spectral-band contribution analysis", "",
             f"Coverage: 10 formal snapshots, {len(tensor_rows)} eligible 2D Muon tensors. Runtime: {elapsed:.2f}s CPU.",
             "", "The production INT4 dynamic b2048 quantizer and production Muon transform are reused unchanged.",
             f"Active modes use sigma/sigma_max >= {SPECTRAL_THRESHOLD:g}; head, centered-middle, and tail are the established 10%-of-active-index bands. An explicit other band accounts for the remaining active modes.",
             "Residual coordinates are E_hat=U.T(Q(M)-M)V. The nine named row/column blocks plus explicit other blocks are orthogonal; high-level named-band attribution uses disjoint row-associated blocks, while cross-band mixing and other-mode energy are reported separately.",
             "Controlled sensitivity uses epsilon=0.001. Magnitude perturbations change only selected singular values; orientation perturbations rotate left singular vectors with a deterministic skew generator. The sensitivity is empirical update relative-L2 / epsilon, not a derivative.",
             "", "## Aggregate band summaries", ""]
    for band in BANDS:
        lines.append(f"- {band}: {by_band[band]}")
    lines += ["", "## Interpretation", "",
              "Sensitivity and actual contribution are intentionally separate. A small high-sensitivity band can have low absolute contribution if its actual INT4 residual is small; restoration ablations are the direct descriptive contribution test.",
              "No causal or additive claim is made: the Muon transform can couple spectral components, and reduced-SVD unresolved residual is kept explicit."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    methodology = """# Methodology

This is a read-only CPU mechanism study over the existing ten FP32 Muon momentum snapshots. Only 2D Muon matrices are eligible. For each matrix, the existing production `int4-dynamic-b2048` blockwise dynamic roundtrip produces Q(M), and the production Newton--Schulz Muon transform is invoked on detached copies.

The FP32 reduced SVD defines active modes by `sigma_i/sigma_max >= 1e-6`. Head, centered middle, and tail use the established deterministic 10%-of-active-index convention from prior reports. The remaining active modes are an explicit `other` accounting band; this preserves the prior named bands without silently dropping modes. The FP32 spectral coordinates `E_hat=U^T(Q(M)-M)V` are partitioned into the nine named row/column blocks plus explicit-other blocks. Their squared energies sum to the projected residual energy; unresolved reduced-SVD energy is reported explicitly. High-level attribution uses disjoint row-associated components so it does not double-count cross-band entries. The all-off-diagonal blocks are additionally reported as cross-band mixing.

Controlled local sensitivity uses relative perturbation epsilon=0.001. Magnitude perturbations change only selected singular values with deterministic weights. Orientation perturbations rotate the selected left singular vectors with a deterministic adjacent-mode skew generator and bisection to match the target Frobenius norm. Sensitivity is the observed production-K=5 update relative-L2 divided by epsilon, and is not asserted to be a derivative. Direct restoration starts from Q(M) and subtracts selected actual row-band residual components. Combinations test non-additivity. Exact-polar readouts are included for restoration rows.

The contribution proxy is `sqrt(actual associated residual energy) * local magnitude sensitivity`; coefficients are not tuned. It is a mechanism heuristic, not an exact decomposition or deployable quantizer. Representative tensors are selected before intervention using low/median/high baseline update direction error.
"""
    (args.output / "methodology.md").write_text(methodology)
    print(f"wrote {args.output} in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
