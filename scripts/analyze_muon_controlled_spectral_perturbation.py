#!/usr/bin/env python3
"""Offline controlled head/tail spectral perturbation study for Muon."""
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
from optim.muon_controlled_spectral_perturbation import (  # noqa: E402
    actual_error_decomposition,
    controlled_row,
    decompose,
    rotation_only,
    singular_value_only,
    spectral_bands,
)
from optim.muon_spectral_sensitivity import quantize, spectral_metrics, transform  # noqa: E402
from optim.muon_update_fidelity import QUANTIZERS, _ratios, load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
EPSILONS = (0.0025, 0.005, 0.01, 0.02, 0.05)
QUANTIZER = "int4-dynamic-b2048"


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
    selected = []
    for seed in SEEDS:
        candidates = sorted((group, values) for (s, group), values in groups.items()
                            if s == seed and set(values) == set(LANDMARKS))
        if candidates:
            group, values = candidates[0]
            selected.extend((seed, update, values[update]) for update in LANDMARKS)
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        if keys:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)


def ident(item: dict, seed: int, update: int) -> dict:
    return {"seed": seed, "update": update,
            "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
            "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")),
            "shape": str(item["shape"])}


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def transform_kwargs(snapshot: dict) -> dict:
    config = snapshot["metadata"]["muon_transform"]
    return {"steps": int(config["steps"]),
            "coefficients": tuple(float(x) for x in config["coefficients"]),
            "eps": float(config["eps"])}


def cached_metrics(source: torch.Tensor, observed: torch.Tensor, reference_update: torch.Tensor,
                   kwargs: dict) -> dict:
    observed_update = transform(observed, **kwargs)
    result = _ratios(source, observed, "raw")
    result.update(_ratios(reference_update, observed_update, "update"))
    return result


def correlation(rows: list[dict], feature: str, target: str) -> dict:
    vals = [(float(r[feature]), float(r[target])) for r in rows if finite(r.get(feature)) and finite(r.get(target))]
    if len(vals) < 2:
        return {"feature": feature, "target": target, "sample_count": len(vals), "pearson": None, "spearman": None}
    x = torch.tensor([a for a, _ in vals], dtype=torch.float64); y = torch.tensor([b for _, b in vals], dtype=torch.float64)
    def corr(a, b):
        ac, bc = a - a.mean(), b - b.mean(); den = ac.norm() * bc.norm()
        return float((ac * bc).sum() / den) if den.item() else None
    def rank(a):
        order = torch.argsort(a, stable=True); out = torch.empty_like(a); out[order] = torch.arange(a.numel(), dtype=a.dtype)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]: j += 1
            if j - i > 1: out[order[i:j]] = (i + j - 1) / 2
            i = j
        return out
    return {"feature": feature, "target": target, "sample_count": len(vals),
            "pearson": corr(x, y), "spearman": corr(rank(x), rank(y))}


def select_representatives(rows: list[dict], max_count: int = 12) -> tuple[set[tuple], dict[tuple, str]]:
    """Select before intervention using condition/error strata and names."""
    key = lambda r: (r["seed"], r["update"], r["parameter_id"])
    by_condition = sorted(rows, key=lambda r: (float(r["effective_condition_number"]), r["seed"], r["update"], r["parameter_id"]))
    by_error = sorted(rows, key=lambda r: (float(r["actual_update_error"]), r["seed"], r["update"], r["parameter_id"]))
    picks: list[tuple[dict, str]] = []
    for values, labels in ((by_condition, ("low_condition", "medium_condition", "high_condition")),
                           (by_error, ("low_update_error", "medium_update_error", "high_update_error"))):
        n = len(values)
        for pos, label in zip((0, (n - 1) // 2, n - 1), labels): picks.append((values[pos], label))
    # Add the first deterministic instance of distinct layer/name prefixes.
    seen_prefixes = set()
    for row in sorted(rows, key=lambda r: (r["parameter_name"], r["seed"], r["update"], r["parameter_id"])):
        prefix = str(row["parameter_name"]).split(".")[0]
        if prefix not in seen_prefixes:
            picks.append((row, "layer_diversity")); seen_prefixes.add(prefix)
        if len(seen_prefixes) >= 6: break
    selected, reasons = set(), {}
    for row, reason in picks:
        item = key(row)
        if item not in selected and len(selected) < max_count:
            selected.add(item); reasons[item] = reason
    return selected, reasons


def make_plots(out: Path, head_tail: list[dict], sv: list[dict], rot: list[dict], corr_rows: list[dict], baseline_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs remain complete.\n"); return
    def agg(rows, band=None, field="update_relative_l2"):
        vals = {}
        for r in rows:
            if band and r["band"] != band: continue
            vals.setdefault(r["epsilon"], []).append(float(r[field]))
        return sorted((e, sum(v) / len(v)) for e, v in vals.items())
    plt.figure(figsize=(7, 5))
    for band, style in (("head", "-"), ("tail", "--")):
        data = agg(head_tail, band); plt.plot([x for x, _ in data], [y for _, y in data], style, marker="o", label=band)
    plt.xlabel("relative perturbation epsilon"); plt.ylabel("post-Muon relative L2"); plt.legend(); plt.tight_layout(); plt.savefig(out / "head_vs_tail_update_distortion.png", dpi=140); plt.close()
    # Per-tensor tail/head ratio at each epsilon.
    plt.figure(figsize=(7, 5)); ratios = {}
    for r in head_tail: ratios.setdefault((r["seed"], r["update"], r["parameter_id"], r["epsilon"]), {})[r["band"]] = r["update_relative_l2"]
    for key, pair in list(ratios.items()):
        if "head" in pair and "tail" in pair and pair["head"]:
            ratios.setdefault("_points", []).append((key[3], pair["tail"] / pair["head"]))
    points = ratios.get("_points", []); plt.scatter([x for x, _ in points], [y for _, y in points], s=10); plt.xlabel("epsilon"); plt.ylabel("tail/head sensitivity ratio"); plt.tight_layout(); plt.savefig(out / "tail_head_sensitivity_ratio.png", dpi=140); plt.close()
    plt.figure(figsize=(7, 5))
    for rows, label, style in ((sv, "singular-value-only", "-"), (rot, "subspace-rotation-only", "--")):
        data = agg(rows, "tail"); plt.plot([x for x, _ in data], [y for _, y in data], style, marker=".", label=label)
    plt.xlabel("epsilon"); plt.ylabel("post-Muon relative L2"); plt.legend(); plt.tight_layout(); plt.savefig(out / "singular_value_vs_rotation.png", dpi=140); plt.close()
    for band in ("head", "tail"):
        data = agg(rot, band); plt.plot([x for x, _ in data], [y for _, y in data], marker=".", label=band)
    plt.xlabel("epsilon"); plt.ylabel("rotation-only post-Muon relative L2"); plt.legend(); plt.tight_layout(); plt.savefig(out / "head_vs_tail_rotation.png", dpi=140); plt.close()
    # Controlled tail sensitivity against the corresponding actual INT4 error.
    actual = {(r["seed"], r["update"], r["parameter_id"]): r.get("actual_update_error") for r in baseline_rows}
    controlled = {}
    for r in head_tail:
        if r["band"] == "tail" and r["epsilon"] == 0.01 and finite(r.get("update_relative_l2")):
            controlled[(r["seed"], r["update"], r["parameter_id"])] = r["update_relative_l2"]
    points = [(actual[k], v) for k, v in controlled.items() if finite(actual.get(k))]
    plt.figure(figsize=(7, 5)); plt.scatter([x for x, _ in points], [y for _, y in points], s=14)
    plt.xlabel("actual INT4 update direction error"); plt.ylabel("controlled tail sensitivity (epsilon=0.01)"); plt.tight_layout(); plt.savefig(out / "controlled_tail_vs_actual_int4.png", dpi=140); plt.close()
    # Layer/name-prefix distribution, retaining the provenance in the CSVs.
    layer_values = {}
    for r in head_tail:
        if r["band"] == "tail" and r["epsilon"] == 0.01:
            parts = str(r["parameter_id"]).split(".")
            layer_values.setdefault(".".join(parts[:3]) if len(parts) >= 3 else parts[0], []).append(r["update_relative_l2"])
    labels = sorted(layer_values); plt.figure(figsize=(8, 5)); plt.boxplot([layer_values[x] for x in labels], labels=labels, showfliers=False)
    plt.xticks(rotation=30, ha="right"); plt.ylabel("tail rotation post-Muon relative L2"); plt.tight_layout(); plt.savefig(out / "per_layer_sensitivity_distribution.png", dpi=140); plt.close()
    plt.figure(figsize=(7, 5)); plt.bar([r["feature"] for r in corr_rows], [r["spearman"] or 0 for r in corr_rows]); plt.xticks(rotation=30, ha="right"); plt.ylabel("Spearman correlation"); plt.tight_layout(); plt.savefig(out / "sensitivity_correlations.png", dpi=140); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/muon_controlled_spectral_perturbation")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(); started = time.perf_counter()
    paths = discover_snapshots(args.reports_root)
    if len(paths) != len(SEEDS) * len(LANDMARKS): raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_rows = []; cached: dict[tuple, tuple] = {}
    print("pass 1: baseline spectral stats and actual INT4 error", flush=True)
    for seed, update, path in paths:
        snapshot = load_snapshot(path); kwargs = transform_kwargs(snapshot)
        for item in snapshot["tensors"]:
            if len(item["shape"]) != 2: continue
            source = decompose(item["tensor"]); q = quantize(source.matrix, QUANTIZER)
            ref_update = transform(source.matrix, **kwargs); q_update = transform(q, **kwargs)
            raw = _ratios(source.matrix, q, "raw"); upd = _ratios(ref_update, q_update, "update")
            sm = spectral_metrics(source.singular_values); bands = spectral_bands(source.singular_values)
            row = ident(item, seed, update) | sm | {"actual_update_error": 1 - upd.get("update_cosine") if finite(upd.get("update_cosine")) else None,
                "actual_update_relative_l2": upd.get("update_relative_l2"), "actual_raw_cosine": raw.get("raw_cosine"),
                "head_mode_count": int(bands.head.numel()), "tail_mode_count": int(bands.tail.numel())}
            row.update({f"{key}": value for key, value in actual_error_decomposition(source, q, head=bands.head, tail=bands.tail).items()})
            baseline_rows.append(row)
            cached[(seed, update, row["parameter_id"])] = (source, ref_update, kwargs, bands)
        del snapshot
    selected, reasons = select_representatives(baseline_rows)
    manifest = []
    for row in baseline_rows:
        key = (row["seed"], row["update"], row["parameter_id"])
        if key in selected: manifest.append(row | {"selection_reason": reasons[key]})
    head_tail = []; sv_rows = []; rot_rows = []
    print(f"pass 2: controlled interventions for {len(selected)} representatives", flush=True)
    for key in sorted(selected):
        source, ref_update, kwargs, bands = cached[key]
        common = {"seed": key[0], "update": key[1], "parameter_id": key[2], "shape": str(list(source.matrix.shape)), "selection_reason": reasons[key]}
        for epsilon in EPSILONS:
            for band_name, indices in (("head", bands.head), ("tail", bands.tail)):
                observed, theta, invalid = rotation_only(source, indices, epsilon)
                row = common | controlled_row(source, observed, kind="equal_energy_rotation", band=band_name, epsilon=epsilon, transform_kwargs=kwargs, theta=theta, invalid=invalid)
                row["seed"], row["update"], row["parameter_id"] = key; head_tail.append(row)
                sv_observed, sv_invalid = singular_value_only(source, indices, epsilon)
                sv_rows.append(common | controlled_row(source, sv_observed, kind="singular_value_only", band=band_name, epsilon=epsilon, transform_kwargs=kwargs, invalid=sv_invalid))
                rot_rows.append(common | controlled_row(source, observed, kind="subspace_rotation_only", band=band_name, epsilon=epsilon, transform_kwargs=kwargs, theta=theta, invalid=invalid))
    for r in head_tail + sv_rows + rot_rows:
        r["seed"] = int(r["seed"]); r["update"] = int(r["update"])
    ratios = []
    for epsilon in EPSILONS:
        grouped = {}
        for r in head_tail:
            grouped.setdefault((r["seed"], r["update"], r["parameter_id"], r["epsilon"]), {})[r["band"]] = r
        for key, pair in grouped.items():
            if "head" in pair and "tail" in pair and finite(pair["head"].get("update_relative_l2")) and float(pair["head"]["update_relative_l2"]) > 1e-12:
                ratios.append({"seed": key[0], "update": key[1], "parameter_id": key[2], "epsilon": key[3], "tail_head_update_error_ratio": float(pair["tail"]["update_relative_l2"]) / float(pair["head"]["update_relative_l2"])})
    ratio_by_key = {(r["seed"], r["update"], r["parameter_id"], r["epsilon"]): r["tail_head_update_error_ratio"] for r in ratios}
    for r in head_tail:
        r["tail_head_update_error_ratio"] = ratio_by_key.get((r["seed"], r["update"], r["parameter_id"], r["epsilon"]))
    # Controlled sensitivity features are computed from the fixed epsilon=.01
    # intervention, after the representative set itself was selected.
    controlled_features = {}
    for r in head_tail:
        if r["band"] == "tail" and r["epsilon"] == 0.01 and finite(r.get("update_relative_l2")):
            controlled_features.setdefault((r["seed"], r["update"], r["parameter_id"]), {})["tail_rotation_sensitivity"] = r["update_relative_l2"]
        if r["band"] == "head" and r["epsilon"] == 0.01 and finite(r.get("update_relative_l2")):
            controlled_features.setdefault((r["seed"], r["update"], r["parameter_id"]), {})["head_rotation_sensitivity"] = r["update_relative_l2"]
    for r in sv_rows:
        if r["band"] == "tail" and r["epsilon"] == 0.01 and finite(r.get("update_relative_l2")):
            controlled_features.setdefault((r["seed"], r["update"], r["parameter_id"]), {})["tail_singular_value_sensitivity"] = r["update_relative_l2"]
    for r in ratios:
        controlled_features.setdefault((r["seed"], r["update"], r["parameter_id"]), {})["tail_head_sensitivity_ratio"] = r["tail_head_update_error_ratio"] if r["epsilon"] == 0.01 else controlled_features.get((r["seed"], r["update"], r["parameter_id"]), {}).get("tail_head_sensitivity_ratio")
    corr_rows = []
    for feature in ("effective_condition_number", "tail_subspace_proxy", "tail_rotation_sensitivity", "tail_singular_value_sensitivity", "tail_head_sensitivity_ratio", "tail_singular_value_error_fraction", "tail_subspace_mixing_fraction"):
        if feature in {"tail_subspace_proxy", "tail_rotation_sensitivity", "tail_singular_value_sensitivity", "tail_head_sensitivity_ratio"}:
            values = {}
            for item, features in controlled_features.items():
                if feature == "tail_subspace_proxy":
                    values[item] = features.get("tail_rotation_sensitivity")
                else:
                    values[item] = features.get(feature)
            work = [r | {feature: values.get((r["seed"], r["update"], r["parameter_id"]))} for r in baseline_rows]
        else: work = baseline_rows
        corr_rows.append({"quantizer": QUANTIZER} | correlation(work, feature, "actual_update_error"))
    write_csv(args.output / "head_tail_equal_energy.csv", head_tail)
    write_csv(args.output / "singular_value_only.csv", sv_rows)
    write_csv(args.output / "subspace_rotation_only.csv", rot_rows)
    write_csv(args.output / "actual_int4_error_decomposition.csv", baseline_rows)
    write_csv(args.output / "sensitivity_correlations.csv", corr_rows)
    write_csv(args.output / "representative_tensor_manifest.csv", manifest)
    if not args.skip_plots: make_plots(args.output, head_tail, sv_rows, rot_rows, corr_rows, baseline_rows)
    runtime = time.perf_counter() - started
    # Aggregate controlled results and write durable method documentation.
    summary_rows = []
    for epsilon in EPSILONS:
        for band in ("head", "tail"):
            rows = [r for r in head_tail if r["epsilon"] == epsilon and r["band"] == band]
            vals = [float(r["update_relative_l2"]) for r in rows if finite(r.get("update_relative_l2"))]
            summary_rows.append((epsilon, band, sum(vals) / len(vals) if vals else None, sorted(vals)[len(vals)//2] if vals else None))
    comparison_rows = []
    for epsilon in EPSILONS:
        for band in ("head", "tail"):
            sv_values = [float(r["update_relative_l2"]) for r in sv_rows if r["epsilon"] == epsilon and r["band"] == band and finite(r.get("update_relative_l2"))]
            rot_values = [float(r["update_relative_l2"]) for r in rot_rows if r["epsilon"] == epsilon and r["band"] == band and finite(r.get("update_relative_l2"))]
            comparison_rows.append((epsilon, band, sum(sv_values) / len(sv_values) if sv_values else None, sum(rot_values) / len(rot_values) if rot_values else None,
                                    (sum(rot_values) / len(rot_values)) / (sum(sv_values) / len(sv_values)) if sv_values and rot_values and sum(sv_values) else None))
    write_csv(args.output / "tail_head_sensitivity_ratios.csv", ratios)
    (args.output / "methodology.md").write_text(f"""# Controlled spectral perturbation methodology\n\nThis read-only offline study used {len(paths)} existing FP32 snapshots and {len(baseline_rows)} eligible 2D tensors. Runtime was {runtime:.2f} seconds on CPU. No training or prior artifact was changed.\n\n## Bands and perturbations\n\nThe head is the smallest leading set explaining 90% of squared singular-value energy, with at least two modes when possible. The tail is its complement. Representatives were selected before intervention from low/median/high effective condition number, low/median/high actual INT4 update error, and deterministic layer-name diversity; duplicate keys were removed.\n\nFor each epsilon in `{EPSILONS}`, head and tail equal-energy perturbations rotate the left singular vectors with a deterministic adjacent-pair skew-symmetric generator. Bisection solves for `||E||_F / ||M||_F = epsilon`. Singular-value-only perturbations add a deterministic positive increment to singular values in the selected band and match the same Frobenius magnitude while leaving U,V fixed. Rotation-only perturbations use the same orthogonal rotation and preserve singular values. The exact production `zeropower_newton_schulz` transform is used for every update metric.\n\nActual INT4 error is projected as `U^T(Q(M)-M)V`; diagonal entries are singular-value error, within-band off-diagonal entries are head/tail mixing, cross-band entries are cross mixing, and reduced-SVD residual is reported separately. No new quantizer is introduced.\n\nAll constructions are deterministic, use no RNG, and record target/actual perturbation norms and singular-value checks. Undefined or degenerate cases are marked invalid rather than replaced with zero.\n""")
    (args.output / "summary.md").write_text(f"""# Summary\n\nAnalyzed {len(paths)} formal snapshots ({len(baseline_rows)} eligible 2D tensors) and ran controlled interventions on {len(selected)} predeclared representatives. Runtime: {runtime:.2f}s CPU.\n\nHead/tail equal-energy aggregate rows are in `head_tail_equal_energy.csv`; singular-value-only and rotation-only controls are in their respective CSVs. Actual INT4 spectral-coordinate decomposition and descriptive correlations are in the remaining CSVs. Interpret correlations as mechanism evidence, not causal or statistical-significance claims.\n\nMean head/tail post-Muon relative-L2 rows (epsilon, band, mean, median): {summary_rows}\n\nMean singular-value-only vs rotation-only post-Muon relative-L2 (epsilon, band, singular-value, rotation, ratio): {comparison_rows}\n\nThe controlled result is a mechanism probe: if tail/head ratios are consistently above one, Muon is spectrally nonuniform for matched-energy orientation perturbations. The comparison between rotation-only and singular-value-only rows indicates whether orientation or magnitude is larger in this controlled construction; it does not establish causality for production quantization.\n""")
    print(f"analyzed {len(paths)} snapshots / {len(baseline_rows)} tensors / {len(selected)} representatives in {runtime:.2f}s", flush=True)


if __name__ == "__main__": main()
