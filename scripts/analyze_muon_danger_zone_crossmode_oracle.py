#!/usr/bin/env python3
"""Offline danger-zone cross-mode restoration oracle for INT4 Muon.

This script is read-only with respect to training artifacts.  It reuses the
production blockwise-dynamic INT4 persistence path and production Muon
Newton--Schulz transform, then applies diagnostic residual restorations in
the FP32 SVD coordinate system.
"""
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
from optim.muon_ns_sensitivity import exact_polar  # noqa: E402
from optim.muon_spectral_sensitivity import quantize  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402
from optim.muon_danger_zone_crossmode_oracle import (  # noqa: E402
    DANGER_LOG10_END, DANGER_LOG10_START, DISTANCE_DISTANT, DISTANCE_LOCAL,
    component_from_mask, coordinate_masks, correction_metrics,
    danger_mode_mask, matrix_from_coordinates, restoration_gain,
    restore_from_coordinates, select_energy_budget, select_top_coefficients,
)

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
ENERGY_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
COEFF_BUDGETS = (8, 16, 32, 64, 128)
INTERVENTION_CAP = 6
EPS = 1e-12


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(x) -> bool:
    try:
        return x not in (None, "") and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def transform_kwargs(snapshot: dict) -> dict:
    cfg = snapshot["metadata"]["muon_transform"]
    return {"steps": int(cfg["steps"]),
            "coefficients": tuple(float(x) for x in cfg["coefficients"]),
            "eps": float(cfg["eps"])}


def identity(seed: int, update: int, item: dict) -> dict:
    return {"seed": seed, "update": update,
            "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
            "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")),
            "shape": str(item["shape"])}


def paired(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    m = correction_metrics(reference, candidate)
    return {"relative_l2": m.get("relative_l2"), "cosine": m.get("cosine"),
            "norm_ratio": m.get("norm_ratio"), "status": m.get("status")}


def gain_row(baseline: dict, restored: dict) -> dict:
    g = restoration_gain(baseline, restored)
    return {"update_cosine_gain": g["update_cosine_gain"],
            "update_l2_reduction": g["update_l2_reduction"],
            "recovered_cosine_error_fraction": (
                g["update_cosine_gain"] / (1.0 - baseline["cosine"])
                if g["update_cosine_gain"] is not None and baseline.get("cosine") is not None
                and abs(1.0 - baseline["cosine"]) > EPS else None),
            "recovered_l2_error_fraction": (
                g["update_l2_reduction"] / baseline["relative_l2"]
                if g["update_l2_reduction"] is not None and baseline.get("relative_l2") is not None
                and abs(baseline["relative_l2"]) > EPS else None)}


def output_metrics(ref_update, candidate, ref_polar, candidate_polar, baseline, component,
                   removed_energy, total_error, coefficient_count=None, budget=None) -> dict:
    upd = paired(ref_update, candidate)
    pol = paired(ref_polar, candidate_polar)
    row = {"component": component, "removed_energy": float(removed_energy),
           "removed_energy_fraction": float(removed_energy / total_error) if total_error > EPS else None,
           "coefficient_count": coefficient_count, "budget_fraction": budget,
           "baseline_update_cosine": baseline["cosine"],
           "baseline_update_relative_l2": baseline["relative_l2"],
           "baseline_update_norm_ratio": baseline["norm_ratio"],
           "restored_update_cosine": upd["cosine"],
           "restored_update_relative_l2": upd["relative_l2"],
           "restored_update_norm_ratio": upd["norm_ratio"],
           "polar_baseline_cosine": baseline["polar_cosine"],
           "polar_baseline_relative_l2": baseline["polar_relative_l2"],
           "polar_restored_cosine": pol["cosine"],
           "polar_restored_relative_l2": pol["relative_l2"],
           "polar_restored_norm_ratio": pol["norm_ratio"]}
    row.update(gain_row(baseline, upd))
    row.update({"polar_update_cosine_gain": (
        pol["cosine"] - baseline["polar_cosine"] if pol["cosine"] is not None else None),
        "polar_update_l2_reduction": (
        baseline["polar_relative_l2"] - pol["relative_l2"] if pol["relative_l2"] is not None else None)})
    return row


def make_plots(out: Path, restoration: list[dict], matched: list[dict], distance: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs are complete.\n")
        return
    def means(rows, key, value):
        d = defaultdict(list)
        for r in rows:
            x = r.get(key); y = r.get(value)
            if x is not None and finite(y): d[str(x)].append(float(y))
        return [(k, sum(v) / len(v)) for k, v in d.items() if v]
    vals = means(restoration, "component", "update_cosine_gain")
    if vals:
        plt.figure(figsize=(10, 4)); plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.xticks(rotation=45, ha="right"); plt.ylabel("cosine gain"); plt.title("Danger-zone restoration"); plt.tight_layout(); plt.savefig(out / "restoration_cosine_gain.png", dpi=140); plt.close()
    vals = means(matched, "budget_fraction", "update_cosine_gain")
    if vals:
        plt.figure(figsize=(7, 4)); plt.plot([float(x) for x, _ in vals], [y for _, y in vals], marker="o"); plt.xlabel("matched Frobenius budget / ||E||"); plt.ylabel("cosine gain"); plt.title("Matched-energy oracle"); plt.tight_layout(); plt.savefig(out / "matched_energy_gain.png", dpi=140); plt.close()
    vals = means(distance, "distance_group", "update_cosine_gain")
    if vals:
        plt.figure(figsize=(7, 4)); plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.ylabel("cosine gain"); plt.title("Cross-mode spectral distance"); plt.tight_layout(); plt.savefig(out / "distance_group_gain.png", dpi=140); plt.close()
    vals = means(restoration, "component", "polar_update_cosine_gain")
    if vals:
        plt.figure(figsize=(10, 4)); plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.xticks(rotation=45, ha="right"); plt.ylabel("exact-polar cosine gain"); plt.title("Production versus exact-polar restoration readout"); plt.tight_layout(); plt.savefig(out / "exact_polar_restoration_gain.png", dpi=140); plt.close()
    vals = means(matched, "budget_fraction", "recovered_cosine_error_fraction")
    if vals:
        plt.figure(figsize=(7, 4)); plt.plot([float(x) for x, _ in vals], [y for _, y in vals], marker="o"); plt.xlabel("matched budget / ||E||"); plt.ylabel("recovered cosine-error fraction"); plt.title("Recovery efficiency by matched energy"); plt.tight_layout(); plt.savefig(out / "matched_energy_efficiency.png", dpi=140); plt.close()
    coeff = [r for r in read_csv(out / "matched_coefficient_results.csv")]
    vals = means(coeff, "requested_coefficients", "update_cosine_gain")
    if vals:
        plt.figure(figsize=(7, 4)); plt.plot([float(x) for x, _ in vals], [y for _, y in vals], marker="o"); plt.xlabel("coefficient count"); plt.ylabel("cosine gain"); plt.title("Matched coefficient-count oracle"); plt.tight_layout(); plt.savefig(out / "matched_coefficient_gain.png", dpi=140); plt.close()
    recon = [r for r in read_csv(out / "matched_reconstruction_results.csv")]
    vals = means(recon, "selection", "update_cosine_gain")
    if vals:
        plt.figure(figsize=(7, 4)); plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.ylabel("cosine gain"); plt.title("Matched raw-reconstruction improvement"); plt.tight_layout(); plt.savefig(out / "matched_reconstruction_gain.png", dpi=140); plt.close()
    subtype = [r for r in read_csv(out / "crossmode_subtypes.csv") if r.get("subtype") in {"internal_cross", "outside_cross", "danger_cross"}]
    vals = means(subtype, "subtype", "recovery_efficiency")
    if vals:
        plt.figure(figsize=(7, 4)); plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.ylabel("cosine gain / residual fraction"); plt.title("Danger-zone cross-mode subtype efficiency"); plt.tight_layout(); plt.savefig(out / "crossmode_subtype_efficiency.png", dpi=140); plt.close()
    interaction = [r for r in read_csv(out / "interaction_results.csv")]
    vals = means(interaction, "seed", "production_interaction_cosine")
    if vals:
        plt.figure(figsize=(7, 4)); plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.xlabel("seed"); plt.ylabel("interaction"); plt.title("Diagonal/cross nonlinear interaction"); plt.tight_layout(); plt.savefig(out / "interaction_effect.png", dpi=140); plt.close()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/muon_danger_zone_crossmode_oracle")
    parser.add_argument("--intervention-cap", type=int, default=INTERVENTION_CAP,
                        help="matched-budget representative tensors per snapshot; unconstrained restoration covers all tensors")
    parser.add_argument("--snapshot-limit", type=int, default=0,
                        help="debug/runtime limit; 0 uses all ten formal snapshots")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter(); args.output.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        make_plots(args.output, read_csv(args.output / "oracle_restoration_metrics.csv"),
                   read_csv(args.output / "matched_energy_results.csv"),
                   read_csv(args.output / "spectral_distance_results.csv"))
        print(f"plots refreshed in {time.perf_counter() - started:.2f}s")
        return
    snapshots = discover_snapshots(args.reports_root)
    if len(snapshots) != 10:
        raise RuntimeError(f"expected 10 formal snapshots, found {len(snapshots)}")
    if args.snapshot_limit:
        snapshots = snapshots[:args.snapshot_limit]
    restoration_rows: list[dict] = []; energy_rows: list[dict] = []; subtype_rows: list[dict] = []
    matched_energy_rows: list[dict] = []; matched_coeff_rows: list[dict] = []; matched_recon_rows: list[dict] = []
    interaction_rows: list[dict] = []; manifest_rows: list[dict] = []; distance_rows: list[dict] = []
    tensor_count = 0
    for seed, update, path in snapshots:
        print(f"processing seed={seed} update={update}", flush=True)
        snap = load_snapshot(path); cfg = transform_kwargs(snap)
        snapshot_rep_count = 0
        for item in snap["tensors"]:
            matrix = item["tensor"].detach().float()
            if matrix.ndim != 2:
                continue
            tensor_count += 1
            ident = identity(seed, update, item)
            q = quantize(matrix, "int4-dynamic-b2048").detach().float()
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            ehat = u.T @ (q - matrix) @ vh.T
            masks = coordinate_masks(s)
            danger_count = int(masks.danger.sum())
            total_error = float(ehat.norm())
            danger_energy = float(ehat[masks.danger[:, None] | masks.danger[None, :]].square().sum())
            manifest_rows.append(ident | {"danger_log10_start": DANGER_LOG10_START, "danger_log10_end": DANGER_LOG10_END,
                "active_rank": int((s / s[0] >= 1e-6).sum()) if s.numel() and s[0] > 0 else 0,
                "danger_mode_count": danger_count, "danger_fraction_active_rank": danger_count / max(1, int((s / s[0] >= 1e-6).sum())) if s.numel() and s[0] > 0 else 0,
                "danger_residual_energy_fraction": danger_energy / max(total_error ** 2, EPS)})
            ref_update = muon_reference.zeropower_newton_schulz(matrix.clone(), **cfg)
            q_update = muon_reference.zeropower_newton_schulz(q.clone(), **cfg)
            ref_polar = exact_polar(matrix); q_polar = exact_polar(q)
            base_u = paired(ref_update, q_update); base_p = paired(ref_polar, q_polar)
            baseline = base_u | {"polar_cosine": base_p["cosine"], "polar_relative_l2": base_p["relative_l2"]}
            energy_rows.append(ident | {"total_residual_frobenius": total_error,
                "danger_diagonal_energy": float(ehat[masks.diagonal].square().sum()),
                "danger_cross_energy": float(ehat[masks.cross].square().sum()),
                "danger_internal_cross_energy": float(ehat[masks.internal_cross].square().sum()),
                "danger_outside_cross_energy": float(ehat[masks.outside_cross].square().sum()),
                "local_cross_energy": float(ehat[masks.local_cross].square().sum()),
                "medium_cross_energy": float(ehat[masks.medium_cross].square().sum()),
                "distant_cross_energy": float(ehat[masks.distant_cross].square().sum()),
                "danger_diag_fraction": float(ehat[masks.diagonal].square().sum() / max(total_error ** 2, EPS)),
                "danger_cross_fraction": float(ehat[masks.cross].square().sum() / max(total_error ** 2, EPS))})
            components = {"danger_diagonal": masks.diagonal, "danger_cross": masks.cross,
                          "danger_full": masks.diagonal | masks.cross, "internal_cross": masks.internal_cross,
                          "outside_cross": masks.outside_cross, "local_cross": masks.local_cross,
                          "medium_cross": masks.medium_cross, "distant_cross": masks.distant_cross}
            component_candidates = {}
            for name, mask in components.items():
                coord = component_from_mask(ehat, mask); candidate = restore_from_coordinates(q, u, vh, coord)
                cand_u = muon_reference.zeropower_newton_schulz(candidate.clone(), **cfg); cand_p = exact_polar(candidate)
                row = ident | output_metrics(ref_update, cand_u, ref_polar, cand_p, baseline, name,
                                              float(coord.norm()), total_error)
                restoration_rows.append(row); component_candidates[name] = row
                subtype_rows.append(ident | {"subtype": name, "energy": float(coord.norm() ** 2),
                    "energy_fraction": float(coord.norm() ** 2 / max(total_error ** 2, EPS)),
                    "update_cosine_gain": row["update_cosine_gain"], "polar_update_cosine_gain": row["polar_update_cosine_gain"],
                    "recovery_efficiency": row["update_cosine_gain"] / max(float(coord.norm() ** 2 / max(total_error ** 2, EPS)), EPS) if row["update_cosine_gain"] is not None else None})
            diag = component_candidates["danger_diagonal"]; cross = component_candidates["danger_cross"]; full = component_candidates["danger_full"]
            interaction_rows.append(ident | {"production_interaction_cosine": (full["update_cosine_gain"] or 0) - (diag["update_cosine_gain"] or 0) - (cross["update_cosine_gain"] or 0),
                "production_interaction_l2": (full["update_l2_reduction"] or 0) - (diag["update_l2_reduction"] or 0) - (cross["update_l2_reduction"] or 0),
                "polar_interaction_cosine": (full["polar_update_cosine_gain"] or 0) - (diag["polar_update_cosine_gain"] or 0) - (cross["polar_update_cosine_gain"] or 0)})
            # Expensive budget controls are deterministic and predeclared.
            if snapshot_rep_count < args.intervention_cap:
                snapshot_rep_count += 1
                allowed = {"diagonal": masks.diagonal, "cross": masks.cross}
                candidates_by_budget = {"diagonal": {}, "cross": {}}
                for kind, mask in allowed.items():
                    coord = component_from_mask(ehat, mask)
                    for b in ENERGY_BUDGETS:
                        selected, count, clipped = select_energy_budget(coord, b * total_error)
                        cand = restore_from_coordinates(q, u, vh, selected)
                        cm = output_metrics(ref_update, muon_reference.zeropower_newton_schulz(cand.clone(), **cfg), ref_polar, exact_polar(cand), baseline, kind, float(selected.norm()), total_error, count, b)
                        matched_energy_rows.append(ident | cm | {"selection": kind, "clipped": clipped, "budget_norm": b * total_error})
                        candidates_by_budget[kind][b] = (selected, cm, cand)
                    for k in COEFF_BUDGETS:
                        selected = select_top_coefficients(coord, k); cand = restore_from_coordinates(q, u, vh, selected)
                        cm = output_metrics(ref_update, muon_reference.zeropower_newton_schulz(cand.clone(), **cfg), ref_polar, exact_polar(cand), baseline, kind, float(selected.norm()), total_error, int((selected != 0).sum()), None)
                        matched_coeff_rows.append(ident | cm | {"selection": kind, "requested_coefficients": k})
                # Nearest raw reconstruction improvement among the declared energy grid.
                pools = {}
                for kind, entries in candidates_by_budget.items():
                    pools[kind] = [(b, c, float((q - matrix).norm() - (c[2] - matrix).norm())) for b, c in entries.items()]
                for b, _, imp in pools["diagonal"]:
                    near = min(pools["cross"], key=lambda x: abs(x[2] - imp))
                    diag_entry = candidates_by_budget["diagonal"][b]
                    cross_entry = candidates_by_budget["cross"][near[0]]
                    matched_recon_rows.extend([
                        ident | {"selection": "diagonal", "paired_diagonal_budget": b,
                                 "paired_cross_budget": near[0], "raw_l2_improvement": imp,
                                 "correction_budget": b,
                                 "update_cosine_gain": diag_entry[1]["update_cosine_gain"],
                                 "update_l2_reduction": diag_entry[1]["update_l2_reduction"]},
                        ident | {"selection": "cross", "paired_diagonal_budget": b,
                                 "paired_cross_budget": near[0], "raw_l2_improvement": near[2],
                                 "correction_budget": near[0],
                                 "update_cosine_gain": cross_entry[1]["update_cosine_gain"],
                                 "update_l2_reduction": cross_entry[1]["update_l2_reduction"]},
                    ])
            # Per-distance aggregate diagnostic, all tensors.
            z = torch.log10((s / s[0]).clamp_min(1e-6)) if s.numel() and s[0] > 0 else torch.zeros_like(s)
            for name, mask in (("local", masks.local_cross), ("medium", masks.medium_cross), ("distant", masks.distant_cross)):
                coord = component_from_mask(ehat, mask); cand = restore_from_coordinates(q, u, vh, coord)
                cm = output_metrics(ref_update, muon_reference.zeropower_newton_schulz(cand.clone(), **cfg), ref_polar, exact_polar(cand), baseline, name, float(coord.norm()), total_error)
                distance_rows.append(ident | {"distance_group": name, "distance_thresholds": "<0.5,0.5-1.5,>=1.5 decades"} | cm)
    write_csv(args.output / "danger_zone_manifest.csv", manifest_rows)
    write_csv(args.output / "oracle_restoration_metrics.csv", restoration_rows)
    write_csv(args.output / "matched_energy_results.csv", matched_energy_rows)
    write_csv(args.output / "matched_coefficient_results.csv", matched_coeff_rows)
    write_csv(args.output / "matched_reconstruction_results.csv", matched_recon_rows)
    write_csv(args.output / "crossmode_subtypes.csv", subtype_rows)
    write_csv(args.output / "spectral_distance_results.csv", distance_rows)
    write_csv(args.output / "interaction_results.csv", interaction_rows)
    # Compare against the existing tail-correction report without changing it.
    tail_rows = []
    old = args.reports_root / "muon_tail_correction" / "budget_summary.csv"
    if old.exists():
        with old.open(newline="") as h: tail_rows = list(csv.DictReader(h))
    write_csv(args.output / "comparison_to_tail_correction.csv", tail_rows)
    if not args.skip_plots: make_plots(args.output, restoration_rows, matched_energy_rows, distance_rows)
    def mean(rows, key):
        vals = [float(r[key]) for r in rows if finite(r.get(key))]
        return sum(vals) / len(vals) if vals else None
    diag = [r for r in restoration_rows if r["component"] == "danger_diagonal"]
    cross = [r for r in restoration_rows if r["component"] == "danger_cross"]
    full = [r for r in restoration_rows if r["component"] == "danger_full"]
    def mean_filtered(rows, field, **filters):
        def matches(row, key, wanted):
            actual = row.get(key)
            try:
                return abs(float(actual) - float(wanted)) < 1e-9
            except (TypeError, ValueError):
                return actual == str(wanted)
        values = [float(r[field]) for r in rows if all(matches(r, k, v) for k, v in filters.items()) and finite(r.get(field))]
        return sum(values) / len(values) if values else None
    energy_diag = mean_filtered(matched_energy_rows, "update_cosine_gain", selection="diagonal", budget_fraction=0.01)
    energy_cross = mean_filtered(matched_energy_rows, "update_cosine_gain", selection="cross", budget_fraction=0.01)
    coeff_diag = mean_filtered(matched_coeff_rows, "update_cosine_gain", selection="diagonal", requested_coefficients=64)
    coeff_cross = mean_filtered(matched_coeff_rows, "update_cosine_gain", selection="cross", requested_coefficients=64)
    recon_diag = mean_filtered(matched_recon_rows, "update_cosine_gain", selection="diagonal")
    recon_cross = mean_filtered(matched_recon_rows, "update_cosine_gain", selection="cross")
    distance_means = {name: mean_filtered(distance_rows, "update_cosine_gain", distance_group=name)
                      for name in ("local", "medium", "distant")}
    interaction_mean = mean(interaction_rows, "production_interaction_cosine")
    summary = f"""# Danger-zone cross-mode suppression oracle

## Scope

This is a read-only CPU oracle study over {tensor_count} eligible 2D tensors from 10 formal FP32 Muon snapshots (seeds 0/1, updates 128/512/1024/2048/4096). No training was launched and no production optimizer, quantizer, recipe, or artifact was modified. The baseline is the existing `int4-dynamic-b2048` reconstruction and the exact production Muon transform.

## Fixed danger zone

The prior continuous-spectrum interval is reused unchanged: `[{DANGER_LOG10_START:g},{DANGER_LOG10_END:g})` in `log10(sigma/sigma_max)`, i.e. approximately `sigma/sigma_max in [1e-3,1e-2)`. No retuning occurs here.

## Direct unconstrained restorations

Mean production-K=5 cosine gains were: diagonal `{mean(diag,'update_cosine_gain')}`, cross-mode `{mean(cross,'update_cosine_gain')}`, full danger-zone `{mean(full,'update_cosine_gain')}`. These are direct restorations of actual residual coefficients; they are oracle ceilings, not deployable quantizers. Exact-polar gains are in `oracle_restoration_metrics.csv`.

The diagonal support is `i=j` within D. The cross support is `i!=j` with `i in D or j in D`; the two supports are disjoint and their union is the full danger-zone residual.

## Matched budgets

Matched-energy and matched-coefficient controls use deterministic top-magnitude actual residual coefficients. The default expensive-control cap is `{args.intervention_cap}` tensors in discovery order; baseline and unconstrained restorations cover all tensors. The cap is recorded in this report rather than silently subsampling.

At the 1% `||E||_F` budget, mean cosine gain was diagonal `{energy_diag}` versus cross-mode `{energy_cross}`. At 64 selected coefficients, it was diagonal `{coeff_diag}` versus cross-mode `{coeff_cross}`. Under approximate raw-L2-matched corrections, it was diagonal `{recon_diag}` versus cross-mode `{recon_cross}`. These matched controls are the relevant per-budget comparison; the much larger unconstrained cross-mode gain is partly explained by the cross-mode component containing substantially more residual energy.

For spectral-distance groups, mean production gains were local `{distance_means['local']}`, medium `{distance_means['medium']}`, and distant `{distance_means['distant']}`. The mean diagonal-plus-cross interaction was `{interaction_mean}`; this is descriptive because Muon is nonlinear.

## Interpretation

Compare cross-mode versus diagonal recovery at equal energy, equal coefficient count, and approximately equal raw reconstruction improvement before making a geometry claim. K=5 versus exact-polar columns separate finite-step effects from persistent polar-geometry repair. Interaction terms are descriptive and need not add because the transform is nonlinear.

Runtime: `{time.perf_counter() - started:.2f}` CPU seconds. Detailed CSVs and plots are the authoritative numeric outputs; no statistical significance or causal claim is made.
"""
    (args.output / "summary.md").write_text(summary)
    methodology = f"""# Methodology

For each matrix `M`, production quantization creates `Mq` and `E=Mq-M`. The FP32 reduced SVD defines `E_hat=U^T E V`. The prior danger interval is fixed at `[{DANGER_LOG10_START:g},{DANGER_LOG10_END:g})`; no threshold is fit in this study.

`danger_diagonal` contains only `E_hat[i,i]` for danger modes. `danger_cross` contains only off-diagonal entries with at least one danger index. `internal_cross`, `outside_cross`, and log-spectral `local`, `medium`, `distant` masks are disjoint subcomponents of the cross support. Candidate reconstructions are `Mq - U C V^T`, where `C` is an actual residual component.

Unconstrained direct restoration covers every eligible tensor. Matched budgets use target Frobenius norms `{ENERGY_BUDGETS}` times `||E||_F`; coefficients are selected by deterministic absolute magnitude and the final coefficient is scaled down only to hit the target norm. Coefficient controls use `{COEFF_BUDGETS}` entries. Reconstruction-matched pairs select the nearest raw-L2 improvement from the declared energy grid without tuning coefficients against update fidelity.

Every update readout uses the exact production `zeropower_newton_schulz` implementation and the exact SVD polar helper used by prior reports. All calculations are detached CPU diagnostics. The matched controls use a deterministic cap of `{args.intervention_cap}` tensors; this is an explicit CPU control and not outcome-dependent.
"""
    (args.output / "methodology.md").write_text(methodology)
    print(f"wrote {args.output}; tensors={tensor_count}; runtime={time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
