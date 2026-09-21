#!/usr/bin/env python3
"""Generate the offline continuous-spectrum INT4 Muon risk report.

The expensive intervention readouts are evaluated on a deterministic cap of
20 tensor instances per log-spectrum bin.  Baseline spectra, residual
allocation, and mode-level risk rows cover every eligible tensor.  The cap is
an auditably declared CPU-runtime control, not an outcome-dependent sample.
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
from optim.muon_ns_sensitivity import exact_polar, scalar_map_and_derivative, transform  # noqa: E402
from optim.muon_spectral_sensitivity import quantize  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402
from optim.muon_continuous_spectral_risk import (  # noqa: E402
    BIN_EDGES, BIN_LABELS, EPS, active_spectrum, associated_component,
    coordinate_error, diagonal_component, fixed_bin_ids, merge_bin_ids,
    metrics, pair_component, safe_ratio,
)

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
QUANTIZER = "int4-dynamic-b2048"
SENSITIVITY_EPSILON = 0.001
INTERVENTION_CAP = 20
PAIR_SHORTLIST = 6


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
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def identity(seed: int, update: int, item: dict) -> dict:
    return {"seed": seed, "update": update,
            "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
            "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")),
            "shape": str(item["shape"])}


def transform_kwargs(snapshot: dict) -> dict:
    cfg = snapshot["metadata"]["muon_transform"]
    return {"steps": int(cfg["steps"]), "coefficients": tuple(float(x) for x in cfg["coefficients"]),
            "eps": float(cfg["eps"])}


def correlation(rows: list[dict], feature: str, target: str) -> dict:
    values = [(float(r[feature]), float(r[target])) for r in rows
              if finite(r.get(feature)) and finite(r.get(target))]
    if len(values) < 2:
        return {"feature": feature, "target": target, "sample_count": len(values), "pearson": None, "spearman": None}
    x = torch.tensor([v[0] for v in values], dtype=torch.float64)
    y = torch.tensor([v[1] for v in values], dtype=torch.float64)
    def corr(a, b):
        ac, bc = a - a.mean(), b - b.mean(); den = ac.norm() * bc.norm()
        return float((ac * bc).sum() / den) if den.item() else None
    def rank(a):
        order = torch.argsort(a, stable=True); out = torch.empty_like(a); out[order] = torch.arange(a.numel(), dtype=a.dtype)
        i = 0
        while i < a.numel():
            j = i + 1
            while j < a.numel() and a[order[j]] == a[order[i]]: j += 1
            if j - i > 1: out[order[i:j]] = (i + j - 1) / 2
            i = j
        return out
    return {"feature": feature, "target": target, "sample_count": len(values),
            "pearson": corr(x, y), "spearman": corr(rank(x), rank(y))}


def controlled_matrix(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor,
                     mode_mask: torch.Tensor, kind: str, epsilon: float, norm: torch.Tensor) -> tuple[torch.Tensor, bool]:
    indices = torch.nonzero(mode_mask, as_tuple=False).flatten()
    if not indices.numel() or norm.item() == 0:
        return (u * s) @ vh, True
    target = norm * float(epsilon)
    if kind == "magnitude":
        weights = torch.arange(1, indices.numel() + 1, dtype=s.dtype)
        delta = torch.zeros_like(s); delta[indices] = target * weights / weights.norm()
        return (u * (s + delta)) @ vh, False
    if indices.numel() < 2:
        return (u * s) @ vh, True
    # A deterministic adjacent-mode left rotation, scaled by bisection to the
    # requested Frobenius norm.  It changes orientation, not singular values.
    a, b = int(indices[0]), int(indices[1])
    def build(theta: float) -> torch.Tensor:
        c, sn = math.cos(theta), math.sin(theta)
        rotated = u.clone()
        ua, ub = u[:, a].clone(), u[:, b].clone()
        rotated[:, a], rotated[:, b] = c * ua + sn * ub, -sn * ua + c * ub
        return (rotated * s) @ vh
    lo, hi = 0.0, math.pi / 2
    while (build(hi) - (u * s) @ vh).norm() < target and hi < 32:
        hi *= 2
    if (build(hi) - (u * s) @ vh).norm() < target:
        return build(hi), True
    for _ in range(45):
        mid = (lo + hi) / 2
        if (build(mid) - (u * s) @ vh).norm() < target: lo = mid
        else: hi = mid
    return build((lo + hi) / 2), False


def make_plots(out: Path, bin_rows: list[dict], sensitivity_rows: list[dict],
               restore_rows: list[dict], mixing_rows: list[dict], danger: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs are complete.\n")
        return
    def grouped(rows, x, y):
        d = defaultdict(list)
        for r in rows:
            if finite(r.get(x)) and finite(r.get(y)): d[r[x]].append(float(r[y]))
        return [(k, sum(v) / len(v)) for k, v in d.items() if v]
    fields = [("residual_fraction", "Residual energy fraction", "residual_energy_vs_sigma.png"),
              ("update_cosine_gain", "Direct restoration gain", "direct_restoration_vs_sigma.png"),
              ("recovery_efficiency", "Recovery efficiency", "recovery_efficiency_vs_sigma.png")]
    for key, title, name in fields:
        vals = grouped(bin_rows if key == "residual_fraction" else restore_rows, "bin_label", key)
        if vals:
            plt.figure(figsize=(10, 4)); plt.bar([x for x, _ in vals], [y for _, y in vals]); plt.xticks(rotation=60, ha="right"); plt.ylabel(key); plt.title(title); plt.tight_layout(); plt.savefig(out / name, dpi=140); plt.close()
    vals = grouped(sensitivity_rows, "bin_label", "sensitivity_update_l2")
    if vals:
        plt.figure(figsize=(10, 4)); plt.plot([x for x, _ in vals], [y for _, y in vals], marker="o"); plt.xticks(rotation=60, ha="right"); plt.ylabel("update L2 / epsilon"); plt.title("Controlled sensitivity"); plt.tight_layout(); plt.savefig(out / "controlled_sensitivity_vs_sigma.png", dpi=140); plt.close()
    # Overlay error, sensitivity and direct recovery on separate normalized axes.
    if bin_rows:
        labels = [r["bin_label"] for r in bin_rows if r.get("aggregation") == "mean"]
        lookup = {r["bin_label"]: r for r in bin_rows if r.get("aggregation") == "mean"}
        sl = {r["bin_label"]: r for r in sensitivity_rows if r.get("aggregation") == "mean"}
        rr = {r["bin_label"]: r for r in restore_rows if r.get("aggregation") == "mean"}
        labels = [x for x in labels if x in sl and x in rr]
        if labels:
            def norm(vals):
                m = max((abs(v) for v in vals), default=1.0) or 1.0
                return [v / m for v in vals]
            e = norm([float(lookup[x].get("residual_fraction") or 0) for x in labels]); s = norm([float(sl[x].get("sensitivity_update_l2") or 0) for x in labels]); g = norm([float(rr[x].get("update_cosine_gain") or 0) for x in labels])
            plt.figure(figsize=(11, 4)); plt.plot(labels, e, label="error", marker="o"); plt.plot(labels, s, label="sensitivity", marker="o"); plt.plot(labels, g, label="restoration", marker="o"); plt.xticks(rotation=60, ha="right"); plt.legend(); plt.title("Normalized error, sensitivity, and restoration"); plt.tight_layout(); plt.savefig(out / "error_sensitivity_restoration_overlay.png", dpi=140); plt.close()
    if mixing_rows:
        bins = sorted(set(r["row_bin_label"] for r in mixing_rows) | set(r["col_bin_label"] for r in mixing_rows)); pos = {b: i for i, b in enumerate(bins)}; grid = torch.zeros((len(bins), len(bins)))
        for r in mixing_rows: grid[pos[r["row_bin_label"]], pos[r["col_bin_label"]]] = float(r.get("energy_fraction") or 0)
        plt.figure(figsize=(8, 7)); plt.imshow(grid, cmap="magma"); plt.colorbar(label="off-diagonal energy fraction"); plt.xticks(range(len(bins)), bins, rotation=70, ha="right"); plt.yticks(range(len(bins)), bins); plt.title("Cross-scale residual mixing"); plt.tight_layout(); plt.savefig(out / "cross_scale_mixing_heatmap.png", dpi=140); plt.close()
    if danger.get("labels"):
        plt.figure(figsize=(10, 4)); plt.bar(danger["labels"], danger["gain"]); plt.xticks(rotation=60, ha="right"); plt.axhline(0, color="black", lw=.7); plt.title("Danger-zone selection by direct restoration gain"); plt.tight_layout(); plt.savefig(out / "danger_zone_contribution.png", dpi=140); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/muon_continuous_spectral_risk")
    parser.add_argument("--intervention-cap", type=int, default=INTERVENTION_CAP)
    parser.add_argument("--max-snapshots", type=int, default=None, help="diagnostic smoke limit; formal runs use all 10")
    parser.add_argument("--max-tensors", type=int, default=None, help="diagnostic smoke limit per snapshot")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None, help="run one trajectory for bounded CPU execution")
    parser.add_argument("--update", type=int, choices=LANDMARKS, default=None, help="run one landmark for bounded CPU execution")
    parser.add_argument("--append", action="store_true", help="append raw rows to an existing report directory")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(); started = time.perf_counter()
    paths = discover_snapshots(args.reports_root)
    if len(paths) != 10: raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    if args.seed is not None: paths = [p for p in paths if p[0] == args.seed]
    if args.update is not None: paths = [p for p in paths if p[1] == args.update]
    if args.max_snapshots is not None: paths = paths[:max(1, args.max_snapshots)]
    if not paths: raise SystemExit("no snapshots selected")
    args.output.mkdir(parents=True, exist_ok=True)
    mode_rows: list[dict] = []; bin_acc = defaultdict(lambda: defaultdict(float)); bin_counts = defaultdict(int)
    sensitivity_acc = defaultdict(list); restore_acc = defaultdict(list); mixing_acc = defaultdict(float); mixing_counts = defaultdict(int); pair_candidates = defaultdict(float)
    selected_counts = defaultdict(int); tensor_rows: list[dict] = []; restore_rows: list[dict] = []; cross_restore_rows: list[dict] = []; sensitivity_rows: list[dict] = []; seed_stage: list[dict] = []
    all_bin_labels = None
    for seed, update, path in paths:
        snapshot = load_snapshot(path); kwargs = transform_kwargs(snapshot)
        tensor_items = snapshot["tensors"] if args.max_tensors is None else snapshot["tensors"][:max(1, args.max_tensors)]
        for item in tensor_items:
            if len(item["shape"]) != 2: continue
            ident = identity(seed, update, item); matrix = item["tensor"].detach().float();
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False); spec = active_spectrum(s); active_ids = fixed_bin_ids(spec.log10_normalized)
            ids = torch.full((s.numel(),), -1, dtype=torch.long)
            ids[spec.active_indices] = active_ids
            if all_bin_labels is None: all_bin_labels = BIN_LABELS
            q = quantize(matrix, QUANTIZER); ehat = coordinate_error(u, matrix, q, vh); residual = q - matrix
            ref_update = transform(matrix, **kwargs); q_update = transform(q, **kwargs); polar_ref = exact_polar(matrix); polar_q = exact_polar(q)
            baseline = metrics(ref_update, q_update); polar_base = metrics(polar_ref, polar_q); total_e = float(ehat.square().sum())
            tensor_key = (seed, update, ident["parameter_id"])
            tensor_rows.append({**ident, "raw_relative_l2": float(residual.norm() / matrix.norm()), "raw_cosine": float((matrix * q).sum() / (matrix.norm() * q.norm())), "update_error": (1.0 - baseline["cosine"] if baseline["cosine"] is not None else None), "polar_update_error": (1.0 - polar_base["cosine"] if polar_base["cosine"] is not None else None), "effective_rank": int(spec.active_indices.numel()), "baseline_update_cosine": baseline["cosine"], "baseline_update_relative_l2": baseline["relative_l2"], "polar_update_cosine": polar_base["cosine"]})
            # Transfer-function sensitivity descriptors and mode-level residual rows.
            f, deriv = scalar_map_and_derivative(s, matrix_norm=float(matrix.norm()), steps=kwargs["steps"], coefficients=kwargs["coefficients"], eps=kwargs["eps"])
            diag = torch.diag(ehat); row_energy = ehat.square().sum(dim=1); col_energy = ehat.square().sum(dim=0)
            for local, original in enumerate(spec.active_indices.tolist()):
                bid = int(active_ids[local]); label = BIN_LABELS[bid] if bid >= 0 else "below-range"
                mode_rows.append({**ident, "mode": original, "sigma": float(s[original]), "normalized_sigma": float(spec.normalized[local]), "log10_normalized_sigma": float(spec.log10_normalized[local]), "rank_percentile": float(spec.rank_percentile[local]), "local_gap": float(spec.local_gap[local]), "relative_gap": float(spec.relative_gap[local]), "bin_id": bid, "bin_label": label, "diagonal_error_energy": float(diag[original].square()), "left_associated_energy": float(row_energy[original]), "right_associated_energy": float(col_energy[original]), "cross_mode_row_energy": float((row_energy[original] - diag[original].square()).clamp_min(0)), "a_mag": float(abs(deriv[original])), "a_dir": float(abs(f[original]) / s[original].abs().clamp_min(EPS))})
            # Bin allocation (row energy is a disjoint assignment; column is a parallel view).
            for bid, label in enumerate(BIN_LABELS):
                mask = ids == bid
                if not mask.any(): continue
                selected_diag = diag[mask].square().sum(); selected_left = row_energy[mask].sum(); selected_right = col_energy[mask].sum(); selected_cross = (row_energy[mask] - diag[mask].square()).sum()
                vals = bin_acc[(seed, update, label)]; vals["diagonal_energy"] += float(selected_diag); vals["left_energy"] += float(selected_left); vals["right_energy"] += float(selected_right); vals["cross_mode_energy"] += float(selected_cross); vals["total_energy"] += total_e; bin_counts[(seed, update, label)] += 1
                # Actual direct restoration is capped deterministically by discovery order.
                if selected_counts[label] < args.intervention_cap:
                    selected_counts[label] += 1; target = associated_component(u, ehat, vh, mask); diag_target = diagonal_component(u, ehat, vh, mask)
                    for component_name, component in (("diagonal", diag_target), ("associated", target)):
                        corrected = q - component
                        prod = metrics(ref_update, transform(corrected, **kwargs)); pol = metrics(polar_ref, exact_polar(corrected))
                        restore_rows.append({**ident, "bin_id": bid, "bin_label": label, "component": component_name, "selection_status": "selected", "baseline_update_cosine": baseline["cosine"], "baseline_update_relative_l2": baseline["relative_l2"], "update_cosine_gain": (prod["cosine"] - baseline["cosine"] if prod["cosine"] is not None and baseline["cosine"] is not None else None), "update_l2_reduction": (baseline["relative_l2"] - prod["relative_l2"] if prod["relative_l2"] is not None and baseline["relative_l2"] is not None else None), "polar_update_cosine_gain": (pol["cosine"] - polar_base["cosine"] if pol["cosine"] is not None and polar_base["cosine"] is not None else None), "polar_update_l2_reduction": (polar_base["relative_l2"] - pol["relative_l2"] if pol["relative_l2"] is not None and polar_base["relative_l2"] is not None else None), "removed_energy": float(component.square().sum()), "removed_energy_fraction": safe_ratio(float(component.square().sum()), total_e)})
                        restore_acc[(label, component_name)].append(restore_rows[-1])
                    # Deterministic top-energy cross-scale candidates are recorded now,
                    # before looking at restoration gains.
                    for j in range(len(BIN_LABELS)):
                        mask_j = ids == j
                        if j != bid and mask_j.any(): pair_candidates[(label, BIN_LABELS[j])] += float(ehat[mask][:, mask_j].square().sum())
                # One sensitivity representative per bin cap.
                if selected_counts[("sens", label)] < args.intervention_cap:
                    selected_counts[("sens", label)] += 1
                    for kind in ("magnitude", "orientation"):
                        perturbed, invalid = controlled_matrix(u, s, vh, mask, kind, SENSITIVITY_EPSILON, matrix.norm())
                        p = metrics(ref_update, transform(perturbed, **kwargs)); pp = metrics(polar_ref, exact_polar(perturbed))
                        sensitivity_acc[(label, kind)].append({"seed": seed, "update": update, "bin_label": label, "kind": kind, "invalid": invalid, "sensitivity_update_l2": safe_ratio(p["relative_l2"], SENSITIVITY_EPSILON), "sensitivity_update_cosine_error": safe_ratio(1.0 - p["cosine"] if p["cosine"] is not None else None, SENSITIVITY_EPSILON), "polar_sensitivity_update_l2": safe_ratio(pp["relative_l2"], SENSITIVITY_EPSILON), "polar_sensitivity_update_cosine_error": safe_ratio(1.0 - pp["cosine"] if pp["cosine"] is not None else None, SENSITIVITY_EPSILON)})
            # Off-diagonal cross-scale energy matrix, excluding same-bin entries.
            for i, row_label in enumerate(BIN_LABELS):
                mi = ids == i
                if not mi.any(): continue
                for j, col_label in enumerate(BIN_LABELS):
                    if i == j: continue
                    mj = ids == j
                    if mj.any(): mixing_acc[(row_label, col_label)] += float(ehat[mi][:, mj].square().sum()); mixing_counts[(row_label, col_label)] += 1
            # A local, pre-intervention shortlist supplies direct cross-scale
            # restoration readouts without selecting pairs by their gains.
            local_pairs = []
            for i, row_label in enumerate(BIN_LABELS):
                mi = ids == i
                if not mi.any(): continue
                for j, col_label in enumerate(BIN_LABELS):
                    if i == j: continue
                    mj = ids == j
                    if mj.any(): local_pairs.append((float(ehat[mi][:, mj].square().sum()), row_label, col_label, mi, mj))
            for energy, row_label, col_label, mi, mj in sorted(local_pairs, reverse=True)[:2]:
                component = pair_component(u, ehat, vh, mi, mj); corrected = q - component
                prod = metrics(ref_update, transform(corrected, **kwargs)); pol = metrics(polar_ref, exact_polar(corrected))
                cross_restore_rows.append({**ident, "row_bin_label": row_label, "col_bin_label": col_label, "energy": energy, "energy_fraction": safe_ratio(energy, total_e), "selection_status": "top-two-by-pre-intervention-energy", "update_cosine_gain": (prod["cosine"] - baseline["cosine"] if prod["cosine"] is not None and baseline["cosine"] is not None else None), "polar_update_cosine_gain": (pol["cosine"] - polar_base["cosine"] if pol["cosine"] is not None and polar_base["cosine"] is not None else None), "update_l2_reduction": (baseline["relative_l2"] - prod["relative_l2"] if prod["relative_l2"] is not None and baseline["relative_l2"] is not None else None)})
    # Append mode is used to keep each bounded one-snapshot process below the
    # execution watchdog.  Only raw rows are loaded; aggregate rows are
    # regenerated from the combined data below.
    if args.append and args.output.exists():
        mode_rows = read_csv(args.output / "mode_spectral_coordinates.csv") + mode_rows
        old_bins = [r for r in read_csv(args.output / "bin_residual_energy.csv") if r.get("aggregation") == "group"]
        old_sens = [r for r in read_csv(args.output / "bin_controlled_sensitivity.csv") if r.get("aggregation") != "mean"]
        old_restore = [r for r in read_csv(args.output / "bin_restoration_metrics.csv") if r.get("aggregation") != "mean"]
        old_mixing = read_csv(args.output / "cross_scale_mixing.csv")
        bin_rows_old = old_bins
        sensitivity_out_old = old_sens
        restore_out_old = old_restore
        # Current group rows are added below by concatenating these holders.
        bin_acc_old = bin_rows_old
        sensitivity_out = sensitivity_out_old
        restore_rows = restore_out_old + restore_rows
        # Rebuild per-bin aggregate inputs from the raw rows after the current
        # run; old bin rows are already group-level and remain valid.
        bin_rows = bin_acc_old
        bin_rows.extend([])
        for row in old_mixing:
            key = (row.get("row_bin_label"), row.get("col_bin_label")); mixing_acc[key] += float(row.get("energy") or 0.0)
        old_tensor = read_csv(args.output / "tensor_baseline.csv")
        tensor_rows = old_tensor + tensor_rows
    else:
        bin_rows = None
        sensitivity_out = None
        restore_rows = restore_rows

    # Aggregate bin records and direct/sensitivity summaries.
    current_bin_rows = []
    for key, vals in sorted(bin_acc.items()):
        seed, update, label = key; total = vals["total_energy"]
        current_bin_rows.append({"seed": seed, "update": update, "bin_label": label, "tensor_count": bin_counts[key], "diagonal_energy": vals["diagonal_energy"], "left_associated_energy": vals["left_energy"], "right_associated_energy": vals["right_energy"], "cross_mode_energy": vals["cross_mode_energy"], "total_energy": total, "residual_fraction": safe_ratio(vals["left_energy"], total), "aggregation": "group"})
    bin_rows = (bin_rows or []) + current_bin_rows
    mean_bins = []
    for label in BIN_LABELS:
        rows = [r for r in bin_rows if r["bin_label"] == label]
        if rows: mean_bins.append({"bin_label": label, "aggregation": "mean", "tensor_count": sum(int(float(r["tensor_count"])) for r in rows), "residual_fraction": sum(float(r["residual_fraction"]) for r in rows) / len(rows), "diagonal_energy": sum(float(r["diagonal_energy"]) for r in rows) / len(rows), "cross_mode_energy": sum(float(r["cross_mode_energy"]) for r in rows) / len(rows)})
    current_sensitivity_out = []
    for (label, kind), rows in sorted(sensitivity_acc.items()):
        for row in rows: current_sensitivity_out.append(row)
        current_sensitivity_out.append({"bin_label": label, "kind": kind, "aggregation": "mean", "samples": len(rows), **{key: sum(float(r[key]) for r in rows if finite(r.get(key))) / max(1, sum(finite(r.get(key)) for r in rows)) for key in ("sensitivity_update_l2", "sensitivity_update_cosine_error", "polar_sensitivity_update_l2", "polar_sensitivity_update_cosine_error")}})
    sensitivity_out = (sensitivity_out or []) + current_sensitivity_out
    restore_out = list(restore_rows)
    for (label, component), rows in sorted(restore_acc.items()):
        restore_out.append({"bin_label": label, "component": component, "aggregation": "mean", "samples": len(rows), **{key: sum(float(r[key]) for r in rows if finite(r.get(key))) / max(1, sum(finite(r.get(key)) for r in rows)) for key in ("update_cosine_gain", "update_l2_reduction", "polar_update_cosine_gain", "polar_update_l2_reduction", "removed_energy_fraction")}})
    # Deterministic danger zone: narrowest contiguous interval reaching 50% of
    # total positive associated-restoration gain.
    agg_assoc = [r for r in restore_out if r.get("aggregation") == "mean" and r.get("component") == "associated" and finite(r.get("update_cosine_gain"))]
    positive = [max(0.0, float(next((r["update_cosine_gain"] for r in agg_assoc if r["bin_label"] == label), 0.0))) for label in BIN_LABELS]
    threshold = .5 * sum(positive); best = None
    for i in range(len(BIN_LABELS)):
        total = 0.0
        for j in range(i, len(BIN_LABELS)):
            total += positive[j]
            if total >= threshold and (best is None or j - i < best[1] - best[0]): best = (i, j, total); break
    if best is None: best = (0, len(BIN_LABELS) - 1, sum(positive))
    labels = list(BIN_LABELS[best[0]:best[1] + 1]); danger = {"labels": labels, "gain": positive[best[0]:best[1] + 1], "threshold_fraction": .5, "positive_gain_total": sum(positive), "danger_gain": best[2], "start_log10": BIN_EDGES[best[0]], "end_log10": BIN_EDGES[best[1] + 1]}
    mixing_rows = [{"row_bin_label": row, "col_bin_label": col, "energy": value, "energy_fraction": safe_ratio(value, sum(mixing_acc.values()))} for (row, col), value in sorted(mixing_acc.items())]
    top_pairs = sorted(mixing_acc, key=mixing_acc.get, reverse=True)[:PAIR_SHORTLIST]
    correlations = []
    for feature in ("raw_relative_l2", "effective_rank", "update_error", "polar_update_error"):
        if feature != "update_error": correlations.append(correlation(tensor_rows, feature, "update_error"))
    correlations += [correlation(tensor_rows, "update_error", "polar_update_error")]
    # Mode-level risk correlations use each mode's row-associated error against
    # the parent tensor's update error.
    by_key = {(r["seed"], r["update"], r["parameter_id"]): r for r in tensor_rows}
    for row in mode_rows: row["update_error"] = by_key[(row["seed"], row["update"], row["parameter_id"])] ["update_error"]
    for feature in ("diagonal_error_energy", "left_associated_energy", "relative_gap", "a_dir", "a_mag", "log10_normalized_sigma"):
        correlations.append(correlation(mode_rows, feature, "update_error"))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "mode_spectral_coordinates.csv", mode_rows)
    write_csv(args.output / "tensor_baseline.csv", tensor_rows)
    write_csv(args.output / "spectrum_bin_summary.csv", bin_rows + mean_bins)
    write_csv(args.output / "bin_residual_energy.csv", bin_rows + mean_bins)
    write_csv(args.output / "bin_controlled_sensitivity.csv", sensitivity_out)
    write_csv(args.output / "bin_restoration_metrics.csv", restore_out)
    write_csv(args.output / "danger_zone_summary.csv", [danger])
    write_csv(args.output / "cross_scale_mixing.csv", mixing_rows)
    write_csv(args.output / "cross_scale_restoration.csv", cross_restore_rows + [{"row_bin_label": a, "col_bin_label": b, "status": "global_energy_shortlist", "energy": mixing_acc[(a, b)]} for a, b in top_pairs])
    write_csv(args.output / "risk_metric_correlations.csv", correlations)
    # Stable subgroup table.
    stability = []
    for seed in SEEDS:
        for update in LANDMARKS:
            rows = [r for r in restore_out if r.get("aggregation") == "group" and r.get("seed") == seed and r.get("update") == update and r.get("component") == "associated"]
            for label in labels:
                matched = [r for r in rows if r.get("bin_label") == label]
                stability.append({"seed": seed, "update": update, "bin_label": label, "update_cosine_gain": matched[0].get("update_cosine_gain") if matched else None, "in_global_danger_zone": True})
    write_csv(args.output / "seed_stage_stability.csv", stability)
    summary = args.output / "summary.md"
    summary.write_text(f"""# Continuous spectral risk analysis\n\nCoverage: 10 formal snapshots, {len(tensor_rows)} eligible 2D Muon tensors, {len(mode_rows)} active modes. Runtime: {time.perf_counter() - started:.1f} CPU seconds.\n\nProduction INT4 dynamic b2048 and the production Muon transform are reused unchanged. The continuous coordinate is `log10(sigma_i / sigma_max)` for active modes (`sigma_i/sigma_max >= 1e-6`). Fixed bins are {', '.join(BIN_LABELS)}; all were retained because sparse-bin support is reported explicitly.\n\nThe direct intervention cap is deterministic: the first {args.intervention_cap} tensors in snapshot/parameter discovery order per bin are used for bin restoration and sensitivity. Baseline residual allocation covers every eligible tensor.\n\n## Danger zone\n\nThe zone is selected from associated-bin restoration gains by the narrowest contiguous interval reaching 50% of total positive gain. It is `{danger['start_log10']:g}` to `{danger['end_log10']:g}` in log10 normalized singular value, with {danger['danger_gain']:.6g} of {danger['positive_gain_total']:.6g} positive gain in the selected interval. This is a deterministic descriptive rule, not a fitted threshold.\n\nSensitivity alone identifies where perturbations are dangerous; residual energy alone identifies where INT4 error is large; direct restoration measures where realized error matters. The report keeps these quantities separate and does not claim causal mediation. Cross-bin associated restorations overlap on cross-bin entries, so their gains are not additive.\n""")
    (args.output / "methodology.md").write_text("""# Methodology\n\nFor each FP32 snapshot matrix M, compute a reduced FP32 SVD M=U diag(sigma) V^T and production INT4 dynamic reconstruction Q(M). Active modes satisfy sigma/sigma_max >= 1e-6. The primary continuous coordinate is log10(sigma/sigma_max), with local absolute/relative gaps, rank percentile, Newton--Schulz A_mag and A_dir descriptors.\n\nResidual coordinates are E_hat=U^T(Q(M)-M)V. Diagonal-bin energy is the sum of |E_hat[i,i]|^2. Left/right associated energy sums full rows/columns for modes in a bin; these are parallel attributions, not an additive partition. Direct diagonal restoration removes only selected diagonal coordinates. Associated restoration removes every coordinate whose row OR column is selected; different bins overlap on cross-bin entries by design, so direct gains are not expected to add.\n\nControlled sensitivity uses deterministic epsilon=0.001 magnitude perturbations of selected singular values and adjacent left-vector rotations, measured through production K=5 and exact polar. Only the first 20 tensors per bin in deterministic discovery order are used for these expensive interventions. Cross-scale pairs are shortlisted by pre-intervention residual energy.\n\nThe danger zone is the narrowest contiguous fixed-bin interval reaching 50% of positive mean associated-restoration gain. This rule is declared before interpreting outcomes. Pearson and Spearman values are descriptive correlations only.\n""")
    if not args.skip_plots: make_plots(args.output, bin_rows + mean_bins, sensitivity_out, restore_out, mixing_rows, danger)
    print(f"wrote {args.output}; tensors={len(tensor_rows)} modes={len(mode_rows)} runtime={time.perf_counter() - started:.1f}s")


if __name__ == "__main__": main()
