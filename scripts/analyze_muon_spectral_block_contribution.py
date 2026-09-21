#!/usr/bin/env python3
"""Offline 3x3 spectral error-block contribution analysis for INT4 Muon."""
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
from optim.muon_spectral_block_contribution import (  # noqa: E402
    BANDS, GROUPS, PRIMARY_BLOCKS, block_energies, decompose, exact_polar,
    metric_pair, paired_restore_metrics, quantize, restore_blocks,
)
from optim.muon_update_fidelity import load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
PRIMARY = tuple(PRIMARY_BLOCKS)


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
        writer.writeheader()
        writer.writerows(rows)


def finite(value) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def base_identity(seed: int, update: int, item: dict) -> dict:
    return {"seed": seed, "update": update,
            "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
            "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")),
            "shape": str(item["shape"])}


def transform_kwargs(snapshot: dict) -> dict:
    config = snapshot["metadata"]["muon_transform"]
    return {"steps": int(config["steps"]),
            "coefficients": tuple(float(x) for x in config["coefficients"]),
            "eps": float(config["eps"])}


def correlation(rows: list[dict], feature: str, target: str) -> dict:
    values = [(float(row[feature]), float(row[target])) for row in rows
              if finite(row.get(feature)) and finite(row.get(target))]
    if len(values) < 2:
        return {"feature": feature, "target": target, "sample_count": len(values), "pearson": None, "spearman": None}
    x = torch.tensor([v[0] for v in values], dtype=torch.float64)
    y = torch.tensor([v[1] for v in values], dtype=torch.float64)

    def corr(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        den = ac.norm() * bc.norm()
        return float((ac * bc).sum() / den) if den.item() else None

    def rank(a):
        order = torch.argsort(a, stable=True)
        result = torch.empty_like(a)
        result[order] = torch.arange(a.numel(), dtype=a.dtype)
        i = 0
        while i < a.numel():
            j = i + 1
            while j < a.numel() and a[order[j]] == a[order[i]]:
                j += 1
            if j - i > 1:
                result[order[i:j]] = (i + j - 1) / 2
            i = j
        return result

    return {"feature": feature, "target": target, "sample_count": len(values),
            "pearson": corr(x, y), "spearman": corr(rank(x), rank(y))}


def make_plots(out: Path, energy_rows: list[dict], restore_rows: list[dict],
               efficiency_rows: list[dict], grouped_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs are complete.\n")
        return

    def mean(rows, key, group_key):
        values = defaultdict(list)
        for row in rows:
            if finite(row.get(key)):
                values[row[group_key]].append(float(row[key]))
        return [(name, sum(xs) / len(xs)) for name, xs in sorted(values.items())]

    matrix = {name: sum(float(r["energy_fraction"]) for r in energy_rows if r["block"] == name) /
              max(1, sum(1 for r in energy_rows if r["block"] == name)) for name in PRIMARY}
    grid = torch.tensor([[matrix.get(f"{r}{c}", 0.0) for c in "HMT"] for r in "HMT"])
    plt.figure(figsize=(6, 5)); plt.imshow(grid, cmap="magma"); plt.colorbar(label="mean residual fraction")
    plt.xticks(range(3), ["V-head", "V-mid", "V-tail"]); plt.yticks(range(3), ["U-head", "U-mid", "U-tail"])
    for i in range(3):
        for j in range(3): plt.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", color="white")
    plt.title("3x3 INT4 residual energy"); plt.tight_layout(); plt.savefig(out / "spectral_block_energy_heatmap.png", dpi=140); plt.close()

    def heat(rows, key, title, name):
        values = {block: sum(float(r[key]) for r in rows if r["block"] == block and finite(r.get(key))) /
                  max(1, sum(1 for r in rows if r["block"] == block and finite(r.get(key)))) for block in PRIMARY}
        grid = torch.tensor([[values.get(f"{r}{c}", 0.0) for c in "HMT"] for r in "HMT"])
        plt.figure(figsize=(6, 5)); plt.imshow(grid, cmap="viridis"); plt.colorbar(label=key)
        plt.xticks(range(3), ["V-head", "V-mid", "V-tail"]); plt.yticks(range(3), ["U-head", "U-mid", "U-tail"])
        for i in range(3):
            for j in range(3): plt.text(j, i, f"{grid[i,j]:.4f}", ha="center", va="center", color="white")
        plt.title(title); plt.tight_layout(); plt.savefig(out / name, dpi=140); plt.close()

    heat(restore_rows, "update_cosine_gain", "K=5 restoration gain", "production_restoration_heatmap.png")
    heat(efficiency_rows, "cosine_efficiency", "K=5 normalized efficiency", "normalized_efficiency_heatmap.png")
    heat([r for r in restore_rows if r["readout"] == "polar"], "update_cosine_gain", "Exact-polar restoration gain", "polar_restoration_heatmap.png")

    labels = ["all_diag", "all_offdiag", "H_to_M", "M_to_T", "H_to_T"]
    vals = {label: sum(float(r["update_cosine_gain"]) for r in grouped_rows if r["group"] == label and r["readout"] == "production") /
            max(1, sum(1 for r in grouped_rows if r["group"] == label and r["readout"] == "production")) for label in labels}
    plt.figure(figsize=(8, 4)); plt.bar(labels, [vals[x] for x in labels]); plt.ylabel("mean update cosine gain"); plt.title("Grouped restoration")
    plt.tight_layout(); plt.savefig(out / "grouped_restoration.png", dpi=140); plt.close()

    plt.figure(figsize=(7, 5))
    xs, ys = [], []
    for row in energy_rows:
        target = next((x for x in restore_rows if x["seed"] == row["seed"] and x["update"] == row["update"] and x["parameter_id"] == row["parameter_id"] and x["block"] == row["block"] and x["readout"] == "production"), None)
        if target and finite(row.get("energy_fraction")) and finite(target.get("update_cosine_gain")):
            xs.append(float(row["energy_fraction"])); ys.append(float(target["update_cosine_gain"]))
    plt.scatter(xs, ys, s=8); plt.xlabel("block residual energy fraction"); plt.ylabel("update cosine gain"); plt.title("Block energy vs recovery")
    plt.tight_layout(); plt.savefig(out / "block_energy_vs_restoration.png", dpi=140); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/muon_spectral_block_contribution")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    paths = discover_snapshots(args.reports_root)
    if len(paths) != 10:
        raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    args.output.mkdir(parents=True, exist_ok=True)
    energy_rows: list[dict] = []
    restore_rows: list[dict] = []
    efficiency_rows: list[dict] = []
    grouped_rows: list[dict] = []
    tensor_rows: list[dict] = []
    interaction_rows: list[dict] = []
    left_right_rows: list[dict] = []

    for seed, update, path in paths:
        snapshot = load_snapshot(path)
        kwargs = transform_kwargs(snapshot)
        for item in snapshot["tensors"]:
            if len(item["shape"]) != 2:
                continue
            d = decompose(item["tensor"])
            source, q = d.matrix, quantize(d.matrix, "int4-dynamic-b2048")
            prod_ref = muon_reference.zeropower_newton_schulz(source.clone(), **kwargs)
            prod_base = muon_reference.zeropower_newton_schulz(q.clone(), **kwargs)
            polar_ref, polar_base = exact_polar(source), exact_polar(q)
            base = base_identity(seed, update, item)
            raw_base = metric_pair(source, q)
            baseline = metric_pair(prod_ref, prod_base)
            pbase = metric_pair(polar_ref, polar_base)
            energy = block_energies(d, q)
            total = energy.total
            row_base = base | {"effective_rank": int(d.bands["active"].numel()),
                               "total_residual_energy": total,
                               "projected_residual_energy": energy.projected,
                               "unresolved_residual_energy": energy.unresolved,
                               "raw_relative_l2": raw_base["relative_l2"],
                               "raw_cosine": raw_base["cosine"],
                               "baseline_update_cosine": baseline["cosine"],
                               "baseline_update_relative_l2": baseline["relative_l2"],
                               "baseline_polar_cosine": pbase["cosine"],
                               "baseline_polar_relative_l2": pbase["relative_l2"]}
            tensor_rows.append(row_base)
            for block in PRIMARY:
                amount = energy.blocks.get(block, 0.0)
                energy_rows.append(row_base | {"block": block, "energy": amount,
                    "energy_fraction": amount / total if total else None,
                    "accounting": "primary"})
            # Keep the complete 4x4 accounting decomposition alongside the
            # requested 3x3 matrix, including other-mode and unresolved terms.
            for accounting_block, amount in energy.blocks.items():
                if accounting_block in PRIMARY:
                    continue
                energy_rows.append(row_base | {"block": accounting_block, "energy": amount,
                    "energy_fraction": amount / total if total else None,
                    "accounting": "other-associated"})
            energy_rows.append(row_base | {"block": "unresolved", "energy": energy.unresolved,
                "energy_fraction": energy.unresolved / total if total else None,
                "accounting": "unresolved"})

            # Direct primary-block restoration at production K and exact polar.
            candidate_cache: dict[tuple[str, ...], torch.Tensor] = {}
            for block in PRIMARY:
                candidate_cache[(block,)] = restore_blocks(d, q, (block,))
            for group in ("all_diag", "all_offdiag", "H_to_M", "M_to_T", "H_to_T"):
                candidate_cache[GROUPS[group]] = restore_blocks(d, q, GROUPS[group])
            for block in PRIMARY:
                for readout, reference, baseline_update, is_polar in (("production", prod_ref, prod_base, False), ("polar", polar_ref, polar_base, True)):
                    corrected = candidate_cache[(block,)]
                    m = paired_restore_metrics(source, q, corrected, transform_kwargs=kwargs,
                                               reference_update=reference, baseline_update=baseline_update, polar=is_polar)
                    amount = energy.blocks.get(block, 0.0)
                    fraction = amount / total if total else None
                    restore_rows.append(row_base | {"block": block, "readout": readout,
                        "removed_residual_energy": amount, "removed_residual_fraction": fraction, **m})
                    efficiency_rows.append(row_base | {"block": block, "readout": readout,
                        "removed_residual_fraction": fraction,
                        "cosine_efficiency": (m["update_cosine_gain"] / fraction if fraction and m["update_cosine_gain"] is not None else None),
                        "l2_efficiency": (m["update_l2_reduction"] / fraction if fraction and m["update_l2_reduction"] is not None else None)})

            # Grouped, row-associated, and column-associated direct restorations.
            for group in ("all_diag", "all_offdiag", "H_to_M", "M_to_T", "H_to_T", "U_head_rows", "U_middle_rows", "U_tail_rows", "V_head_cols", "V_middle_cols", "V_tail_cols"):
                selected = GROUPS[group]
                corrected = candidate_cache.get(selected, restore_blocks(d, q, selected))
                for readout, reference, baseline_update, is_polar in (("production", prod_ref, prod_base, False), ("polar", polar_ref, polar_base, True)):
                    m = paired_restore_metrics(source, q, corrected, transform_kwargs=kwargs,
                                               reference_update=reference, baseline_update=baseline_update, polar=is_polar)
                    amount = sum(energy.blocks.get(block, 0.0) for block in selected)
                    fraction = amount / total if total else None
                    grouped_rows.append(row_base | {"group": group, "readout": readout,
                        "removed_residual_energy": amount, "removed_residual_fraction": fraction, **m})

            # A small predeclared interaction set; all values are direct restorations.
            interaction_sets = (("MM_plus_M_to_T", ("MM", "MT", "TM")),
                                ("MM_plus_H_to_M", ("MM", "HM", "MH")),
                                ("diag_plus_M_to_T", ("HH", "MM", "TT", "MT", "TM")))
            single_gain = {(r["block"], r["readout"]): r["update_cosine_gain"] for r in restore_rows[-18:] if r["seed"] == seed and r["update"] == update and r["parameter_id"] == base["parameter_id"]}
            for name, selected in interaction_sets:
                corrected = restore_blocks(d, q, selected)
                for readout, reference, baseline_update, is_polar in (("production", prod_ref, prod_base, False), ("polar", polar_ref, polar_base, True)):
                    m = paired_restore_metrics(source, q, corrected, transform_kwargs=kwargs,
                                               reference_update=reference, baseline_update=baseline_update, polar=is_polar)
                    gains = sum(float(single_gain.get((block, readout), 0.0) or 0.0) for block in selected)
                    interaction_rows.append(row_base | {"interaction": name, "readout": readout,
                        "update_cosine_gain": m["update_cosine_gain"], "sum_single_gains": gains,
                        "interaction_term": (m["update_cosine_gain"] - gains if m["update_cosine_gain"] is not None else None)})

            for group in ("U_head_rows", "U_middle_rows", "U_tail_rows", "V_head_cols", "V_middle_cols", "V_tail_cols"):
                matching = next((r for r in grouped_rows[::-1] if r["seed"] == seed and r["update"] == update and r["parameter_id"] == base["parameter_id"] and r["group"] == group and r["readout"] == "production"), None)
                if matching:
                    left_right_rows.append(row_base | {"group": group, "update_cosine_gain": matching["update_cosine_gain"], "update_l2_reduction": matching["update_l2_reduction"]})
        print(f"processed seed={seed} update={update}", flush=True)

    for row in tensor_rows:
        row["baseline_update_error"] = (1.0 - row["baseline_update_cosine"]
                                         if finite(row.get("baseline_update_cosine")) else None)
    corr_rows = []
    for feature in ("raw_relative_l2", "baseline_update_relative_l2", "total_residual_energy"):
        corr_rows.append(correlation(tensor_rows, feature, "baseline_update_error"))
    for block in PRIMARY:
        subset = [r for r in energy_rows if r["block"] == block]
        for row in subset:
            row["baseline_update_error"] = (1.0 - row["baseline_update_cosine"]
                                             if finite(row.get("baseline_update_cosine")) else None)
        corr_rows.append(correlation(subset, "energy_fraction", "baseline_update_error") | {"block": block})

    write_csv(args.output / "spectral_block_energy.csv", energy_rows)
    write_csv(args.output / "spectral_block_restoration.csv", restore_rows)
    write_csv(args.output / "spectral_block_efficiency.csv", efficiency_rows)
    write_csv(args.output / "grouped_restoration.csv", grouped_rows)
    write_csv(args.output / "left_right_association.csv", left_right_rows)
    write_csv(args.output / "interaction_terms.csv", interaction_rows)
    write_csv(args.output / "tensor_block_summary.csv", tensor_rows)
    write_csv(args.output / "spectral_block_correlations.csv", corr_rows)
    if not args.skip_plots:
        make_plots(args.output, energy_rows, restore_rows, efficiency_rows, grouped_rows)

    def avg(rows, field, filt=None):
        values = [float(r[field]) for r in rows if (filt is None or filt(r)) and finite(r.get(field))]
        return sum(values) / len(values) if values else None

    def med(rows, field, filt=None):
        values = sorted(float(r[field]) for r in rows if (filt is None or filt(r)) and finite(r.get(field)))
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2

    lines = ["# 3x3 spectral block contribution analysis", "",
             f"Coverage: 10 formal snapshots, {len(tensor_rows)} eligible 2D Muon tensors. Runtime: {time.perf_counter() - started:.2f}s CPU.",
             "Production INT4 dynamic b2048 and production Muon Newton--Schulz are reused unchanged.",
             "The FP32 reduced SVD defines the established active/head/middle/tail bands; primary blocks are HH, HM, HT, MH, MM, MT, TH, TM, TT. Other-mode blocks and unresolved reduced-SVD energy are retained for accounting.",
             "For block ij, E_ij = U_i (U_i.T E V_j) V_j.T. Direct restoration subtracts exactly E_ij from Q(M), and gains are measured against both production K=5 and exact polar readouts.", "", "## Mean primary block results", ""]
    for block in PRIMARY:
        er = avg(energy_rows, "energy_fraction", lambda r, b=block: r["block"] == b)
        gain = avg(restore_rows, "update_cosine_gain", lambda r, b=block: r["block"] == b and r["readout"] == "production")
        median_gain = med(restore_rows, "update_cosine_gain", lambda r, b=block: r["block"] == b and r["readout"] == "production")
        pgain = avg(restore_rows, "update_cosine_gain", lambda r, b=block: r["block"] == b and r["readout"] == "polar")
        eta = avg(efficiency_rows, "cosine_efficiency", lambda r, b=block: r["block"] == b and r["readout"] == "production")
        median_energy = med(energy_rows, "energy_fraction", lambda r, b=block: r["block"] == b)
        lines.append(f"- {block}: mean_energy_fraction={er}, median_energy_fraction={median_energy}, mean_K5_gain={gain}, median_K5_gain={median_gain}, mean_polar_gain={pgain}, mean_efficiency={eta}")
    lines += ["", "## Grouped interpretation", "", "Grouped restoration is direct and nonlinear; sums of individual gains are not treated as exact additive contributions.", "Diagonal and off-diagonal groups, H<->M, M<->T, H<->T, and left/right row/column associations are recorded in grouped_restoration.csv."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    (args.output / "methodology.md").write_text("""# Methodology

This is a read-only CPU analysis of the existing ten formal FP32 Muon momentum snapshots. Only 2D tensors are eligible. Each tensor is quantized through the production INT4 blockwise-dynamic b2048 path. No training state is modified.

The reduced FP32 SVD M=U Sigma V.T defines active modes with sigma/sigma_max >= 1e-6 and the established 10% head, centered-middle, and tail index bands. In FP32 spectral coordinates E_hat=U.T(Q(M)-M)V, each accounting block is E_ij=U_i E_hat_ij V_j.T. The nine primary 3x3 blocks and all other-associated blocks are orthogonal, so squared Frobenius energies do not double-count. Total, projected, and unresolved energy are retained.

For a direct ablation, Q_restore=Q-E_ij (or a selected group of blocks) is passed through the production K=5 Muon transform. The same candidates are also evaluated through the reduced-SVD exact polar factor. Update cosine gain and relative-L2 reduction are measured against the unquantized FP32 reference. Normalized efficiency divides cosine gain or L2 reduction by the removed residual-energy fraction; unstable zero-energy ratios are left unavailable.

The predeclared grouped tests are all diagonal, all off-diagonal, H<->M, M<->T, H<->T, left-row groups, and right-column groups. Three interaction sets test direct gain against the sum of single-block gains. These are descriptive ablations: Muon is nonlinear and no additive causal decomposition is claimed.
""")
    print(f"wrote {args.output} in {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
