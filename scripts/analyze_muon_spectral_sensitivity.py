#!/usr/bin/env python3
"""Offline spectral-sensitivity analysis for formal FP32 Muon snapshots.

This command only reads ``reports/`` snapshots and writes a new analysis
directory. It never starts training. SVDs are processed one tensor at a time
so the full formal set does not require retaining hundreds of decompositions.
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
from optim.muon_spectral_sensitivity import (  # noqa: E402
    QUANTIZER_TO_SIMULATION, band_slices, conditioning_intervention, decompose,
    quantize, quantized_spectral_metrics, spectral_error_decomposition,
    spectral_metrics, subspace_metrics, update_metrics,
)
from optim.muon_update_fidelity import QUANTIZERS as PRODUCTION_QUANTIZERS  # noqa: E402
from optim.muon_update_fidelity import _ratios, load_snapshot  # noqa: E402

QUANTIZERS = tuple(QUANTIZER_TO_SIMULATION)
LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
TAUS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
FORMAL_GROUP_SIZE = len(LANDMARKS)


def discover_snapshots(root: Path) -> list[tuple[int, int, Path]]:
    """Find complete seed/update groups under reports, excluding smoke groups.

    A group is the directory containing ``muon_momentum_snapshots``. Requiring
    all five landmarks prevents the restored two-update smoke run from being
    accidentally mixed into the formal study. If multiple complete groups are
    present, the lexicographically first group for each seed is selected.
    """
    groups: dict[tuple[int, str], dict[int, Path]] = {}
    for path in sorted(root.rglob("update_*.pt")):
        if "muon_momentum_snapshots" not in path.parts:
            continue
        try:
            snapshot = load_snapshot(path)
            metadata = snapshot["metadata"]
            seed = int(metadata["seeds"]["seed"])
            update = int(metadata["update"])
        except (KeyError, ValueError, RuntimeError, OSError):
            continue
        if seed not in SEEDS or update not in LANDMARKS:
            continue
        group = str(path.parent)
        groups.setdefault((seed, group), {})[update] = path
    selected: list[tuple[int, int, Path]] = []
    for seed in SEEDS:
        candidates = sorted((group, values) for (candidate_seed, group), values in groups.items()
                            if candidate_seed == seed and set(values) == set(LANDMARKS))
        if not candidates:
            continue
        group, values = candidates[0]
        selected.extend((seed, update, values[update]) for update in LANDMARKS)
    return selected


def _id(item: dict, seed: int, update: int) -> dict:
    return {
        "seed": seed, "update": update,
        "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
        "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")),
        "shape": str(item["shape"]),
    }


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
        writer.writeheader()
        writer.writerows(rows)


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def correlation(rows: list[dict], x_key: str, y_key: str) -> dict:
    """Return descriptive Pearson/Spearman coefficients with explicit N."""
    values = [(float(row[x_key]), float(row[y_key])) for row in rows
              if _finite(row.get(x_key)) and _finite(row.get(y_key))]
    if len(values) < 2:
        return {"feature": x_key, "target": y_key, "sample_count": len(values),
                "pearson": None, "spearman": None}
    x = torch.tensor([v[0] for v in values], dtype=torch.float64)
    y = torch.tensor([v[1] for v in values], dtype=torch.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = xc.square().sum().sqrt() * yc.square().sum().sqrt()
    pearson = float((xc * yc).sum() / denom) if denom.item() else None
    # Average ranks are deterministic and handle ties correctly.
    def ranks(value: torch.Tensor) -> torch.Tensor:
        ordered = torch.argsort(value, stable=True)
        result = torch.empty_like(value)
        result[ordered] = torch.arange(value.numel(), dtype=value.dtype)
        i = 0
        while i < value.numel():
            j = i + 1
            while j < value.numel() and value[ordered[j]] == value[ordered[i]]:
                j += 1
            if j - i > 1:
                result[ordered[i:j]] = (i + j - 1) / 2
            i = j
        return result
    xr, yr = ranks(x), ranks(y)
    xrc, yrc = xr - xr.mean(), yr - yr.mean()
    rank_denom = xrc.square().sum().sqrt() * yrc.square().sum().sqrt()
    spearman = float((xrc * yrc).sum() / rank_denom) if rank_denom.item() else None
    return {"feature": x_key, "target": y_key, "sample_count": len(values),
            "pearson": pearson, "spearman": spearman}


def _transform_kwargs(metadata: dict) -> dict:
    config = metadata["muon_transform"]
    return {"steps": int(config["steps"]),
            "coefficients": tuple(float(x) for x in config["coefficients"]),
            "eps": float(config["eps"])}


def _flat_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()
            if isinstance(value, (str, int, float)) or value is None}


def _selection(source_rows: list[dict], nearest_rows: list[dict]) -> tuple[set[tuple], dict[tuple, str]]:
    """Predeclare representatives from pre-intervention features only."""
    by_key = {(r["seed"], r["update"], r["parameter_id"]): r for r in source_rows}
    nearest_by_key = {(r["seed"], r["update"], r["parameter_id"]): r for r in nearest_rows}
    chosen: list[tuple[tuple, str]] = []
    finite_cond = lambda r: float(r["effective_condition_number"]) if _finite(r.get("effective_condition_number")) else float("inf")
    ascending = sorted(source_rows, key=lambda r: (finite_cond(r), r["seed"], r["update"], r["parameter_id"]))
    for row, reason in ((ascending[0], "low_condition"),
                        (ascending[len(ascending) // 2], "medium_condition"),
                        (ascending[-1], "high_condition")):
        chosen.append(((row["seed"], row["update"], row["parameter_id"]), reason))
    errors = sorted(nearest_rows, key=lambda r: (float(r["update_direction_error"]) if _finite(r.get("update_direction_error")) else float("inf"),
                                                 r["seed"], r["update"], r["parameter_id"]))
    if errors:
        chosen.append(((errors[0]["seed"], errors[0]["update"], errors[0]["parameter_id"]), "low_update_error"))
        chosen.append(((errors[-1]["seed"], errors[-1]["update"], errors[-1]["parameter_id"]), "high_update_error"))
    selected: set[tuple] = set()
    reasons: dict[tuple, str] = {}
    for key, reason in chosen:
        if key in by_key and key not in selected:
            selected.add(key)
            reasons[key] = reason
    return selected, reasons


def _make_plots(output: Path, source_rows: list[dict], quant_rows: list[dict],
                decomposition_rows: list[dict], subspace_rows: list[dict],
                intervention_rows: list[dict], spectra: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (output / "plots_unavailable.txt").write_text("matplotlib is unavailable; CSV outputs remain complete.\n")
        return
    # The first three are the requested tensor-level relationship plots.
    features = (("log_effective_condition_number", "effective_condition_vs_update_error.png", "log10 effective condition number"),
                ("tail_relative_spectral_perturbation", "tail_relative_perturbation_vs_update_error.png", "tail relative spectral perturbation"),
                ("tail_subspace_distortion", "tail_subspace_vs_update_error.png", "tail subspace projection distance"))
    for feature, filename, xlabel in features:
        plt.figure(figsize=(7, 5))
        for qname in QUANTIZERS:
            rows = [row for row in quant_rows if row["quantizer"] == qname and _finite(row.get(feature))]
            plt.scatter([row[feature] for row in rows], [row["update_direction_error"] for row in rows], s=8, alpha=.65, label=qname)
        plt.xlabel(xlabel); plt.ylabel("1 - post-Muon update cosine"); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(output / filename, dpi=140); plt.close()
    # Singular spectrum examples: retain only two arrays to avoid report-memory growth.
    if spectra:
        for label, values in spectra.items():
            plt.figure(figsize=(7, 5)); plt.semilogy(range(1, len(values) + 1), values, label=label)
            plt.xlabel("singular-value index"); plt.ylabel("singular value"); plt.legend(); plt.tight_layout()
            plt.savefig(output / f"singular_spectrum_{label.replace('/', '_')}.png", dpi=140); plt.close()
        if "FP32" in spectra and "INT4-dynamic" in spectra:
            plt.figure(figsize=(7, 5));
            plt.semilogy(range(1, len(spectra["FP32"]) + 1), spectra["FP32"], label="FP32")
            plt.semilogy(range(1, len(spectra["INT4-dynamic"]) + 1), spectra["INT4-dynamic"], label="INT4 dynamic")
            plt.xlabel("singular-value index"); plt.ylabel("singular value"); plt.legend(); plt.tight_layout()
            plt.savefig(output / "fp32_vs_int4_singular_spectra.png", dpi=140); plt.close()
    # Conditioning intervention: one curve per selected tensor, with INT8 as a control.
    for metric, filename, ylabel in (("update_cosine", "conditioning_tau_vs_update_cosine.png", "post-Muon update cosine"),
                                     ("update_relative_l2", "conditioning_tau_vs_update_relative_l2.png", "post-Muon relative L2")):
        plt.figure(figsize=(7, 5))
        for key in sorted({(row["seed"], row["update"], row["parameter_id"]) for row in intervention_rows}):
            for qname, style in (("int4-dynamic-b2048", "-"), ("int8-linear-b2048", "--")):
                rows = [row for row in intervention_rows if (row["seed"], row["update"], row["parameter_id"]) == key and row["quantizer"] == qname]
                rows.sort(key=lambda r: r["tau_ratio"])
                if rows:
                    plt.plot([r["tau_ratio"] for r in rows], [r[metric] for r in rows], style, marker=".", alpha=.7,
                             label=f"s{key[0]} u{key[1]} {qname.split('-')[0]}" if metric == "update_cosine" else None)
        plt.xlabel("tau / sigma_max"); plt.ylabel(ylabel)
        if metric == "update_cosine": plt.legend(fontsize=6, ncol=2)
        plt.tight_layout(); plt.savefig(output / filename, dpi=140); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/muon_spectral_sensitivity")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    paths = discover_snapshots(args.reports_root)
    expected = len(SEEDS) * len(LANDMARKS)
    if len(paths) != expected:
        raise SystemExit(f"expected {expected} formal snapshots, found {len(paths)}: {paths}")
    args.output.mkdir(parents=True, exist_ok=True)

    # Pass 1 determines all conditioning/update features and the predeclared
    # intervention subset without retaining matrices or quantized decompositions.
    source_rows: list[dict] = []
    nearest_rows: list[dict] = []
    for seed, update, path in paths:
        print(f"pass1 seed={seed} update={update}", flush=True)
        snapshot = load_snapshot(path)
        for item in snapshot["tensors"]:
            if len(item["shape"]) != 2:
                continue
            row_id = _id(item, seed, update)
            source = decompose(item["tensor"])
            row = row_id | spectral_metrics(source.singular_values)
            source_rows.append(row)
            nearest = quantize(source.matrix, "int4-dynamic-b2048")
            update_row = update_metrics(source.matrix, nearest, transform_kwargs=_transform_kwargs(snapshot["metadata"]))
            nearest_rows.append(row_id | {"update_direction_error": 1 - update_row.get("update_cosine")
                                          if _finite(update_row.get("update_cosine")) else None})
        del snapshot
    selected, selection_reasons = _selection(source_rows, nearest_rows)

    quant_rows: list[dict] = []
    decomposition_rows: list[dict] = []
    subspace_rows: list[dict] = []
    spectra: dict[str, list[float]] = {}
    intervention_sources: dict[tuple, tuple[object, dict]] = {}

    # Pass 2 computes all quantized spectra and retains only the five selected
    # source decompositions for the controlled tau intervention.
    for seed, update, path in paths:
        print(f"pass2 seed={seed} update={update}", flush=True)
        snapshot = load_snapshot(path)
        kwargs = _transform_kwargs(snapshot["metadata"])
        for item in snapshot["tensors"]:
            if len(item["shape"]) != 2:
                continue
            row_id = _id(item, seed, update)
            key = (seed, update, row_id["parameter_id"])
            source = decompose(item["tensor"])
            if key in selected:
                intervention_sources[key] = (source, kwargs)
            for quantizer in QUANTIZERS:
                quantized = quantize(source.matrix, quantizer)
                spectral = quantized_spectral_metrics(source, quantized)
                raw = _ratios(source.matrix, quantized, "raw")
                update_metrics_row = update_metrics(source.matrix, quantized, transform_kwargs=kwargs)
                source_metrics = spectral["source_metrics"]
                quantized_metrics = spectral["quantized_metrics"]
                decomp = spectral_error_decomposition(source, quantized)
                subspaces = {}
                for band, sl in band_slices(source.singular_values.numel()).items():
                    subspaces[band] = {
                        "left": subspace_metrics(source.u[:, sl], spectral["quantized"].u[:, sl]),
                        "right": subspace_metrics(source.vh.T[:, sl], spectral["quantized"].vh.T[:, sl]),
                    }
                    for side in ("left", "right"):
                        subspace_rows.append(row_id | {"quantizer": quantizer, "band": band, "side": side} | subspaces[band][side])
                tail_left = subspaces["tail"]["left"]["projection_distance_normalized"]
                tail_right = subspaces["tail"]["right"]["projection_distance_normalized"]
                tail_subspace = ((tail_left + tail_right) / 2 if _finite(tail_left) and _finite(tail_right)
                                 else tail_left if _finite(tail_left) else tail_right)
                tail_mode = decomp["tail_relative_mode_perturbation_mean"]
                qrow = row_id | {"quantizer": quantizer,
                                 "bits": PRODUCTION_QUANTIZERS[quantizer][1], "codebook": PRODUCTION_QUANTIZERS[quantizer][2],
                                 "block_size": PRODUCTION_QUANTIZERS[quantizer][3],
                                 "update_direction_error": 1 - update_metrics_row.get("update_cosine")
                                 if _finite(update_metrics_row.get("update_cosine")) else None,
                                 "effective_condition_number": source_metrics["effective_condition_number"],
                                 "effective_rank": source_metrics["effective_rank"],
                                 "log_effective_condition_number": math.log10(source_metrics["effective_condition_number"])
                                 if _finite(source_metrics["effective_condition_number"]) and source_metrics["effective_condition_number"] > 0 else None,
                                 "tail_relative_spectral_perturbation": tail_mode,
                                 "tail_subspace_distortion": tail_subspace}
                qrow.update(_flat_metrics("source", source_metrics))
                qrow.update(_flat_metrics("quantized", quantized_metrics))
                qrow.update({"spectral_norm_error": spectral["spectral_norm_error"],
                             "frobenius_error": spectral["frobenius_error"],
                             "relative_singular_value_l2": spectral["relative_singular_value_l2"],
                             "effective_rank_change": spectral["effective_rank_change"],
                             "condition_number_change": spectral["condition_number_change"]})
                qrow.update({"raw_relative_l2": raw.get("raw_relative_l2"), "raw_cosine": raw.get("raw_cosine"),
                             "raw_norm_ratio": raw.get("raw_norm_ratio"),
                             "update_relative_l2": update_metrics_row.get("update_relative_l2"),
                             "update_cosine": update_metrics_row.get("update_cosine"),
                             "update_norm_ratio": update_metrics_row.get("update_norm_ratio"),
                             "muon_update_relative_l2": update_metrics_row.get("muon_update_relative_l2"),
                             "muon_update_cosine": update_metrics_row.get("muon_update_cosine"),
                             "muon_update_norm_ratio": update_metrics_row.get("muon_update_norm_ratio")})
                quant_rows.append(qrow)
                decomposition_rows.append(row_id | {"quantizer": quantizer} | decomp)
                if key in selected and quantizer == "int4-dynamic-b2048" and not spectra:
                    spectra = {"FP32": source.singular_values.detach().cpu().tolist(),
                               "INT4-dynamic": spectral["quantized"].singular_values.detach().cpu().tolist()}
        del snapshot

    # Per-tensor correlation rows use only source features and exact post-Muon metrics.
    correlation_rows: list[dict] = []
    for quantizer in QUANTIZERS:
        subset = [row for row in quant_rows if row["quantizer"] == quantizer]
        for feature in ("log_effective_condition_number", "tail_relative_spectral_perturbation",
                        "tail_subspace_distortion", "effective_rank"):
            correlation_rows.append({"quantizer": quantizer} | correlation(subset, feature, "update_direction_error"))

    # Apply the controlled intervention only after representative selection.
    intervention_rows: list[dict] = []
    for key in sorted(selected):
        source, kwargs = intervention_sources[key]
        reason = selection_reasons[key]
        for tau_ratio in TAUS:
            for quantizer in ("int4-dynamic-b2048", "int8-linear-b2048"):
                result = conditioning_intervention(source, tau_ratio, quantizer=quantizer, transform_kwargs=kwargs)
                intervention_rows.append({
                    "seed": key[0], "update": key[1], "parameter_id": key[2], "shape": str(list(source.matrix.shape)),
                    "selection_reason": reason, "quantizer": quantizer, "tau_ratio": tau_ratio,
                    "naive_condition_number": result["conditioned"]["naive_condition_number"],
                    "effective_condition_number": result["conditioned"]["effective_condition_number"],
                    "effective_rank": result["conditioned"]["effective_rank"],
                    "raw_relative_l2": result.get("raw_relative_l2"), "raw_cosine": result.get("raw_cosine"),
                    "raw_norm_ratio": result.get("raw_norm_ratio"),
                    "update_relative_l2": result.get("update_relative_l2"), "update_cosine": result.get("update_cosine"),
                    "update_norm_ratio": result.get("update_norm_ratio"),
                })

    output = args.output
    write_csv(output / "tensor_spectral_metrics.csv", source_rows)
    write_csv(output / "quantized_spectral_metrics.csv", quant_rows)
    write_csv(output / "spectral_error_decomposition.csv", decomposition_rows)
    write_csv(output / "subspace_distortion.csv", subspace_rows)
    write_csv(output / "spectral_update_correlations.csv", correlation_rows)
    write_csv(output / "conditioning_intervention.csv", intervention_rows)
    if not args.skip_plots:
        _make_plots(output, source_rows, quant_rows, decomposition_rows, subspace_rows, intervention_rows, spectra)

    runtime = time.perf_counter() - started
    dynamic_rows = [row for row in quant_rows if row["quantizer"] == "int4-dynamic-b2048"]
    strongest = []
    for row in correlation_rows:
        if row["quantizer"] == "int4-dynamic-b2048" and _finite(row.get("spearman")):
            strongest.append(row)
    strongest.sort(key=lambda row: abs(float(row["spearman"])), reverse=True)
    intervention_dynamic = [row for row in intervention_rows if row["quantizer"] == "int4-dynamic-b2048"]
    by_tau = {}
    for row in intervention_dynamic:
        by_tau.setdefault(row["tau_ratio"], []).append(row)
    intervention_summary = [(tau, sum(row["update_cosine"] for row in rows) / len(rows),
                             sum(row["update_relative_l2"] for row in rows) / len(rows),
                             sum(row["effective_condition_number"] for row in rows) / len(rows))
                           for tau, rows in sorted(by_tau.items())]
    methodology = f"""# Muon spectral-sensitivity analysis

This is a read-only offline mechanism study over {len(paths)} existing FP32
Muon snapshots and {len(source_rows)} eligible 2D Muon tensors. No training,
optimizer mutation, or existing report modification was performed. Runtime on
this CPU run was {runtime:.2f} seconds.

## Definitions

For each FP32 matrix M, the reduced SVD is M=U diag(sigma) V^T in FP32.
`effective_rank` counts sigma_i/sigma_max >= {1e-6:g}; `effective_condition_number`
is sigma_max divided by the smallest value meeting that threshold. The naive
condition number is sigma_max/sigma_min and is left empty when sigma_min is
zero. Effective-rank entropy is exp(-sum p log p), p=sigma/sum(sigma), and
stable rank is sum(sigma^2)/sigma_max^2. Percentiles are ordinary singular
value quantiles. Tail energy fractions use the last ceil(10%) and ceil(25%)
index modes.

The exact existing blockwise production quantizers are applied with block size
2048, absmax scale, existing codebooks, clipping, and nearest rounding:
`int8-linear-b2048`, `int4-linear-b2048`, and `int4-dynamic-b2048`. The exact
production `zeropower_newton_schulz` transform is used for post-Muon metrics,
with its steps/coefficients/epsilon read from each snapshot provenance.

For E=Q(M)-M, `E_hat=U^T E V`; diagonal terms are relative to sigma_i when
sigma_i/sigma_max >= {1e-6:g}. Top, centered-middle, and tail bands are each
10% of the singular-index range (at least one mode). Projection/subspace
metrics use principal angles and normalized Frobenius distance between the
left and right singular subspace projectors; this avoids one-to-one vector
comparisons for clustered singular values. Any reduced-basis residual is
reported as `unresolved_error_energy`.

## Controlled intervention

Representatives were selected *before* intervention: lowest, median, and
highest effective condition number plus lowest and highest nearest INT4
dynamic update-direction error, deduplicated. The tau grid is
`tau/sigma_max = {', '.join(f'{x:g}' for x in TAUS)}`. Each M_tau keeps U,V
fixed and replaces sigma_i by max(sigma_i,tau). INT8 linear is a healthy
low-distortion control. This is an oracle/mechanism intervention, not a
deployable quantizer and not tuned against training loss.

## Correlations and interpretation

Pearson/Spearman values in `spectral_update_correlations.csv` are descriptive
associations over tensor/landmark pairs; they do not establish causality or
statistical significance. The strongest dynamic-INT4 Spearman associations
were: {[(r['feature'], r['spearman'], r['sample_count']) for r in strongest[:3]]}.

Intervention dynamic-INT4 means (tau, update cosine, update relative L2,
effective condition) were: {intervention_summary}. Use these controls to
decide whether conditioning changes fidelity rather than inferring from
correlation alone. Non-finite or undefined metrics are empty fields.
"""
    (output / "methodology.md").write_text(methodology)
    summary = f"""# Summary

Analyzed {len(paths)} formal snapshots (seeds 0/1, updates {list(LANDMARKS)})
and {len(source_rows)} eligible 2D tensors. Non-2D snapshot entries were
excluded from SVD and post-Muon analysis because production Muon only
orthogonalizes 2D matrices. Runtime was {runtime:.2f} seconds on CPU.

See `tensor_spectral_metrics.csv`, `quantized_spectral_metrics.csv`,
`spectral_error_decomposition.csv`, `subspace_distortion.csv`,
`spectral_update_correlations.csv`, and `conditioning_intervention.csv`.
Correlations are descriptive only. The controlled tau intervention is the
primary mechanism check; it keeps singular-vector orientation fixed and uses
the unchanged production quantizers and Muon transform.
"""
    (output / "summary.md").write_text(summary)
    print(f"analyzed {len(paths)} snapshots / {len(source_rows)} tensors in {runtime:.2f}s")


if __name__ == "__main__":
    main()
