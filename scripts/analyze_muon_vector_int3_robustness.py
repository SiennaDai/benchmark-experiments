#!/usr/bin/env python3
"""Offline robustness closure for structurally conditioned 2-D Muon INT3 VQ.

Only the already-studied top-k conditioner, BF16 factors, contiguous pairing,
2048-scalar p98 scaling, and deterministic MSE k-means codebooks are evaluated.
This script does not import or modify training paths.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_muon_int3_practical_scale import (  # noqa: E402
    discover, factorized_topk, quant_metrics,
    transform_kwargs,
)
from analyze_muon_vector_int3_residual import (  # noqa: E402
    calibration_sample, eligible_items,
)
from optim import muon_reference  # noqa: E402
from optim.muon_ns_sensitivity import exact_polar  # noqa: E402
from optim.muon_spectral_sensitivity import decompose  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402
from optim.muon_vector_int3 import (  # noqa: E402
    fit_kmeans, normalize_vector_blocks, pair_values, quantize_vectors,
    vector_scales, vector_storage_bits,
)
from optim.muon_vector_int3_robustness import (  # noqa: E402
    align_codebook, codebook_occupancy_stats, pack_indices,
    packed_storage_report, unpack_indices,
)

OUT = ROOT / "reports/muon_vector_int3_robustness"
RANKS = (4, 8)
WORDS = (32, 64, 128)
LANDMARKS = (128, 512, 1024, 2048, 4096)
CAL_TENSORS = (1, 2, 4, 8, 16)
VECTORS_PER_TENSOR = (300, 600, 1200, 2400)
BLOCK_SIZE = 2048


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def numeric_row(row):
    out = {}
    for key, value in row.items():
        if value == "":
            out[key] = None
        else:
            try:
                out[key] = int(value) if value.lstrip("-").isdigit() else float(value)
            except (ValueError, AttributeError):
                out[key] = value
    return out


def occupancy_stats_from_counts(counts):
    """Compute occupancy descriptors directly, without expanding all labels."""
    c = torch.as_tensor(counts, dtype=torch.float64)
    total = int(c.sum().item())
    if total:
        p = c[c > 0] / total
        entropy = float(-(p * p.log2()).sum())
        top = torch.sort(c, descending=True).values
        top1 = float(top[0] / total)
        top5 = float(top[:5].sum() / total)
    else:
        entropy = top1 = top5 = 0.0
    words = int(c.numel())
    return {"total_indices": total, "codewords": words,
            "used_codewords": int((c > 0).sum()),
            "dead_codewords": int((c == 0).sum()),
            "occupancy_entropy_bits": entropy,
            "occupancy_entropy_normalized": entropy / math.log2(words) if words > 1 else 0.0,
            "top1_occupancy_fraction": top1,
            "top5_occupancy_fraction": top5,
            "counts": [int(x) for x in c.tolist()]}


def load_inputs(root: Path):
    paths = discover(root)
    indexed = {}
    for seed, update, path in paths:
        indexed[(int(seed), int(update))] = (path, load_snapshot(path))
    expected = {(s, u) for s in (0, 1) for u in LANDMARKS}
    missing = expected - set(indexed)
    if missing:
        raise FileNotFoundError(f"missing expected formal snapshots: {sorted(missing)}")
    return indexed


def select_training_items(snapshot, count):
    # Same deterministic tensor sampling rule used by the previous VQ study.
    return calibration_sample(eligible_items(snapshot), count)


def collect_calibration_vectors(inputs, seed, rank, tensors_per_landmark, max_vectors):
    """Return per-landmark/per-tensor vectors after the fixed block p98 path."""
    result = {}
    for update in LANDMARKS:
        _, snap = inputs[(seed, update)]
        for item in select_training_items(snap, tensors_per_landmark):
            x = item["tensor"].detach().float()
            d = decompose(x)
            residual = x - (d.u[:, :rank] * d.singular_values[:rank]) @ d.vh[:rank]
            pairs, _, _, _ = pair_values(residual, "contiguous")
            normalized = normalize_vector_blocks(pairs, "p98")
            if max_vectors and normalized.shape[0] > max_vectors:
                ix = torch.linspace(0, normalized.shape[0] - 1, max_vectors).round().long()
                normalized = normalized[ix]
            key = (int(update), str(item.get("parameter_id", item.get("name"))))
            result[key] = normalized.cpu()
    return result


def cache_calibration_population(inputs):
    """Compute each calibration tensor's top-k residual pairs only once.

    The nested tensor-count/vector-count sweeps reuse these cached normalized
    vectors. Sampling each requested count still uses the original evenly
    spaced rules, so this is only a compute optimization.
    """
    cache = {}
    for seed in (0, 1):
        for update in LANDMARKS:
            _, snap = inputs[(seed, update)]
            # The largest calibration design uses 16 tensors/landmark; never
            # decompose the remaining evaluation tensors for codebook fitting.
            items = select_training_items(snap, max(CAL_TENSORS))
            for item in items:
                x = item["tensor"].detach().float()
                d = decompose(x)
                ident = str(item.get("parameter_id", item.get("name")))
                for rank in RANKS:
                    residual = x - (d.u[:, :rank] * d.singular_values[:rank]) @ d.vh[:rank]
                    pairs, _, _, _ = pair_values(residual, "contiguous")
                    normalized = normalize_vector_blocks(pairs, "p98")
                    sampled = {}
                    for count in VECTORS_PER_TENSOR:
                        if normalized.shape[0] > count:
                            ix = torch.linspace(0, normalized.shape[0] - 1,
                                                count).round().long()
                            sampled[count] = normalized[ix].cpu()
                        else:
                            sampled[count] = normalized.cpu()
                    cache[(seed, int(update), ident, rank)] = sampled
            print(f"cached calibration vectors seed={seed} update={update}", flush=True)
    return cache


def fit_calibration_codebook(inputs, seed, rank, words, tensors_per_landmark,
                             max_vectors, *, cached_vectors=None, max_fit_samples=30_000):
    if cached_vectors is None:
        samples = collect_calibration_vectors(inputs, seed, rank, tensors_per_landmark, max_vectors)
    else:
        samples = {}
        for update in LANDMARKS:
            _, snap = inputs[(seed, update)]
            items = select_training_items(snap, tensors_per_landmark)
            for item in items:
                ident = str(item.get("parameter_id", item.get("name")))
                z = cached_vectors[(seed, int(update), ident, rank)][max_vectors]
                samples[(int(update), ident)] = z
    merged = torch.cat(list(samples.values()), dim=0)
    cb, meta = fit_kmeans(merged, words, seed=2026, iterations=24,
                          max_samples=max_fit_samples)
    # Calibration-space distortion and occupancy are computed only on this
    # calibration population; evaluation tensors do not affect fitting.
    nearest = torch.cdist(merged, cb).argmin(dim=1)
    recon = cb[nearest]
    mse = float((merged - recon).square().sum(dim=1).mean())
    occ = codebook_occupancy_stats(nearest, words)
    info = {"calibration_seed": seed, "rank": rank, "codewords": words,
            "tensors_per_landmark": tensors_per_landmark,
            "max_vectors_per_tensor": max_vectors,
            "calibration_tensor_instances": len(samples),
            "calibration_vectors": int(merged.shape[0]),
            "calibration_normalized_vector_mse": mse,
            "calibration_used_codewords": occ["used_codewords"],
            "calibration_dead_codewords": occ["dead_codewords"],
            "calibration_occupancy_entropy_bits": occ["occupancy_entropy_bits"]}
    return cb.float().cpu(), info


def config_key(seed, rank, words, tensors, vectors):
    return f"s{seed}_k{rank}_w{words}_t{tensors}_v{vectors}"


def build_codebooks(inputs, args, out):
    cache_path = out / "calibration_codebooks.pt"
    signature = {"tensors": list(CAL_TENSORS), "vectors": list(VECTORS_PER_TENSOR),
                 "words": list(WORDS), "ranks": list(RANKS), "kmeans_seed": 2026,
                 "kmeans_iterations": 24, "fit_cap": 30000,
                 "calibration_max_vectors": args.calibration_max_vectors,
                 "sampling": "direct-even-grid-from-full-vector-set-v2"}
    if cache_path.exists() and not args.recalibrate:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("signature") == signature:
            print("loaded robustness codebook cache", flush=True)
            return cached["codebooks"], cached["metadata"]
    specs = set()
    for seed in (0, 1):
        for rank in RANKS:
            for words in WORDS:
                specs.add((seed, rank, words, 8, args.calibration_max_vectors))
            for count in CAL_TENSORS:
                specs.add((seed, rank, 64, count, args.calibration_max_vectors))
            for vecs in VECTORS_PER_TENSOR:
                specs.add((seed, rank, 64, 8, vecs))
    codebooks, metadata = {}, {}
    cached_vectors = cache_calibration_population(inputs)
    for seed, rank, words, tensors, vecs in sorted(specs):
        key = config_key(seed, rank, words, tensors, vecs)
        cb, info = fit_calibration_codebook(inputs, seed, rank, words, tensors, vecs,
                                             cached_vectors=cached_vectors)
        codebooks[key] = cb
        metadata[key] = info
        print(f"calibrated {key}: nvec={info['calibration_vectors']} mse={info['calibration_normalized_vector_mse']:.6g}", flush=True)
    torch.save({"signature": signature, "codebooks": codebooks,
                "metadata": metadata}, cache_path)
    write_csv(out / "calibration_manifest.csv", list(metadata.values()))
    return codebooks, metadata


def codebook_specs(args):
    """Map one fitted codebook to all analysis roles in which it participates."""
    specs = {}
    for seed in (0, 1):
        for rank in RANKS:
            for words in WORDS:
                key = config_key(seed, rank, words, 8, args.calibration_max_vectors)
                specs[key] = {"seed": seed, "rank": rank, "words": words,
                              "tensors": 8, "vectors": args.calibration_max_vectors,
                              "groups": ["frontier", "cross_split"]}
            for count in CAL_TENSORS:
                key = config_key(seed, rank, 64, count, args.calibration_max_vectors)
                if key not in specs:
                    specs[key] = {"seed": seed, "rank": rank, "words": 64,
                                  "tensors": count, "vectors": args.calibration_max_vectors,
                                  "groups": []}
                specs[key]["groups"].append("sample_size")
            for vecs in VECTORS_PER_TENSOR:
                key = config_key(seed, rank, 64, 8, vecs)
                if key not in specs:
                    specs[key] = {"seed": seed, "rank": rank, "words": 64,
                                  "tensors": 8, "vectors": vecs, "groups": []}
                specs[key]["groups"].append("vector_samples")
    return specs


def evaluation_plan(eval_seed, specs):
    plan = defaultdict(list)
    for key, spec in specs.items():
        train_seed = spec["seed"]
        default = spec["tensors"] == 8 and spec["vectors"] == 1200
        if default or eval_seed != train_seed:
            plan[spec["rank"]].append(key)
    return plan


def make_references(root):
    scalar_path = root / "muon_int3_practical_scale" / "tensor_level_results.csv"
    structural_path = root / "muon_structural_decomposition_mechanism" / "update_fidelity_by_rank.csv"
    int8_path = root / "muon_structural_storage_pareto" / "factor_precision_fidelity.csv"
    scalar = {}
    for row in read_csv(scalar_path):
        if row.get("method") == "percentile:p=98":
            key = (int(row["seed"]), int(row["update"]), row["parameter_id"], int(row["k"]))
            scalar[key] = row
    int4, int8 = {}, {}
    for row in read_csv(structural_path):
        if row.get("kind") == "structural" and row.get("rank_kind") == "fixed":
            key = (int(row["seed"]), int(row["update"]), row["parameter_id"], int(row["rank"]))
            int4[key] = row
    for row in read_csv(int8_path):
        if row.get("configuration") == "direct_int8":
            key = (int(row["seed"]), int(row["update"]), row["parameter_id"])
            int8[key] = row
    return scalar, int4, int8


def evaluate_snapshot(eval_seed, update, snapshot, plan, specs, codebooks, metadata,
                      refs, polar_ids, occupancy_acc):
    scalar_ref, int4_ref, int8_ref = refs
    rows = []
    items = eligible_items(snapshot)
    kwargs = transform_kwargs(snapshot)
    for item_index, item in enumerate(items):
        x = item["tensor"].detach().float()
        d = decompose(x)
        ref_update = muon_reference.zeropower_newton_schulz(x.clone(), **kwargs)
        ident = str(item.get("parameter_id", item.get("name")))
        for rank in RANKS:
            c_hat = factorized_topk(d.u, d.singular_values, d.vh, rank)
            residual = x - (d.u[:, :rank] * d.singular_values[:rank]) @ d.vh[:rank]
            pairs, _, _, _ = pair_values(residual, "contiguous")
            scales = vector_scales(pairs, "p98", block_size=BLOCK_SIZE)
            target_ids = plan[rank]
            for key in target_ids:
                spec = specs[key]
                cb = codebooks[key]
                if (spec["tensors"] != 8 or spec["vectors"] != 1200) and eval_seed == spec["seed"]:
                    continue
                q_pairs, q_scales, indices = quantize_vectors(
                    pairs, cb, scales=scales, scale_method="p98", block_size=BLOCK_SIZE)
                # Eligible Muon matrices are even-sized in the formal snapshots;
                # retain deterministic handling if a future tensor has one tail value.
                if residual.numel() % 2:
                    q_residual = torch.empty_like(residual).reshape(-1)
                    pidx, sidx, _, _ = pair_values(residual, "contiguous")
                    pair_index, single_index = __import__("optim.muon_vector_int3", fromlist=["pairing_indices"]).pairing_indices(tuple(residual.shape), "contiguous")
                    from optim.muon_vector_int3 import unpair_values
                    q_single = residual.reshape(-1)[single_index]
                    q_residual = unpair_values(q_pairs, q_single, tuple(residual.shape), pair_index, single_index)
                else:
                    from optim.muon_vector_int3 import unpair_values, pairing_indices
                    pair_index, single_index = pairing_indices(tuple(residual.shape), "contiguous")
                    q_residual = unpair_values(q_pairs, torch.empty(0, dtype=q_pairs.dtype), tuple(residual.shape), pair_index, single_index)
                estimate = c_hat + q_residual
                out_update = muon_reference.zeropower_newton_schulz(estimate.clone(), **kwargs)
                storage = vector_storage_bits(tuple(x.shape), codewords=spec["words"],
                                              lowrank_rank=rank, block_size=BLOCK_SIZE)
                base = {"calibration_seed": spec["seed"], "evaluation_seed": eval_seed,
                        "split_role": "in_split" if spec["seed"] == eval_seed else "held_out",
                        "rank": rank, "codewords": spec["words"],
                        "nominal_bits_per_value": math.ceil(math.log2(spec["words"])) / 2,
                        "calibration_tensors_per_landmark": spec["tensors"],
                        "max_vectors_per_tensor": spec["vectors"],
                        "codebook_id": key, "update": update, "parameter_id": ident,
                        "parameter_name": item.get("name", ident), "shape": str(tuple(x.shape)),
                        "numel": x.numel(), "total_bits_unamortized": storage["total_bits_unamortized"],
                        "shared_codebook_bits": storage["shared_codebook_bits"],
                        "fp32_bits": storage["fp32_bits"],
                        "storage_ratio_unamortized": storage["storage_ratio_unamortized"],
                        "codebook_calibration_vector_mse": metadata[key]["calibration_normalized_vector_mse"],
                        **quant_metrics(residual, q_residual, "residual"),
                        **quant_metrics(x, estimate, "full_state_raw"),
                        **quant_metrics(ref_update, out_update, "update")}
                reference_key = (eval_seed, update, ident, rank)
                if reference_key in scalar_ref:
                    s = scalar_ref[reference_key]
                    base["scalar_int3_update_cosine"] = float(s["update_cosine"])
                    base["delta_vs_scalar_int3"] = base["update_cosine"] - float(s["update_cosine"])
                if reference_key in int4_ref:
                    s = int4_ref[reference_key]
                    base["structural_int4_update_cosine"] = float(s["update_cosine"])
                    base["delta_vs_structural_int4"] = base["update_cosine"] - float(s["update_cosine"])
                int8_key = (eval_seed, update, ident)
                if int8_key in int8_ref:
                    base["direct_int8_update_cosine"] = float(int8_ref[int8_key]["update_cosine"])
                    base["delta_vs_direct_int8"] = base["update_cosine"] - base["direct_int8_update_cosine"]
                normed = normalize_vector_blocks(pairs, "p98")
                base["normalized_vector_mse"] = float((normed - cb[indices]).square().sum(1).mean())
                base["packed_index_count"] = int(indices.numel())
                occ_key = (key, eval_seed)
                counts = torch.bincount(indices.detach().cpu(), minlength=spec["words"])
                occupancy_acc[occ_key] = occupancy_acc.get(occ_key, torch.zeros_like(counts)) + counts
                do_polar = eval_seed != spec["seed"] and spec["tensors"] == 8 and spec["vectors"] == 1200 and spec["words"] in WORDS and ident in polar_ids
                if do_polar:
                    base.update(quant_metrics(exact_polar(x), exact_polar(estimate), "exact_polar"))
                rows.append(base)
    return rows


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    output = []
    for group_key, rr in sorted(groups.items(), key=lambda z: tuple(str(x) for x in z[0])):
        cos = [float(r["update_cosine"]) for r in rr]
        sizes = [int(r["numel"]) for r in rr]
        row = dict(zip(keys, group_key)); row.update({"tensor_instances": len(rr),
            "mean_update_cosine": statistics.mean(cos), "median_update_cosine": statistics.median(cos),
            "state_size_weighted_update_cosine": sum(c*s for c,s in zip(cos,sizes))/sum(sizes),
            "p10_update_cosine": sorted(cos)[int(.10*(len(cos)-1))],
            "p25_update_cosine": sorted(cos)[int(.25*(len(cos)-1))],
            "p75_update_cosine": sorted(cos)[int(.75*(len(cos)-1))],
            "p90_update_cosine": sorted(cos)[int(.90*(len(cos)-1))],
            "mean_update_relative_l2": statistics.mean(float(r["update_relative_l2"]) for r in rr),
            "mean_normalized_vector_mse": statistics.mean(float(r["normalized_vector_mse"]) for r in rr),
            "mean_codebook_calibration_vector_mse": statistics.mean(float(r["codebook_calibration_vector_mse"]) for r in rr)})
        pol = [float(r["exact_polar_cosine"]) for r in rr if r.get("exact_polar_cosine") not in (None, "")]
        if pol:
            row["exact_polar_tensor_count"] = len(pol)
            row["mean_exact_polar_cosine"] = statistics.mean(pol)
        output.append(row)
    return output


def write_outputs(out, rows, codebooks, metadata, occupancy_acc, inputs, refs):
    codebook_specs_by_id = {r["codebook_id"]: r for r in rows}
    forward = [r for r in rows if r["calibration_seed"] == 0 and r["evaluation_seed"] == 1 and r["calibration_tensors_per_landmark"] == 8 and r["max_vectors_per_tensor"] == 1200]
    reverse = [r for r in rows if r["calibration_seed"] == 1 and r["evaluation_seed"] == 0 and r["calibration_tensors_per_landmark"] == 8 and r["max_vectors_per_tensor"] == 1200]
    cross = [r for r in rows if r["calibration_tensors_per_landmark"] == 8 and r["max_vectors_per_tensor"] == 1200]
    frontier = [r for r in cross if r["calibration_seed"] != r["evaluation_seed"]]
    sample = [r for r in rows if r["codewords"] == 64 and r["max_vectors_per_tensor"] == 1200 and r["calibration_seed"] != r["evaluation_seed"]]
    vector = [r for r in rows if r["codewords"] == 64 and r["calibration_tensors_per_landmark"] == 8 and r["calibration_seed"] != r["evaluation_seed"]]
    write_csv(out/"forward_split.csv", forward)
    write_csv(out/"reverse_split.csv", reverse)
    cross_summary=aggregate(cross, ["calibration_seed","evaluation_seed","rank","codewords"])
    write_csv(out/"sample_size_results.csv", aggregate(sample,["calibration_seed","evaluation_seed","rank","calibration_tensors_per_landmark"]))
    write_csv(out/"vector_sample_results.csv", aggregate(vector,["calibration_seed","evaluation_seed","rank","max_vectors_per_tensor"]))
    write_csv(out/"full_bit_frontier.csv", aggregate(frontier,["calibration_seed","evaluation_seed","rank","codewords","nominal_bits_per_value"]))

    # Direct per-tensor win rates, calculated only against matched cached rows.
    win_rows=[]
    for (train_seed, eval_seed, rank, words), rr in defaultdict(list, {
        key:[r for r in frontier if (r["calibration_seed"],r["evaluation_seed"],r["rank"],r["codewords"])==key]
        for key in {(r["calibration_seed"],r["evaluation_seed"],r["rank"],r["codewords"]) for r in frontier}
    }).items():
        sv=[float(r["delta_vs_scalar_int3"]) for r in rr if r.get("delta_vs_scalar_int3") is not None]
        iv=[float(r["delta_vs_structural_int4"]) for r in rr if r.get("delta_vs_structural_int4") is not None]
        win_rows.append({"calibration_seed":train_seed,"evaluation_seed":eval_seed,"rank":rank,"codewords":words,
                         "tensor_instances":len(rr),"win_rate_vs_scalar_int3":sum(x>0 for x in sv)/len(sv) if sv else None,
                         "win_rate_vs_structural_int4":sum(x>0 for x in iv)/len(iv) if iv else None,
                         "mean_delta_vs_scalar":statistics.mean(sv) if sv else None,
                         "median_delta_vs_scalar":statistics.median(sv) if sv else None,
                         "worst_decile_delta_vs_scalar":sorted(sv)[int(.1*(len(sv)-1))] if sv else None,
                         "mean_delta_vs_int4":statistics.mean(iv) if iv else None})
    write_csv(out/"tensor_win_rates.csv",win_rows)

    # Paired in-split minus held-out transfer gap.
    transfer=[]
    for seed in (0,1):
        for rank in RANKS:
            for words in WORDS:
                inside=[r for r in cross if r["calibration_seed"]==seed and r["evaluation_seed"]==seed and r["rank"]==rank and r["codewords"]==words]
                outside=[r for r in cross if r["calibration_seed"]==seed and r["evaluation_seed"]==1-seed and r["rank"]==rank and r["codewords"]==words]
                if inside and outside:
                    transfer.append({"calibration_seed":seed,"rank":rank,"codewords":words,
                        "in_split_cosine":statistics.mean(float(r["update_cosine"]) for r in inside),
                        "heldout_cosine":statistics.mean(float(r["update_cosine"]) for r in outside),
                        "transfer_gap":statistics.mean(float(r["update_cosine"]) for r in inside)-statistics.mean(float(r["update_cosine"]) for r in outside),
                        "in_split_mse":statistics.mean(float(r["normalized_vector_mse"]) for r in inside),
                        "heldout_mse":statistics.mean(float(r["normalized_vector_mse"]) for r in outside)})
    write_csv(out/"transfer_gaps.csv",transfer)
    transfer_map={(r["calibration_seed"],r["rank"],r["codewords"]):r["transfer_gap"] for r in transfer}
    for r in cross_summary:
        r["in_split_minus_heldout_transfer_gap"] = transfer_map.get((r["calibration_seed"],r["rank"],r["codewords"]))
    write_csv(out/"cross_split_transfer.csv",cross_summary)

    split_gain_rows=[]
    for rank in RANKS:
        for words in WORDS:
            means={}
            for train,ev,label in ((0,1,"forward"),(1,0,"reverse")):
                q=[r for r in frontier if r["calibration_seed"]==train and r["evaluation_seed"]==ev and r["rank"]==rank and r["codewords"]==words]
                means[label]=statistics.mean(float(r["update_cosine"]) for r in q)
                scalar_mean=statistics.mean(float(r["scalar_int3_update_cosine"]) for r in q if r.get("scalar_int3_update_cosine") is not None)
                means[label+"_scalar"]=scalar_mean
                means[label+"_gain"]=means[label]-scalar_mean
            split_gain_rows.append({"rank":rank,"codewords":words,**means,
                "gain_difference_forward_minus_reverse":means["forward_gain"]-means["reverse_gain"]})
    write_csv(out/"split_gain_comparison.csv",split_gain_rows)

    # Held-out occupancy over all 150 tensor instances, not an entropy coder claim.
    occ_rows=[]
    for (key, eval_seed), counts in sorted(occupancy_acc.items()):
        spec=next(r for r in rows if r["codebook_id"]==key)
        stats=occupancy_stats_from_counts(counts)
        occ_rows.append({"codebook_id":key,"calibration_seed":spec["calibration_seed"],"evaluation_seed":eval_seed,
                         "rank":spec["rank"],"codewords":spec["codewords"],
                         "calibration_tensors_per_landmark":spec["calibration_tensors_per_landmark"],
                         "max_vectors_per_tensor":spec["max_vectors_per_tensor"],
                         **{k:v for k,v in stats.items() if k!="counts"},"counts":json.dumps(counts.tolist())})
    write_csv(out/"codebook_occupancy.csv",occ_rows)

    polar_rows = [r for r in rows if r.get("exact_polar_cosine") not in (None, "")
                  and r["calibration_tensors_per_landmark"] == 8
                  and r["max_vectors_per_tensor"] == 1200
                  and r["calibration_seed"] != r["evaluation_seed"]]
    write_csv(out/"polar_check.csv", polar_rows)

    # Metadata-inclusive storage, shared codebook paid once over each eval set.
    storage_rows=[]
    for (train,ev,rank,words),(rr) in defaultdict(list, {
        key:[r for r in frontier if (r["calibration_seed"],r["evaluation_seed"],r["rank"],r["codewords"])==key]
        for key in {(r["calibration_seed"],r["evaluation_seed"],r["rank"],r["codewords"]) for r in frontier}
    }).items():
        storage=0;fp=0
        for update in LANDMARKS:
            snap_rows=[r for r in rr if int(r["update"])==update]
            if not snap_rows: continue
            storage += sum(int(r["total_bits_unamortized"]) for r in snap_rows)-(len(snap_rows)-1)*int(snap_rows[0]["shared_codebook_bits"])
            fp += sum(int(r["fp32_bits"]) for r in snap_rows)
        bits=math.ceil(math.log2(words))
        payload_count=sum(int(r["packed_index_count"]) for r in rr)
        packed=sum(packed_storage_report(int(r["packed_index_count"]),bits)["packed_bytes"] for r in rr)
        parts=[vector_storage_bits(tuple(map(int, ast.literal_eval(r["shape"]))),
                                   codewords=words,lowrank_rank=rank,block_size=BLOCK_SIZE)
               for r in rr]
        storage_rows.append({"calibration_seed":train,"evaluation_seed":ev,"rank":rank,"codewords":words,
                             "residual_bits_per_value":bits/2,"tensor_instances":len(rr),
                             "total_metadata_inclusive_bits":storage,"fp32_bits":fp,
                             "storage_ratio_vs_fp32":storage/fp,"compression_ratio_vs_fp32":fp/storage,
                             "shared_codebook_bits":int(rr[0]["shared_codebook_bits"]),
                             "snapshot_count":len({int(r["update"]) for r in rr}),
                             "residual_index_bits":sum(int(p["pair_index_bits"]) for p in parts),
                             "scale_metadata_bits":sum(int(p["scale_bits"]) for p in parts),
                             "lowrank_bf16_factor_bits":sum(int(p["factor_bits"]) for p in parts),
                             "fixed_representation_metadata_bits":sum(int(p["metadata_bits"])-int(p["scale_bits"]) for p in parts),
                             "amortized_global_codebook_bits":int(rr[0]["shared_codebook_bits"])*len({int(r["update"]) for r in rr}),
                             "packed_index_count":payload_count,"realized_packed_index_bytes_per_tensor_sum":packed,
                             "theoretical_index_bits":payload_count*bits})
    write_csv(out/"storage_summary.csv",storage_rows)

    # Codebook permutation matching for each rank at the frozen default budget.
    stability=[]; aligned_plot=[]
    for rank in RANKS:
        a=codebooks[config_key(0,rank,64,8,1200)]
        b=codebooks[config_key(1,rank,64,8,1200)]
        aligned, permutation=align_codebook(a,b)
        displacement=(a-aligned).norm(dim=1)
        an=a[1:]; bn=aligned[1:]
        ra=an.norm(dim=1); rb=bn.norm(dim=1)
        aa=torch.atan2(an[:,1],an[:,0]); ab=torch.atan2(bn[:,1],bn[:,0])
        cov_a=torch.cov(a.T); cov_b=torch.cov(b.T)
        stability.append({"rank":rank,"codewords":64,"aligned_codewords":len(permutation),
            "mean_codeword_displacement":float(displacement.mean()),"median_codeword_displacement":float(displacement.median()),
            "max_codeword_displacement":float(displacement.max()),
            "cov_seed0_00":float(cov_a[0,0]),"cov_seed0_01":float(cov_a[0,1]),"cov_seed0_11":float(cov_a[1,1]),
            "cov_seed1_00":float(cov_b[0,0]),"cov_seed1_01":float(cov_b[0,1]),"cov_seed1_11":float(cov_b[1,1]),
            "mean_radius_seed0":float(ra.mean()),"mean_radius_seed1":float(rb.mean()),
            "median_radius_seed0":float(ra.median()),"median_radius_seed1":float(rb.median()),
            "mean_angle_resultant_seed0":float(torch.sqrt(aa.cos().mean()**2+aa.sin().mean()**2)),
            "mean_angle_resultant_seed1":float(torch.sqrt(ab.cos().mean()**2+ab.sin().mean()**2)),
            "codebook_covariance_frobenius_difference":float((cov_a-cov_b).norm())})
        for i in range(a.shape[0]):
            va=a[i]; vb=aligned[i]
            aligned_plot.append({"rank":rank,"reference_seed":0,"candidate_seed":1,
                                 "codeword":i,
                                 "reference_x":float(va[0]),"reference_y":float(va[1]),
                                 "aligned_x":float(vb[0]),"aligned_y":float(vb[1]),
                                 "reference_radius":float(va.norm()),"aligned_radius":float(vb.norm()),
                                 "reference_angle_radians":float(torch.atan2(va[1],va[0])),
                                 "aligned_angle_radians":float(torch.atan2(vb[1],vb[0])),
                                 "displacement":float(displacement[i])})
    write_csv(out/"codebook_stability.csv",stability)
    write_csv(out/"codebook_alignment.csv",aligned_plot)

    # Metadata for held-out calibration/reference methods.
    cal_rows=[]
    cache_meta={}
    for r in read_csv(out/"calibration_manifest.csv"):
        cache_meta[(int(r["calibration_seed"]),int(r["rank"]),int(r["codewords"]),int(r["tensors_per_landmark"]),int(r["max_vectors_per_tensor"]))]=r
    for row in rows:
        ident=(row["calibration_seed"],row["rank"],row["codewords"],row["calibration_tensors_per_landmark"],row["max_vectors_per_tensor"])
        if ident in cache_meta:
            row["codebook_calibration_mse"] = float(cache_meta[ident]["calibration_normalized_vector_mse"])
    # Per-codebook evaluation distribution summary, plus cached references.
    reference_rows=[]
    scalar, int4, int8=refs
    scalar_storage={}
    scalar_storage_path=Path("reports/muon_int3_practical_scale")/"storage_summary.csv"
    for rr in read_csv(scalar_storage_path):
        key=(int(rr["seed"]),int(rr["update"]),rr["parameter_id"],int(rr["rank"]))
        scalar_storage[key]=float(rr["storage_ratio_vs_fp32"])
    precision_rows=read_csv(Path("reports/muon_structural_storage_pareto")/"factor_precision_fidelity.csv")
    structural_storage={}
    for rr in precision_rows:
        if rr.get("configuration") in {"fixed_4_bf16","fixed_8_bf16"}:
            structural_storage[(int(rr["seed"]),int(rr["update"]),rr["parameter_id"],int(rr["rank"]))]=float(rr["storage_ratio_vs_fp32"])
    int8_storage={}
    for rr in precision_rows:
        if rr.get("configuration")=="direct_int8":
            int8_storage[(int(rr["seed"]),int(rr["update"]),rr["parameter_id"])]=float(rr["storage_ratio_vs_fp32"])
    for seed in (0,1):
        for rank in RANKS:
            scalar_values=[]; int4_values=[]; int8_values=[]
            scalar_storage_values=[]; int4_storage_values=[]; int8_storage_values=[]
            for update in LANDMARKS:
                snap=inputs[(seed,update)][1]
                for item in eligible_items(snap):
                    ident=str(item.get("parameter_id",item.get("name")))
                    key=(seed,update,ident,rank)
                    if key in scalar:
                        scalar_values.append(float(scalar[key]["update_cosine"]))
                        if key in scalar_storage:
                            scalar_storage_values.append(scalar_storage[key])
                        elif scalar[key].get("storage_ratio_vs_fp32_unamortized"):
                            scalar_storage_values.append(float(scalar[key]["storage_ratio_vs_fp32_unamortized"]))
                    if key in int4:
                        int4_values.append(float(int4[key]["update_cosine"]))
                        if key in structural_storage: int4_storage_values.append(structural_storage[key])
                    key8=(seed,update,ident)
                    if key8 in int8:
                        int8_values.append(float(int8[key8]["update_cosine"]))
                        if key8 in int8_storage: int8_storage_values.append(int8_storage[key8])
            reference_rows.extend([
                {"evaluation_seed":seed,"rank":rank,"reference":"best_practical_scalar_INT3_p98","tensor_instances":len(scalar_values),"mean_update_cosine":statistics.mean(scalar_values) if scalar_values else None,"mean_storage_ratio_vs_fp32":statistics.mean(scalar_storage_values) if scalar_storage_values else None},
                {"evaluation_seed":seed,"rank":rank,"reference":"structural_INT4","tensor_instances":len(int4_values),"mean_update_cosine":statistics.mean(int4_values) if int4_values else None,"mean_storage_ratio_vs_fp32":statistics.mean(int4_storage_values) if int4_storage_values else None},
                {"evaluation_seed":seed,"rank":rank,"reference":"direct_INT8","tensor_instances":len(int8_values),"mean_update_cosine":statistics.mean(int8_values) if int8_values else None,"mean_storage_ratio_vs_fp32":statistics.mean(int8_storage_values) if int8_storage_values else None},
            ])
    write_csv(out/"reference_comparison.csv",reference_rows)

    # Auditable reproduction record against the immediately preceding VQ study.
    old_cache_path=Path("reports/muon_vector_int3_residual/calibration_codebooks.pt")
    old_summary_path=Path("reports/muon_vector_int3_residual/heldout_summary.csv")
    reproduction=[]
    if old_cache_path.exists():
        old_cache=torch.load(old_cache_path,map_location="cpu",weights_only=False)["codebooks"]
        for rank in RANKS:
            for words in WORDS:
                old_cb=old_cache[(rank,"mse",words)].float()
                new_cb=codebooks[config_key(0,rank,words,8,1200)].float()
                reproduction.append({"check":"forward_codebook_vs_prior_cache","rank":rank,"codewords":words,
                    "max_abs_codeword_difference":float((old_cb-new_cb).abs().max()),
                    "bitwise_tensor_equal":torch.equal(old_cb,new_cb)})
    if old_summary_path.exists():
        old_rows=read_csv(old_summary_path)
        for rank in RANKS:
            match=[r for r in old_rows if r.get("seed")=="1" and r.get("role")=="held_out"
                   and int(r.get("rank",-1))==rank and r.get("method")=="mse64_contiguous"]
            current=[r for r in frontier if int(r["calibration_seed"])==0 and int(r["evaluation_seed"])==1
                     and int(r["rank"])==rank and int(r["codewords"])==64]
            if match and current:
                old_mean=float(match[0]["mean_update_cosine"])
                new_mean=statistics.mean(float(r["update_cosine"]) for r in current)
                reproduction.append({"check":"forward_heldout_mean_vs_prior_report","rank":rank,"codewords":64,
                    "prior_mean_update_cosine":old_mean,"current_mean_update_cosine":new_mean,
                    "absolute_difference":abs(old_mean-new_mean),"within_0.001":abs(old_mean-new_mean)<=.001})
    write_csv(out/"forward_reproduction_check.csv",reproduction)

    # Fixed-size synthetic pack/unpack sanity plus actual per-tensor byte padding.
    packing=[]
    for bits, words in ((5,32),(6,64),(7,128)):
        count=1009
        gen=torch.Generator(device="cpu").manual_seed(2026+bits)
        idx=torch.randint(0,words,(count,),generator=gen)
        packed=pack_indices(idx,bits); decoded=unpack_indices(packed,count,bits)
        report=packed_storage_report(count,bits)
        packing.append({"codewords":words,"bits_per_index":bits,"synthetic_count":count,
                        "roundtrip_exact":torch.equal(idx,decoded),**report})
    write_csv(out/"packing_sanity.csv",packing)

    (out/"methodology.md").write_text("""# Methodology

## Frozen representation

Every eligible formal 2D Muon momentum tensor is decomposed using its FP32 truncated SVD. The rank-4 or rank-8 top-k component is stored as BF16 factors using the same `factorized_topk` routine as the prior structural/VQ study. The residual is paired contiguously (row-major adjacent scalars), grouped in blocks of 2048 scalar values (1024 pairs), normalized by the within-block 98th percentile of absolute scalar values, and clipped coordinatewise to `[-1,1]`. A learned shared 2D Euclidean-MSE codebook is then applied by deterministic nearest-neighbor assignment. Codebooks reserve exact zero and use seeded k-means++ initialization followed by at most 24 Lloyd updates, seed 2026, and at most 30,000 calibration vectors.

## Splits and sampling

The primary forward split fits on seed 0 and evaluates on seed 1; the reverse split fits on seed 1 and evaluates on seed 0. Cross-split transfer additionally evaluates each trained codebook on its own calibration seed, explicitly labeled in-split and never used as primary evidence. For each landmark, the tensor subset is selected by endpoint-inclusive evenly spaced indices rounded to nearest integers, matching the previous VQ experiment. The default is 8 tensors per landmark and at most 1200 vectors per selected tensor. Calibration-size sweeps use 1/2/4/8/16 tensors per landmark; vector-count sweeps use 300/600/1200/2400 vectors per tensor. Codebook fitting is isolated to the declared calibration seed.

## Fidelity and storage

The reconstructed state is BF16-factorized top-k plus the dequantized residual. K=5 readouts call the exact production `zeropower_newton_schulz` implementation with transform parameters loaded from each snapshot. Exact-polar diagnostics use the existing analysis utility and a deterministic identity subset formed by the union of the first four eligible tensor identities at each of the ten snapshots; only held-out default codebooks are measured, and the resulting tensor-update count is recorded in `polar_check.csv`. Aggregate cosine statistics include arithmetic mean, median, unweighted quantiles, and state-size-weighted mean. Tensor-instance observations repeat parameters across five landmarks and are therefore descriptive, not independent statistical samples.

Storage uses the existing `vector_storage_bits` accounting: fixed-width residual indices (5/6/7 bits per pair for 32/64/128 words), one FP32 p98 scalar scale per 2048 residual values, BF16 low-rank factors, fixed metadata, and a shared FP32 2D codebook. In the aggregate over five landmarks, the shared codebook is charged once per snapshot state (once across its 30 eligible tensors), not once per tensor. A separate bitstream check packs indices little-endian and verifies exact round-trip; this is not a kernel or runtime benchmark. Occupancy summaries describe codeword use and are not entropy-coded storage estimates.

## Robustness quantities and limits

Transfer gap is the in-split mean update cosine minus held-out mean for a fixed trained codebook/rank/size. Tensor win rates compare matched tensor instances with cached scalar INT3 and structural INT4 references. Codebook stability aligns codeword permutations with minimum total squared Euclidean distance before displacement and distribution summaries. The two-seed experiment tests transfer across these trajectories only; it does not establish broad population generalization. CPU runtime and offline factor extraction are not estimates of training-time GPU cost.
""")

    # Summary and plots are generated by a compact post-processing helper.
    write_summary_and_plots(out, rows, forward, reverse, frontier, sample, vector,
                             stability, occ_rows, storage_rows, reference_rows,
                             codebooks, metadata, inputs)


def write_summary_and_plots(out, rows, forward, reverse, frontier, sample, vector,
                            stability, occupancy, storage, refs, codebooks, metadata, inputs):
    groups=aggregate(frontier,["calibration_seed","evaluation_seed","rank","codewords"])
    polar_rows=[r for r in rows if r.get("exact_polar_cosine") not in (None,"")
                and r["calibration_tensors_per_landmark"]==8
                and r["max_vectors_per_tensor"]==1200
                and r["calibration_seed"]!=r["evaluation_seed"]]
    split_gain_rows=[]
    for rank in RANKS:
        for words in WORDS:
            values={}
            for train,ev,label in ((0,1,"forward"),(1,0,"reverse")):
                rr=[r for r in frontier if r["calibration_seed"]==train and r["evaluation_seed"]==ev and r["rank"]==rank and r["codewords"]==words]
                values[label]=statistics.mean(float(r["update_cosine"]) for r in rr)
                scalar_values=[float(r["scalar_int3_update_cosine"]) for r in rr if r.get("scalar_int3_update_cosine") is not None]
                values[label+"_gain"]=values[label]-statistics.mean(scalar_values)
            split_gain_rows.append({"rank":rank,"codewords":words,**values,
                "gain_difference_forward_minus_reverse":values["forward_gain"]-values["reverse_gain"]})
    held0=[r for r in groups if r["calibration_seed"]==0 and r["evaluation_seed"]==1]
    held1=[r for r in groups if r["calibration_seed"]==1 and r["evaluation_seed"]==0]
    def find(group, train, ev, rank, words):
        return next(r for r in group if r["calibration_seed"]==train and r["evaluation_seed"]==ev and r["rank"]==rank and r["codewords"]==words)
    # Match prior scalar-vs-VQ forward run; failure is a protocol validation stop.
    prior={(4,64):.8258,(8,64):.8516}
    for rank in RANKS:
        got=find(groups,0,1,rank,64)["mean_update_cosine"]
        if abs(got-prior[(rank,64)]) > .001:
            raise RuntimeError(f"forward reproduction failed for k={rank}: got {got:.6f}, expected near {prior[(rank,64)]:.4f}")
    with (out/"summary.md").open("w") as f:
        f.write("# Structurally conditioned vector INT3 robustness closure\n\n")
        f.write("CPU-only offline matched-state study. Fixed method: ordinary top-k SVD, BF16 low-rank factors, contiguous pairing, 2048-scalar p98 normalization with clipping to [-1,1], deterministic 2D Euclidean MSE k-means (seed 2026). Calibration/evaluation data are separated by trajectory seed. No training or production path was changed.\n\n")
        f.write("## Full held-out bit frontier\n\n| calibration → eval | k | words | bits/value | n | mean cosine | size-weighted | median [p25,p75] | mean update rel-L2 | exact-polar subset cosine | storage/FP32 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for train,ev,role in ((0,1,"forward held-out"),(1,0,"reverse held-out")):
            for rank in RANKS:
                for words in WORDS:
                    r=find(groups,train,ev,rank,words)
                    sr=next(x for x in storage if x["calibration_seed"]==train and x["evaluation_seed"]==ev and x["rank"]==rank and x["codewords"]==words)
                    pc=r.get("mean_exact_polar_cosine")
                    pc_text=f"{pc:.5f}" if pc is not None else "subset n/a"
                    f.write(f"| {role} | {rank} | {words} | {math.ceil(math.log2(words))/2:.1f} | {r['tensor_instances']} | {r['mean_update_cosine']:.5f} | {r['state_size_weighted_update_cosine']:.5f} | {r['median_update_cosine']:.5f} [{r['p25_update_cosine']:.5f},{r['p75_update_cosine']:.5f}] | {r['mean_update_relative_l2']:.5f} | {pc_text} | {sr['storage_ratio_vs_fp32']:.5f} |\n")
        f.write("\n## Decision interpretation\n\n")
        for rank in RANKS:
            a=find(groups,0,1,rank,64); b=find(groups,1,0,rank,64)
            scalar_ref=next(x for x in refs if x["evaluation_seed"]==1 and x["rank"]==rank and x["reference"]=="best_practical_scalar_INT3_p98")
            scalar_rev=next(x for x in refs if x["evaluation_seed"]==0 and x["rank"]==rank and x["reference"]=="best_practical_scalar_INT3_p98")
            int4_f=next(x for x in refs if x["evaluation_seed"]==1 and x["rank"]==rank and x["reference"]=="structural_INT4")
            int4_r=next(x for x in refs if x["evaluation_seed"]==0 and x["rank"]==rank and x["reference"]=="structural_INT4")
            f.write(f"- k={rank}: forward 64-word cosine {a['mean_update_cosine']:.4f}, gain over scalar {a['mean_update_cosine']-scalar_ref['mean_update_cosine']:+.4f}, gap to structural INT4 {a['mean_update_cosine']-int4_f['mean_update_cosine']:+.4f}; reverse cosine {b['mean_update_cosine']:.4f}, gain over scalar {b['mean_update_cosine']-scalar_rev['mean_update_cosine']:+.4f}, gap to structural INT4 {b['mean_update_cosine']-int4_r['mean_update_cosine']:+.4f}. Held-out counts are {a['tensor_instances']} and {b['tensor_instances']} respectively.\n")
        primary=[]
        for train,ev in ((0,1),(1,0)):
            for rank in RANKS:
                v=find(groups,train,ev,rank,64)
                scalar_ref=next(x for x in refs if x["evaluation_seed"]==ev and x["rank"]==rank and x["reference"]=="best_practical_scalar_INT3_p98")
                primary.append((v["mean_update_cosine"]-scalar_ref["mean_update_cosine"],v["mean_update_cosine"],rank,train,ev))
        bidirectional_core_gate=all(gain>=.04 for gain,_,_,_,_ in primary) and all(cos>=.84 for _,cos,rank,_,_ in primary if rank==8)
        f.write("\nBidirectional 64-word criterion (both ranks gain at least +0.04 over scalar and k=8 reaches 0.84 in each held-out direction): **" + ("passes" if bidirectional_core_gate else "does not pass") + ".**\n\n")
        f.write("The scale/fidelity plot and `storage_summary.csv` use the same codebook/representation metadata accounting for both split directions; the 64-word point is nominally 3 bits per residual scalar and includes BF16 factors, p98 scales, metadata, and the global codebook amortized once per snapshot state. `packing_sanity.csv` separately verifies exact 5/6/7-bit index packing and realized byte padding.\n\n")
        f.write("\nThis supports robustness only if both split directions retain the predeclared gain and k=8 fidelity gate. Codebook sample-size/vector-count sweeps, codebook transfer, full held-out 32/64/128 frontiers, and packed-index sanity are reported in the adjacent CSVs. Repeated tensors across landmarks are correlated; summaries are descriptive, not significance tests. Calibration/codebook MSE is not a deployment objective.\n\n")
        f.write("## Split-gain comparison\n\n| k | words | forward cosine | forward vs scalar | reverse cosine | reverse vs scalar | gain difference (F−R) |\n|---:|---:|---:|---:|---:|---:|---:|\n")
        for q in split_gain_rows:
            f.write(f"| {q['rank']} | {q['codewords']} | {q['forward']:.5f} | {q['forward_gain']:+.5f} | {q['reverse']:.5f} | {q['reverse_gain']:+.5f} | {q['gain_difference_forward_minus_reverse']:+.5f} |\n")
        f.write("\n## Calibration data sensitivity\n\n| k | calibration seed | tensors/landmark | held-out cosine | held-out vector MSE | calibration vector MSE |\n|---:|---:|---:|---:|---:|---:|\n")
        sample_stats=aggregate(sample,["calibration_seed","evaluation_seed","rank","calibration_tensors_per_landmark"])
        for q in sample_stats:
            f.write(f"| {q['rank']} | {q['calibration_seed']} | {q['calibration_tensors_per_landmark']} | {q['mean_update_cosine']:.5f} | {q['mean_normalized_vector_mse']:.6f} | {q['mean_codebook_calibration_vector_mse']:.6f} |\n")
        f.write("\n## Vector-count sensitivity\n\n| k | calibration seed | vectors/tensor | held-out cosine | held-out vector MSE | calibration vector MSE |\n|---:|---:|---:|---:|---:|---:|\n")
        vector_stats=aggregate(vector,["calibration_seed","evaluation_seed","rank","max_vectors_per_tensor"])
        for q in vector_stats:
            f.write(f"| {q['rank']} | {q['calibration_seed']} | {q['max_vectors_per_tensor']} | {q['mean_update_cosine']:.5f} | {q['mean_normalized_vector_mse']:.6f} | {q['mean_codebook_calibration_vector_mse']:.6f} |\n")
        f.write("\n## References and codebook diagnostics\n\nCached scalar INT3, structural INT4, and direct INT8 means are in `reference_comparison.csv`; they are included only where tensor keys/protocols match. `codebook_stability.csv` reports permutation-aligned displacement and coarse radial/angular/covariance summaries; `codebook_alignment.csv` retains per-word coordinates and polar coordinates; `codebook_occupancy.csv` reports held-out counts, used/dead words, entropy-like occupancy, and top shares (not entropy coding).\n\n")
        f.write(f"Full evaluated rows: {len(rows)}. Default frontier held-out rows per direction/rank/codeword count: 150. CPU analysis runtime for this invocation is recorded separately by the runner. Exact polar was evaluated only on the deterministic subset encoded in `polar_check.csv`.\n")
    # Plots
    try:
        import matplotlib.pyplot as plt
        def splitplot():
            fig,ax=plt.subplots(figsize=(8,5))
            for train,ev,label in ((0,1,"seed0→seed1"),(1,0,"seed1→seed0")):
                rr=[r for r in groups if r["calibration_seed"]==train and r["evaluation_seed"]==ev and r["rank"]==8]
                rr=sorted(rr,key=lambda x:x["codewords"])
                ax.plot([math.ceil(math.log2(r["codewords"]))/2 for r in rr],[r["mean_update_cosine"] for r in rr],marker="o",label=label)
            ax.set_xlabel("residual bits/value");ax.set_ylabel("held-out K=5 update cosine");ax.legend();fig.tight_layout();fig.savefig(out/"forward_reverse_frontier.png",dpi=150);plt.close(fig)
        splitplot()
        fig,ax=plt.subplots(figsize=(8,5))
        for train,ev,label in ((0,1,"seed0→seed1"),(1,0,"seed1→seed0")):
            for rank,marker in ((4,"o"),(8,"s")):
                rr=sorted([r for r in groups if r["calibration_seed"]==train and r["evaluation_seed"]==ev and r["rank"]==rank],key=lambda x:x["codewords"])
                ax.plot([r["codewords"] for r in rr],[r["mean_update_cosine"] for r in rr],marker=marker,label=f"{label}, k={rank}")
        ax.set_xscale("log",base=2);ax.set_xticks(WORDS);ax.set_xticklabels([str(w) for w in WORDS]);ax.set_xlabel("codewords (5/6/7 bits per vector index)");ax.set_ylabel("held-out K=5 update cosine");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"full_bit_frontier_both_ranks.png",dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5))
        for train,ev,label in ((0,1,"forward"),(1,0,"reverse")):
            rr=[r for r in frontier if r["calibration_seed"]==train and r["evaluation_seed"]==ev and r["rank"] in RANKS]
            vals=[float(r["delta_vs_scalar_int3"]) for r in rr if r.get("delta_vs_scalar_int3") is not None]
            ax.hist(vals,bins=24,alpha=.5,label=label)
        ax.axvline(0,color="black",lw=1);ax.set_xlabel("tensor-level VQ − scalar INT3 update cosine");ax.set_ylabel("tensor instances");ax.legend();fig.tight_layout();fig.savefig(out/"tensor_gain_histogram.png",dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
        for ax,rank in zip(axes,RANKS):
            for train,ev,label in ((0,1,"forward"),(1,0,"reverse")):
                rr=[r for r in frontier if r["calibration_seed"]==train and r["evaluation_seed"]==ev and r["rank"]==rank and r["codewords"]==64]
                vals=[float(r["delta_vs_scalar_int3"]) for r in rr if r.get("delta_vs_scalar_int3") is not None]
                ax.hist(vals,bins=20,alpha=.5,label=label)
            ax.axvline(0,color="black",lw=1);ax.set_title(f"k={rank}, 64 words");ax.set_xlabel("update-cosine gain vs scalar INT3")
        axes[0].set_ylabel("tensor instances");axes[1].legend();fig.tight_layout();fig.savefig(out/"64word_tensor_gain_by_rank.png",dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5))
        refs_plot=refs
        for rank in RANKS:
            q=[r for r in refs_plot if r["evaluation_seed"]==1 and r["rank"]==rank]
            by={r["reference"]:r["mean_update_cosine"] for r in q}
            held=find(groups,0,1,rank,64)
            labels=["scalar INT3", "2D VQ 3.0b/v", "structural INT4", "direct INT8"]
            keys=["best_practical_scalar_INT3_p98", None, "structural_INT4", "direct_INT8"]
            vals=[by.get(keys[0]),held["mean_update_cosine"],by.get(keys[2]),by.get(keys[3])]
            ax.plot(labels,vals,marker="o",label=f"k={rank}")
        ax.set_ylabel("held-out K=5 update cosine");ax.legend();fig.tight_layout();fig.savefig(out/"reference_fidelity_comparison.png",dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5))
        for train,ev,label,marker in ((0,1,"forward","o"),(1,0,"reverse","s")):
            rr=aggregate([r for r in frontier if r["calibration_seed"]==train and r["evaluation_seed"]==ev],
                         ["calibration_seed","evaluation_seed","rank","codewords"])
            storage_for=[r for r in storage if r["calibration_seed"]==train and r["evaluation_seed"]==ev]
            for rank in RANKS:
                q=sorted([r for r in rr if r["rank"]==rank],key=lambda r:r["codewords"])
                for r in q:
                    sr=next(s for s in storage_for if s["rank"]==rank and s["codewords"]==r["codewords"])
                    ax.scatter(sr["storage_ratio_vs_fp32"],r["mean_update_cosine"],marker=marker,label=f"{label}, k={rank}, {r['codewords']}w")
        ax.set_xlabel("metadata-inclusive storage / FP32");ax.set_ylabel("held-out K=5 update cosine");ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/"storage_fidelity_frontier.png",dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5))
        for rank in RANKS:
            for train,ev,label,marker in ((0,1,"forward 2D VQ","o"),(1,0,"reverse 2D VQ","s")):
                q=next(r for r in groups if r["calibration_seed"]==train and r["evaluation_seed"]==ev and r["rank"]==rank and r["codewords"]==64)
                sr=next(s for s in storage if s["calibration_seed"]==train and s["evaluation_seed"]==ev and s["rank"]==rank and s["codewords"]==64)
                ax.scatter(sr["storage_ratio_vs_fp32"],q["mean_update_cosine"],marker=marker,label=f"{label}, k={rank}")
            eval_seed=1
            for ref in refs:
                if ref["evaluation_seed"]==eval_seed and ref["rank"]==rank:
                    ax.scatter(float(ref["mean_storage_ratio_vs_fp32"]),float(ref["mean_update_cosine"]),marker="*",s=100,label=f"{ref['reference']}, k={rank}")
        ax.set_xlabel("metadata-inclusive storage / FP32");ax.set_ylabel("K=5 update cosine");ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/"reference_storage_fidelity_frontier.png",dpi=150);plt.close(fig)
        for rank in RANKS:
            c0=codebooks[config_key(0,rank,64,8,1200)]
            c1,_=align_codebook(c0,codebooks[config_key(1,rank,64,8,1200)])
            fig,axes=plt.subplots(2,2,figsize=(9,7))
            for col,(cb,title) in enumerate(((c0,"seed 0"),(c1,"seed 1 aligned"))):
                nonzero=cb[cb.norm(dim=1)>0]
                axes[0,col].hist(nonzero.norm(dim=1).numpy(),bins=16,alpha=.75)
                axes[0,col].set_title(title+" radius");axes[0,col].set_xlabel("codeword radius")
                axes[1,col].hist(torch.atan2(nonzero[:,1],nonzero[:,0]).numpy(),bins=16,alpha=.75)
                axes[1,col].set_title(title+" angle");axes[1,col].set_xlabel("angle (radians)")
            fig.tight_layout();fig.savefig(out/f"codebook_radial_angular_distribution_k{rank}.png",dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        for r in RANKS:
            rr=[x for x in sample if int(x["rank"])==r]
            g=aggregate(rr,["calibration_seed","evaluation_seed","rank","calibration_tensors_per_landmark"])
            for train,ev in ((0,1),(1,0)):
                x=[a for a in g if a["calibration_seed"]==train and a["evaluation_seed"]==ev]
                x.sort(key=lambda a:a["calibration_tensors_per_landmark"])
                ax.plot([a["calibration_tensors_per_landmark"] for a in x],[a["mean_update_cosine"] for a in x],marker="o",label=f"k={r}, {train}→{ev}")
        ax.set_xlabel("calibration tensors per landmark");ax.set_ylabel("held-out update cosine");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"calibration_sample_size.png",dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        for r in RANKS:
            rr=aggregate([x for x in vector if int(x["rank"])==r],["calibration_seed","evaluation_seed","rank","max_vectors_per_tensor"])
            for train,ev in ((0,1),(1,0)):
                x=sorted([a for a in rr if a["calibration_seed"]==train and a["evaluation_seed"]==ev],key=lambda a:a["max_vectors_per_tensor"])
                ax.plot([a["max_vectors_per_tensor"] for a in x],[a["mean_update_cosine"] for a in x],marker="o",label=f"k={r}, {train}→{ev}")
        ax.set_xlabel("maximum calibration vectors/tensor");ax.set_ylabel("held-out update cosine");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"vector_sample_count.png",dpi=150);plt.close(fig)
        for rank in RANKS:
            a=codebooks[config_key(0,rank,64,8,1200)]; b=codebooks[config_key(1,rank,64,8,1200)]
            aligned,_=align_codebook(a,b)
            fig,ax=plt.subplots(figsize=(6,6));ax.scatter(a[:,0],a[:,1],label="seed 0",s=25);ax.scatter(aligned[:,0],aligned[:,1],marker="x",label="seed 1 aligned",s=25);ax.set_aspect("equal");ax.legend();ax.set_title(f"k={rank} 64-word codebook alignment");fig.tight_layout();fig.savefig(out/f"codebook_alignment_k{rank}.png",dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        for rank in RANKS:
            occ=[r for r in occupancy if r["rank"]==rank and r["calibration_tensors_per_landmark"]==8 and r["max_vectors_per_tensor"]==1200 and r["codewords"]==64]
            if occ:ax.hist([r["occupancy_entropy_normalized"] for r in occ],bins=12,alpha=.5,label=f"k={rank}")
        ax.set_xlabel("normalized codeword occupancy entropy (descriptive)");ax.set_ylabel("eval populations");ax.legend();fig.tight_layout();fig.savefig(out/"occupancy_entropy.png",dpi=150);plt.close(fig)
        polar=polar_rows
        fig,ax=plt.subplots(figsize=(7,5))
        for train,ev,label in ((0,1,"forward"),(1,0,"reverse")):
            rr=aggregate([r for r in polar if r["calibration_seed"]==train and r["evaluation_seed"]==ev],["calibration_seed","evaluation_seed","rank","codewords"])
            for rank in RANKS:
                q=sorted([r for r in rr if r["rank"]==rank],key=lambda x:x["codewords"])
                ax.plot([r["codewords"] for r in q],[r.get("mean_exact_polar_cosine",float("nan")) for r in q],marker="o",label=f"k={rank} {label}")
        ax.set_xlabel("codewords");ax.set_ylabel("subset exact-polar cosine");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"exact_polar_frontier.png",dpi=150);plt.close(fig)
    except Exception as exc:
        print(f"plot generation warning: {exc}", flush=True)


def run(args):
    torch.set_num_threads(args.threads)
    start=time.perf_counter(); wall_start=time.time(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    inputs=load_inputs(Path(args.reports_root))
    codebooks, metadata=build_codebooks(inputs,args,out)
    specs=codebook_specs(args)
    partial=out/"tensor_eval.partial.csv"; occ_cache=out/"occupancy_accumulator.pt"
    rows=[]; completed=set()
    if args.resume and partial.exists():
        rows=[numeric_row(r) for r in read_csv(partial)]
        completed={(int(r["evaluation_seed"]),int(r["update"])) for r in rows}
    occupancy_acc=torch.load(occ_cache,map_location="cpu",weights_only=True) if args.resume and occ_cache.exists() else {}
    refs=make_references(Path(args.reports_root))
    polar_ids=set()
    for seed in (0,1):
        for update in LANDMARKS:
            _,snap=inputs[(seed,update)]
            polar_ids.update(str(x.get("parameter_id",x.get("name")))
                             for x in eligible_items(snap)[:args.polar_tensors_per_update])
    for eval_seed in (0,1):
        plan=evaluation_plan(eval_seed,specs)
        for update in LANDMARKS:
            if (eval_seed,update) in completed:
                print(f"resume skipped eval seed={eval_seed} update={update}",flush=True);continue
            _,snapshot=inputs[(eval_seed,update)]
            batch=evaluate_snapshot(eval_seed,update,snapshot,plan,specs,codebooks,metadata,refs,polar_ids,occupancy_acc)
            rows.extend(batch)
            write_csv(partial,rows)
            torch.save(occupancy_acc,occ_cache)
            print(f"evaluated eval seed={eval_seed} update={update}; rows={len(rows)}",flush=True)
    write_outputs(out,rows,codebooks,metadata,occupancy_acc,inputs,refs)
    elapsed=(time.time()-args.runtime_start_epoch) if args.runtime_start_epoch is not None else (time.time()-wall_start)
    (out/"runtime_seconds.txt").write_text(f"{elapsed:.3f}\n")
    print(f"robustness closure complete: {out}; seconds={elapsed:.1f}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reports-root",default=str(ROOT/"reports"));p.add_argument("--outdir",default=str(OUT))
    p.add_argument("--threads",type=int,default=4);p.add_argument("--calibration-max-vectors",type=int,default=1200)
    p.add_argument("--polar-tensors-per-update",type=int,default=4)
    p.add_argument("--resume",action="store_true");p.add_argument("--recalibrate",action="store_true")
    p.add_argument("--runtime-start-epoch",type=float,default=None,
                   help="optional wall-clock epoch for cumulative runtime across an interrupted/resumed analysis")
    args=p.parse_args();run(args)


if __name__=="__main__":main()
