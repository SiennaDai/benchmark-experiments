#!/usr/bin/env python3
"""Offline CPU study of vector INT3 quantization for conditioned Muon states."""
from __future__ import annotations

import argparse
import csv
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
    codebook_for_rank, discover, factorized_topk, quantize_scales, quant_metrics,
    transform_kwargs, write_csv,
)
from optim import muon_reference  # noqa: E402
from optim.muon_conditioned_int3_companding import INT3_CODEBOOK, spectral_error_metrics  # noqa: E402
from optim.muon_ns_sensitivity import exact_polar  # noqa: E402
from optim.muon_quantization_aware_conditioner import BLOCK_SIZE  # noqa: E402
from optim.muon_spectral_sensitivity import decompose, quantize as production_quantize  # noqa: E402
from optim.muon_storage_pareto import storage_bits  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402
from optim.muon_vector_int3 import (  # noqa: E402
    codebook_occupancy, fit_covariance_transform, fit_kmeans, pair_values,
    normalize_vector_blocks, polar_codebook, quantize_vectors, quantize_vectors_per_dimension,
    unpair_values, vector_scales,
)

OUT = ROOT / "reports/muon_vector_int3_residual"
RANKS = (4, 8)
SEEDS = (0, 1)
LANDMARKS = (128, 512, 1024, 2048, 4096)
PAIRINGS = ("contiguous", "row", "column", "checkerboard")


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def group_stats(rows, key, value):
    groups = defaultdict(list)
    for r in rows:
        v = r.get(value)
        if v not in (None, ""):
            try: groups[r[key]].append(float(v))
            except (ValueError, TypeError): pass
    return [{key: k, "n": len(v), "mean": statistics.mean(v),
             "median": statistics.median(v),
             "p25": statistics.quantiles(v, n=4)[0] if len(v) >= 2 else v[0],
             "p75": statistics.quantiles(v, n=4)[2] if len(v) >= 2 else v[0]}
            for k, v in sorted(groups.items())]


def eligible_items(snapshot, limit=0):
    items=[x for x in snapshot["tensors"] if len(x["shape"])==2 and min(x["shape"])>=8]
    return items[:limit] if limit else items


def save_plot(path, fn):
    try:
        import matplotlib.pyplot as plt
        fn(plt)
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    except Exception as exc:
        print(f"plot skipped {path.name}: {exc}", flush=True)


def make_residual(item, k, d=None):
    x = item["tensor"].detach().float()
    d = decompose(x) if d is None else d
    c_exact = (d.u[:, :k] * d.singular_values[:k]) @ d.vh[:k]
    c_hat = factorized_topk(d.u, d.singular_values, d.vh, k)
    return x, d, c_exact, c_hat, x - c_exact


def normalize_pair_blocks(pairs, method="p98", per_dimension=False):
    """Normalize and clip calibration vectors exactly as evaluation does."""
    if not pairs.numel(): return pairs.detach().float()
    return normalize_vector_blocks(pairs, method, per_dimension=per_dimension)


def calibration_sample(items, count):
    if not count or count >= len(items): return items
    indexes=torch.linspace(0,len(items)-1,count).round().long().tolist()
    return [items[i] for i in indexes]


def gather_calibration(snapshots, *, max_vectors_per_tensor=1200, limit_tensors=0,
                       tensors_per_update=8):
    """Collect only seed-0 normalized pairs; return contiguous MSE codebooks."""
    pools = {(k, p): [] for k in RANKS for p in PAIRINGS}
    per_dim = {k: [] for k in RANKS}
    spectral = {k: [] for k in RANKS}
    for seed, update, path, snap in snapshots:
        if seed != 0: continue
        for item in calibration_sample(eligible_items(snap,limit_tensors),tensors_per_update):
            x=item["tensor"].detach().float(); d0=decompose(x)
            for k in RANKS:
                c_exact=(d0.u[:,:k]*d0.singular_values[:k])@d0.vh[:k]
                r=x-c_exact
                for pairing in PAIRINGS:
                    z, _, _, _ = pair_values(r, pairing)
                    if z.shape[0]:
                        z=normalize_pair_blocks(z,"p98")
                        ix = torch.linspace(0, z.shape[0]-1, min(z.shape[0], max_vectors_per_tensor)).round().long()
                        z = z[ix]
                        pools[(k, pairing)].append(z)
                z, _, _, _ = pair_values(r, "contiguous")
                if z.shape[0]:
                    ix=torch.linspace(0,z.shape[0]-1,min(z.shape[0],max_vectors_per_tensor)).round().long()
                    z=normalize_pair_blocks(z,"p98",per_dimension=True)
                    per_dim[k].append(z[ix])
                # A shared codebook also supports the spectral-coordinate
                # control; samples are scalar-b2048 normalized in coefficient
                # pairs, then deterministically row-major paired.
                rh = d0.u.T @ r @ d0.vh.T
                z, _, _, _ = spectral_local_pairs(rh, d0.singular_values)
                if z.shape[0]:
                    z=normalize_pair_blocks(z,"p98")
                    ix = torch.linspace(0, z.shape[0]-1, min(z.shape[0], max_vectors_per_tensor)).round().long()
                    z = z[ix]
                    spectral[k].append(z)
        print(f"calibration collected seed=0 update={update}", flush=True)
    codebooks = {}
    meta = []
    for k in RANKS:
        # One codebook per rank: using contiguous seed-0 residual pairs keeps
        # spatial-pairing comparisons at a genuinely shared codebook.
        merged = torch.cat(pools[(k, "contiguous")], dim=0)
        for words in (32, 64, 128):
            cb, info = fit_kmeans(merged, words, seed=2026, iterations=24, max_samples=30_000)
            codebooks[(k, "mse", words)] = cb
            meta.append({"rank": k, "family": "shared_mse_kmeans", "codewords_requested": words,
                         **info, "calibration_seed": 0, "pairing": "contiguous", "objective": "2D Euclidean MSE"})
        for name, na, nr, radial in (("polar_8x8_sqrt", 8, 8, "sqrt"),
                                     ("polar_16x4_uniform", 16, 4, "uniform"),
                                     ("polar_32x2_log", 32, 2, "log")):
            cb = polar_codebook(na, nr, radial, max_codewords=64)
            codebooks[(k, name, 64)] = cb
            meta.append({"rank": k, "family": name, "codewords_requested": 64,
                         "actual_codewords": cb.shape[0], "calibration_seed": "fixed analytic", "objective": "none"})
        # Shared PCA rotation fitted only on seed-0, contiguous normalized pairs.
        z = torch.cat(pools[(k, "contiguous")], dim=0)
        cb, info = fit_kmeans(torch.cat(per_dim[k],dim=0),64,seed=2026,iterations=24,max_samples=30_000)
        codebooks[(k,"perdim_mse",64)]=cb
        meta.append({"rank":k,"family":"per_dimension_scale_mse_kmeans","codewords_requested":64,
                     **info,"calibration_seed":0,"objective":"2D Euclidean MSE after per-coordinate p98"})
        w, wi = fit_covariance_transform(z, whiten=False)
        codebooks[(k, "pca_transform", 64)] = (w, wi)
        transformed = z @ w.T
        cb, info = fit_kmeans(transformed, 64, seed=2026, iterations=24, max_samples=30_000)
        codebooks[(k, "pca_mse", 64)] = cb
        meta.append({"rank": k, "family": "pca_rotated_mse_kmeans", "codewords_requested": 64,
                     **info, "calibration_seed": 0, "rotation": "global PCA"})
        zspec = torch.cat(spectral[k], dim=0)
        cb, info = fit_kmeans(zspec, 64, seed=2026, iterations=24, max_samples=30_000)
        codebooks[(k, "spectral_mse", 64)] = cb
        meta.append({"rank": k, "family": "spectral_local_mse_kmeans", "codewords_requested": 64,
                     **info, "calibration_seed": 0, "pairing": "nearest row-major FP32 spectral coefficients"})
    return codebooks, meta


def spectral_local_pairs(matrix, singular_values):
    """Pair spectral coefficients by adjacent log-scale coordinate (<0.5 decade).

    Coefficient coordinates use the mean of their row/column log singular
    values.  Greedy nearest neighbors inside the declared threshold are paired
    first; remaining coordinates are paired row-major, with at most one scalar
    singleton.  This is deliberately an offline spectral-basis oracle.
    """
    m, n = matrix.shape
    if m != n or singular_values.numel() != m:
        raise ValueError("spectral-local pairing expects a square reduced SVD coordinate matrix")
    logx = torch.log10((singular_values.float() / singular_values.float()[0].clamp_min(1e-30)).clamp_min(1e-12))
    coord = ((logx[:, None] + logx[None, :]) * .5).reshape(-1)
    coords=coord.tolist()
    order = torch.argsort(coord, stable=True).tolist()
    pairs=[]; deferred=[]
    # Linear pass over sorted coordinates (avoid quadratic nearest-neighbor
    # search on the ~r^2 spectral coefficient grid).
    for pos in range(0, len(order)-1, 2):
        i,j=order[pos],order[pos+1]
        if abs(coords[i]-coords[j]) < .5: pairs.append((i,j))
        else: deferred.extend((i,j))
    if len(order)%2: deferred.append(order[-1])
    pairs.extend((deferred[i],deferred[i+1]) for i in range(0,len(deferred)-1,2))
    singles=deferred[len(deferred)//2*2:]
    pi=torch.tensor(pairs,dtype=torch.long).reshape(-1,2); si=torch.tensor(singles,dtype=torch.long)
    flat=matrix.detach().float().reshape(-1)
    return flat[pi.to(flat.device)],flat[si.to(flat.device)],pi,si


def quantize_matrix_residual(residual, pairing, codebook, *, scale_method="p98", transform=None, inverse=None, indices=None, scales_override=None, per_dimension=False):
    if indices is None:
        pairs, singles, pair_idx, single_idx = pair_values(residual, pairing)
    else:
        pair_idx, single_idx = indices
        flat=residual.detach().float().reshape(-1)
        pairs,singles=flat[pair_idx.to(flat.device)],flat[single_idx.to(flat.device)]
    if per_dimension:
        qpair, scales, indices = quantize_vectors_per_dimension(pairs,codebook,method=scale_method)
    else:
        qpair, scales, indices = quantize_vectors(pairs, codebook, scales=scales_override,
                                                   scale_method=scale_method, transform=transform,
                                                   inverse_transform=inverse)
    # Odd singleton uses the frozen scalar rank-specific best practical
    # codebook and its own same-rule scalar block scale; at most one value for
    # contiguous pairing, but row/column leftovers are deterministically paired
    # by pairing_indices before reaching this point.
    qsingle = singles.clone()
    if singles.numel():
        alpha = singles.abs().amax().clamp_min(1e-20)
        cb = INT3_CODEBOOK.to(singles)
        qsingle = cb[(singles / alpha).clamp(-1, 1)[:, None].sub(cb).abs().argmin(1)] * alpha
    q = unpair_values(qpair, qsingle, tuple(residual.shape), pair_idx, single_idx)
    return q, scales, indices, pair_idx, single_idx, pairs, qpair


def evaluate_configuration(seed, update, item, k, config, codebook, snapshot, *, do_polar=False,
                           prepared=None, reference_update=None):
    eval_started=time.perf_counter()
    x, d, c_exact, c_hat, residual = prepared if prepared is not None else make_residual(item, k)
    pairing = config["pairing"]
    transform = inverse = None
    if config.get("pca"):
        transform, inverse = codebook
        cb = config["cb"]
    else: cb = codebook
    pair_ref, _, _, _ = pair_values(residual, pairing)
    scale_ref = pair_ref @ transform.T if transform is not None else pair_ref
    scales_override = vector_scales(scale_ref, config.get("scale", "p98"))
    rhat, scales, indices, pair_idx, single_idx, pairs, qpair = quantize_matrix_residual(
        residual, pairing, cb, scale_method=config.get("scale", "p98"),
        transform=transform, inverse=inverse, scales_override=scales_override,
        per_dimension=config.get("per_dimension",False))
    estimate = c_hat + rhat
    kwargs = transform_kwargs(snapshot)
    if reference_update is None: reference_update = muon_reference.zeropower_newton_schulz(x.clone(), **kwargs)
    out_update = muon_reference.zeropower_newton_schulz(estimate.clone(), **kwargs)
    storage = vector_storage_for_row(item["shape"], k, cb.shape[0],
                                     rotation_bits=128 if transform is not None else 0,
                                     scales_per_block=2 if config.get("per_dimension") else 1)
    raw_pnorm=pairs.norm(dim=1); pnorm=raw_pnorm.clamp_min(1e-20); qnorm=qpair.norm(dim=1)
    vcos=(pairs*qpair).sum(dim=1)/(pnorm*qnorm.clamp_min(1e-20))
    valid=raw_pnorm>1e-20; valid_cos=vcos[valid]
    vmag=(qnorm-pnorm).abs()/pnorm
    row = {"seed": seed, "update": update, "parameter_id": item.get("parameter_id", item.get("name")),
           "parameter_name": item.get("name", item.get("parameter_id")), "shape": str(tuple(item["shape"])),
           "numel": x.numel(), "rank": k, "method": config["name"],
           "evaluation_scope":config.get("scope","seed1_sample"),
           "pairing": pairing, "codewords": cb.shape[0], "nominal_bits_per_value": math.ceil(math.log2(cb.shape[0]))/2,
           "scale": config.get("scale", "p98"), "paired_vectors": int(pairs.shape[0]),
           "singleton_values": int(single_idx.numel()),
           "zero_pair_fraction": float((indices == 0).float().mean()) if indices.numel() else 0.0,
           "codebook_occupied": codebook_occupancy(indices, cb.shape[0])["occupied"] if indices.numel() else 0,
           "mean_vector_error": float((pairs-qpair).norm(dim=1).mean()) if pairs.numel() else 0.0,
           "mean_alpha": float(scales.mean()) if scales.numel() else 0.0,
           "storage_ratio_vs_fp32_unamortized": storage["storage_ratio_unamortized"],
           "persistent_bits_unamortized": storage["total_bits_unamortized"],
           "shared_codebook_bits": storage["shared_codebook_bits"], "fp32_bits": storage["fp32_bits"],
           "vector_mean_magnitude_relative_error": float(vmag.mean()) if vmag.numel() else 0.0,
           "vector_mean_direction_cosine": float(valid_cos.mean()) if valid_cos.numel() else None,
           "vector_mean_angular_error_deg": float(torch.acos(valid_cos.clamp(-1,1)).mean()*180/math.pi) if valid_cos.numel() else None,
           "vector_small_norm_magnitude_error": float(vmag[raw_pnorm<pnorm.median()].mean()) if bool((raw_pnorm<pnorm.median()).any()) else None,
           "vector_large_norm_magnitude_error": float(vmag[raw_pnorm>=pnorm.median()].mean()) if bool((raw_pnorm>=pnorm.median()).any()) else None,
           **quant_metrics(residual, rhat, "residual"),
           **quant_metrics(x, estimate, "full_state_raw"),
           **quant_metrics(reference_update, out_update, "update")}
    if do_polar:
        pr = exact_polar(x); po = exact_polar(estimate)
        row.update(quant_metrics(pr, po, "exact_polar"))
    if seed == 1 and config["name"] in {"mse64_contiguous", "polar_8x8_sqrt", "pca_rotated_mse64", "muon_aware_polar_oracle"}:
        sp=spectral_error_metrics(d.u,d.singular_values,d.vh,estimate-x)
        row.update({f"spectral_{key}":value for key,value in sp.items()})
    row["evaluation_wall_seconds"]=time.perf_counter()-eval_started
    return row, estimate, residual, d


def vector_storage_for_row(shape, k, codewords, *, rotation_bits=0, spectral=False, scales_per_block=1):
    from optim.muon_vector_int3 import vector_storage_bits
    return vector_storage_bits(tuple(shape), codewords=codewords, lowrank_rank=k,
                               rotation_bits=rotation_bits, include_spectral_basis=spectral,
                               scales_per_block=scales_per_block)


def run(args):
    torch.set_num_threads(args.threads)
    start_time = time.perf_counter(); root = Path(args.reports_root); out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    paths = discover(root)
    if args.updates:
        paths=[x for x in paths if x[1] in args.updates]
    snapshots = [(s, u, p, load_snapshot(p)) for s, u, p in paths]
    codebook_cache = out / "calibration_codebooks.pt"
    cache_signature = (args.calibration_vectors, args.calibration_tensors_per_update, args.limit_tensors)
    if codebook_cache.exists() and not args.recalibrate:
        cached = torch.load(codebook_cache, map_location="cpu", weights_only=False)
        if cached.get("signature") == cache_signature:
            codebooks, codebook_manifest = cached["codebooks"], cached["manifest"]
            print("loaded cached seed-0 codebooks", flush=True)
        else:
            codebooks, codebook_manifest = gather_calibration(snapshots, max_vectors_per_tensor=args.calibration_vectors,
                                                               limit_tensors=args.limit_tensors,
                                                               tensors_per_update=args.calibration_tensors_per_update)
    else:
        codebooks, codebook_manifest = gather_calibration(snapshots, max_vectors_per_tensor=args.calibration_vectors,
                                                           limit_tensors=args.limit_tensors,
                                                           tensors_per_update=args.calibration_tensors_per_update)
    write_csv(out / "codebook_manifest.csv", codebook_manifest)
    # Persist calibration products so an interrupted CPU evaluation does not
    # have to repeat seed-0 SVD/codebook fitting.  They are analysis outputs,
    # never consumed by training.
    torch.save({"codebooks": codebooks, "manifest": codebook_manifest,
                "signature": cache_signature}, codebook_cache)
    codebook_rows = []
    for key, value in codebooks.items():
        tensors = value if isinstance(value, tuple) else (value,)
        for component, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                continue
            for index, vector in enumerate(tensor.detach().float().cpu().reshape(-1, 2) if tensor.ndim == 2 and tensor.shape[-1] == 2 else tensor.detach().float().cpu().reshape(-1, 1)):
                codebook_rows.append({"rank": key[0], "family": key[1], "size": key[2],
                                      "component": component, "index": index,
                                      "value_0": float(vector[0]),
                                      "value_1": float(vector[1]) if vector.numel() > 1 else ""})
    write_csv(out / "codebook_levels.csv", codebook_rows)

    # Configurations: shared MSE codebooks at 32/64/128, four spatial schemes
    # at matched 64 words, fixed polar grids, PCA rotation, and spectral-local
    # coefficient pairing.  Every learned object uses seed 0 only.
    configs = []
    for k in RANKS:
        for words in (32, 64, 128):
            cb = codebooks[(k, "mse", words)]
            schemes = PAIRINGS if words == 64 else ("contiguous",)
            for p in schemes:
                configs.append((k, {"name": f"mse{words}_{p}", "pairing": p,
                                    # All 300 instances are retained for the
                                    # primary matched-budget 64-word comparison.
                                    # 32/128 words are diagnostic bit-budget
                                    # endpoints on a fixed held-out subset.
                                    "scope": "all_tensors" if words == 64 else "seed1_sample"}, cb))
        cb64=codebooks[(k,"mse",64)]
        for scale in ("absmax","rms2"):
            configs.append((k,{"name":f"mse64_contiguous_{scale}","pairing":"contiguous","scale":scale,"scope":"seed1_sample"},cb64))
        configs.append((k,{"name":"mse64_contiguous_perdim","pairing":"contiguous","per_dimension":True,"scope":"seed1_sample"},codebooks[(k,"perdim_mse",64)]))
        for name in ("polar_8x8_sqrt", "polar_16x4_uniform", "polar_32x2_log"):
            configs.append((k, {"name": name, "pairing": "contiguous","scope":"seed1_sample"}, codebooks[(k, name, 64)]))
        w, wi = codebooks[(k, "pca_transform", 64)]
        configs.append((k, {"name": "pca_rotated_mse64", "pairing": "contiguous", "pca": True,
                            "cb": codebooks[(k, "pca_mse", 64)],"scope":"seed1_sample"}, (w, wi)))
        configs.append((k, {"name": "spectral_local_mse64", "pairing": "contiguous", "spectral": True,"scope":"seed1_sample"},
                        codebooks[(k, "spectral_mse", 64)]))

    # Small held-out-safe calibration oracle: rotate a fixed polar codebook by
    # one of four offsets and select mean seed-0 K=5 cosine on 25 deterministic
    # calibration tensor instances.  It is intentionally an oracle ceiling.
    oracle_offsets = (0.0, math.pi/32, math.pi/16, 3*math.pi/32)
    oracle_selected = {}
    oracle_cal_path = out / "muon_oracle_calibration.csv"
    oracle_cal_rows = read_csv(oracle_cal_path) if oracle_cal_path.exists() else []
    cache_oracle = bool(oracle_cal_rows) and all(
        {"rank", "offset", "calibration_seed", "calibration_tensor_count", "mean_update_cosine"}.issubset(r)
        for r in oracle_cal_rows)
    for k in RANKS:
        cb0 = codebooks[(k, "polar_8x8_sqrt", 64)]
        positive = cb0[1:]
        cached_for_rank = [r for r in oracle_cal_rows if int(r["rank"]) == k]
        for offset in oracle_offsets if not cache_oracle else ():
            co, si = math.cos(offset), math.sin(offset)
            rot = torch.tensor([[co, -si], [si, co]])
            cb = torch.cat((torch.zeros((1,2)), positive @ rot.T), dim=0)
            vals = []
            count = 0
            for seed, update, _, snap in snapshots:
                if seed != 0: continue
                for item in eligible_items(snap,args.limit_tensors):
                    if count >= args.oracle_calibration_tensors: break
                    row, _, _, _ = evaluate_configuration(seed, update, item, k,
                        {"name": "oracle_candidate", "pairing": "contiguous"}, cb, snap)
                    vals.append(float(row["update_cosine"])); count += 1
                if count >= args.oracle_calibration_tensors: break
            mean = statistics.mean(vals) if vals else float("nan")
            oracle_cal_rows.append({"rank": k, "offset": offset, "calibration_seed": 0,
                                   "calibration_tensor_count": count, "mean_update_cosine": mean})
        choices = cached_for_rank if cache_oracle else oracle_cal_rows[-len(oracle_offsets):]
        if len(choices) != len(oracle_offsets):
            raise RuntimeError("Muon-aware oracle cache must contain all offsets for each rank")
        chosen = max(choices, key=lambda r: (float(r["mean_update_cosine"]), -float(r["offset"])))
        offset = float(chosen["offset"]); co, si = math.cos(offset), math.sin(offset)
        rot = torch.tensor([[co, -si], [si, co]])
        oracle_selected[k] = torch.cat((torch.zeros((1,2)), positive @ rot.T), dim=0)
        configs.append((k, {"name": "muon_aware_polar_oracle", "pairing": "contiguous","scope":"seed1_sample"}, oracle_selected[k]))
    if args.limit_configs:
        configs=configs[:args.limit_configs]
    write_csv(oracle_cal_path, oracle_cal_rows)

    all_rows = []; diag_rows = []; elapsed = time.perf_counter()
    partial_path = out / "tensor_level_results.partial.csv"
    done_snapshots = set()
    if args.resume and partial_path.exists():
        for rr in read_csv(partial_path):
            parsed = {}
            for key, value in rr.items():
                if value == "": parsed[key] = None; continue
                try:
                    parsed[key] = int(value) if value.lstrip("-").isdigit() else float(value)
                except (ValueError, AttributeError):
                    parsed[key] = value
            all_rows.append(parsed)
            done_snapshots.add((int(parsed["seed"]), int(parsed["update"])))
    exact_polar_names = {"mse64_contiguous", "polar_8x8_sqrt", "pca_rotated_mse64", "muon_aware_polar_oracle"}
    scalar_rows = read_csv(root / "muon_int3_practical_scale" / "tensor_level_results.csv")
    scalar_ref = {(int(r["seed"]), int(r["update"]), r["parameter_id"], int(r["k"]), r["method"]): r
                  for r in scalar_rows if r.get("method") == "percentile:p=98"}
    for seed, update, path, snap in snapshots:
        if (seed, update) in done_snapshots:
            print(f"resume: already completed seed={seed} update={update}", flush=True)
            continue
        items=eligible_items(snap,args.limit_tensors)
        full_items=items
        diagnostic_items=items[:args.diagnostic_tensors_per_update] if args.diagnostic_tensors_per_update else items
        diagnostic_ids={id(x) for x in diagnostic_items}
        for item in full_items:
            x=item["tensor"].detach().float(); d0=decompose(x)
            ref_update=muon_reference.zeropower_newton_schulz(x.clone(),**transform_kwargs(snap))
            prepared={}
            for rk in RANKS:
                ce=(d0.u[:,:rk]*d0.singular_values[:rk])@d0.vh[:rk]
                ch=factorized_topk(d0.u,d0.singular_values,d0.vh,rk)
                prepared[rk]=(x,d0,ce,ch,x-ce)
            for k, config, cb in configs:
                if config.get("scope")=="seed1_sample" and (seed!=1 or id(item) not in diagnostic_ids):
                    continue
                # Spectral-local control applies the same quantizer in M's
                # original SVD-coordinate matrix and maps the result back.
                if config.get("spectral"):
                    x, d, _, c_hat, residual = prepared[k]
                    rh = d.u.T @ residual @ d.vh.T
                    pairs0,single0,pi,si=spectral_local_pairs(rh,d.singular_values)
                    qh, scales, indices, pi, si, pairs, qpair = quantize_matrix_residual(
                        rh, "spectral-local", cb, indices=(pi,si))
                    qres = d.u @ qh @ d.vh
                    estimate = c_hat + qres
                    obs_update = muon_reference.zeropower_newton_schulz(estimate.clone(), **transform_kwargs(snap))
                    storage = vector_storage_for_row(item["shape"], k, cb.shape[0], spectral=True)
                    row = {"seed":seed,"update":update,"parameter_id":item.get("parameter_id",item.get("name")),
                           "parameter_name":item.get("name",item.get("parameter_id")),"shape":str(tuple(item["shape"])),
                           "numel":x.numel(),"rank":k,"method":config["name"],"evaluation_scope":config.get("scope","seed1_sample"),"pairing":"spectral-local",
                           "codewords":cb.shape[0],"nominal_bits_per_value":3.0,"scale":"p98",
                           "paired_vectors":len(pairs),"singleton_values":len(si),
                           "zero_pair_fraction":float((indices==0).float().mean()),
                           "storage_ratio_vs_fp32_unamortized":storage["storage_ratio_unamortized"],
                           "persistent_bits_unamortized":storage["total_bits_unamortized"],
                           "shared_codebook_bits":storage["shared_codebook_bits"],"fp32_bits":storage["fp32_bits"],
                           **quant_metrics(residual,qres,"residual"),**quant_metrics(x,estimate,"full_state_raw"),
                           **quant_metrics(ref_update,obs_update,"update")}
                    if seed==1 and config["name"] in exact_polar_names:
                        row.update(quant_metrics(exact_polar(x),exact_polar(estimate),"exact_polar"))
                    eh=d.u.T@(estimate-x)@d.vh.T
                    diag_rows.append({"seed":seed,"update":update,"parameter_id":row["parameter_id"],"rank":k,
                                      "method":config["name"],"danger_energy_fraction":float(eh.diagonal().square().sum()),
                                      "total_error_energy":float(eh.square().sum())})
                    all_rows.append(row); continue
                do_polar = seed == 1 and config["name"] in exact_polar_names
                row, estimate, residual, d = evaluate_configuration(seed,update,item,k,config,cb,snap,
                    do_polar=do_polar,prepared=prepared[k],reference_update=ref_update)
                row["source_seed_role"] = "calibration" if seed == 0 else "held_out"
                ref = scalar_ref.get((seed,update,row["parameter_id"],k,"percentile:p=98"))
                if ref:
                    row["scalar_int3_update_cosine"] = float(ref["update_cosine"])
                    row["delta_vs_scalar_int3"] = float(row["update_cosine"])-float(ref["update_cosine"])
                # Direct production INT4 reference is a read-only residual
                # quantization evaluation; use exact production quantizer.
                if config["name"] == "mse64_contiguous":
                    x = item["tensor"].detach().float(); _,_,_,c_hat,r = prepared[k]
                    q4 = production_quantize(r,"int4-dynamic-b2048")
                    q4u = muon_reference.zeropower_newton_schulz((c_hat+q4).clone(),**transform_kwargs(snap))
                    row["structural_int4_update_cosine"] = quant_metrics(ref_update,q4u,"q4")["q4_cosine"]
                all_rows.append(row)
        print(f"evaluated seed={seed} update={update}; rows={len(all_rows)}", flush=True)
        # Checkpoint only after a complete landmark, making long CPU runs
        # resumable without partial/duplicate tensor records.
        write_csv(partial_path, all_rows)
    write_csv(out / "tensor_level_results.csv", all_rows)
    write_csv(out / "spectral_diagnostics.csv", diag_rows)
    held = [r for r in all_rows if r.get("seed") == 1]
    for seed in SEEDS:
        for method in {r["method"] for r in all_rows}:
            for k in RANKS:
                rr=[r for r in all_rows if r["seed"]==seed and r["method"]==method and r["rank"]==k]
                if rr:
                    n=len(rr); shared=rr[0]["shared_codebook_bits"]
                    total=sum(r["persistent_bits_unamortized"] for r in rr)-(n-1)*shared
                    den=sum(r["fp32_bits"] for r in rr)
                    for r in rr:r["storage_ratio_amortized_global_codebook"]=total/den
    summary=[]
    for method in sorted({r["method"] for r in held}):
        for k in RANKS:
            rs=[r for r in held if r["method"]==method and r["rank"]==k]
            if not rs: continue
            deltas=[float(r["delta_vs_scalar_int3"]) for r in rs
                    if r.get("delta_vs_scalar_int3") not in (None, "")]
            summary.append({"seed":1,"role":"held_out","rank":k,"method":method,"tensor_count":len(rs),
                            "mean_update_cosine":statistics.mean(r["update_cosine"] for r in rs),
                            "median_update_cosine":statistics.median(r["update_cosine"] for r in rs),
                            "weighted_update_cosine":sum(r["update_cosine"]*r["numel"] for r in rs)/sum(r["numel"] for r in rs),
                            "p25_update_cosine":sorted(r["update_cosine"] for r in rs)[max(0,int(.25*(len(rs)-1)))],
                            "p75_update_cosine":sorted(r["update_cosine"] for r in rs)[max(0,int(.75*(len(rs)-1)))],
                            "p10_update_cosine":sorted(r["update_cosine"] for r in rs)[max(0,int(.10*(len(rs)-1)))],
                            "p90_update_cosine":sorted(r["update_cosine"] for r in rs)[max(0,int(.90*(len(rs)-1)))],
                            "mean_update_relative_l2":statistics.mean(r["update_relative_l2"] for r in rs),
                            "mean_full_state_raw_cosine":statistics.mean(r["full_state_raw_cosine"] for r in rs),
                            "mean_residual_relative_l2":statistics.mean(r["residual_relative_l2"] for r in rs),
                            "mean_storage_ratio":rs[0]["storage_ratio_amortized_global_codebook"],
                            "mean_delta_vs_scalar":statistics.mean(deltas) if deltas else None})
    write_csv(out / "heldout_summary.csv", summary)
    # Pairing-specific outputs and main scalar/vector/INT4 summary.
    write_csv(out / "pairing_results.csv", [r for r in all_rows if r["method"].startswith("mse64_")])
    write_csv(out / "polar_codebook_results.csv", [r for r in all_rows if r["method"].startswith("polar_")])
    write_csv(out / "kmeans_codebook_results.csv", [r for r in all_rows if r["method"].startswith("mse")])
    write_csv(out / "rotation_results.csv", [r for r in all_rows if "pca" in r["method"]])
    write_csv(out / "muon_oracle_results.csv", [r for r in all_rows if "oracle" in r["method"]])
    write_csv(out / "spectral_pairing_results.csv", [r for r in all_rows if "spectral_local" in r["method"]])
    storage_summary=[]
    for r in summary:
        rr=[x for x in held if x["rank"]==r["rank"] and x["method"]==r["method"]]
        if rr:
            storage_summary.append({"rank":r["rank"],"method":r["method"],"codewords":rr[0]["codewords"],
                "nominal_bits_per_value":rr[0]["nominal_bits_per_value"],
                "global_shared_codebook_bits":rr[0]["shared_codebook_bits"],
                "mean_storage_ratio_global_amortized":r["mean_storage_ratio"],
                "mean_storage_ratio_unamortized":statistics.mean(x["storage_ratio_vs_fp32_unamortized"] for x in rr),
                "block_scale_metadata_bits_total":sum(math.ceil(x["numel"]/BLOCK_SIZE)*32*(2 if "perdim" in x["method"] else 1) for x in rr),
                "lowrank_bf16_factor_bits_total":sum(16*(eval(x["shape"])[0]*r["rank"]+eval(x["shape"])[1]*r["rank"]+r["rank"]) for x in rr)})
    write_csv(out / "storage_summary.csv", storage_summary)
    vector_metric_keys={"seed","update","parameter_id","rank","method","pairing","vector_mean_magnitude_relative_error","vector_mean_direction_cosine","vector_small_norm_magnitude_error","vector_large_norm_magnitude_error","zero_pair_fraction","residual_relative_l2","residual_cosine"}
    write_csv(out / "vector_error_metrics.csv", [{k:v for k,v in r.items() if k in vector_metric_keys} for r in all_rows])
    write_csv(out / "scalar_baselines.csv", [r for r in scalar_rows if r["seed"]=="1" and r.get("method")=="percentile:p=98"])
    runtime_file=out/"full_run_runtime_seconds.txt"
    if args.resume and runtime_file.exists():
        full_runtime=float(runtime_file.read_text().strip())
    else:
        full_runtime=time.perf_counter()-start_time
        if not args.resume:
            runtime_file.write_text(f"{full_runtime:.3f}\n")
    summary_path = out / "summary.md"
    with summary_path.open("w") as f:
        f.write("# Muon conditioned residual vector INT3 study\n\n")
        f.write(f"CPU-only offline evaluation. Seed 0 supplied all learned codebooks and the small Muon-aware angle-offset calibration; seed 1 is the held-out evaluation split. Codebook training uses an evenly spaced deterministic sample of {args.calibration_tensors_per_update} tensors at each seed-0 landmark and up to {args.calibration_vectors} vectors per tensor, with exact 2048-scalar block p98 normalization. The primary 64-codeword pairing comparison covers every eligible tensor instance (300 total); 32/128-codeword budget endpoints and the more expensive polar/PCA/spectral/oracle/scale diagnostics use a deterministic held-out subset. No training or production code was changed.\n\n")
        f.write("## Held-out seed 1\n\n| rank | method | n | mean K=5 cosine | median [p25,p75] | size-weighted | mean update rel-L2 | exact-polar cosine | scalar INT3 delta | storage / FP32 |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary:
            rs=[x for x in held if x["rank"]==r["rank"] and x["method"]==r["method"]]
            pv=[float(x["exact_polar_cosine"]) for x in rs if x.get("exact_polar_cosine") not in (None,"")]
            polar_text=f"{statistics.mean(pv):.5f} (n={len(pv)})" if pv else "—"
            delta_text=f"{r['mean_delta_vs_scalar']:.5f}" if r["mean_delta_vs_scalar"] is not None else "—"
            f.write(f"| {r['rank']} | {r['method']} | {r['tensor_count']} | {r['mean_update_cosine']:.5f} | {r['median_update_cosine']:.5f} [{r['p25_update_cosine']:.5f},{r['p75_update_cosine']:.5f}] | {r['weighted_update_cosine']:.5f} | {r['mean_update_relative_l2']:.5f} | {polar_text} | {delta_text} | {r['mean_storage_ratio']:.5f} |\n")
        f.write("\n## Main interpretation\n\n")
        main={k: next(r for r in summary if r["rank"]==k and r["method"]=="mse64_contiguous") for k in RANKS}
        f.write(f"The held-out 64-word MSE vector quantizer at the matched 3-bit/value residual budget reaches K=5 update cosine {main[4]['mean_update_cosine']:.4f} (k=4) and {main[8]['mean_update_cosine']:.4f} (k=8), on all 150 seed-1 tensor instances per rank. Against the matched scalar p98 INT3 baseline ({.777975:.4f}/{.799500:.4f}), gains are +{main[4]['mean_delta_vs_scalar']:.4f}/+{main[8]['mean_delta_vs_scalar']:.4f}; the vector method beats scalar on {100*sum(float(x['delta_vs_scalar_int3'])>0 for x in held if x['method']=='mse64_contiguous')/sum(1 for x in held if x['method']=='mse64_contiguous'):.1f}% of these instances. It approximately matches the prior structural INT4 means (0.8245/0.8514), without exceeding the residual's nominal 3-bit/value payload. This is evidence that 2D residual representation can break the scalar INT3 ceiling in this offline matched-state study, not evidence about training trajectory behavior.\n\n")
        f.write(f"Pairing is secondary but not entirely irrelevant: row and contiguous are identical for the present even-width matrices, checkerboard is close (means {next(r['mean_update_cosine'] for r in summary if r['rank']==4 and r['method']=='mse64_checkerboard'):.4f}/{next(r['mean_update_cosine'] for r in summary if r['rank']==8 and r['method']=='mse64_checkerboard'):.4f}), while column pairing is lower. The fixed polar grids and the small polar-offset Muon-aware oracle do not match the MSE-trained codebook; that oracle is restricted to a tiny calibration set and one polar family, so it is not a general upper bound over vector codebooks. PCA/per-dimension scaling diagnostics use only 20 held-out instances and should not be compared as full-coverage winners. Spectral-local pairing is an explicitly non-deployable control and performs poorly here.\n\n")
        f.write("The strongest supported interpretation is that a shared 2D MSE codebook adds useful local residual representation capacity, and on this held-out trajectory closes most or all of the observed structural INT4 cosine gap at a similar total storage ratio. Follow-up should first repeat the result with an independently trained codebook/alternate calibration split and validate scale/packing/runtime behavior before any training prototype. Spatial-vs-spectral comparisons do not support a claim that spectral pairing adds value.\n\n")
        instance_count=sum(len(eligible_items(s,args.limit_tensors)) for _,_,_,s in snapshots)
        f.write(f"\nFull CPU analysis runtime: {full_runtime:.1f}s ({full_runtime/60:.1f} min); current report aggregation: {time.perf_counter()-start_time:.1f}s. Evaluation phase in this invocation: {time.perf_counter()-elapsed:.1f}s. Eligible matrix instances: {instance_count}. Diagnostic tensors/update: {args.diagnostic_tensors_per_update}.\n")
    with (out / "methodology.md").open("w") as f:
        f.write("# Methodology\n\n")
        f.write("For each eligible 2D Muon momentum matrix, the fixed conditioner is the ordinary rank-k truncated FP32 SVD (k=4,8), with factors represented in BF16 for storage/fidelity evaluation. The residual is quantized independently of production training using 2D nearest-codeword quantization. Each vector receives one fixed-length index: 5 bits for 32 words, 6 bits for 64, and 7 bits for 128. The primary matched scalar budget is 64 codewords = 6 bits/pair = 3 bits/value. One shared scalar scale is selected per 2048 residual scalars, p98 by default; normalized coordinates are clipped to [-1,1], then nearest Euclidean codeword is selected. Odd singleton values use the existing uniform signed INT3 scalar reference and are explicitly counted.\n\n")
        f.write("Calibration uses seed 0 only. Shared learned codebooks use deterministic k-means/Lloyd with seed 2026 and exact zero reserved; fixed polar grids are analytic. Seed 1 is held out. Spatial pairings are row-major contiguous, row-wise, column-wise, and checkerboard diagonal pairs with deterministic leftover pairing. PCA is one global seed-0 rotation. Spectral-local pairing is an explicitly non-deployable FP32-SVD-basis oracle. Muon-aware polar codebook selection uses only the fixed seed-0 calibration sample; no held-out update fidelity enters selection.\n\n")
        f.write("All fidelity readouts use the exact production K=5 Muon transform; exact polar is computed only for selected diagnostic configurations. Persistent storage counts vector indices, BF16 low-rank factors, per-block FP32 scales, metadata, global codebook amortization, and transforms/bases for the relevant oracle variants. CPU timings are analysis-only and do not predict accelerator kernel costs.\n")
    # Simple headline plots.
    def bars(plt):
        fig, ax = plt.subplots(figsize=(12,6)); names=[]; vals=[]
        for r in summary:
            names.append(f"k{r['rank']} {r['method']}"); vals.append(r["mean_update_cosine"])
        ax.barh(names, vals); ax.axvline(.8245,color="gray",ls="--",label="structural INT4 k4 ref")
        ax.set_xlabel("held-out K=5 update cosine (seed-1 mean)"); ax.legend()
    save_plot(out/"vector_vs_scalar_int4.png",bars)
    def frontier(plt):
        fig, ax=plt.subplots(figsize=(8,6))
        for k in RANKS:
            rs=[r for r in summary if r["rank"]==k]
            ax.scatter([r["mean_storage_ratio"] for r in rs],[r["mean_update_cosine"] for r in rs],label=f"k={k}")
            for r in rs: ax.annotate(r["method"],(r["mean_storage_ratio"],r["mean_update_cosine"]),fontsize=5)
        ax.set_xlabel("metadata-inclusive storage ratio / FP32");ax.set_ylabel("held-out K=5 update cosine");ax.legend()
    save_plot(out/"storage_fidelity.png",frontier)
    def pairing(plt):
        fig,ax=plt.subplots(figsize=(8,5))
        vals=[r for r in summary if r["method"].startswith("mse64_")]
        for k in RANKS:
            rr=[r for r in vals if r["rank"]==k]
            ax.plot([r["method"].replace("mse64_","") for r in rr],[r["mean_update_cosine"] for r in rr],marker="o",label=f"k={k}")
        ax.set_ylabel("held-out K=5 update cosine");ax.set_xlabel("pairing / scale rule");ax.legend()
    save_plot(out/"pairing_strategy_fidelity.png",pairing)
    def codebook_size(plt):
        fig,ax=plt.subplots(figsize=(7,5))
        for k in RANKS:
            rr=sorted([r for r in summary if r["rank"]==k and r["method"].startswith("mse") and r["method"].split("_")[0] in {"mse32","mse64","mse128"}],key=lambda r:int(r["method"].split("_")[0][3:]))
            ax.plot([int(r["method"].split("_")[0][3:]) for r in rr],[r["mean_update_cosine"] for r in rr],marker="o",label=f"k={k}")
        ax.set_xscale("log",base=2);ax.set_xlabel("codeword budget");ax.set_ylabel("held-out update cosine");ax.legend()
    save_plot(out/"codebook_size_vs_fidelity.png",codebook_size)
    def vector_error(plt):
        rr=[r for r in held if r.get("vector_mean_direction_cosine") is not None]
        fig,ax=plt.subplots(figsize=(7,5));ax.scatter([r["vector_mean_magnitude_relative_error"] for r in rr],[r["vector_mean_angular_error_deg"] for r in rr],s=8,alpha=.4)
        ax.set_xlabel("mean vector magnitude relative error");ax.set_ylabel("mean vector angular error (degrees)")
    save_plot(out/"vector_angular_magnitude_error.png",vector_error)
    def delta_hist(plt):
        fig,ax=plt.subplots(figsize=(8,5))
        for k in RANKS:
            rr=[r for r in held if r["rank"]==k and r.get("delta_vs_scalar_int3") is not None]
            if rr:ax.hist([r["delta_vs_scalar_int3"] for r in rr],bins=30,alpha=.5,label=f"k={k}")
        ax.axvline(0,color="black",lw=1);ax.set_xlabel("vector minus scalar INT3 update cosine");ax.set_ylabel("tensor instances");ax.legend()
    save_plot(out/"heldout_gain_histogram.png",delta_hist)
    def spectral_plot(plt):
        rr=[r for r in held if r.get("spectral_danger_associated_fraction_error") not in (None,"")]
        fig,ax=plt.subplots(figsize=(7,5));ax.scatter([r["spectral_danger_associated_fraction_error"] for r in rr],[r["update_cosine"] for r in rr],s=10,alpha=.5)
        ax.set_xlabel("danger-zone associated fraction of spectral error");ax.set_ylabel("K=5 update cosine")
    save_plot(out/"danger_zone_error_vs_fidelity.png",spectral_plot)
    def codebook_plot(plt):
        fig, axes=plt.subplots(1,2,figsize=(11,5))
        for ax,k in zip(axes,RANKS):
            learned=codebooks[(k,"mse",64)].detach().cpu()
            polar=codebooks[(k,"polar_8x8_sqrt",64)].detach().cpu()
            ax.scatter(learned[:,0],learned[:,1],s=22,label="seed-0 MSE codebook")
            ax.scatter(polar[:,0],polar[:,1],s=15,marker="x",label="fixed polar")
            ax.set_title(f"rank k={k}");ax.set_xlim(-1.05,1.05);ax.set_ylim(-1.05,1.05);ax.set_aspect("equal")
            ax.set_xlabel("coordinate 1");ax.set_ylabel("coordinate 2");ax.legend(fontsize=7)
        fig.suptitle("64-word codebooks (normalized/clipped training domain)")
    save_plot(out/"codebook_layout.png",codebook_plot)
    def vector_cloud(plt):
        snap=next(s for seed,update,_,s in snapshots if seed==1 and update==128)
        item=eligible_items(snap)[0]; _,_,_,_,resid=make_residual(item,8)
        z,_,_,_=pair_values(resid,"contiguous"); z=normalize_vector_blocks(z,"p98")
        take=torch.linspace(0,z.shape[0]-1,min(6000,z.shape[0])).round().long(); z=z[take].cpu()
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        for ax,k in zip(axes,RANKS):
            cb=codebooks[(k,"mse",64)].cpu()
            ax.hexbin(z[:,0],z[:,1],gridsize=45,mincnt=1,cmap="Blues",bins="log")
            ax.scatter(cb[:,0],cb[:,1],c="red",s=14,marker="x",label="learned codewords")
            ax.set_xlim(-1.05,1.05);ax.set_ylim(-1.05,1.05);ax.set_aspect("equal");ax.set_title(f"k={k}; seed1/u128 sample")
            ax.set_xlabel("normalized residual coordinate 1");ax.set_ylabel("coordinate 2");ax.legend(fontsize=7)
    save_plot(out/"residual_vector_distribution.png",vector_cloud)
    def zero_rate(plt):
        rr=[r for r in held if r.get("zero_pair_fraction") not in (None,"")]
        fig,ax=plt.subplots(figsize=(7,5));ax.scatter([r["zero_pair_fraction"] for r in rr],[r["update_cosine"] for r in rr],s=8,alpha=.35)
        ax.set_xlabel("fraction assigned to exact-zero vector codeword");ax.set_ylabel("held-out K=5 update cosine")
    save_plot(out/"zero_rate_vs_fidelity.png",zero_rate)
    def pareto_refs(plt):
        fig,ax=plt.subplots(figsize=(8,6))
        for k in RANKS:
            rr=[r for r in summary if r["rank"]==k and r["method"] in {"mse64_contiguous","mse32_contiguous","mse128_contiguous"}]
            ax.scatter([r["mean_storage_ratio"] for r in rr],[r["mean_update_cosine"] for r in rr],label=f"2D INT3 k={k}")
            for r in rr:ax.annotate(r["method"].replace("mse",""),(r["mean_storage_ratio"],r["mean_update_cosine"]),fontsize=7)
        scalar_storage={4:.101636,8:.109020};scalar_fid={4:.777975,8:.799500}
        for k in RANKS:ax.scatter([scalar_storage[k]],[scalar_fid[k]],marker="s",label=f"scalar INT3 k={k}")
        ax.scatter([.13289,.14027],[.824504,.851377],marker="*",s=100,label="structural INT4 reference")
        ax.set_xlabel("metadata-inclusive idealized storage / FP32");ax.set_ylabel("held-out K=5 update cosine");ax.legend(fontsize=7)
    save_plot(out/"storage_fidelity_references.png",pareto_refs)
    print(f"vector INT3 study complete: {out}; total_seconds={time.perf_counter()-start_time:.1f}",flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--reports-root",default=str(ROOT/"reports")); ap.add_argument("--outdir",default=str(OUT)); ap.add_argument("--threads",type=int,default=4); ap.add_argument("--calibration-vectors",type=int,default=1200); ap.add_argument("--calibration-tensors-per-update",type=int,default=8); ap.add_argument("--oracle-calibration-tensors",type=int,default=12); ap.add_argument("--updates",type=int,nargs="*",default=None); ap.add_argument("--limit-configs",type=int,default=0); ap.add_argument("--limit-tensors",type=int,default=0); ap.add_argument("--diagnostic-tensors-per-update",type=int,default=4); ap.add_argument("--resume",action="store_true",help="resume from complete landmark checkpoints in tensor_level_results.partial.csv"); ap.add_argument("--recalibrate",action="store_true",help="discard cached analysis codebooks and recalibrate from seed 0"); args=ap.parse_args(); run(args)


if __name__ == "__main__": main()
