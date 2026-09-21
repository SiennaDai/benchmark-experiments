#!/usr/bin/env python3
"""Offline quantization-aware spectral conditioner study for Muon."""
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
from optim.muon_quantization_aware_conditioner import (  # noqa: E402
    cross_scale_fractions,
    greedy_select,
    quantize_residual,
    selected_reconstruction,
)
from optim.muon_spectral_sensitivity import quantize  # noqa: E402
from optim.muon_storage_pareto import storage_bits  # noqa: E402
from optim.muon_structural_decomposition import decompose, danger_zone_energy  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
RANKS = (4, 8)
METHODS = ("topk_svd", "range_aware", "quant_error_aware", "muon_update_aware")
BITS = (4, 3)
EPS = 1e-12


def discover_snapshots(root: Path):
    groups = {}
    for path in sorted(root.rglob("update_*.pt")):
        if "muon_momentum_snapshots" not in path.parts:
            continue
        try:
            snap = load_snapshot(path)
            seed = int(snap["metadata"]["seeds"]["seed"]); update = int(snap["metadata"]["update"])
        except Exception:
            continue
        if seed in SEEDS and update in LANDMARKS:
            groups.setdefault((seed, str(path.parent)), {})[update] = path
    out = []
    for seed in SEEDS:
        choices = sorted((g, v) for (s, g), v in groups.items() if s == seed and set(v) == set(LANDMARKS))
        if choices:
            values = choices[0][1]; out.extend((seed, u, values[u]) for u in LANDMARKS)
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path):
    if not path.exists(): return []
    with path.open(newline="") as h: return list(csv.DictReader(h))


def finite(value):
    try: return value not in (None, "") and math.isfinite(float(value))
    except (TypeError, ValueError): return False


def kwargs(snapshot):
    c = snapshot["metadata"]["muon_transform"]
    return {"steps": int(c["steps"]), "coefficients": tuple(float(x) for x in c["coefficients"]), "eps": float(c["eps"])}


def ratios(reference, candidate):
    a, b = reference.float(), candidate.float(); an, bn = a.norm(), b.norm()
    return {"relative_l2": float((b-a).norm()/an) if an.item() else None,
            "cosine": float((a*b).sum()/(an*bn)) if an.item() and bn.item() else None,
            "norm_ratio": float(bn/an) if an.item() else None}


def evaluate(x, candidate, ref_update, ref_polar, kw, *, polar=True):
    raw = ratios(x, candidate)
    update = muon_reference.zeropower_newton_schulz(candidate.detach().clone(), **kw)
    upd = ratios(ref_update, update)
    pol = ratios(ref_polar, exact_polar(candidate)) if polar else {"relative_l2": None, "cosine": None, "norm_ratio": None}
    return {"raw_relative_l2": raw["relative_l2"], "raw_cosine": raw["cosine"], "raw_norm_ratio": raw["norm_ratio"],
            "update_relative_l2": upd["relative_l2"], "update_cosine": upd["cosine"], "update_norm_ratio": upd["norm_ratio"],
            "exact_polar_relative_l2": pol["relative_l2"], "exact_polar_cosine": pol["cosine"], "exact_polar_norm_ratio": pol["norm_ratio"]}


def ident(seed, update, item):
    return {"seed": seed, "update": update, "parameter_id": item.get("parameter_id", item.get("name", "unknown")),
            "parameter_name": item.get("name", item.get("parameter_id", "unknown")), "shape": str(item["shape"])}


def block_absmax(x, block=2048):
    flat = x.float().reshape(-1)
    return torch.stack([flat[i:i+block].abs().amax() for i in range(0, flat.numel(), block)]) if flat.numel() else torch.zeros(0)


def run(args):
    started = time.perf_counter(); paths = discover_snapshots(args.reports_root)
    if len(paths) != 10: raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    if args.snapshot_limit: paths = paths[:args.snapshot_limit]
    ranks = (4, 8, 16) if args.include_k16 else RANKS
    bits = (4, 3) if not args.int4_only else (4,)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    manifest=[]; selected=[]; fidelity=[]; storage=[]; conditioning=[]; danger=[]; mixing=[]; oracle=[]
    tensor_rows=[]; eval_count=0
    for seed, update, path in paths:
        snap = load_snapshot(path); kw = kwargs(snap)
        print(f"processing seed={seed} update={update}", flush=True)
        items = snap["tensors"][:args.tensor_limit] if args.tensor_limit else snap["tensors"]
        for item in items:
            if len(item["shape"]) != 2: continue
            x = item["tensor"].detach().float(); state = decompose(x); m,n=x.shape
            ref_u = muon_reference.zeropower_newton_schulz(x.clone(), **kw); ref_p = exact_polar(x)
            direct4 = quantize(x, "int4-dynamic-b2048").float()
            direct4m = evaluate(x, direct4, ref_u, ref_p, kw)
            for bits_value in bits:
                direct = direct4 if bits_value == 4 else quantize_residual(x, bits_value)
                direct_m = direct4m if bits_value == 4 else evaluate(x, direct, ref_u, ref_p, kw)
                active = len(state.singular_values)
                pool = min(args.candidate_pool, active)
                base = ident(seed, update, item)
                requested = [m for m in args.methods.split(",") if m]
                methods = ["topk_svd"] + [m for m in requested if m != "topk_svd"]
                methods = [m for m in methods if m in METHODS]
                max_rank = min(max(ranks), active)
                selected_max = {}
                for method in methods:
                    if method == "topk_svd":
                        continue
                    sel = greedy_select(x, state.u, state.singular_values, state.vh, max_rank, bits_value, method,
                                        candidate_pool=pool, reference_update=ref_u, transform_kwargs=kw)
                    selected_max[method] = sel
                    eval_count += sel.candidate_evaluations
                for rank in ranks:
                    k = min(rank, active)
                    top_modes = tuple(range(k))
                    candidates = {"topk_svd": top_modes}
                    for method, sel in selected_max.items():
                        candidates[method] = sel.modes[:k]
                        manifest.append({**base, "bits": bits_value, "rank": k, "method": method, "candidate_pool": pool,
                                         "candidate_evaluations": sel.candidate_evaluations, "objective_start": sel.objective_start,
                                         "objective_final": sel.objective_final, "selection_rank": max_rank})
                    manifest.append({**base, "bits": bits_value, "rank": k, "method": "topk_svd", "candidate_pool": pool,
                                     "candidate_evaluations": 0, "objective_start": None, "objective_final": None, "selection_rank": k})
                    for method, modes in candidates.items():
                        _, qres, candidate = selected_reconstruction(x, state.u, state.singular_values, state.vh, modes, bits_value)
                        mm = evaluate(x, candidate, ref_u, ref_p, kw, polar=True)
                        gain = mm["update_cosine"] - direct_m["update_cosine"] if finite(mm["update_cosine"]) and finite(direct_m["update_cosine"]) else None
                        top_gain = None
                        if method != "topk_svd":
                            top_modes_candidate = selected_reconstruction(x, state.u, state.singular_values, state.vh, top_modes, bits_value)[2]
                            top_mm = evaluate(x, top_modes_candidate, ref_u, ref_p, kw, polar=True)
                            top_gain = mm["update_cosine"] - top_mm["update_cosine"]
                        row={**base,"bits":bits_value,"rank":k,"method":method,"modes":" ".join(map(str,modes)),"candidate_pool":pool,
                             "direct_update_cosine":direct_m["update_cosine"],"delta_vs_direct":gain,"delta_vs_topk":top_gain,
                             "relative_extra_recovery_vs_topk": (top_gain/(1-top_mm["update_cosine"])) if top_gain is not None and finite(top_mm.get("update_cosine")) and abs(1-top_mm["update_cosine"])>EPS else None, **mm}
                        fidelity.append(row); tensor_rows.append(row)
                        sb = storage_bits((m,n),k,"bf16",include_metadata=True,residual_bits=bits_value)
                        storage.append({**base,"bits":bits_value,"rank":k,"method":method,"persistent_bits":sb["metadata_inclusive_bits"],
                                        "storage_ratio_vs_fp32":sb["metadata_inclusive_bits"]/sb["fp32_bits"],"compression_ratio_vs_fp32":sb["fp32_bits"]/sb["metadata_inclusive_bits"]})
                        resid=x-subset_component_safe(state, modes); scales=block_absmax(resid); source=block_absmax(x)
                        conditioning.append({**base,"bits":bits_value,"rank":k,"method":method,"residual_relative_norm":float(resid.norm()/x.norm()) if x.norm().item() else None,
                                             "residual_spectral_norm":float(torch.linalg.matrix_norm(resid,ord=2)),"mean_block_absmax":float(scales.mean()),"direct_mean_block_absmax":float(source.mean()),
                                             "mean_block_absmax_ratio_vs_direct":float(scales.mean()/source.mean()) if source.mean().item() else None,
                                             "residual_quant_error":float((qres-resid).norm()),"mean_quant_step_proxy":float(scales.mean()/3 if bits_value==4 else scales.mean()/3)})
                        dz=danger_zone_energy(state,x-candidate); dm=danger_zone_energy(state,x-direct)
                        danger.append({**base,"bits":bits_value,"rank":k,"method":method,"danger_fraction_of_error":dz["danger_fraction_of_error"],"danger_fraction_of_matrix":dz["danger_fraction_of_matrix"],"direct_danger_fraction_of_error":dm["danger_fraction_of_error"]})
                        cm=cross_scale_fractions(state,x-candidate); mixing.append({**base,"bits":bits_value,"rank":k,"method":method,**cm})
                        if method in ("topk_svd","range_aware","quant_error_aware","muon_update_aware"):
                            for mode in modes:
                                selected.append({**base,"bits":bits_value,"rank":k,"method":method,"mode":mode,"sigma_ratio":float(state.singular_values[mode]/state.singular_values[0]) if state.singular_values[0].item() else None,
                                                 "danger_zone":-3 <= math.log10(max(float(state.singular_values[mode]/state.singular_values[0]),1e-6)) < -2})
                # one oracle headroom row per tensor/bit/rank is emitted above
                oracle.append({**base,"bits":bits_value,"rank":k,"topk_update_cosine":None})
    write_csv(out/"conditioner_manifest.csv", manifest); write_csv(out/"selected_modes.csv", selected); write_csv(out/"fidelity_results.csv", fidelity)
    write_csv(out/"storage_matched_summary.csv", storage); write_csv(out/"residual_conditioning.csv", conditioning); write_csv(out/"danger_zone_results.csv", danger); write_csv(out/"cross_scale_mixing.csv", mixing)
    write_csv(out/"int3_results.csv", [r for r in fidelity if int(r["bits"]) == 3]); write_csv(out/"oracle_headroom.csv", fidelity); write_csv(out/"tensor_level_results.csv", tensor_rows)
    if not args.skip_plots:
        write_plots(out, fidelity, storage, conditioning)
    (out/"methodology.md").write_text("""# Quantization-aware spectral conditioner methodology

`topk_svd` retains the first k FP32 singular components. `range_aware` greedily chooses singular components minimizing mean b2048 residual block absmax. `quant_error_aware` greedily minimizes the exact residual quantization Frobenius error. `muon_update_aware` greedily maximizes production K=5 update cosine and is an offline oracle, not a deployable method. All methods use exactly k modes and BF16 factor storage for the primary comparison.

INT4 delegates to the existing production `int4-dynamic-b2048` implementation. INT3 is an offline analogous signed dynamic b2048 roundtrip: each block uses absmax scale and nearest values from `{-1,-2/3,-1/3,0,1/3,2/3,1}`; the eighth three-bit code is reserved so zero is exact. It is not a production quantizer. Candidate selection is deterministic, uses the first 32 active modes by default, and ties resolve to the lowest singular index. No selection objective uses downstream fidelity except the explicitly labeled Muon-update-aware oracle.

Persistent storage uses the existing metadata-inclusive BF16 factor accounting; temporary greedy candidate tensors and Muon transforms are not persistent state.
""")
    (out/"summary.md").write_text(f"# Quantization-aware spectral conditioner study\n\nCoverage: {len(paths)} snapshots; tensor-level rows: {len(fidelity)}; candidate evaluations: {eval_count}; CPU runtime: {time.perf_counter()-started:.1f}s.\n\nThe report compares top-k SVD, deterministic residual-range selection, residual-quantization-error selection, and a K=5 Muon update-aware oracle at identical rank/storage. INT4 uses the unchanged production quantizer. INT3 is explicitly offline and uses the documented seven-level symmetric dynamic codebook. See CSV files and plots for per-tensor and aggregate results.\n")
    print(f"wrote {out}; rows={len(fidelity)} evaluations={eval_count} runtime={time.perf_counter()-started:.1f}s", flush=True)


def subset_component_safe(state, modes):
    if not modes: return torch.zeros((state.u.shape[0], state.vh.shape[1]), dtype=torch.float32)
    idx=torch.tensor(list(modes),dtype=torch.long)
    return (state.u[:,idx]*state.singular_values[idx])@state.vh[idx]


def write_plots(out, fidelity, storage, conditioning):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out/"plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs are complete.\n"); return
    grouped=defaultdict(list)
    for r in fidelity:
        if finite(r.get("update_cosine")): grouped[(r["bits"],r["rank"],r["method"])].append(float(r["update_cosine"]))
    labels=[]; vals=[]
    for key, values in sorted(grouped.items()): labels.append(f"b{key[0]} k{key[1]} {key[2]}"); vals.append(sum(values)/len(values))
    if vals:
        plt.figure(figsize=(14,5)); plt.bar(range(len(vals)),vals); plt.xticks(range(len(vals)),labels,rotation=75,ha="right",fontsize=7); plt.ylabel("K=5 update cosine"); plt.tight_layout(); plt.savefig(out/"conditioner_vs_update_cosine.png",dpi=130); plt.close()
    pts=[]
    for r,s in zip(fidelity,storage):
        if finite(r.get("update_cosine")): pts.append((float(s["storage_ratio_vs_fp32"]),float(r["update_cosine"]),r["method"],r["bits"]))
    if pts:
        plt.figure(figsize=(7,5));
        for method in sorted(set(p[2] for p in pts)):
            q=[p for p in pts if p[2]==method]; plt.scatter([p[0] for p in q],[p[1] for p in q],s=4,label=method)
        plt.xlabel("persistent storage / FP32"); plt.ylabel("K=5 update cosine"); plt.legend(); plt.tight_layout(); plt.savefig(out/"storage_fidelity_frontier.png",dpi=130); plt.close()
    if conditioning:
        d=defaultdict(list)
        for r in conditioning:
            if finite(r.get("mean_block_absmax_ratio_vs_direct")): d[r["method"]].append(float(r["mean_block_absmax_ratio_vs_direct"]))
        if d:
            plt.figure(figsize=(8,4)); plt.bar(list(d),[sum(v)/len(v) for v in d.values()]); plt.xticks(rotation=35,ha="right"); plt.ylabel("mean residual/direct block absmax"); plt.tight_layout(); plt.savefig(out/"block_absmax_vs_conditioner.png",dpi=130); plt.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--reports-root",type=Path,default=ROOT/"reports"); ap.add_argument("--output",type=Path,default=ROOT/"reports/muon_quantization_aware_conditioner"); ap.add_argument("--snapshot-limit",type=int,default=0); ap.add_argument("--tensor-limit",type=int,default=0); ap.add_argument("--candidate-pool",type=int,default=8,help="deterministic active-mode candidate pool (default 8 for CPU coverage)"); ap.add_argument("--include-k16",action="store_true"); ap.add_argument("--int4-only",action="store_true"); ap.add_argument("--methods",default="range_aware,quant_error_aware,muon_update_aware",help="comma-separated non-top-k methods"); ap.add_argument("--skip-plots",action="store_true"); args=ap.parse_args()
    run(args)


if __name__ == "__main__": main()
