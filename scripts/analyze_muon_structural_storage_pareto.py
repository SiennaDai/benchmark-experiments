#!/usr/bin/env python3
"""Offline storage--fidelity Pareto analysis for structural INT4 Muon.

The low-rank factor is diagnostic side information only.  The script never
touches an optimizer and uses the production quantizer and Muon transform on
detached snapshot tensors.
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
from optim.muon_update_fidelity import load_snapshot  # noqa: E402
from optim.muon_spectral_sensitivity import quantize  # noqa: E402
from optim.muon_structural_decomposition import (  # noqa: E402
    ENERGY_TARGETS, FIXED_RANKS, decompose, energy_rank, truncated,
)
from optim.muon_storage_pareto import (  # noqa: E402
    BLOCK_SIZE, FACTOR_VARIANTS, cast_factors, direct_storage_bits,
    pareto_mask, storage_bits,
)
from optim import muon_reference  # noqa: E402
from optim.muon_ns_sensitivity import exact_polar  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
Q4 = "int4-dynamic-b2048"
Q8 = "int8-linear-b2048"
RANK_GRID = (0, 1, 2, 4, 8, 16)
TARGETS = (0.90, 0.95, 0.98)


def discover_snapshots(root: Path):
    groups = {}
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
    out = []
    for seed in SEEDS:
        candidates = sorted((g, v) for (s, g), v in groups.items()
                            if s == seed and set(v) == set(LANDMARKS))
        if candidates:
            values = candidates[0][1]
            out.extend((seed, u, values[u]) for u in LANDMARKS)
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def finite(v):
    try:
        return v is not None and v != "" and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def ident(seed, update, item):
    return {"seed": seed, "update": update,
            "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
            "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")),
            "shape": str(item["shape"])}


def transform(x, kw):
    return muon_reference.zeropower_newton_schulz(x.detach().clone(), **kw)


def compare(ref, candidate, ref_update, kw, ref_polar):
    ref, candidate = ref.float(), candidate.float()
    rn, cn = ref.norm(), candidate.norm()
    raw = {"raw_relative_l2": float((candidate - ref).norm() / rn) if rn.item() else None,
           "raw_cosine": float((ref * candidate).sum() / (rn * cn)) if rn.item() and cn.item() else None,
           "raw_norm_ratio": float(cn / rn) if rn.item() else None}
    cu = transform(candidate, kw); un, cn = ref_update.norm(), cu.norm()
    raw.update({"update_relative_l2": float((cu-ref_update).norm()/un) if un.item() else None,
                "update_cosine": float((ref_update*cu).sum()/(un*cn)) if un.item() and cn.item() else None,
                "update_norm_ratio": float(cn/un) if un.item() else None})
    cp = exact_polar(candidate); pn, cn = ref_polar.norm(), cp.norm()
    raw.update({"exact_polar_relative_l2": float((cp-ref_polar).norm()/pn) if pn.item() else None,
                "exact_polar_cosine": float((ref_polar*cp).sum()/(pn*cn)) if pn.item() and cn.item() else None,
                "exact_polar_norm_ratio": float(cn/pn) if pn.item() else None})
    return raw


def avg(rows, field, weight_field=None):
    vals = [(float(r[field]), float(r[weight_field]) if weight_field and finite(r.get(weight_field)) else 1.0)
            for r in rows if finite(r.get(field))]
    if not vals:
        return None
    return sum(v*w for v,w in vals) / sum(w for _,w in vals)


def aggregate(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(row)
    out = []
    metrics = ("storage_idealized_bits", "storage_metadata_bits", "storage_ratio_vs_fp32",
               "compression_ratio_vs_fp32", "extra_storage_vs_int4", "update_cosine",
               "update_relative_l2", "exact_polar_cosine", "exact_polar_relative_l2")
    for key, values in grouped.items():
        row = {k:v for k,v in zip(group_keys,key)}
        total_fp32 = sum(float(r["fp32_bits"]) for r in values if finite(r.get("fp32_bits")))
        for metric in metrics:
            if metric.startswith("storage_") and metric.endswith("bits"):
                row[metric+"_sum"] = sum(float(r[metric]) for r in values if finite(r.get(metric)))
            elif metric in {"storage_ratio_vs_fp32", "compression_ratio_vs_fp32", "extra_storage_vs_int4"}:
                raw = sum(float(r["storage_metadata_bits"])*float(r["fp32_bits"])**0 for r in values if finite(r.get("storage_metadata_bits")))
                # Weighted ratios are recomputed from global bits below.
                if metric == "storage_ratio_vs_fp32":
                    row[metric] = sum(float(r["storage_metadata_bits"]) for r in values) / total_fp32 if total_fp32 else None
                elif metric == "compression_ratio_vs_fp32":
                    row[metric] = total_fp32 / sum(float(r["storage_metadata_bits"]) for r in values) if raw else None
                else:
                    direct = sum(float(r["direct_int4_metadata_bits"]) for r in values)
                    row[metric] = (sum(float(r["storage_metadata_bits"]) for r in values)-direct)/direct if direct else None
            else:
                row[metric+"_weighted"] = avg(values, metric, "numel")
                row[metric+"_mean"] = avg(values, metric)
                sorted_vals = sorted(float(r[metric]) for r in values if finite(r.get(metric)))
                row[metric+"_median"] = sorted_vals[len(sorted_vals)//2] if sorted_vals else None
        row["tensor_count"] = len(values)
        row["numel_total"] = sum(int(r["numel"]) for r in values)
        out.append(row)
    return out


def plot_outputs(out: Path, rows: list[dict], global_rows: list[dict], adaptive: list[dict]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs remain complete.\n")
        return
    def points(condition=None, y="update_cosine"):
        data = [(float(r["storage_ratio_vs_fp32"]), float(r[y]), str(r.get("configuration", r.get("rank_label", ""))))
                for r in rows if (condition is None or condition(r)) and finite(r.get(y)) and finite(r.get("storage_ratio_vs_fp32"))]
        return data
    plt.figure(figsize=(7,5))
    for variant in FACTOR_VARIANTS:
        p = points(lambda r: r.get("factor_precision") == variant)
        if p: plt.scatter([a for a,_,_ in p],[b for _,b,_ in p],s=5,label=variant)
    plt.xlabel("storage / FP32"); plt.ylabel("K=5 update cosine"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/"memory_vs_update_cosine.png",dpi=140); plt.close()
    for y,name,label in [("exact_polar_cosine","memory_vs_exact_polar_cosine.png","exact polar cosine"),("update_relative_l2","memory_vs_update_relative_l2.png","update relative L2")]:
        plt.figure(figsize=(7,5)); p=points(y=y);
        if p: plt.scatter([a for a,_,_ in p],[b for _,b,_ in p],s=5,alpha=.3)
        plt.xlabel("storage / FP32"); plt.ylabel(label); plt.tight_layout(); plt.savefig(out/name,dpi=140); plt.close()
    # Curves over fixed/energy labels using global weighted metrics.
    for metric,name in [("update_cosine_weighted","fixed_rank_curves.png"),("exact_polar_cosine_weighted","energy_rank_curves.png")]:
        plt.figure(figsize=(9,5))
        for variant in FACTOR_VARIANTS:
            vals=[r for r in global_rows if r.get("factor_precision")==variant and r.get("rank_label") not in {"direct_int4","direct_int8"}]
            vals=sorted(vals,key=lambda r: float(r.get("rank") or 0))
            if vals: plt.plot([float(r["storage_ratio_vs_fp32"]) for r in vals],[float(r[metric]) for r in vals],marker="o",label=variant)
        plt.xlabel("storage / FP32"); plt.ylabel(metric); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/name,dpi=140); plt.close()
    if adaptive:
        plt.figure(figsize=(7,5)); p=[(float(r["storage_ratio_vs_fp32"]),float(r["aggregate_update_cosine"])) for r in adaptive if finite(r.get("aggregate_update_cosine"))];
        if p: plt.scatter([x for x,_ in p],[y for _,y in p]);
        plt.xlabel("storage / FP32"); plt.ylabel("adaptive aggregate K=5 cosine"); plt.tight_layout(); plt.savefig(out/"adaptive_rank_oracle.png",dpi=140); plt.close()


@torch.no_grad()
def run(args):
    started=time.perf_counter(); paths=discover_snapshots(args.reports_root)
    if len(paths) != 10: raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    if args.snapshot_limit: paths=paths[:args.snapshot_limit]
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    tensor_rows=[]; fidelity=[]; mechanism=[]; shape_rows=[]; direct_rows=[]
    for seed, update, path in paths:
        snap=load_snapshot(path); meta=snap["metadata"]; cfg=meta["muon_transform"]
        kw={"steps":int(cfg["steps"]),"coefficients":tuple(float(x) for x in cfg["coefficients"]),"eps":float(cfg["eps"])}
        print(f"processing seed={seed} update={update}",flush=True)
        tensor_items = snap["tensors"][:args.tensor_limit] if args.tensor_limit else snap["tensors"]
        for item in tensor_items:
            if len(item["shape"]) != 2: continue
            x=item["tensor"].detach().float(); m,n=x.shape; numel=x.numel(); ident0=ident(seed,update,item)
            st=decompose(x); direct=quantize(x,Q4).float(); refu=transform(x,kw); refp=exact_polar(x)
            direct_m=compare(x,direct,refu,kw,refp)
            q8=quantize(x,Q8).float(); int8_m=compare(x,q8,refu,kw,refp)
            direct_bits=direct_storage_bits((m,n),include_metadata=True)
            base={**ident0,"rank_label":"direct_int4","rank":0,"factor_precision":"none","configuration":"direct_int4","numel":numel,"fp32_bits":direct_bits["fp32_bits"],"direct_int4_metadata_bits":direct_bits["int4_metadata_inclusive_bits"],"storage_idealized_bits":direct_bits["int4_idealized_bits"],"storage_metadata_bits":direct_bits["int4_metadata_inclusive_bits"],"storage_ratio_vs_fp32":direct_bits["int4_metadata_inclusive_bits"]/direct_bits["fp32_bits"],"compression_ratio_vs_fp32":direct_bits["fp32_bits"]/direct_bits["int4_metadata_inclusive_bits"],"extra_storage_vs_int4":0.0,**direct_m}
            direct_rows.append(base)
            # Baseline INT8 is retained as a comparison point, not a structural factor condition.
            i8={**ident0,"rank_label":"direct_int8","rank":0,"factor_precision":"none","configuration":"direct_int8","numel":numel,"fp32_bits":direct_bits["fp32_bits"],"direct_int4_metadata_bits":direct_bits["int4_metadata_inclusive_bits"],"storage_idealized_bits":8*numel,"storage_metadata_bits":8*numel+32*((numel+BLOCK_SIZE-1)//BLOCK_SIZE)+152,"storage_ratio_vs_fp32":(8*numel+32*((numel+BLOCK_SIZE-1)//BLOCK_SIZE)+152)/direct_bits["fp32_bits"],"compression_ratio_vs_fp32":direct_bits["fp32_bits"]/(8*numel+32*((numel+BLOCK_SIZE-1)//BLOCK_SIZE)+152),"extra_storage_vs_int4":None,**int8_m}
            direct_rows.append(i8)
            for label, rank, kind, target in [(f"fixed_{k}",min(k,len(st.singular_values)),"fixed",None) for k in RANK_GRID if k>0] + [(f"energy_{int(t*100)}",energy_rank(st.singular_values,t),"energy",t) for t in ENERGY_TARGETS]:
                rank=min(rank,m,n); low,res=truncated(st,rank); qres=quantize(res,Q4).float();
                for variant in FACTOR_VARIANTS:
                    u,s,vh=cast_factors(st.u[:,:rank],st.singular_values[:rank],st.vh[:rank],variant)
                    low_hat=(u*s)@vh if rank else torch.zeros_like(x); candidate=low_hat+qres
                    mm=compare(x,candidate,refu,kw,refp); bits=storage_bits((m,n),rank,variant,include_metadata=True)
                    row={**ident0,"rank_label":label,"rank":rank,"rank_kind":kind,"energy_target":target,"factor_precision":variant,"configuration":f"{label}_{variant}","numel":numel,"fp32_bits":bits["fp32_bits"],"direct_int4_metadata_bits":bits["direct_int4_metadata_bits"],"storage_idealized_bits":bits["idealized_bits"],"storage_metadata_bits":bits["metadata_inclusive_bits"],"storage_ratio_vs_fp32":bits["metadata_inclusive_bits"]/bits["fp32_bits"],"compression_ratio_vs_fp32":bits["fp32_bits"]/bits["metadata_inclusive_bits"],"extra_storage_vs_int4":(bits["metadata_inclusive_bits"]-bits["direct_int4_metadata_bits"])/bits["direct_int4_metadata_bits"],"low_rank_energy_fraction":float(st.singular_values[:rank].square().sum()/st.singular_values.square().sum()) if st.singular_values.numel() and rank else 0.0,**mm}
                    fidelity.append(row); tensor_rows.append({k:row[k] for k in ("seed","update","parameter_id","shape","rank_label","rank","factor_precision","numel","fp32_bits","storage_idealized_bits","storage_metadata_bits","storage_ratio_vs_fp32","compression_ratio_vs_fp32","extra_storage_vs_int4")})
                # Mechanism rows are precision-independent because residual is identical.
                block_scales=lambda z: torch.stack([z.reshape(-1)[i:i+BLOCK_SIZE].abs().amax() for i in range(0,numel,BLOCK_SIZE)])
                ds=block_scales(x); rs=block_scales(res)
                mechanism.append({**ident0,"rank_label":label,"rank":rank,"residual_over_matrix_norm":float(res.norm()/x.norm()) if x.norm().item() else None,"max_abs_residual":float(res.abs().max()),"max_abs_matrix":float(x.abs().max()),"block_absmax_ratio_mean":float(rs.mean()/ds.mean()) if ds.mean().item() else None,"block_absmax_ratio_median":float(rs.median()/ds.median()) if ds.median().item() else None,"danger_zone_residual_norm_fraction":float(res.norm()/x.norm()) if x.norm().item() else None})
    # Add rank-0/direct and all per-tensor factor rows to storage tables.
    # Fidelity rows already carry the full storage accounting; the compact
    # tensor_rows list is retained only for the per-tensor storage artifact.
    storage_rows=fidelity+direct_rows
    global_rows=aggregate(storage_rows,["rank_label","factor_precision","configuration"])
    for row in global_rows:
        label = str(row.get("rank_label", ""))
        if label.startswith("fixed_"):
            try:
                row["rank"] = int(label.split("_", 1)[1])
            except ValueError:
                row["rank"] = None
        elif label.startswith("energy_"):
            try:
                row["energy_target"] = int(label.split("_", 1)[1]) / 100.0
            except ValueError:
                row["energy_target"] = None
    # Pareto flags for global configs; all three objective views are emitted.
    configs=[]
    for r in global_rows:
        if finite(r.get("update_cosine_weighted")): configs.append(r)
    for metric in ("update_cosine_weighted","exact_polar_cosine_weighted","update_relative_l2_weighted"):
        flags=pareto_mask(configs, "storage_ratio_vs_fp32", metric,
                          higher_is_better=metric != "update_relative_l2_weighted")
        for r,flag in zip(configs,flags): r[f"pareto_{metric}"]=flag
    frontier = []
    for metric in ("update_cosine_weighted", "exact_polar_cosine_weighted", "update_relative_l2_weighted"):
        for row in configs:
            if row.get(f"pareto_{metric}"):
                frontier.append({"objective": metric, **row})
    # Keep direct rows separate in the fidelity artifact, while global summary includes baselines.
    write_csv(out/"tensor_storage_cost.csv",storage_rows)
    write_csv(out/"factor_precision_fidelity.csv",fidelity+direct_rows)
    write_csv(out/"global_storage_summary.csv",global_rows)
    write_csv(out/"pareto_frontier.csv", frontier)
    energy=[r for r in global_rows if str(r.get("rank_label","")).startswith("energy_")]
    write_csv(out/"energy_rank_storage.csv",energy)
    # Per-configuration threshold and compression summaries.
    thresholds=[]; constraints=[]
    for variant in list(FACTOR_VARIANTS)+["none"]:
        cand=[r for r in global_rows if r.get("factor_precision")==variant and finite(r.get("update_cosine_weighted"))]
        for target in (0.80,0.85,0.90,0.92):
            hit=[r for r in cand if float(r["update_cosine_weighted"])>=target]
            best=min(hit,key=lambda r:float(r["storage_ratio_vs_fp32"])) if hit else None
            thresholds.append({"factor_precision":variant,"target_update_cosine":target,"rank_label":best.get("rank_label") if best else None,"storage_ratio_vs_fp32":best.get("storage_ratio_vs_fp32") if best else None,"update_cosine":best.get("update_cosine_weighted") if best else None})
        for compression in (2,4,6):
            hit=[r for r in cand if finite(r.get("compression_ratio_vs_fp32")) and float(r["compression_ratio_vs_fp32"])>=compression]
            best=max(hit,key=lambda r:float(r["update_cosine_weighted"])) if hit else None
            constraints.append({"factor_precision":variant,"compression_constraint":compression,"rank_label":best.get("rank_label") if best else None,"storage_ratio_vs_fp32":best.get("storage_ratio_vs_fp32") if best else None,"update_cosine":best.get("update_cosine_weighted") if best else None})
    write_csv(out/"fidelity_thresholds.csv",thresholds); write_csv(out/"compression_constraints.csv",constraints)
    # Shape dependence uses one row per tensor/condition, with low-rank overhead explicit.
    for r in storage_rows:
        if r.get("factor_precision") not in FACTOR_VARIANTS: continue
        sh=eval(r["shape"],{"__builtins__":{}},{}); m,n=map(int,sh); r2={"seed":r["seed"],"update":r["update"],"parameter_id":r["parameter_id"],"shape":r["shape"],"m":m,"n":n,"aspect_ratio":max(m,n)/min(m,n),"min_dim":min(m,n),"rank":r["rank"],"factor_precision":r["factor_precision"],"side_info_bits":int(r["storage_idealized_bits"])-4*m*n,"side_info_over_residual":(int(r["storage_idealized_bits"])-4*m*n)/(4*m*n)}; shape_rows.append(r2)
    write_csv(out/"shape_dependence.csv",shape_rows)
    # Adaptive rank oracle from factor rows, separately for each factor precision and target.
    adaptive=[]
    for variant in FACTOR_VARIANTS:
        for target in TARGETS:
            policy=[]
            unreachable = 0
            for key in sorted({(r["seed"],r["update"],r["parameter_id"]) for r in fidelity if r.get("factor_precision")==variant}):
                rr=[r for r in fidelity if r.get("factor_precision")==variant and (r["seed"],r["update"],r["parameter_id"])==key and r.get("rank_label","").startswith("fixed_")]
                rr=[r for r in rr if float(r.get("rank",0)) in RANK_GRID[1:] and finite(r.get("update_cosine"))]
                hit=sorted([r for r in rr if float(r["update_cosine"])>=target],key=lambda r:int(r["rank"]))
                if hit:
                    chosen = hit[0]
                elif rr:
                    # Keep a complete state policy while making unreachable
                    # targets explicit; rank-16 is the best available fallback.
                    chosen = max(rr, key=lambda r: int(r["rank"]))
                    unreachable += 1
                else:
                    chosen = None
                if chosen:
                    policy.append(chosen)
            if policy:
                fp=sum(float(r["fp32_bits"]) for r in policy); sb=sum(float(r["storage_metadata_bits"]) for r in policy)
                adaptive.append({"factor_precision":variant,"target_update_cosine":target,"tensor_count":len(policy),"unreachable_tensor_count":unreachable,"total_storage_bits":sb,"total_fp32_bits":fp,"storage_ratio_vs_fp32":sb/fp,"compression_ratio_vs_fp32":fp/sb,"aggregate_update_cosine":sum(float(r["update_cosine"])*int(r["numel"]) for r in policy)/sum(int(r["numel"]) for r in policy),"mean_rank":sum(int(r["rank"]) for r in policy)/len(policy)})
    write_csv(out/"adaptive_rank_oracle.csv",adaptive)
    # Marginal curves from weighted global rows.
    marginal=[]
    for variant in FACTOR_VARIANTS:
        rr={int(r["rank"]):r for r in global_rows if r.get("factor_precision")==variant and r.get("rank") is not None}
        for a,b in zip(RANK_GRID[:-1],RANK_GRID[1:]):
            if a in rr and b in rr:
                db=float(rr[b]["storage_metadata_bits_sum"])-float(rr[a]["storage_metadata_bits_sum"]); dy=float(rr[b]["update_cosine_weighted"])-float(rr[a]["update_cosine_weighted"])
                marginal.append({"factor_precision":variant,"from_rank":a,"to_rank":b,"delta_update_cosine":dy,"delta_storage_bits":db,"delta_cosine_per_bit":dy/db if db else None,"delta_cosine_per_storage_ratio":dy/(float(rr[b]["storage_ratio_vs_fp32"])-float(rr[a]["storage_ratio_vs_fp32"])) if float(rr[b]["storage_ratio_vs_fp32"])-float(rr[a]["storage_ratio_vs_fp32"]) else None})
    write_csv(out/"marginal_storage_returns.csv",marginal)
    # Join existing mechanism report where available.
    mech_rows=[]
    prior=ROOT/"reports/muon_structural_decomposition_mechanism/resolution_analysis.csv"
    prior_rows=[]
    if prior.exists():
        with prior.open() as h: prior_rows=list(csv.DictReader(h))
    for r in mechanism:
        match=next((p for p in prior_rows if p.get("seed")==str(r["seed"]) and p.get("update")==str(r["update"]) and p.get("parameter_id")==r["parameter_id"] and p.get("rank_label")==r["rank_label"]),{})
        mech_rows.append({**r,"prior_update_cosine_gain":match.get("update_cosine_gain"),"prior_scale_reduction_ratio":match.get("scale_reduction_ratio"),"prior_danger_error_fraction":match.get("danger_error_fraction")})
    write_csv(out/"mechanism_frontier_link.csv",mech_rows)
    if not args.skip_plots:
        plot_outputs(out,fidelity,global_rows,adaptive)
    # Human-readable methodology and summary are intentionally concise and
    # generated from the same rows used by downstream analyses.
    best=max((r for r in global_rows if r.get("factor_precision") in FACTOR_VARIANTS and finite(r.get("update_cosine_weighted"))),key=lambda r:float(r["update_cosine_weighted"]),default=None)
    unique_tensors = len({(r.get("seed"), r.get("update"), r.get("parameter_id")) for r in tensor_rows})
    lines=["# Structural INT4 Muon storage--fidelity Pareto analysis", "", f"Coverage: {len(paths)} snapshots; eligible Muon tensors: {unique_tensors}; condition rows: {len(tensor_rows)}.", "", "Storage uses dense INT4 residual payload plus U(mk), sigma(k), V(nk). Metadata-inclusive estimates add one FP32 b2048 absmax scale per residual block and fixed dimensions/rank/precision/quantizer identifiers; no kernel overhead is inferred.", "", "FP32 factor fidelity is the previous oracle condition. BF16/FP16 and mixed conditions round-trip factors before reconstruction, while the residual is always Q4(M-M_k) from the original FP32 decomposition.", "", f"Highest observed weighted K=5 cosine configuration: {best.get('configuration') if best else 'unavailable'} ({best.get('update_cosine_weighted') if best else 'unavailable'}), storage ratio {best.get('storage_ratio_vs_fp32') if best else 'unavailable'}.", "", "This is an offline headroom analysis; adaptive ranks and exact factors are not a deployable method."]
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    (out/"methodology.md").write_text("""# Methodology\n\nFor M in R^(m x n), structural storage is 4mn + b_U mk + b_sigma k + b_V nk bits. Metadata-inclusive storage adds 32 bits per b2048 residual block and 152 fixed decoder bits (dimensions, rank, precision labels, quantizer family, and block size). FP32 baseline is 32mn bits.\n\nThe structural reconstruction is Mhat = cast(U_k) cast(Sigma_k) cast(V_k)^T + Q4(M-M_k). The production b2048 dynamic quantizer and zeropower_newton_schulz are called on detached copies. K=5 and exact polar metrics are computed against the original FP32 snapshot. Pareto filtering uses lower metadata-inclusive storage and higher fidelity. Whole-optimizer totals are not inferred because auxiliary AdamW state is not present in these Muon-only snapshots.\n""")
    print(f"wrote {out}; tensors={len(tensor_rows)} runtime={time.perf_counter()-started:.1f}s",flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--reports-root",type=Path,default=ROOT/"reports"); ap.add_argument("--output",type=Path,default=ROOT/"reports/muon_structural_storage_pareto"); ap.add_argument("--snapshot-limit",type=int,default=0); ap.add_argument("--tensor-limit",type=int,default=0); ap.add_argument("--skip-plots",action="store_true"); ap.add_argument("--plots-only",action="store_true")
    args=ap.parse_args()
    if args.plots_only:
        def read(path):
            with path.open(newline="") as h:
                return list(csv.DictReader(h))
        fidelity_rows = read(args.output / "factor_precision_fidelity.csv")
        global_rows = read(args.output / "global_storage_summary.csv")
        configs = [r for r in global_rows if finite(r.get("update_cosine_weighted"))]
        frontier = []
        for metric in ("update_cosine_weighted", "exact_polar_cosine_weighted", "update_relative_l2_weighted"):
            for row, flag in zip(configs, pareto_mask(configs, "storage_ratio_vs_fp32", metric,
                                                     higher_is_better=metric != "update_relative_l2_weighted")):
                if flag:
                    frontier.append({"objective": metric, **row})
        write_csv(args.output / "pareto_frontier.csv", frontier)
        plot_outputs(args.output, fidelity_rows, global_rows,
                     read(args.output / "adaptive_rank_oracle.csv"))
    else:
        run(args)


if __name__ == "__main__": main()
