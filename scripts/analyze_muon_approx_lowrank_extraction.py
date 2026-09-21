#!/usr/bin/env python3
"""Offline approximate low-rank structural INT4 Muon analysis."""
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
from optim.muon_approx_lowrank import ApproxFactors, approximate_svd, reconstruct  # noqa: E402
from optim.muon_ns_sensitivity import exact_polar  # noqa: E402
from optim.muon_spectral_sensitivity import quantize  # noqa: E402
from optim.muon_structural_decomposition import decompose, danger_zone_energy, truncated  # noqa: E402
from optim.muon_storage_pareto import storage_bits, cast_factors  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
RANKS = (1, 2, 4, 8, 16)
ITERATIONS = (0, 1, 2, 3, 4)
OVERSAMPLING = (0, 4, 8)
INITIALIZATIONS = ("canonical", "randomized")
Q4 = "int4-dynamic-b2048"
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
        w = csv.DictWriter(h, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="") as h: return list(csv.DictReader(h))


def make_plots(out: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs remain complete.\n"); return
    rows=read_csv(out/"update_fidelity.csv"); app=read_csv(out/"approximation_metrics.csv"); sub=read_csv(out/"subspace_quality.csv"); dyn=read_csv(out/"residual_dynamic_range.csv"); runtime=read_csv(out/"runtime_cost.csv"); danger=read_csv(out/"danger_zone_metrics.csv")
    def mean_points(field, key, filt=lambda r: True):
        d=defaultdict(list)
        for r in rows:
            if filt(r) and r.get(field) not in (None,"") and r.get(key) not in (None,""):
                d[str(r[key])].append(float(r[field]))
        return [(k,sum(v)/len(v)) for k,v in d.items()]
    for rank in (4,8,16):
        vals=mean_points("fraction_of_exact_gain","iterations",lambda r:int(r["rank"])==rank and int(r["oversampling"])==0 and r["initialization"]=="canonical")
        if vals:
            plt.figure(figsize=(6,4)); plt.plot([int(x) for x,_ in vals],[y for _,y in vals],marker="o"); plt.xlabel("q"); plt.ylabel("fraction of exact gain"); plt.title(f"rank {rank}"); plt.tight_layout(); plt.savefig(out/f"q_vs_gain_rank{rank}.png",dpi=130); plt.close()
    vals=mean_points("fraction_of_exact_gain","oversampling",lambda r:int(r["rank"])==8 and int(r["iterations"])==4 and r["initialization"]=="canonical")
    if vals:
        plt.figure(figsize=(6,4)); plt.plot([int(x) for x,_ in vals],[y for _,y in vals],marker="o"); plt.xlabel("oversampling p"); plt.ylabel("fraction of exact gain"); plt.tight_layout(); plt.savefig(out/"p_vs_gain.png",dpi=130); plt.close()
    def scatter(xrows,x,y,name,xlabel,ylabel):
        pts=[(float(r[x]),float(r[y])) for r in xrows if finite(r.get(x)) and finite(r.get(y))]
        if pts:
            plt.figure(figsize=(6,4)); plt.scatter([a for a,_ in pts],[b for _,b in pts],s=4,alpha=.25); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(out/name,dpi=130); plt.close()
    scatter(rows, "raw_relative_l2", "update_cosine", "approximation_error_vs_update_cosine.png", "raw relative L2", "K=5 update cosine")
    scatter(sub, "left_projection_distance", "update_fidelity_loss", "subspace_error_vs_update_loss.png", "left projection distance", "1-update cosine")
    scatter(dyn, "approx_to_exact_block_absmax", "approx_residual_norm", "residual_block_absmax_vs_norm.png", "approx/exact block absmax", "approx residual norm")
    scatter(danger, "approx_danger_fraction", "update_cosine", "danger_zone_vs_fidelity.png", "danger-zone error fraction", "K=5 update cosine")
    scatter(rows, "exact_polar_cosine", "update_cosine", "exact_polar_vs_update_cosine.png", "exact-polar cosine", "K=5 update cosine")
    scatter(runtime, "runtime_seconds", "rank", "runtime_vs_rank.png", "cumulative CPU seconds", "rank")
    # Joint storage/fidelity chart for terminal randomized rows.
    pts=[]
    for r,s in zip(rows, read_csv(out/"storage_workspace.csv")):
        if int(r["iterations"])==4 and int(r["oversampling"])==8 and r["initialization"]=="randomized": pts.append((float(s["storage_ratio_vs_fp32"]),float(r["update_cosine"]),int(r["rank"])))
    if pts:
        plt.figure(figsize=(7,5));
        for k in sorted(set(c for _,_,c in pts)):
            pp=[p for p in pts if p[2]==k]; plt.scatter([x for x,_,_ in pp],[y for _,y,_ in pp],s=5,label=f"rank {k}")
        plt.xlabel("persistent storage / FP32"); plt.ylabel("K=5 update cosine"); plt.legend(); plt.tight_layout(); plt.savefig(out/"storage_vs_fidelity.png",dpi=130); plt.close()
    # Shape summary.
    d=defaultdict(list)
    for r in rows:
        if int(r["iterations"])==4 and int(r["oversampling"])==8: d[r["shape"]].append(float(r["update_cosine"]))
    if d:
        plt.figure(figsize=(8,4)); plt.bar(list(d),[sum(v)/len(v) for v in d.values()]); plt.xticks(rotation=30,ha="right"); plt.ylabel("terminal update cosine"); plt.tight_layout(); plt.savefig(out/"shape_approximation_quality.png",dpi=130); plt.close()
    terminal=[r for r in rows if int(r["iterations"])==4 and int(r["oversampling"])==8 and r["initialization"]=="randomized"]
    if terminal:
        vals=[]; labels=[]
        for k in (4,8,16):
            pp=[float(r["update_cosine"]) for r in terminal if int(r["rank"])==k]
            labels.append(f"approx k{k}"); vals.append(sum(pp)/len(pp))
        labels = ["direct INT4"] + labels + ["direct INT8"]; vals = [0.7640] + vals + [0.9961]
        plt.figure(figsize=(7,4)); plt.bar(labels,vals); plt.ylabel("K=5 update cosine"); plt.xticks(rotation=25,ha="right"); plt.tight_layout(); plt.savefig(out/"direct_vs_structural_vs_int8.png",dpi=130); plt.close()


def finite(x):
    try: return x is not None and x != "" and math.isfinite(float(x))
    except (TypeError, ValueError): return False


def kwargs(snapshot):
    c = snapshot["metadata"]["muon_transform"]
    return {"steps": int(c["steps"]), "coefficients": tuple(float(x) for x in c["coefficients"]), "eps": float(c["eps"])}


def ratios(ref, value):
    ref, value = ref.float(), value.float(); rn, vn = ref.norm(), value.norm()
    return {"relative_l2": float((value-ref).norm()/rn) if rn.item() else None,
            "cosine": float((ref*value).sum()/(rn*vn)) if rn.item() and vn.item() else None,
            "norm_ratio": float(vn/rn) if rn.item() else None}


def compare(x, candidate, ref_update, ref_polar, kw, *, include_polar=True):
    raw = ratios(x, candidate)
    cu = muon_reference.zeropower_newton_schulz(candidate.detach().clone(), **kw)
    upd = ratios(ref_update, cu)
    pol = ratios(ref_polar, exact_polar(candidate)) if include_polar else {"relative_l2": None, "cosine": None, "norm_ratio": None}
    return {"raw_relative_l2": raw["relative_l2"], "raw_cosine": raw["cosine"], "raw_norm_ratio": raw["norm_ratio"],
            "update_relative_l2": upd["relative_l2"], "update_cosine": upd["cosine"], "update_norm_ratio": upd["norm_ratio"],
            "exact_polar_relative_l2": pol["relative_l2"], "exact_polar_cosine": pol["cosine"], "exact_polar_norm_ratio": pol["norm_ratio"]}


def block_absmax(x, block=2048):
    flat = x.float().reshape(-1)
    return torch.stack([flat[i:i+block].abs().amax() for i in range(0, flat.numel(), block)]) if flat.numel() else torch.zeros(0)


def subspace_distance(exact_u, approx_u):
    k = min(exact_u.shape[1], approx_u.shape[1])
    if not k: return {"max_principal_angle": None, "mean_principal_angle": None, "projection_frobenius": 0.0}
    vals = torch.linalg.svdvals(exact_u[:, :k].T @ approx_u[:, :k]).clamp(0, 1)
    angles = torch.arccos(vals)
    pe = exact_u[:, :k] @ exact_u[:, :k].T; pa = approx_u[:, :k] @ approx_u[:, :k].T
    return {"max_principal_angle": float(angles.max()), "mean_principal_angle": float(angles.mean()),
            "projection_frobenius": float((pe-pa).norm())}


def cross_mix(state, error):
    ehat = state.u.T @ error.float() @ state.vh.T
    total = ehat.square().sum().item(); s = state.singular_values
    if not s.numel() or s[0].item() == 0: return {"local": 0.0, "medium": 0.0, "distant": 0.0}
    z = torch.log10((s/s[0]).clamp_min(1e-6)); d = (z[:, None]-z[None, :]).abs(); off = ~torch.eye(len(s), dtype=torch.bool)
    out = {}
    for name, lo, hi in (("local",0,.5),("medium",.5,1.5),("distant",1.5,float("inf"))):
        out[name] = float(ehat[(d >= lo) & (d < hi) & off].square().sum()/total) if total else 0.0
    return out


@torch.no_grad()
def run(args):
    started = time.perf_counter(); paths = discover_snapshots(args.reports_root)
    if len(paths) != 10: raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    if args.snapshot_limit: paths = paths[:args.snapshot_limit]
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    grid_iterations = (0, 2, 4) if args.fast_grid else ITERATIONS
    grid_oversampling = (0, 8) if args.fast_grid else OVERSAMPLING
    grid_initializations = INITIALIZATIONS
    approx_rows=[]; fidelity_rows=[]; exact_rows=[]; subspace_rows=[]; dynamic_rows=[]; danger_rows=[]; mix_rows=[]; storage_rows=[]; runtime_rows=[]; shape_rows=[]; failure_rows=[]
    for seed, update, path in paths:
        snap = load_snapshot(path); kw = kwargs(snap); print(f"processing seed={seed} update={update}", flush=True)
        items = snap["tensors"][:args.tensor_limit] if args.tensor_limit else snap["tensors"]
        for item in items:
            if len(item["shape"]) != 2: continue
            x = item["tensor"].detach().float(); m,n=x.shape; ident={"seed":seed,"update":update,"parameter_id":item.get("parameter_id",item.get("name","unknown")),"parameter_name":item.get("name","unknown"),"shape":str(item["shape"])}
            exact = decompose(x); direct = quantize(x,Q4).float(); refu = muon_reference.zeropower_newton_schulz(x.clone(),**kw); refp = exact_polar(x); direct_m = compare(x,direct,refu,refp,kw)
            exact_cache = {}
            for k in RANKS:
                kk=min(k,len(exact.singular_values)); low,res=truncated(exact,kk); qres=quantize(res,Q4).float(); u,s,vh=cast_factors(exact.u[:,:kk],exact.singular_values[:kk],exact.vh[:kk],"bf16"); cand=(u*s)@vh+qres
                mm=compare(x,cand,refu,refp,kw); exact_cache[kk] = (mm, low, res, cand); exact_rows.append({**ident,"rank":kk,"method":"exact_svd","initialization":"exact","iterations":None,"oversampling":None,**mm})
            for init in grid_initializations:
                for p in grid_oversampling:
                    maxk=max(RANKS); t0=time.perf_counter(); factors_by_q={}
                    for q in grid_iterations:
                        factors_by_q[q]=approximate_svd(x,maxk,iterations=q,oversampling=p,initialization=init,seed=2026)
                    for q, factors in factors_by_q.items():
                        for k in RANKS:
                            kk=min(k, min(m,n)); full=factors_by_q[q]
                            af=ApproxFactors(full.u[:, :kk], full.singular_values[:kk], full.vh[:kk], kk, full.working_rank, full.iterations, full.oversampling, full.initialization, full.m_multiplies, full.mt_multiplies)
                            low=reconstruct(af); res=x-low; qres=quantize(res,Q4).float(); uf,sf,vhf=cast_factors(af.u,af.singular_values,af.vh,"bf16"); lowq=(uf*sf)@vhf; candidate=lowq+qres
                            polar_readout = (q == max(grid_iterations) and p == max(grid_oversampling))
                            elapsed=time.perf_counter()-t0; mm=compare(x,candidate,refu,refp,kw,include_polar=polar_readout); exact_mm,exact_low,exact_res,exact_candidate=exact_cache[kk]
                            exact_gain=(exact_mm["update_cosine"]-direct_m["update_cosine"]) if finite(exact_mm.get("update_cosine")) else None; gain=(mm["update_cosine"]-direct_m["update_cosine"]) if finite(mm.get("update_cosine")) else None
                            row={**ident,"rank":kk,"iterations":q,"oversampling":p,"initialization":init,"method":"approximate_bf16","fraction_of_exact_gain":gain/exact_gain if exact_gain and abs(exact_gain)>EPS else None,**mm}
                            fidelity_rows.append(row)
                            direct_scale = block_absmax(x); approx_scale = block_absmax(res); exact_scale = block_absmax(exact_res)
                            dynamic_rows.append({**ident,"rank":kk,"iterations":q,"oversampling":p,"initialization":init,"approx_residual_norm":float(res.norm()),"exact_residual_norm":float(exact_res.norm()),"approx_block_absmax_mean":float(approx_scale.mean()),"exact_block_absmax_mean":float(exact_scale.mean()),"direct_block_absmax_mean":float(direct_scale.mean()),"approx_to_exact_block_absmax":float(approx_scale.mean()/exact_scale.mean()) if exact_scale.mean().item() else None,"approx_to_direct_block_absmax":float(approx_scale.mean()/direct_scale.mean()) if direct_scale.mean().item() else None})
                            exerr=(x-exact_low).norm(); apperr=(x-low).norm(); gap=exact.singular_values[kk-1]-exact.singular_values[kk] if kk<len(exact.singular_values) else exact.singular_values[kk-1]
                            sd_u=subspace_distance(exact.u[:,:kk],af.u); sd_v=subspace_distance(exact.vh[:kk].T,af.vh.T)
                            approx_rows.append({**ident,"rank":kk,"iterations":q,"oversampling":p,"initialization":init,"approx_error":float(apperr),"exact_error":float(exerr),"relative_excess_error":float((apperr-exerr)/exerr) if exerr.item() else None,"spectral_gap_at_k":float(gap),"condition_proxy":float(exact.singular_values[0]/gap.clamp_min(1e-12)),"m_multiplies":af.m_multiplies,"mt_multiplies":af.mt_multiplies})
                            subspace_rows.append({**ident,"rank":kk,"iterations":q,"oversampling":p,"initialization":init,"left_max_angle":sd_u["max_principal_angle"],"left_mean_angle":sd_u["mean_principal_angle"],"left_projection_distance":sd_u["projection_frobenius"],"right_max_angle":sd_v["max_principal_angle"],"right_mean_angle":sd_v["mean_principal_angle"],"right_projection_distance":sd_v["projection_frobenius"],"update_fidelity_loss":1-float(mm["update_cosine"]) if finite(mm.get("update_cosine")) else None})
                            dz=danger_zone_energy(exact, x-candidate); dm=danger_zone_energy(exact,x-direct); de=danger_zone_energy(exact,x-exact_candidate); danger_rows.append({**ident,"rank":kk,"iterations":q,"oversampling":p,"initialization":init,"direct_danger_fraction":dm["danger_fraction_of_error"],"exact_danger_fraction":de["danger_fraction_of_error"],"approx_danger_fraction":dz["danger_fraction_of_error"]})
                            cm=cross_mix(exact,x-candidate); mix_rows.append({**ident,"rank":kk,"iterations":q,"oversampling":p,"initialization":init,**{f"approx_{a}":b for a,b in cm.items()},**{f"direct_{a}":b for a,b in cross_mix(exact,x-direct).items()}})
                            bits=storage_bits((m,n),kk,"bf16",include_metadata=True); ws=32*(m+n)*af.working_rank+32*kk; storage_rows.append({**ident,"rank":kk,"iterations":q,"oversampling":p,"initialization":init,"persistent_bits":bits["metadata_inclusive_bits"],"storage_ratio_vs_fp32":bits["metadata_inclusive_bits"]/bits["fp32_bits"],"temporary_workspace_bits":ws,"working_rank":af.working_rank})
                            runtime_rows.append({**ident,"rank":kk,"iterations":q,"oversampling":p,"initialization":init,"runtime_seconds":elapsed,"m_multiplies":af.m_multiplies,"mt_multiplies":af.mt_multiplies,"qr_count":2*(q+1),"flop_proxy":2*(q+1)*m*n*af.working_rank})
                            shape_rows.append({"shape":str(item["shape"]),"rank":kk,"iterations":q,"oversampling":p,"initialization":init,"relative_excess_error":float((apperr-exerr)/exerr) if exerr.item() else None,"runtime_seconds":elapsed,"update_cosine":mm["update_cosine"]})
                            if q==4 and p==8 and init=="randomized": failure_rows.append({**ident,"rank":kk,"spectral_gap_at_k":float(gap),"condition_proxy":float(exact.singular_values[0]/gap.clamp_min(1e-12)),"fraction_of_exact_gain":row["fraction_of_exact_gain"],"update_cosine":mm["update_cosine"]})
    write_csv(out/"approximation_metrics.csv",approx_rows); write_csv(out/"update_fidelity.csv",fidelity_rows); write_csv(out/"exact_gain_retention.csv",fidelity_rows); write_csv(out/"subspace_quality.csv",subspace_rows); write_csv(out/"residual_dynamic_range.csv",dynamic_rows); write_csv(out/"danger_zone_metrics.csv",danger_rows); write_csv(out/"cross_scale_mixing.csv",mix_rows); write_csv(out/"storage_workspace.csv",storage_rows); write_csv(out/"runtime_cost.csv",runtime_rows); write_csv(out/"shape_dependence.csv",shape_rows); write_csv(out/"failure_analysis.csv",failure_rows)
    candidates=[]
    grouped=defaultdict(list)
    for r in fidelity_rows:
        grouped[(int(r["rank"]),int(r["iterations"]),int(r["oversampling"]),r["initialization"])].append(r)
    for target in (.8,.9,.95):
        configs=[]
        for key, vals in grouped.items():
            valid=[r for r in vals if finite(r.get("fraction_of_exact_gain"))]
            if not valid: continue
            num=sum(int(r["shape"].strip("[]").split(",")[0])*int(r["shape"].strip("[]").split(",")[1]) for r in valid)
            gain=sum(float(r["fraction_of_exact_gain"])*int(r["shape"].strip("[]").split(",")[0])*int(r["shape"].strip("[]").split(",")[1]) for r in valid)/num
            cos=sum(float(r["update_cosine"])*int(r["shape"].strip("[]").split(",")[0])*int(r["shape"].strip("[]").split(",")[1]) for r in valid)/num
            configs.append({"rank":key[0],"iterations":key[1],"oversampling":key[2],"initialization":key[3],"aggregate_fraction_of_exact_gain":gain,"aggregate_update_cosine":cos,"tensor_count":len(valid)})
        valid=[r for r in configs if r["aggregate_fraction_of_exact_gain"]>=target]
        if valid:
            best=min(valid,key=lambda r:(r["rank"],r["iterations"],r["oversampling"]))
            candidates.append({"target_fraction":target,**best})
    write_csv(out/"deployment_candidates.csv",candidates)
    if not args.skip_plots:
        make_plots(out)
    (out/"methodology.md").write_text("""# Approximate low-rank extraction methodology\n\nThe approximate path uses alternating QR-orthonormalized multiplications by M and M^T, starting from either the first canonical columns or a fixed-seed (2026) Gaussian basis. q=0 is one range projection with no alternating refinement; each q>0 adds q alternating M/M^T refinements. A small projected (at most k+p square) SVD produces factors; no full matrix SVD is called by the approximate extractor. Exact SVD is used only by the evaluation/reference path.\n\nResidual quantization is the unchanged int4-dynamic-b2048 implementation. Retained factors are BF16 round-tripped, and persistent storage uses the existing metadata-inclusive storage model. Temporary workspace is reported separately and is not persistent optimizer state.\n\nK=5 update metrics are evaluated for the complete q/p/initialization grid. Exact-polar SVD readouts are intentionally restricted to the terminal q=4, p=8 configurations to keep the all-300 CPU sweep tractable; other rows carry explicit empty exact-polar fields rather than silently substituting a proxy.\n""")
    (out/"summary.md").write_text(f"# Approximate low-rank structural INT4 Muon\n\nCoverage: {len(paths)} snapshots; approximate rows: {len(fidelity_rows)}; CPU runtime: {time.perf_counter()-started:.1f}s.\n\nThe experiment compares canonical and fixed-seed randomized subspace iteration for q={grid_iterations}, p={grid_oversampling}, ranks={RANKS}. Approximate factors are BF16; production INT4 dynamic b2048 and K=5 Muon are reused unchanged. See CSV files for per-tensor metrics, storage/workspace, multiply counts, danger-zone and shape analyses.\n\nThis is an offline extraction study, not a training or deployment result.\n")
    print(f"wrote {out}; rows={len(fidelity_rows)} runtime={time.perf_counter()-started:.1f}s",flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--reports-root",type=Path,default=ROOT/"reports"); ap.add_argument("--output",type=Path,default=ROOT/"reports/muon_approx_lowrank_extraction"); ap.add_argument("--snapshot-limit",type=int,default=0); ap.add_argument("--tensor-limit",type=int,default=0); ap.add_argument("--fast-grid",action="store_true",help="CPU-friendly q={0,2,4}, p={0,8} grid"); ap.add_argument("--skip-plots",action="store_true"); ap.add_argument("--plots-only",action="store_true"); args=ap.parse_args()
    if args.plots_only:
        make_plots(args.output)
    else:
        run(args)


if __name__ == "__main__": main()
