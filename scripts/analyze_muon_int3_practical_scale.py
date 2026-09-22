#!/usr/bin/env python3
"""CPU-only practical block-scale study for conditioned INT3 Muon residuals.

All scale selection is based on residual values/MSE.  The only Muon calls are
post-selection evaluation readouts; they never influence a scale choice.
"""
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
sys.path.insert(0, str(ROOT / "src"))
from optim import muon_reference  # noqa: E402
from optim.muon_conditioned_int3_companding import (  # noqa: E402
    INT3_CODEBOOK, mulaw_inverse, mulaw_transform, nearest_levels,
)
from optim.muon_int3_scale_selection import block_diagnostics, block_scales  # noqa: E402
from optim.muon_ns_sensitivity import exact_polar  # noqa: E402
from optim.muon_quantization_aware_conditioner import BLOCK_SIZE  # noqa: E402
from optim.muon_spectral_sensitivity import quantize as production_quantize  # noqa: E402
from optim.muon_spectral_sensitivity import decompose  # noqa: E402
from optim.muon_storage_pareto import storage_bits  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
RANKS = (4, 8)
EPS = 1e-20
FIXED_FRACTIONS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
PERCENTILES = (95.0, 97.0, 98.0, 99.0, 99.5, 99.9, 100.0)
RMS_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
ROBUST_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
ORACLE_GRID = (.50,.60,.70,.80,.90,.95,1.0,1.05,1.10,1.20,1.35,1.50)
ANALYTIC_CACHE = {}


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def discover(root: Path):
    found = defaultdict(dict)
    for path in sorted(root.rglob("update_*.pt")):
        if "muon_momentum_snapshots" not in path.parts:
            continue
        try:
            snap = load_snapshot(path)
            seed = int(snap["metadata"]["seeds"]["seed"])
            update = int(snap["metadata"]["update"])
        except Exception:
            continue
        if seed in SEEDS and update in LANDMARKS:
            found[(seed, str(path.parent))][update] = path
    selected = []
    for seed in SEEDS:
        groups = sorted((key, val) for key, val in found.items()
                        if key[0] == seed and set(val) == set(LANDMARKS))
        if not groups:
            raise FileNotFoundError(f"could not find all formal snapshots for seed {seed} under {root}")
        for update in LANDMARKS:
            selected.append((seed, update, groups[0][1][update]))
    return selected


def transform_kwargs(snapshot):
    c = snapshot["metadata"]["muon_transform"]
    return {"steps": int(c["steps"]), "coefficients": tuple(float(x) for x in c["coefficients"]), "eps": float(c["eps"])}


def factorized_topk(u, s, vh, k):
    u = u[:, :k].to(torch.bfloat16).float()
    s = s[:k].to(torch.bfloat16).float()
    vh = vh[:k].to(torch.bfloat16).float()
    return (u * s) @ vh


def codebook_for_rank(k):
    # Frozen from the previous held-out study: k=4 mu=5 slightly beat global
    # Lloyd-Max; k=8 global Lloyd-Max slightly beat mu=5. No retuning here.
    if k == 4:
        return {"family": "mulaw_mu5", "codebook": INT3_CODEBOOK,
                "mu": 5.0, "levels": "implicit_mu5_inverse"}
    return {"family": "global_lloyd_max", "codebook": torch.tensor(
        [-1., -0.4198408, -0.1791116, 0., 0.1791116, 0.4198408, 1.]),
        "mu": None, "levels": "-1,-0.4198408,-0.1791116,0,0.1791116,0.4198408,1"}


def quantize_scales(value, scales, cb_info):
    """Apply fixed scale(s) and the rank-frozen scalar codebook."""
    flat = value.detach().float().reshape(-1)
    scales = scales.detach().float().reshape(-1)
    result = torch.empty_like(flat)
    cb = cb_info["codebook"].to(flat)
    for bi, start in enumerate(range(0, flat.numel(), BLOCK_SIZE)):
        block = flat[start:start + BLOCK_SIZE]
        alpha = scales[bi]
        if not block.numel() or not bool(block.abs().any()):
            result[start:start + block.numel()] = 0
        else:
            if alpha.item() <= 0:
                raise ValueError("nonzero block requires positive scale")
            z = (block / alpha).clamp(-1, 1)
            if cb_info["mu"] is not None:
                yq = nearest_levels(mulaw_transform(z, cb_info["mu"]), cb)
                q = mulaw_inverse(yq, cb_info["mu"])
            else:
                q = nearest_levels(z, cb)
            result[start:start + block.numel()] = q * alpha
    return result.reshape_as(value)


def candidate_scales(value, cb_info):
    """Return named candidate alpha vectors, each alpha being block-local."""
    x = value.detach().float().reshape(-1)
    blocks = [x[i:i + BLOCK_SIZE] for i in range(0, x.numel(), BLOCK_SIZE)]
    names, scales = [], []
    families = {
        "fixed_absmax_fraction": [(f"c={c:.2f}", "absmax", c, None) for c in FIXED_FRACTIONS],
        "percentile": [(f"p={p:g}", "percentile", 1.0, p) for p in PERCENTILES],
        "rms": [(f"c={c:g}", "rms", c, None) for c in RMS_MULTIPLIERS],
        "std": [(f"c={c:g}", "std", c, None) for c in RMS_MULTIPLIERS],
        "median_abs": [(f"c={c:g}", "median_abs", c, None) for c in ROBUST_MULTIPLIERS],
        "mad": [(f"c={c:g}", "mad", c, None) for c in ROBUST_MULTIPLIERS],
    }
    for family, rules in families.items():
        for label, method, mult, pct in rules:
            alpha = block_scales(value, method, multiplier=mult,
                                 percentile=pct if pct is not None else 100.0,
                                 block_size=BLOCK_SIZE)
            names.append((family, label)); scales.append(alpha)
    # Precomputed model-fit multipliers: deterministic numerical integration
    # for the fixed selected codebook/compander, never per-block optimization.
    cache_key=(cb_info["family"],cb_info["mu"],tuple(float(v) for v in cb_info["codebook"].tolist()))
    if cache_key not in ANALYTIC_CACHE:
        ANALYTIC_CACHE[cache_key]=distribution_clip_multipliers(cb_info)
    gaussian_mult, laplace_mult = ANALYTIC_CACHE[cache_key]
    for family, method, multiplier in (("analytic_gaussian", "rms", gaussian_mult),
                                       ("analytic_laplacian", "median_abs", laplace_mult)):
        names.append((family, f"multiplier={multiplier:.5g}"))
        scales.append(block_scales(value, method, multiplier=multiplier, block_size=BLOCK_SIZE))
    return names, torch.stack(scales) if scales else torch.empty((0, 0))


def distribution_clip_multipliers(cb_info):
    """One-time deterministic expected-MSE scale fit for N(0,1) and Laplace."""
    cb = cb_info["codebook"]
    mu = cb_info["mu"]
    # Midpoint quantiles avoid RNG and represent each distribution equally.
    n = 32768
    p = (torch.arange(n, dtype=torch.float64) + .5) / n
    normal = (math.sqrt(2.0) * torch.erfinv(2 * p - 1)).float()
    laplace = torch.where(p < .5, torch.log(2 * p), -torch.log(2 * (1 - p))).float()
    values = {"gaussian": normal, "laplacian": laplace}
    result = []
    for family, x in values.items():
        base = x.square().mean().sqrt() if family == "gaussian" else x.abs().median()
        best = (math.inf, None)
        for multiplier in torch.linspace(.75, 6.0, 106).tolist():
            alpha = base * multiplier
            z = (x / alpha).clamp(-1, 1)
            if mu is not None:
                q = mulaw_inverse(nearest_levels(mulaw_transform(z, mu), cb), mu) * alpha
            else:
                q = nearest_levels(z, cb) * alpha
            mse = float((q - x).square().mean())
            if mse < best[0]: best = (mse, multiplier)
        result.append(float(best[1]))
    return tuple(result)


def batched_candidate_mse(value, scales, cb_info, candidate_chunk=4):
    """Evaluate many alpha rules at once; returns total SSE per candidate."""
    x = value.detach().float().reshape(-1)
    nblocks = (x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = torch.zeros((nblocks, BLOCK_SIZE), dtype=torch.float32, device=x.device)
    for i, start in enumerate(range(0, x.numel(), BLOCK_SIZE)):
        b = x[start:start + BLOCK_SIZE]; padded[i, :b.numel()] = b
    errors = []
    cb = cb_info["codebook"].to(x)
    for first in range(0, scales.shape[0], candidate_chunk):
        alpha = scales[first:first + candidate_chunk].to(x.device).unsqueeze(-1)
        z = (padded.unsqueeze(0) / alpha.clamp_min(torch.finfo(torch.float32).tiny)).clamp(-1, 1)
        if cb_info["mu"] is not None:
            yq = nearest_levels(mulaw_transform(z, cb_info["mu"]), cb)
            q = mulaw_inverse(yq, cb_info["mu"]) * alpha
        else:
            q = nearest_levels(z, cb) * alpha
        errors.extend((q - padded.unsqueeze(0)).square().sum(dim=(1, 2)).cpu().tolist())
    return errors


def local_mse_search(value, cb_info, fractions):
    """Block-local MSE scale grid (no Muon metric enters the choice)."""
    flat=value.detach().float().reshape(-1)
    nb=math.ceil(flat.numel()/BLOCK_SIZE); padded=torch.zeros((nb,BLOCK_SIZE),dtype=torch.float32,device=flat.device)
    for bi,start in enumerate(range(0,flat.numel(),BLOCK_SIZE)):
        b=flat[start:start+BLOCK_SIZE]; padded[bi,:b.numel()]=b
    maxima=padded.abs().amax(dim=1); fracs=sorted(set(float(v) for v in fractions))
    alpha_grid=torch.tensor(fracs,dtype=torch.float32,device=flat.device)[:,None]*maxima[None,:]
    cb=cb_info["codebook"].to(flat); mse=torch.empty_like(alpha_grid)
    for ci in range(alpha_grid.shape[0]):
        alpha=alpha_grid[ci].clamp_min(torch.finfo(torch.float32).tiny)[:,None]
        z=(padded/alpha).clamp(-1,1)
        if cb_info["mu"] is not None:
            q=mulaw_inverse(nearest_levels(mulaw_transform(z,cb_info["mu"]),cb),cb_info["mu"])*alpha
        else:q=nearest_levels(z,cb)*alpha
        mse[ci]=(q-padded).square().sum(dim=1)
    best=mse.argmin(dim=0); alphas=alpha_grid.gather(0,best[None,:]).squeeze(0)
    alphas=torch.where(maxima==0,torch.zeros_like(alphas),alphas)
    q=quantize_scales(value,alphas,cb_info)
    errs=mse.gather(0,best[None,:]).squeeze(0)
    return q,alphas,errs.tolist()


def coarse_to_fine(value, cb_info):
    """At most five scales/block: coarse 0.6/0.8/1.0 then two refinements."""
    x = value.detach().float().reshape(-1); q = torch.empty_like(x); alphas=[]; counts=[]
    for start in range(0, x.numel(), BLOCK_SIZE):
        b = x[start:start+BLOCK_SIZE]; m = float(b.abs().amax())
        if m == 0:
            q[start:start+b.numel()] = 0; alphas.append(0.0); counts.append(0); continue
        candidates = [m*f for f in (.6,.8,1.0)]
        scored=[]
        for a in candidates:
            qq=quantize_scales(b,torch.tensor([a]),cb_info); scored.append((float((qq-b).square().sum()),a,qq))
        _, center, _ = min(scored,key=lambda t:(t[0],t[1]))
        for a in sorted(set((center*.9,center*1.1))):
            qq=quantize_scales(b,torch.tensor([a]),cb_info); scored.append((float((qq-b).square().sum()),a,qq))
        err,alpha,qq=min(scored,key=lambda t:(t[0],t[1]))
        q[start:start+b.numel()]=qq; alphas.append(alpha); counts.append(5)
    return q.reshape_as(value),torch.tensor(alphas),counts


def quant_metrics(ref, estimate, prefix):
    a=ref.detach().float(); b=estimate.detach().float(); an=a.norm(); bn=b.norm()
    return {f"{prefix}_relative_l2":float((b-a).norm()/an) if an.item() else None,
            f"{prefix}_cosine":float((a*b).sum()/(an*bn)) if an.item() and bn.item() else None,
            f"{prefix}_norm_ratio":float(bn/an) if an.item() else None}


def block_step_count(tensor):
    return math.ceil(tensor.numel()/BLOCK_SIZE)


def make_row(seed, update, name, item, k, method, cb_info, residual, c_hat,
             ref_update, ref_polar, transform, *, q=None, scales=None,
             candidate_count=None, q4_metrics=False, exact_polar_metrics=False):
    row_started=time.perf_counter()
    if q is None:
        q=quantize_scales(residual,scales,cb_info)
    estimate=c_hat+q
    row={"seed":seed,"update":update,"parameter_id":item.get("parameter_id",item.get("name","unknown")),
         "parameter_name":item.get("name",item.get("parameter_id","unknown")),"shape":str(tuple(item["shape"])),
         "numel":item["tensor"].numel(),
         "k":k,"codebook_family":cb_info["family"],"method":method,
         "candidate_scales_per_block":candidate_count or 1,
         "number_of_blocks":block_step_count(residual),
         "storage_ratio_vs_fp32":storage_bits(item["shape"],k,"bf16",include_metadata=True,residual_bits=3)["metadata_inclusive_bits"]/(32*item["tensor"].numel()),
         "residual_relative_l2":float((q-residual).norm()/residual.norm()) if residual.norm().item() else None,
         "residual_cosine":float((q*residual).sum()/(q.norm()*residual.norm())) if q.norm().item() and residual.norm().item() else None,
         "residual_mse":float((q-residual).square().mean()),
         **quant_metrics(item["tensor"],estimate,"full_state_raw")}
    abs_scales=block_scales(residual,"absmax")
    row.update({"scale_over_absmax_mean":float((scales/abs_scales.clamp_min(1e-20)).mean()) if scales.numel() else None,
                "scale_over_absmax_median":float((scales/abs_scales.clamp_min(1e-20)).median()) if scales.numel() else None,
                "scale_mean":float(scales.mean()) if scales.numel() else None,
                "block_absmax_mean":float(abs_scales.mean()) if abs_scales.numel() else None})
    diagnostics=block_diagnostics(residual,q,scales,block_size=BLOCK_SIZE)
    row.update(diagnostics)
    out_update=transform(estimate)
    row.update(quant_metrics(ref_update,out_update,"update"))
    if exact_polar_metrics and ref_polar is not None:
        row.update(quant_metrics(ref_polar,exact_polar(estimate),"exact_polar"))
    if q4_metrics:
        q4=production_quantize(residual,"int4-dynamic-b2048")
        est4=c_hat+q4
        row.update({"structural_int4_update_cosine":quant_metrics(ref_update,transform(est4),"q4")["q4_cosine"],
                    "structural_int4_relative_storage":storage_bits(item["shape"],k,"bf16",include_metadata=True,residual_bits=4)["metadata_inclusive_bits"]/(32*item["tensor"].numel())})
    row.update({"estimated_block_passes":candidate_count or 1,"estimated_scale_reductions":candidate_count or 1,
                "scale_scalar_metadata_count":int(scales.numel()),"evaluation_wall_seconds":time.perf_counter()-row_started})
    return row,estimate


def chosen_rules(seed0_sse, global_meta):
    choices={}
    for k, totals in seed0_sse.items():
        for family, entries in totals.items():
            label=min(entries,key=lambda key:(entries[key],key))
            choices[(k,family)]=label
    return choices


def plot_outputs(outdir, rows, heldout, candidate_rows, temporal_rows, warm_rows, spectral_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plots skipped: matplotlib unavailable ({exc})",flush=True); return
    methods=defaultdict(list)
    for r in heldout: methods[r["method"]].append(r)
    means={m:sum(float(x["update_cosine"]) for x in v)/len(v) for m,v in methods.items() if v}
    names=sorted(means,key=means.get)
    fig,ax=plt.subplots(figsize=(12,6)); ax.barh(names,[means[n] for n in names]); ax.set_xlabel("K=5 update cosine (mean)"); fig.tight_layout(); fig.savefig(outdir/"scale_method_update_cosine.png",dpi=160); plt.close(fig)
    baseline={r["parameter_id"]+str(r["update"])+str(r["k"]):r["update_cosine"] for r in heldout if r["method"]=="absmax"}
    oracle={r["parameter_id"]+str(r["update"])+str(r["k"]):r["update_cosine"] for r in heldout if r["method"]=="mse_oracle"}
    rec=[]
    for m,v in methods.items():
        vals=[]
        for r in v:
            key=r["parameter_id"]+str(r["update"])+str(r["k"])
            den=1-float(baseline.get(key,0)); num=float(r["update_cosine"])-float(baseline.get(key,0)); oracle_gap=float(oracle.get(key,0))-float(baseline.get(key,0))
            if oracle_gap>1e-9: vals.append(num/oracle_gap)
        if vals: rec.append((m,statistics.mean(vals)))
    rec.sort(key=lambda z:z[1]); fig,ax=plt.subplots(figsize=(10,5)); ax.barh([x[0] for x in rec],[x[1] for x in rec]); ax.set_xlabel("MSE-oracle scale headroom recovery"); fig.tight_layout(); fig.savefig(outdir/"oracle_headroom_recovery.png",dpi=160); plt.close(fig)
    for family,filename,xlabel in (("percentile","percentile_vs_residual_mse.png","percentile"),("rms","rms_multiplier_vs_residual_mse.png","RMS multiplier")):
        fam=[r for r in candidate_rows.get(family,[]) if r["seed"]==1]; grouped=defaultdict(list)
        for r in fam:
            try: grouped[float(r["candidate"].split("=")[1])].append(float(r["residual_mse"]))
            except Exception: pass
        if grouped:
            xs=sorted(grouped); ys=[statistics.mean(grouped[x]) for x in xs]
            fig,ax=plt.subplots(figsize=(7,4)); ax.plot(xs,ys,marker="o"); ax.set_xlabel(xlabel); ax.set_ylabel("held-out mean residual MSE"); fig.tight_layout(); fig.savefig(outdir/filename,dpi=160); plt.close(fig)
    for ykey,filename,ylabel in (("clipping_fraction","clipping_vs_update_cosine.png","clipping fraction"),("zero_fraction","zero_fraction_vs_update_cosine.png","mapped-zero fraction")):
        fig,ax=plt.subplots(figsize=(7,5))
        for method,vals in methods.items():
            points=[r for r in vals if r.get(ykey) is not None]
            if points: ax.scatter([float(r[ykey]) for r in points],[float(r["update_cosine"]) for r in points],s=7,label=method,alpha=.45)
        ax.set_xlabel(ylabel); ax.set_ylabel("K=5 update cosine"); ax.legend(fontsize=5,ncol=2); fig.tight_layout(); fig.savefig(outdir/filename,dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for method,vals in methods.items():
        if vals: ax.scatter([r["scale_over_absmax_mean"] for r in vals],[r["update_cosine"] for r in vals],s=7,alpha=.45,label=method)
    ax.set_xlabel("mean selected alpha / block absmax"); ax.set_ylabel("K=5 update cosine"); ax.legend(fontsize=5,ncol=2); fig.tight_layout(); fig.savefig(outdir/"selected_scale_ratio_vs_fidelity.png",dpi=160); plt.close(fig)
    # Block-scale distribution and direct practical-vs-grid-oracle comparison.
    fig,ax=plt.subplots(figsize=(8,5))
    for method in ("absmax","percentile:p=98","local_grid_4","mse_oracle"):
        vals=[float(r["scale_over_absmax_mean"]) for r in heldout if r["method"]==method and r.get("scale_over_absmax_mean") is not None]
        if vals: ax.hist(vals,bins=35,histtype="step",density=True,label=method)
    ax.set_xlabel("selected alpha / block absmax"); ax.set_ylabel("density across tensors"); ax.legend(); fig.tight_layout(); fig.savefig(outdir/"selected_scale_oracle_distributions.png",dpi=160); plt.close(fig)
    scale_lookup={(r["seed"],r["update"],r["parameter_id"],r["k"],r["method"]):float(r["scale_over_absmax_mean"]) for r in heldout if r.get("scale_over_absmax_mean") not in (None,"")}
    fig,ax=plt.subplots(figsize=(6,6))
    for method in ("percentile:p=98","rms:c=2.5","local_grid_4","coarse_to_fine"):
        pts=[]
        for r in heldout:
            if r["method"]!=method:continue
            key=(r["seed"],r["update"],r["parameter_id"],r["k"],"mse_oracle")
            if key in scale_lookup: pts.append((scale_lookup[(r["seed"],r["update"],r["parameter_id"],r["k"],method)],scale_lookup[key]))
        if pts: ax.scatter([a for a,b in pts],[b for a,b in pts],s=8,label=method,alpha=.5)
    ax.plot([0,1.6],[0,1.6],"k--",lw=1); ax.set_xlabel("practical alpha / absmax"); ax.set_ylabel("matched-codebook MSE-grid alpha / absmax"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(outdir/"practical_scale_vs_oracle_scale.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter([r["candidate_scales_per_block"] for r in heldout],[r["update_cosine"] for r in heldout],s=9,alpha=.4); ax.set_xlabel("scale candidates per block"); ax.set_ylabel("K=5 update cosine"); fig.tight_layout(); fig.savefig(outdir/"candidate_count_vs_fidelity.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter([r["candidate_scales_per_block"] for r in heldout],[r["evaluation_wall_seconds"] for r in heldout],s=9,alpha=.4); ax.set_xlabel("scale candidates per block"); ax.set_ylabel("per-record CPU quantize+Muon evaluation seconds (search excluded)"); fig.tight_layout(); fig.savefig(outdir/"candidate_count_vs_cpu_time.png",dpi=160); plt.close(fig)
    if temporal_rows:
        fig,ax=plt.subplots(figsize=(7,5)); ax.hist([r["mean_abs_log_drift"] for r in temporal_rows],bins=30); ax.set_xlabel("mean absolute log alpha drift between landmarks"); ax.set_ylabel("tensor transitions"); fig.tight_layout(); fig.savefig(outdir/"temporal_scale_drift.png",dpi=160); plt.close(fig)
    if warm_rows:
        wg=defaultdict(list)
        for r in warm_rows: wg[r["method"]].append(float(r["residual_sse"]))
        fig,ax=plt.subplots(figsize=(7,4)); ns=sorted(wg); ax.bar(ns,[statistics.mean(wg[n]) for n in ns]); ax.set_ylabel("mean residual SSE (matched tensors)"); fig.tight_layout(); fig.savefig(outdir/"warmstart_vs_fresh_search.png",dpi=160); plt.close(fig)
        wg_cos=defaultdict(list)
        for r in heldout:
            if r["method"] in {"warmstart_previous_oracle_3","fresh_local_3"}: wg_cos[r["method"]].append(float(r["update_cosine"]))
        if wg_cos:
            ns=sorted(wg_cos); fig,ax=plt.subplots(figsize=(7,4)); ax.bar(ns,[statistics.mean(wg_cos[n]) for n in ns]); ax.set_ylabel("held-out mean K=5 update cosine"); fig.tight_layout(); fig.savefig(outdir/"warmstart_fidelity_vs_fresh.png",dpi=160); plt.close(fig)
    # The global percentile/RMS candidates are MSE-selected on seed 0; their
    # held-out fidelity is shown only for the selected global parameter.
    selected_global=[r for r in heldout if r["method"].startswith(("percentile:","rms:","std:","fixed_absmax_fraction:"))]
    if selected_global:
        grouped=defaultdict(list)
        for r in selected_global: grouped[(int(r["k"]),r["method"].split(":")[0])].append(float(r["update_cosine"]))
        labels=[f"k{k} {m}" for k,m in sorted(grouped)]; vals=[statistics.mean(grouped[x]) for x in sorted(grouped)]
        fig,ax=plt.subplots(figsize=(9,4)); ax.bar(labels,vals); ax.set_ylabel("held-out K=5 update cosine (seed-0 MSE selection)"); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(outdir/"selected_global_rule_fidelity.png",dpi=160); plt.close(fig)
    if spectral_rows:
        lookup={(r["seed"],r["update"],r["parameter_id"],r["k"],r["method"]):r for r in heldout}
        fig,ax=plt.subplots(figsize=(7,5))
        for method in sorted({r["method"] for r in spectral_rows}):
            g=[r for r in spectral_rows if r["method"]==method and r["seed"]==1]
            matched=[(r,lookup.get((r["seed"],r["update"],r["parameter_id"],r["k"],method))) for r in g]
            matched=[(a,b) for a,b in matched if b]
            if matched: ax.scatter([a["danger_associated_fraction_error"] for a,b in matched],[b["update_cosine"] for a,b in matched],s=8,label=method,alpha=.4)
        ax.set_xlabel("danger-zone associated residual error fraction"); ax.set_ylabel("K=5 update cosine"); ax.legend(); fig.tight_layout(); fig.savefig(outdir/"danger_zone_error_vs_fidelity.png",dpi=160); plt.close(fig)
    polar_path=outdir/"exact_polar_best_global_practical.csv"
    if polar_path.exists():
        polar_rows=typed_csv(polar_path)
        fig,ax=plt.subplots(figsize=(7,5))
        for k in (4,8):
            g=[r for r in polar_rows if int(r["k"])==k]
            if g: ax.hist([float(r["exact_polar_cosine"]) for r in g],bins=30,histtype="step",density=True,label=f"p98 k={k}")
        ax.set_xlabel("exact-polar cosine"); ax.set_ylabel("density"); ax.legend(); fig.tight_layout(); fig.savefig(outdir/"exact_polar_best_global_distribution.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for method in sorted({r["method"] for r in heldout}):
        g=[r for r in heldout if r["method"]==method]
        if g and g[0].get("storage_ratio_vs_fp32") is not None:
            ax.scatter([g[0]["storage_ratio_vs_fp32"]],[statistics.mean(float(r["update_cosine"]) for r in g)],label=method)
    ax.set_xlabel("idealized metadata-inclusive storage / FP32"); ax.set_ylabel("held-out mean K=5 cosine"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(outdir/"storage_fidelity_structural_int4.png",dpi=160); plt.close(fig)


def run(args):
    if args.replot_only:
        replot_existing_report(Path(args.outdir)); return
    if args.polar_best_global:
        evaluate_best_global_polar(args.reports_root,args.outdir); return
    if args.spectral_best_global:
        evaluate_best_global_spectral(args.reports_root,args.outdir); return
    torch.set_num_threads(args.threads)
    root=Path(args.reports_root); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    paths=discover(root)
    if args.limit_snapshots: paths=[p for p in paths if p[1] in args.limit_snapshots]
    snapshots=[]
    for seed,update,path in paths:
        snap=load_snapshot(path)
        snapshots.append((seed,update,path,snap))
    start_time=time.perf_counter()
    calibration=defaultdict(lambda:defaultdict(float)); candidate_manifest={}; counts=defaultdict(int)
    # Calibration pass: only seed 0, and only residual MSE is used to choose
    # global rules. Scale candidates are evaluated in vectorized chunks.
    for seed,update,path,snap in snapshots:
        if seed!=0: continue
        matrix_index=0
        for item in snap["tensors"]:
            if len(item["shape"])!=2 or min(item["shape"])<min(RANKS): continue
            if args.limit_tensors and matrix_index>=args.limit_tensors: break
            matrix_index+=1
            x=item["tensor"].detach().float(); state=decompose(x)
            for k in RANKS:
                residual=x-(state.u[:,:k]*state.singular_values[:k])@state.vh[:k]
                cb=codebook_for_rank(k); names,alphas=candidate_scales(residual,cb)
                errs=batched_candidate_mse(residual,alphas,cb)
                for (family,label),err in zip(names,errs):
                    calibration[k][family,label]+=err
                    candidate_manifest[(k,family,label)]=True
            counts[(seed,update)]+=1
        print(f"calibration seed=0 update={update}: {counts[(seed,update)]} matrices",flush=True)
    choices={}
    for k,familylabel in ((k,key) for k in RANKS for key in {f for kk,f,_ in candidate_manifest if kk==k}):
        labels=[label for kk,f,label in candidate_manifest if kk==k and f==familylabel]
        best=min(labels,key=lambda label:(calibration[k][(familylabel,label)],label))
        choices[(k,familylabel)]=best
    # Save calibration curve with seed-0 objective values only.
    calibration_rows=[]
    for (k,family,label),err in sorted(((k,f,l),v) for k,d in calibration.items() for (f,l),v in d.items()):
        calibration_rows.append({"k":k,"family":family,"candidate":label,"seed0_total_residual_sse":err,
                                 "selected_on_seed0_mse":choices.get((k,family))==label})
    write_csv(outdir/"global_scale_calibration.csv",calibration_rows)

    # Cache model-fit analytic multipliers for each rank's frozen codebook.
    cb_by_k={k:codebook_for_rank(k) for k in RANKS}
    analytic={k:distribution_clip_multipliers(cb_by_k[k]) for k in RANKS}
    for k in RANKS:
        ci=cb_by_k[k]; ANALYTIC_CACHE[(ci["family"],ci["mu"],tuple(float(v) for v in ci["codebook"].tolist()))]=analytic[k]
    results=[]; temporal_traces=defaultdict(list); warm_rows=[]; spectral_rows=[]; storage_rows=[]; oracle_rows=[]
    all_rule_rows=defaultdict(list); total_matrices=0; eval_started=time.perf_counter()
    previous_scales={}
    for seed,update,path,snap in snapshots:
        kwargs=transform_kwargs(snap)
        matrix_index=0
        for item in snap["tensors"]:
            if len(item["shape"])!=2 or min(item["shape"])<min(RANKS): continue
            if args.limit_tensors and matrix_index>=args.limit_tensors: break
            matrix_index+=1
            x=item["tensor"].detach().float(); state=decompose(x); ref_update=muon_reference.zeropower_newton_schulz(x.clone(),**kwargs)
            try: ref_polar=exact_polar(x)
            except RuntimeError: ref_polar=None
            total_matrices+=1
            for k in RANKS:
                c_exact=(state.u[:,:k]*state.singular_values[:k])@state.vh[:k]
                residual=x-c_exact; c_hat=factorized_topk(state.u,state.singular_values,state.vh,k)
                cb=cb_by_k[k]; names,scales_matrix=candidate_scales(residual,cb)
                alphas_by_name={name:scales_matrix[i] for i,name in enumerate(names)}
                # Candidate MSE on evaluation is reported but never used for a
                # seed-1 choice. MSE candidate pass does not call Muon.
                candidate_errors=batched_candidate_mse(residual,scales_matrix,cb)
                for (family,label),err,alpha in zip(names,candidate_errors,scales_matrix):
                    all_rule_rows[family].append({"seed":seed,"update":update,"parameter_id":item.get("parameter_id",item.get("name")),"k":k,"candidate":label,"residual_mse":err/residual.numel(),"selected_on_seed0":choices.get((k,family))==label})
                # Primary scale rules: one MSE-calibrated global candidate per
                # family, plus fixed reference and all three local grids.
                selected=[]
                alpha_abs=block_scales(residual,"absmax")
                selected.append(("absmax",alpha_abs,1,True))
                for family,outname in (("fixed_absmax_fraction","fixed_absmax_fraction"),("percentile","percentile"),("rms","rms"),("std","std"),("median_abs","robust_median"),("mad","robust_mad")):
                    label=choices[(k,family)]; selected.append((f"{outname}:{label}",alphas_by_name[(family,label)],1,False))
                for name,grid in (("local_grid_2",(.8,1.0)),("local_grid_3",(.75,.9,1.0)),("local_grid_4",(.6,.75,.9,1.0))):
                    q,alpha,errlist=local_mse_search(residual,cb,grid); selected.append((name,alpha,len(grid),False,q))
                qcf,ac,fcounts=coarse_to_fine(residual,cb); selected.append(("coarse_to_fine",ac,5,False,qcf))
                # Analytic model fits; the multiplier was computed once using
                # deterministic Gaussian/Laplace quadrature for this codebook.
                for family,method,mult in (("analytic_gaussian","rms",analytic[k][0]),("analytic_laplacian","median_abs",analytic[k][1])):
                    alpha=block_scales(residual,method,multiplier=mult); selected.append((f"{family}:m={mult:.4g}",alpha,1,False))
                q_oracle,alpha_oracle,oracle_err=local_mse_search(residual,cb,ORACLE_GRID); selected.append(("mse_oracle",alpha_oracle,len(ORACLE_GRID),False,q_oracle))
                # Uniform reference uses absmax and the original signed 7-level codebook.
                uniform_cb={"family":"uniform_int3_reference","codebook":INT3_CODEBOOK,"mu":None}
                selected.append(("uniform_int3_reference",alpha_abs,1,True,None,uniform_cb))
                q_prior_oracle,alpha_prior_oracle,_=local_mse_search(residual,uniform_cb,ORACLE_GRID)
                selected.append(("prior_uniform_mse_oracle",alpha_prior_oracle,len(ORACLE_GRID),False,q_prior_oracle,uniform_cb))
                # Temporal warm-start: previous landmark's oracle alpha, then
                # three block-local candidates around it. This uses current
                # residual MSE only, with naturally aligned parameter/block ids.
                key=(seed,k,item.get("parameter_id",item.get("name")),str(tuple(x.shape)))
                prev=previous_scales.get(key)
                if prev is not None and prev.numel()==alpha_oracle.numel():
                    # Explicit per-block warm-start multipliers, not max-based.
                    qw=torch.empty_like(residual.reshape(-1)); aw=[]; total_err=[]
                    flat=residual.reshape(-1)
                    for bi,bstart in enumerate(range(0,flat.numel(),BLOCK_SIZE)):
                        block=flat[bstart:bstart+BLOCK_SIZE]; base=float(prev[bi])
                        if not base or not bool(block.abs().any()):
                            base=float(block.abs().amax())
                        cans=sorted(set(base*z for z in (.8,1.0,1.2)))
                        opts=[]
                        for aa in cans:
                            qq=quantize_scales(block,torch.tensor([aa]),cb); opts.append((float((qq-block).square().sum()),aa,qq))
                        er,aa,qq=min(opts,key=lambda v:(v[0],v[1])); qw[bstart:bstart+block.numel()]=qq; aw.append(aa); total_err.append(er)
                    qw=qw.reshape_as(residual); aw=torch.tensor(aw); previous_alpha=alpha_oracle
                    selected.append(("warmstart_previous_oracle_3",aw,3,False,qw))
                    # Fresh local 3-candidate search provides the matched comparison.
                    fresh_q,fresh_alpha,_=local_mse_search(residual,cb,(.75,.9,1.0))
                    selected.append(("fresh_local_3",fresh_alpha,3,False,fresh_q))
                    for label,aa,qq in (("warmstart_previous_oracle_3",aw,qw),("fresh_local_3",fresh_alpha,fresh_q)):
                        err=float((qq-residual).square().sum())
                        warm_rows.append({"seed":seed,"update":update,"parameter_id":key[2],"k":k,"method":label,"residual_sse":err,
                                          "candidate_evaluations_per_block":3,"previous_scale_available":True})
                previous_scales[key]=alpha_oracle.detach().clone()
                temporal_traces[key].append((update,alpha_oracle.detach().clone()))
                # Calculate fidelity for selected scales only. Exact polar is
                # emitted for a compact predeclared representative set.
                polar_names={"absmax","mse_oracle","local_grid_4"}
                for entry in selected:
                    name,alpha,nc,reference_scale,*rest=entry
                    use_cb=rest[1] if len(rest)>1 else cb
                    q=rest[0] if rest and isinstance(rest[0],torch.Tensor) else None
                    if name=="uniform_int3_reference" and q is None:
                        q=quantize_scales(residual,alpha,use_cb)
                    row,estimate=make_row(seed,update,name,item,k,name,use_cb,residual,c_hat,ref_update,
                                          ref_polar,lambda a:muon_reference.zeropower_newton_schulz(a.clone(),**kwargs),
                                          q=q,scales=alpha,candidate_count=nc,
                                          exact_polar_metrics=name.split(":")[0] in polar_names)
                    row["source_seed_role"]="calibration" if seed==0 else "held_out"
                    row["transform_steps"]=kwargs["steps"]
                    results.append(row)
                    if name=="mse_oracle":
                        oracle_rows.append({**row,"scale_over_absmax_mean":float((alpha/(block_scales(residual,"absmax").clamp_min(1e-20))).mean())})
                # Production INT4 reference uses the unchanged quantizer on
                # the same residual and conditioner; no INT4 scale rule is changed.
                q4=production_quantize(residual,"int4-dynamic-b2048").float(); est4=c_hat+q4
                q4_start=time.perf_counter()
                q4polar=quant_metrics(ref_polar,exact_polar(est4),"exact_polar") if ref_polar is not None else {}
                q4row={"seed":seed,"update":update,"parameter_id":key[2],"parameter_name":item.get("name",key[2]),
                       "shape":str(tuple(x.shape)),"numel":x.numel(),"k":k,"method":"structural_int4_reference",
                       "codebook_family":"production_dynamic_int4","candidate_scales_per_block":1,
                       "number_of_blocks":block_step_count(residual),**quant_metrics(x,est4,"full_state_raw"),
                       "residual_relative_l2":float((q4-residual).norm()/residual.norm()) if residual.norm().item() else None,
                       "residual_cosine":float((q4*residual).sum()/(q4.norm()*residual.norm())) if q4.norm().item() and residual.norm().item() else None,
                       "residual_mse":float((q4-residual).square().mean()),
                       "zero_fraction":float((q4==0).float().mean()),"clipping_fraction":None,
                       "mean_absolute_error":float((q4-residual).abs().mean()),"scale_over_absmax_mean":1.0,
                       "scale_over_absmax_median":1.0,"scale_mean":float(block_scales(residual,"absmax").mean()),
                       "block_absmax_mean":float(block_scales(residual,"absmax").mean()),
                       **quant_metrics(ref_update,muon_reference.zeropower_newton_schulz(est4.clone(),**kwargs),"update"),**q4polar,
                       "evaluation_wall_seconds":time.perf_counter()-q4_start,"estimated_block_passes":1,
                       "estimated_scale_reductions":1,"scale_scalar_metadata_count":block_step_count(residual),
                       "storage_ratio_vs_fp32":storage_bits(item["shape"],k,"bf16",include_metadata=True,residual_bits=4)["metadata_inclusive_bits"]/(32*x.numel())}
                results.append(q4row)
                    # Store baseline and strongest practical for plots/summaries.
                # Spectral diagnostics are secondary and computed for a subset
                # of representative scale methods to limit extra projections.
                from optim.muon_conditioned_int3_companding import spectral_error_metrics
                for row in results[-len(selected):]:
                    if row["method"] not in {"absmax","mse_oracle","local_grid_4"}: continue
                    name=row["method"]; match=next(e for e in selected if e[0]==name)
                    _,alpha,_,_,*rest=match; use_cb=rest[1] if len(rest)>1 else cb
                    q=rest[0] if rest and isinstance(rest[0],torch.Tensor) else quantize_scales(residual,alpha,use_cb)
                    spectral_rows.append({"seed":seed,"update":update,"parameter_id":key[2],"k":k,"method":name,
                                          **spectral_error_metrics(state.u,state.singular_values,state.vh,q-residual)})
                # Storage does not depend on the scale rule: one FP32 scale per
                # b2048 residual block plus BF16 low-rank factors and metadata.
                st=storage_bits(item["shape"],k,"bf16",include_metadata=True,residual_bits=3)
                storage_rows.append({"seed":seed,"update":update,"parameter_id":key[2],"k":k,**st,
                                     "storage_ratio_vs_fp32":st["metadata_inclusive_bits"]/st["fp32_bits"],
                                     "compression_vs_fp32":st["fp32_bits"]/st["metadata_inclusive_bits"],
                                     "scale_metadata_bits":32*block_step_count(residual),"methods_share_same_storage":True})
            print(f"evaluated seed={seed} update={update}: matrices={matrix_index}",flush=True)

    # CSV families: requested deliverable names plus a common tensor-level table.
    write_csv(outdir/"tensor_level_results.csv",results)
    write_csv(outdir/"absmax_baseline.csv",[r for r in results if r["method"]=="absmax"])
    name_to_file={"fixed_absmax_fraction":"fixed_absmax_fraction.csv","percentile":"percentile_results.csv","rms":"rms_results.csv","std":"std_results.csv","median_abs":"robust_scale_results.csv","mad":"robust_scale_results.csv","analytic_gaussian":"analytic_distribution_results.csv","analytic_laplacian":"analytic_distribution_results.csv"}
    for family,filename in name_to_file.items():
        rows=list(all_rule_rows.get(family,[]))
        if rows:
            write_csv(outdir/filename,rows)
    write_csv(outdir/"local_grid_results.csv",[r for r in results if r["method"].startswith("local_grid") or r["method"]=="fresh_local_3"])
    write_csv(outdir/"coarse_to_fine_results.csv",[r for r in results if r["method"]=="coarse_to_fine"])
    write_csv(outdir/"oracle_comparison.csv",oracle_rows)
    write_csv(outdir/"warmstart_scale_results.csv",warm_rows)
    write_csv(outdir/"spectral_diagnostics.csv",spectral_rows)
    write_csv(outdir/"storage_summary.csv",storage_rows)
    global_storage=[]
    grouped_storage=defaultdict(list)
    for row in storage_rows: grouped_storage[(row["seed"],row["update"],row["k"])].append(row)
    for (seed,update,k),group in sorted(grouped_storage.items()):
        total_bits=sum(int(r["metadata_inclusive_bits"]) for r in group); fpbits=sum(int(r["fp32_bits"]) for r in group)
        int4bits=sum(int(r["direct_int4_metadata_bits"]) for r in group)
        global_storage.append({"seed":seed,"update":update,"k":k,"tensor_count":len(group),"structural_int3_metadata_bits":total_bits,
                               "fp32_bits":fpbits,"direct_int4_bits":int4bits,"ratio_vs_fp32":total_bits/fpbits,
                               "compression_vs_fp32":fpbits/total_bits,"ratio_vs_int4":total_bits/int4bits})
    write_csv(outdir/"global_storage_summary.csv",global_storage)

    # Candidate MSE on seed-1 is an explicitly held-out curve; no selected
    # parameter was chosen using it.
    heldout_candidate=[r for fam in all_rule_rows.values() for r in fam if r["seed"]==1]
    write_csv(outdir/"heldout_candidate_mse.csv",heldout_candidate)
    # Summary grouped by seed/rank/method, weighted state cosine uses matrix
    # Frobenius weighting; unweighted means are retained alongside it.
    summary=[]
    for seed in SEEDS:
        for k in RANKS:
            methods=sorted({r["method"] for r in results if r["seed"]==seed and r["k"]==k})
            for method in methods:
                group=[r for r in results if r["seed"]==seed and r["k"]==k and r["method"]==method]
                vals=[float(r["update_cosine"]) for r in group if r["update_cosine"] is not None]
                weighted_num=sum(float(r["update_cosine"])*int(r["numel"]) for r in group if r["update_cosine"] is not None)
                weighted_den=sum(int(r["numel"]) for r in group if r["update_cosine"] is not None)
                summary.append({"seed":seed,"role":"calibration" if seed==0 else "held_out","k":k,"method":method,
                                "tensor_count":len(group),"mean_update_cosine":statistics.mean(vals) if vals else None,
                                "median_update_cosine":statistics.median(vals) if vals else None,
                                "weighted_update_cosine":weighted_num/weighted_den if weighted_den else None,
                                "mean_update_relative_l2":statistics.mean(float(r["update_relative_l2"]) for r in group if r["update_relative_l2"] is not None) if group else None,
                                "mean_full_state_raw_relative_l2":statistics.mean(float(r["full_state_raw_relative_l2"]) for r in group if r["full_state_raw_relative_l2"] is not None) if group else None,
                                "mean_residual_relative_l2":statistics.mean(float(r["residual_relative_l2"]) for r in group if r["residual_relative_l2"] is not None) if group else None,
                                "mean_zero_fraction":statistics.mean(float(r["zero_fraction"]) for r in group if r.get("zero_fraction") is not None) if group else None,
                                "mean_clipping_fraction":statistics.mean(float(r["clipping_fraction"]) for r in group if r.get("clipping_fraction") is not None) if any(r.get("clipping_fraction") is not None for r in group) else None,
                                "mean_candidate_scales_per_block":statistics.mean(float(r["candidate_scales_per_block"]) for r in group) if group else None,
                                "mean_exact_polar_cosine":statistics.mean(float(r["exact_polar_cosine"]) for r in group if r.get("exact_polar_cosine") is not None) if any(r.get("exact_polar_cosine") is not None for r in group) else None})
    write_csv(outdir/"summary_by_method.csv",summary)
    write_csv(outdir/"heldout_summary.csv",[r for r in summary if r["seed"]==1])
    # Recovery relative to absmax and the MSE oracle on the same tensor.
    recovery=[]
    bykey={(r["seed"],r["update"],r["parameter_id"],r["k"],r["method"]):r for r in results}
    for r in results:
        if r["method"] in {"absmax","mse_oracle"}: continue
        key=(r["seed"],r["update"],r["parameter_id"],r["k"])
        base=bykey.get((*key,"absmax")); oracle=bykey.get((*key,"mse_oracle"))
        if base and oracle and oracle["update_cosine"]!=base["update_cosine"]:
            r2={"seed":r["seed"],"update":r["update"],"parameter_id":r["parameter_id"],"k":r["k"],"method":r["method"],
                "scale_headroom_recovery":(r["update_cosine"]-base["update_cosine"])/(oracle["update_cosine"]-base["update_cosine"]),
                "update_cosine_gain_vs_absmax":r["update_cosine"]-base["update_cosine"],
                "oracle_update_cosine":oracle["update_cosine"],"baseline_update_cosine":base["update_cosine"]}
            recovery.append(r2)
    write_csv(outdir/"oracle_headroom_recovery.csv",recovery)
    temporal_rows=temporal_rows_from_traces(temporal_traces)
    write_csv(outdir/"temporal_scale_stability.csv",temporal_rows)
    plot_outputs(outdir,results,[r for r in results if r["seed"]==1],all_rule_rows,temporal_rows,warm_rows,spectral_rows)
    elapsed=time.perf_counter()-start_time
    summary_text=make_summary(summary,choices,analytic,total_matrices,elapsed,eval_started,recovery)
    (outdir/"summary.md").write_text(summary_text)
    (outdir/"methodology.md").write_text(methodology(choices,cb_by_k,analytic))
    print(f"analysis complete: matrices={total_matrices} elapsed={elapsed/60:.1f} min output={outdir}",flush=True)


def temporal_rows_from_traces(traces):
    rows=[]
    for key,series in traces.items():
        series=sorted(series,key=lambda z:z[0])
        for (u0,a0),(u1,a1) in zip(series,series[1:]):
            if a0.numel()!=a1.numel():continue
            ratio=a1/a0.clamp_min(1e-20); drift=torch.log(a1.clamp_min(1e-20))-torch.log(a0.clamp_min(1e-20))
            rows.append({"seed":key[0],"k":key[1],"parameter_id":key[2],"shape":key[3],"previous_update":u0,"update":u1,
                         "aligned_block_count":a1.numel(),"mean_alpha_ratio":float(ratio.mean()),"median_alpha_ratio":float(ratio.median()),
                         "mean_abs_relative_change":float((ratio-1).abs().mean()),"mean_abs_log_drift":float(drift.abs().mean()),
                         "spearman_block_rank_stability":spearman(a0,a1)})
    return rows


def spearman(a,b):
    if a.numel()<2:return None
    ra=torch.argsort(torch.argsort(a.float())).float(); rb=torch.argsort(torch.argsort(b.float())).float()
    return float(torch.corrcoef(torch.stack((ra,rb)))[0,1])


def make_summary(summary,choices,analytic,total_matrices,elapsed,eval_started,recovery):
    lookup={(r["seed"],r["k"],r["method"]):r for r in summary}
    lines=["# Practical block-scale selection for structurally conditioned INT3 Muon","",
           f"Offline CPU analysis covered {total_matrices} matrix instances (both seeds, requested landmarks) and ranks 4/8. Runtime: {elapsed/60:.1f} minutes. No training or production optimizer/quantizer behavior was changed.","",
           "Seed 0 alone selected all global scale parameters by aggregate residual SSE. Seed 1 is the held-out evaluation. Scale rules use fixed b2048 blocks and fixed rank-specific codebooks; update fidelity is evaluation-only.","",
           "## Held-out update cosine","","| Rank | Method | Mean cosine | Weighted cosine | Mean update relative-L2 |","|---:|---|---:|---:|---:|"]
    for k in RANKS:
        for method in ("absmax","fixed_absmax_fraction:","percentile:","rms:","std:","robust_median:","robust_mad:","local_grid_2","local_grid_3","local_grid_4","coarse_to_fine","mse_oracle"):
            rows=[r for r in summary if r["seed"]==1 and r["k"]==k and r["method"].startswith(method)]
            for row in rows:
                lines.append(f"| {k} | {row['method']} | {row['mean_update_cosine']:.4f} | {row['weighted_update_cosine']:.4f} | {row['mean_update_relative_l2']:.4f} |")
    practical=[r for r in summary if r["seed"]==1 and r["k"]==8 and r["method"] not in {"absmax","mse_oracle","prior_uniform_mse_oracle","structural_int4_reference","uniform_int3_reference"}]
    practical=[r for r in practical if r["mean_candidate_scales_per_block"] is not None and r["mean_candidate_scales_per_block"]<=4]
    best=max(practical,key=lambda r:r["mean_update_cosine"]) if practical else None
    best_rec=[r["scale_headroom_recovery"] for r in recovery if r["seed"]==1 and r["k"]==8 and best and r["method"]==best["method"]]
    gate=bool(best and (best["mean_update_cosine"]>=.79 or (best_rec and statistics.mean(best_rec)>=.75)))
    lines += ["","## Training gate and interpretation","",
              f"Best held-out k=8 practical rule under the <=4-candidate criterion: `{best['method']}` at mean cosine {best['mean_update_cosine']:.4f}; paired scale-headroom recovery {statistics.mean(best_rec):.1%} (if available). Gate satisfied: **{gate}**. This uses the stated descriptive threshold and is not a universal deployment criterion.",
              "Consult `summary_by_method.csv`, `oracle_headroom_recovery.csv`, `warmstart_scale_results.csv`, and `global_scale_calibration.csv` for the full tensor-level evidence.","",
              "No training run was launched. CPU timing is not a GPU-kernel cost estimate."]
    return "\n".join(lines)+"\n"


def methodology(choices,cb_by_k,analytic):
    lines=["# Methodology","","- Conditioner: ordinary exact top-k truncated SVD, ranks 4 and 8.","- Stored conditioner: BF16 U, singular values, and Vh factors reconstructed in FP32 for evaluation.","- Residual: original FP32 matrix minus the exact FP32 top-k component; all tested residual scales share this residual.","- Block partition: flattened row-major blocks of 2048 values; final short block retained.","- Scale semantics: alpha is a positive clipping bound. Normalize by alpha, clip to [-1,1], assign nearest fixed levels (ties to lower level), then multiply by alpha. All-zero blocks remain exactly zero.",
           "- k=4 uses the previously held-out-selected shared μ-law μ=5 codebook transform; k=8 uses the previously held-out-selected global Lloyd-Max levels. They are frozen before this study; scale-only comparisons never retune levels jointly.",
           "- Seed 0 is the global calibration split. Fixed absmax fraction, percentile, RMS, population std, median-absolute, and MAD multipliers are selected by total residual SSE only. Seed 1 update metrics are held out.",
           "- Local grid selects a block scale by minimizing that block's scalar residual SSE; it does not inspect Muon outputs. The 2/3/4 candidates are absmax-relative. Matched-codebook MSE reference reuses the previous study's exact 12-point grid {0.50,0.60,0.70,0.80,0.90,0.95,1.00,1.05,1.10,1.20,1.35,1.50}; a separate uniform-codebook run reproduces prior oracle results. These are finite-grid references, not continuous optima.",
           "- Coarse-to-fine uses three coarse absmax multiples (.6,.8,1.0), then two refinements around the current SSE winner, for five candidate quantizations per block.",
           f"- Deterministic model-fit clipping multipliers from fixed midpoint-quantile numerical integration: Gaussian RMS multiplier={analytic[4][0]:.5g} (k4), {analytic[8][0]:.5g} (k8); Laplace median-absolute multiplier={analytic[4][1]:.5g} (k4), {analytic[8][1]:.5g} (k8).",
           "- Temporal stability aligns identical parameter names, shapes, and flattened block indices across the five sparse landmarks; these are not consecutive updates. Warm-start uses the previous landmark's MSE-oracle scale and a three-candidate {0.8,1.0,1.2} local search.",
           "- Exact-polar readouts are emitted for absmax, local-grid-4, matched-codebook MSE grid, structural INT4, and seed-0-MSE-selected global p98. Production K=5 Newton-Schulz is used for all fidelity readouts.",
           "- Per-record evaluation time covers quantization/metric/transform evaluation after scale selection; scale-statistic/search work is represented by candidate counts and included in total CPU runtime, not in that per-record timer.",
           "- Every scale method has identical nominal INT3 payload, one FP32 scale per 2048-value residual block, BF16 rank-k factors, and the same decoder metadata. No physical packing/kernel cost is measured.","",
           "## Seed-0 selected global parameters","","```text"]
    for key,label in sorted(choices.items()): lines.append(f"k={key[0]} {key[1]}: {label}")
    lines += ["```","","The selected parameters are determined solely by seed-0 residual MSE, not update fidelity. A full parameter/seed-1 MSE curve is stored in `global_scale_calibration.csv` and `heldout_summary.csv`."]
    return "\n".join(lines)+"\n"


def replot_existing_report(outdir):
    def read(name):
        path=outdir/name
        if not path.exists(): return []
        with path.open(newline="") as f: return list(csv.DictReader(f))
    def typed(rows):
        result=[]
        for row in rows:
            item={}
            for key,value in row.items():
                if value=="": item[key]=None; continue
                try:item[key]=float(value)
                except (TypeError,ValueError):item[key]=value
            result.append(item)
        return result
    tensor=typed(read("tensor_level_results.csv")); held=[r for r in tensor if r.get("seed")==1]
    candidates=defaultdict(list)
    for family,filename in (("percentile","percentile_results.csv"),("rms","rms_results.csv"),
                            ("fixed_absmax_fraction","fixed_absmax_fraction.csv"),("std","std_results.csv"),
                            ("median_abs","robust_scale_results.csv"),("mad","robust_scale_results.csv")):
        for r in typed(read(filename)):
            if r.get("seed")==1:
                r["family"]=family; candidates[family].append(r)
    plot_outputs(outdir,tensor,held,candidates,typed(read("temporal_scale_stability.csv")),
                 typed(read("warmstart_scale_results.csv")),typed(read("spectral_diagnostics.csv")))


def typed_csv(path):
    with Path(path).open(newline="") as f: rows=list(csv.DictReader(f))
    out=[]
    for row in rows:
        item={}
        for k,v in row.items():
            if v=="":item[k]=None
            else:
                try:item[k]=float(v)
                except ValueError:item[k]=v
        out.append(item)
    return out


def evaluate_best_global_polar(reports_root,outdir):
    """Complete the exact-polar readout for the seed-0-MSE-selected p98 rule."""
    rows=[]; started=time.perf_counter()
    for seed,update,path in discover(Path(reports_root)):
        if seed!=1:continue
        snap=load_snapshot(path)
        for item in snap["tensors"]:
            if len(item["shape"])!=2 or min(item["shape"])<8:continue
            x=item["tensor"].detach().float(); state=decompose(x)
            for k in RANKS:
                cb=codebook_for_rank(k)
                c=(state.u[:,:k]*state.singular_values[:k])@state.vh[:k]
                c_hat=factorized_topk(state.u,state.singular_values,state.vh,k)
                residual=x-c
                alpha=block_scales(residual,"percentile",percentile=98.0)
                estimate=c_hat+quantize_scales(residual,alpha,cb)
                pm=quant_metrics(exact_polar(x),exact_polar(estimate),"exact_polar")
                rows.append({"seed":seed,"update":update,"parameter_id":item.get("parameter_id",item.get("name")),
                             "shape":str(tuple(x.shape)),"k":k,"method":"percentile_p98_seed0_mse_selected",**pm})
        print(f"exact-polar p98 readout seed=1 update={update}",flush=True)
    write_csv(Path(outdir)/"exact_polar_best_global_practical.csv",rows)
    print(f"exact-polar p98 completed: {len(rows)} rows, {time.perf_counter()-started:.1f}s",flush=True)


def evaluate_best_global_spectral(reports_root,outdir):
    """Compute danger-zone/cross-scale diagnostics for the selected p98 rule."""
    from optim.muon_conditioned_int3_companding import spectral_error_metrics
    rows=[]
    for seed,update,path in discover(Path(reports_root)):
        if seed!=1:continue
        snap=load_snapshot(path)
        for item in snap["tensors"]:
            if len(item["shape"])!=2 or min(item["shape"])<8:continue
            x=item["tensor"].detach().float(); state=decompose(x)
            for k in RANKS:
                c=(state.u[:,:k]*state.singular_values[:k])@state.vh[:k]; residual=x-c
                alpha=block_scales(residual,"percentile",percentile=98.0)
                q=quantize_scales(residual,alpha,codebook_for_rank(k)); err=q-residual
                rows.append({"seed":seed,"update":update,"parameter_id":item.get("parameter_id",item.get("name")),"k":k,
                             "method":"percentile:p=98",**spectral_error_metrics(state.u,state.singular_values,state.vh,err)})
        print(f"spectral p98 readout seed=1 update={update}",flush=True)
    path=Path(outdir)/"spectral_diagnostics.csv"; existing=typed_csv(path) if path.exists() else []
    existing=[r for r in existing if r.get("method") not in {"percentile_p98_seed0_mse_selected","percentile:p=98"}]
    write_csv(path,existing+rows)
    print(f"spectral p98 completed: {len(rows)} rows",flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports-root",type=Path,default=ROOT/"reports")
    p.add_argument("--outdir",type=Path,default=ROOT/"reports/muon_int3_practical_scale")
    p.add_argument("--threads",type=int,default=min(4,torch.get_num_threads()))
    p.add_argument("--limit-snapshots",type=int,nargs="*",default=None,help="optional landmark subset for development only")
    p.add_argument("--limit-tensors",type=int,default=0,help="optional deterministic matrix prefix per snapshot for development only")
    p.add_argument("--replot-only",action="store_true",help="regenerate plots from existing report CSVs without rerunning analysis")
    p.add_argument("--polar-best-global",action="store_true",help="compute held-out exact-polar metrics for selected global p98 rule only")
    p.add_argument("--spectral-best-global",action="store_true",help="compute held-out spectral diagnostics for selected global p98 rule only")
    return p.parse_args()


if __name__=="__main__":
    run(parse_args())
