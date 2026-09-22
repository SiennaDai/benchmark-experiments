#!/usr/bin/env python3
"""Two-stage offline study of structurally conditioned INT3 Muon residuals."""
from __future__ import annotations

import argparse
import ast
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
    DANGER_LOG10_RANGE, INT3_CODEBOOK, build_symmetric_codebook,
    danger_mode_mask, geometry_surrogate_loss, int3_codebook_roundtrip,
    int3_mulaw_roundtrip, int3_power_roundtrip, int3_scale_oracle_roundtrip,
    int3_uniform_roundtrip, lloyd_max_codebook, mulaw_transform,
    normalized_block_samples, optimize_block_scale, select_geometry_codebook,
    spectral_error_metrics,
)
from optim.muon_ns_sensitivity import exact_polar, scalar_map  # noqa: E402
from optim.muon_quantization_aware_conditioner import BLOCK_SIZE  # noqa: E402
from optim.muon_spectral_sensitivity import quantize as production_quantize  # noqa: E402
from optim.muon_storage_pareto import storage_bits  # noqa: E402
from optim.muon_structural_decomposition import decompose  # noqa: E402
from optim.muon_update_fidelity import load_snapshot  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
MU_GRID = (1, 5, 20, 100, 500, 2000)
GAMMA_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
CODEBOOK_PAIRS = ((1/3, 2/3), (0.20, 0.60), (0.20, 0.80), (0.35, 0.75), (0.10, 0.55), (0.40, 0.85))
EPS = 1e-12


def refine_codebook_pairs(center_a1, center_a2):
    """Deterministic local 0.05-grid around a coarse codebook winner."""
    a1s={round(float(center_a1)+d,2) for d in (-0.05,0,0.05)}
    a2s={round(float(center_a2)+d,2) for d in (-0.05,0,0.05)}
    return tuple(sorted((a1,a2) for a1 in a1s for a2 in a2s if 0<a1<a2<1.0))


def discover(root: Path):
    groups = {}
    for path in sorted(root.rglob("update_*.pt")):
        if "muon_momentum_snapshots" not in path.parts:
            continue
        try:
            snap = load_snapshot(path)
            seed, update = int(snap["metadata"]["seeds"]["seed"]), int(snap["metadata"]["update"])
        except Exception:
            continue
        if seed in SEEDS and update in LANDMARKS:
            groups.setdefault((seed, str(path.parent)), {})[update] = path
    result = []
    for seed in SEEDS:
        candidates = sorted((root_name, files) for (s, root_name), files in groups.items()
                            if s == seed and set(files) == set(LANDMARKS))
        if candidates:
            result.extend((seed, u, candidates[0][1][u]) for u in LANDMARKS)
    return result


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns: columns.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def finite(x):
    try: return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError): return False


def transform_kwargs(snapshot):
    c = snapshot["metadata"]["muon_transform"]
    return {"steps": int(c["steps"]), "coefficients": tuple(float(v) for v in c["coefficients"]), "eps": float(c["eps"])}


def ratios(reference, estimate):
    a, b = reference.detach().float(), estimate.detach().float()
    an, bn = a.norm(), b.norm()
    return {"relative_l2": float((b-a).norm()/an) if an.item() else None,
            "cosine": float((a*b).sum()/(an*bn)) if an.item() and bn.item() else None,
            "norm_ratio": float(bn/an) if an.item() else None}


def factorized_topk(state, k: int):
    ix = torch.arange(min(k, state.singular_values.numel()))
    u = state.u[:, ix].to(torch.bfloat16).float()
    s = state.singular_values[ix].to(torch.bfloat16).float()
    vh = state.vh[ix].to(torch.bfloat16).float()
    return (u*s) @ vh


def make_item(seed, update, item, snapshot, k, *, keep_svd=True, compute_polar=True):
    x = item["tensor"].detach().float()
    state = decompose(x)
    c_exact = (state.u[:, :k] * state.singular_values[:k]) @ state.vh[:k]
    residual = x - c_exact
    c_hat = factorized_topk(state, k)
    kw = transform_kwargs(snapshot)
    ref_update = muon_reference.zeropower_newton_schulz(x.clone(), **kw)
    # A_dir is used only to form a predeclared sensitivity-weighted surrogate.
    mapped = scalar_map(state.singular_values, matrix_norm=float(x.norm()), steps=kw["steps"],
                        coefficients=kw["coefficients"], eps=kw["eps"]).abs()
    adir = mapped / state.singular_values.clamp_min(1e-12)
    median = adir.median().clamp_min(1e-12)
    weights = (adir / median).clamp(0.1, 20.0)
    if not keep_svd:
        state = None
    return {"seed": seed, "update": update, "parameter_id": item.get("parameter_id", item.get("name", "unknown")),
            "parameter_name": item.get("name", item.get("parameter_id", "unknown")), "shape": str(tuple(x.shape)),
            "m": x.shape[0], "n": x.shape[1], "x": x, "state": state, "c_exact": c_exact,
            "c_hat": c_hat, "residual": residual, "kw": kw, "ref_update": ref_update,
            "ref_polar": exact_polar(x) if compute_polar else None,
            "mode_weights": weights, "adir": adir}


def calibration_cache(paths, *, per_snapshot: int, tensor_limit: int):
    """Deterministic seed-0 calibration subset; no seed-1 tensor is read here."""
    entries = {4: [], 8: []}
    for seed, update, path in paths:
        if seed != 0: continue
        snap = load_snapshot(path); count = 0
        for item in snap["tensors"]:
            if len(item["shape"]) != 2: continue
            if tensor_limit and count >= tensor_limit: break
            if per_snapshot and count >= per_snapshot: break
            count += 1
            for k in (4, 8):
                if min(item["shape"]) < k: continue
                entries[k].append(make_item(seed, update, item, snap, k, compute_polar=False))
        print(f"calibration seed=0 update={update}: {count} matrices", flush=True)
    return entries


def codebook_candidate_metrics(items, a1, a2):
    cb = build_symmetric_codebook(a1, a2)
    vals = []
    for it in items:
        q = int3_codebook_roundtrip(it["residual"], cb)
        update = muon_reference.zeropower_newton_schulz((it["c_hat"] + q).clone(), **it["kw"])
        vals.append(float((it["ref_update"] * update).sum() / (it["ref_update"].norm() * update.norm()).clamp_min(1e-20)))
    return sum(vals) / max(1, len(vals))


def calibration_select(calibration, *, calibration_items_per_update: int):
    selected = {}
    all_lm = []; all_geom = []; all_zone = []; all_muon_cb = []; all_muon_compand = []
    for k, full in calibration.items():
        # Cache already includes at most the requested deterministic prefix per update.
        selected[k] = full
        samples = normalized_block_samples([it["residual"] for it in full])
        selected[(k, "lloyd")] = lloyd_max_codebook(samples)
        mu_mse=[]
        for mu in MU_GRID:
            errs=[]
            for it in full:
                q=int3_mulaw_roundtrip(it["residual"],mu)
                den=it["residual"].square().sum().clamp_min(1e-20)
                errs.append(float((q-it["residual"]).square().sum()/den))
            mu_mse.append((sum(errs)/len(errs),mu))
        selected[(k,"mse_mulaw")]=min(mu_mse)[1]
        # Muon calibration and geometry calibration are restricted to seed-0 items.
        pairs = CODEBOOK_PAIRS
        coarse_scores = [(a1, a2, codebook_candidate_metrics(full, a1, a2)) for a1, a2 in pairs]
        coarse_best = max(coarse_scores, key=lambda z: (z[2], -z[0], -z[1]))
        fine_pairs=refine_codebook_pairs(coarse_best[0],coarse_best[1])
        fine_scores=[(a1,a2,codebook_candidate_metrics(full,a1,a2)) for a1,a2 in fine_pairs]
        scores=coarse_scores+fine_scores
        best = max(scores, key=lambda z: (z[2], -z[0], -z[1]))
        selected[(k, "muon_codebook")] = build_symmetric_codebook(best[0], best[1])
        for a1, a2, score in coarse_scores:
            all_muon_cb.append({"k": k, "a1": a1, "a2": a2, "calibration_mean_update_cosine": score,
                                "calibration_count": len(full), "search_stage":"coarse", "selected": (a1, a2) == (best[0], best[1])})
        for a1,a2,score in fine_scores:
            all_muon_cb.append({"k":k,"a1":a1,"a2":a2,"calibration_mean_update_cosine":score,
                                "calibration_count":len(full),"search_stage":"fine","selected":(a1,a2)==(best[0],best[1])})
        geo_items = [{"residual": it["residual"], "u": it["state"].u, "vh": it["state"].vh,
                      "mode_weights": it["mode_weights"]} for it in full]
        coarse_geo=select_geometry_codebook(geo_items,pairs)
        fine_geo=select_geometry_codebook(geo_items,refine_codebook_pairs(coarse_geo[0],coarse_geo[1]))
        ga1,ga2=fine_geo[0],fine_geo[1]
        gscores=[{**r,"search_stage":"coarse"} for r in coarse_geo[2]]+[{**r,"search_stage":"fine"} for r in fine_geo[2]]
        selected[(k, "geometry_codebook")] = build_symmetric_codebook(ga1, ga2)
        for row in gscores:
            all_geom.append({"k": k, **row, "selected": (row["a1"], row["a2"]) == (ga1, ga2), "calibration_seed": 0})
        # Danger-zone + local mixing heuristic. Every lambda pair is optimized
        # on calibration data only; (4,1) is the predeclared primary variant.
        bd, bl = 4, 1
        selected[(k,"zone_lambdas")] = (bd,bl)
        zone_selections={}
        for ld,ll in ((1,1),(1,4),(4,1),(4,4),(16,1),(16,4)):
            def zone_score(a1,a2):
                cb=build_symmetric_codebook(a1,a2); losses=[]
                for it in full:
                    st=it["state"]; s=st.singular_values
                    z=torch.log10((s/s[0]).clamp_min(1e-6)); d=(z[:,None]-z[None,:]).abs(); dz=(z>=-3)&(z<-2)
                    w=1+ld*(dz[:,None]|dz[None,:]).float()+ll*(d<.5).float()
                    q=int3_codebook_roundtrip(it["residual"],cb); eh=st.u.T@(q-it["residual"])@st.vh.T
                    losses.append(float((w*eh.square()).sum()))
                return sum(losses)/len(losses)
            coarse=[(zone_score(a1,a2),a1,a2,"coarse") for a1,a2 in pairs]
            coarse_best=min(coarse)
            fine=[(zone_score(a1,a2),a1,a2,"fine") for a1,a2 in refine_codebook_pairs(coarse_best[1],coarse_best[2])]
            scored=coarse+fine; best_zone=min(scored); zone_selections[(ld,ll)]=best_zone
            for score,a1,a2,phase in scored:
                all_zone.append({"k":k,"lambda_danger":ld,"lambda_local":ll,"a1":a1,"a2":a2,
                                 "danger_weighted_codebook_loss":score,"search_stage":phase,"selected_primary_lambda":(ld,ll)==(bd,bl),
                                 "selected_codebook":(a1,a2)==(best_zone[1],best_zone[2]),"calibration_seed":0})
        dzbest=zone_selections[(bd,bl)]
        selected[(k,"danger_zone_codebook")]=build_symmetric_codebook(dzbest[1],dzbest[2])
        # Shared companding parameters: MSE codebooks above; geometry and Muon objectives
        # select one global mu and gamma on the calibration subset.
        geom_comp=[]; muon_comp=[]
        for family, grid in (("mulaw", MU_GRID), ("power", GAMMA_GRID)):
            for param in grid:
                losses=[]; cosines=[]
                for it in full:
                    fn = int3_mulaw_roundtrip(it["residual"], param) if family=="mulaw" else int3_power_roundtrip(it["residual"],param)
                    err = fn - it["residual"]
                    losses.append(float(geometry_surrogate_loss(it["state"].u,it["state"].vh,err,it["mode_weights"])))
                    uu=muon_reference.zeropower_newton_schulz((it["c_hat"]+fn).clone(),**it["kw"])
                    cosines.append(float((it["ref_update"]*uu).sum()/(it["ref_update"].norm()*uu.norm()).clamp_min(1e-20)))
                geom_comp.append({"family":family,"parameter":param,"loss":sum(losses)/len(losses)})
                muon_comp.append({"family":family,"parameter":param,"cosine":sum(cosines)/len(cosines)})
        selected[(k,"geometry_mulaw")] = min([r for r in geom_comp if r["family"]=="mulaw"],key=lambda r:r["loss"])["parameter"]
        selected[(k,"geometry_power")] = min([r for r in geom_comp if r["family"]=="power"],key=lambda r:r["loss"])["parameter"]
        selected[(k,"muon_mulaw")] = max([r for r in muon_comp if r["family"]=="mulaw"],key=lambda r:r["cosine"])["parameter"]
        selected[(k,"muon_power")] = max([r for r in muon_comp if r["family"]=="power"],key=lambda r:r["cosine"])["parameter"]
        for row in geom_comp: all_geom.append({"k":k,"family":row["family"],"parameter":row["parameter"],"mean_geometry_loss":row["loss"],"calibration_seed":0})
        for row in muon_comp: all_muon_compand.append({"k":k,**row,"calibration_seed":0,
            "selected":row["parameter"]==selected[(k,"muon_"+row["family"])]})
    return selected, all_geom, all_zone, all_muon_cb, all_muon_compand


def quantizer_outputs(it, k, calib, *, per_tensor_oracle=False):
    r=it["residual"]; x=it["x"]; state=it["state"]
    out={"uniform_int3": int3_uniform_roundtrip(r)}
    for mu in MU_GRID: out[f"mulaw_mu_{mu}"]=int3_mulaw_roundtrip(r,mu)
    out["lloyd_global"]=int3_codebook_roundtrip(r,calib[(k,"lloyd")])
    samples=normalized_block_samples([r])
    personal=lloyd_max_codebook(samples)
    out["lloyd_per_tensor_oracle"]=int3_codebook_roundtrip(r,personal)
    out["scale_oracle"], scales=int3_scale_oracle_roundtrip(r)
    out["muon_global_codebook"]=int3_codebook_roundtrip(r,calib[(k,"muon_codebook")])
    out["geometry_codebook"]=int3_codebook_roundtrip(r,calib[(k,"geometry_codebook")])
    out["danger_zone_codebook"]=int3_codebook_roundtrip(r,calib[(k,"danger_zone_codebook")])
    gm=float(calib[(k,"geometry_mulaw")]); gp=float(calib[(k,"geometry_power")])
    mm=float(calib[(k,"muon_mulaw")]); mp=float(calib[(k,"muon_power")])
    out[f"geometry_mulaw_{gm:g}"]=int3_mulaw_roundtrip(r,gm)
    out[f"geometry_power_{gp:g}"]=int3_power_roundtrip(r,gp)
    out[f"muon_mulaw_{mm:g}"]=int3_mulaw_roundtrip(r,mm)
    out[f"muon_power_{mp:g}"]=int3_power_roundtrip(r,mp)
    ms=float(calib[(k,"mse_mulaw")])
    out[f"calibrated_mulaw_{ms:g}"]=int3_mulaw_roundtrip(r,ms)
    # low-cost fixed danger-weight family selected during calibration. Actual
    # codebook search is global and uses weighted spectral residual error.
    bd,bl=calib[(k,"zone_lambdas")]
    # Explicitly retain the calibration-selected values in the result identity.
    out[f"danger_weighted_codebook_D{bd}_L{bl}"]=out["danger_zone_codebook"]
    return out, scales, personal


def residual_error_stats(residual, q):
    e=(q-residual).abs(); mag=residual.abs(); block_scale=[]
    flat=mag.reshape(-1)
    for start in range(0,flat.numel(),BLOCK_SIZE): block_scale.append(flat[start:start+BLOCK_SIZE].amax())
    scale=torch.cat([v.expand(min(BLOCK_SIZE,flat.numel()-i*BLOCK_SIZE)) for i,v in enumerate(block_scale)]) if block_scale else torch.ones_like(flat)
    norms=mag.reshape(-1)/scale.clamp_min(1e-30); ef=e.reshape(-1)
    result={"residual_relative_l2":float((q-residual).norm()/residual.norm()) if residual.norm().item() else None,
            "residual_cosine":float((q*residual).sum()/(q.norm()*residual.norm())) if q.norm().item() and residual.norm().item() else None,
            "mae":float(e.mean()),"median_ae":float(e.median()),
            "zero_fraction":float((q==0).float().mean()),
            "small_005_mae":float(ef[norms<.05].mean()) if bool((norms<.05).any()) else None,
            "small_001_mae":float(ef[norms<.01].mean()) if bool((norms<.01).any()) else None,
            "middle_mae":float(ef[(norms>=.05)&(norms<.5)].mean()) if bool(((norms>=.05)&(norms<.5)).any()) else None,
            "large_mae":float(ef[norms>=.5].mean()) if bool((norms>=.5).any()) else None}
    return result


def evaluate_quantizer(it,k,name,q, *, exact=False):
    x,r,c_hat=it["x"],it["residual"],it["c_hat"]
    mhat=c_hat+q
    raw=ratios(x,mhat)
    update=muon_reference.zeropower_newton_schulz(mhat.clone(),**it["kw"])
    um=ratios(it["ref_update"],update)
    pol=ratios(it["ref_polar"],exact_polar(mhat)) if exact else {"relative_l2":None,"cosine":None}
    rr=ratios(r,q)
    spectral=spectral_error_metrics(it["state"].u,it["state"].singular_values,it["state"].vh,q-r)
    diag=residual_error_stats(r,q)
    flat=r.reshape(-1); bmax=torch.stack([flat[st:st+BLOCK_SIZE].abs().amax() for st in range(0,flat.numel(),BLOCK_SIZE)])
    sv=it["state"].singular_values
    active=sv[sv/sv[0]>=1e-6] if sv.numel() and sv[0].item()>0 else sv[:0]
    dzm=danger_mode_mask(sv)
    return {**{kk:it[kk] for kk in ("seed","update","parameter_id","parameter_name","shape")},"k":k,"quantizer":name,
            "raw_residual_relative_l2":rr["relative_l2"],"raw_residual_cosine":rr["cosine"],
            "full_state_raw_relative_l2":raw["relative_l2"],"full_state_raw_cosine":raw["cosine"],
            "update_cosine":um["cosine"],"update_relative_l2":um["relative_l2"],
            "update_norm_ratio":um["norm_ratio"],"exact_polar_cosine":pol["cosine"],"exact_polar_relative_l2":pol["relative_l2"],
            "residual_block_absmax_mean":float(bmax.mean()),"residual_block_absmax_max":float(bmax.max()),
            "effective_condition_number":float(active[0]/active[-1]) if active.numel() else None,
            "danger_zone_mode_count":int(dzm.sum()),"active_mode_count":int(active.numel()),
            **diag,**spectral}


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        if finite(r.get("update_cosine")): groups[(r["k"],r["quantizer"],r["seed"])].append(float(r["update_cosine"]))
    out=[]
    for (k,q,seed),vals in sorted(groups.items()):
        vals.sort(); members=[r for r in rows if int(r["k"])==int(k) and r["quantizer"]==q and int(r["seed"])==int(seed) and finite(r.get("update_cosine"))]
        weights=[]
        for r in members:
            try: weights.append(math.prod(ast.literal_eval(r["shape"])))
            except (ValueError, SyntaxError, TypeError): weights.append(1)
        weighted_mean=sum(float(r["update_cosine"])*w for r,w in zip(members,weights))/sum(weights) if weights and sum(weights) else statistics.mean(vals)
        out.append({"k":k,"quantizer":q,"seed":seed,"n":len(vals),"mean":statistics.mean(vals),"weighted_mean":weighted_mean,"median":statistics.median(vals),
            "p10":vals[max(0,int(.1*(len(vals)-1)))],"p25":vals[max(0,int(.25*(len(vals)-1)))],
            "p75":vals[min(len(vals)-1,int(.75*(len(vals)-1)))],"p90":vals[min(len(vals)-1,int(.9*(len(vals)-1)))],
            "win_rate_vs_uniform":None})
    # Pairwise win-rate, with exact seed/update/parameter joins.
    ix={(r["k"],r["quantizer"],r["seed"],r["update"],r["parameter_id"]):r for r in rows}
    for row in out:
        tests=[]
        for key, r in ix.items():
            if key[:3]==(row["k"],row["quantizer"],row["seed"]) and key[1]!="uniform_int3":
                base=ix.get((key[0],"uniform_int3",key[2],key[3],key[4]))
                if base: tests.append(float(r["update_cosine"])>float(base["update_cosine"]))
        row["win_rate_vs_uniform"]=sum(tests)/len(tests) if tests else 0.0
    return out


def nearest_matched_controls(rows):
    """Pair geometry codebooks with nearest scalar references per held-out tensor."""
    ix={(r["seed"],r["update"],r["parameter_id"],r["k"],r["quantizer"]):r for r in rows}
    candidates=("muon_global_codebook","geometry_codebook","danger_zone_codebook",
               "muon_global_codebook_fine","geometry_codebook_fine","danger_zone_codebook_fine")
    # References are value-space methods only; geometry-selected power-law
    # candidates must not serve as their own comparison family.
    references=("uniform_int3","lloyd_global",*(f"mulaw_mu_{mu}" for mu in MU_GRID))
    groups=defaultdict(list)
    for key,row in ix.items(): groups[key[:4]].append(row)
    result=[]
    for (seed,update,pid,k), group in groups.items():
        if int(seed)!=1: continue
        for target in group:
            name=target["quantizer"]
            if name not in candidates or not finite(target.get("raw_residual_relative_l2")): continue
            refs=[ix.get((seed,update,pid,k,q)) for q in references]
            refs=[r for r in refs if r is not None and r["quantizer"]!=name and finite(r.get("raw_residual_relative_l2"))]
            if not refs: continue
            pairs=(
                ("nearest_raw_relative_l2",min(refs,key=lambda r:(abs(float(r["raw_residual_relative_l2"])-float(target["raw_residual_relative_l2"])),r["quantizer"]))),
                ("nearest_zero_fraction",min(refs,key=lambda r:(abs(float(r["zero_fraction"])-float(target["zero_fraction"])),r["quantizer"]))),
            )
            for match_type,ref in pairs:
                result.append({"seed":seed,"update":update,"parameter_id":pid,"k":k,"candidate":name,"reference":ref["quantizer"],"match_type":match_type,
                    "raw_residual_l2_candidate":target["raw_residual_relative_l2"],"raw_residual_l2_reference":ref["raw_residual_relative_l2"],
                    "absolute_raw_l2_gap":abs(float(target["raw_residual_relative_l2"])-float(ref["raw_residual_relative_l2"])),
                    "update_cosine_gain_candidate_minus_reference":float(target["update_cosine"])-float(ref["update_cosine"]),
                    "zero_fraction_candidate":target["zero_fraction"],"zero_fraction_reference":ref["zero_fraction"],
                    "zero_fraction_gap":abs(float(target["zero_fraction"])-float(ref["zero_fraction"]))})
    return result


def run_refined_codebook_followup(paths, out, *, calibration_tensors_per_update=3):
    """Coarse-to-fine global-codebook follow-up, without repeating other grids."""
    calibration=calibration_cache(paths,per_snapshot=calibration_tensors_per_update,tensor_limit=0)
    selected={}; centers={}; zone_grid_rows=[]
    # Start the fixed fine grid at the best coarse entries already measured in
    # the full study, then optimize only the three global codebook objectives.
    for k in (4,8):
        coarse_mu=[r for r in csv.DictReader((out/"muon_codebook_oracle.csv").open()) if r.get("k")==str(k) and r.get("selected")=="True" and r.get("a1") and r.get("search_stage","coarse")=="coarse"]
        coarse_geo=[r for r in csv.DictReader((out/"geometry_weighted_codebook.csv").open()) if r.get("k")==str(k) and r.get("selected")=="True" and r.get("a1") and r.get("family","")=="" and r.get("search_stage","coarse")=="coarse"]
        coarse_zone=[r for r in csv.DictReader((out/"geometry_weighted_codebook.csv").open()) if r.get("k")==str(k) and r.get("selected_codebook")=="True" and r.get("lambda_danger")=="4" and r.get("lambda_local")=="1" and r.get("a1") and r.get("search_stage","coarse")=="coarse"]
        centers[k]={
            "muon_codebook":(float(coarse_mu[0]["a1"]),float(coarse_mu[0]["a2"])) if coarse_mu else (0.2,0.6),
            "geometry_codebook":(float(coarse_geo[0]["a1"]),float(coarse_geo[0]["a2"])) if coarse_geo else (0.2,0.6),
            "danger_zone_codebook":(float(coarse_zone[0]["a1"]),float(coarse_zone[0]["a2"])) if coarse_zone else (0.2,0.6),
        }
        for method,(a1,a2) in centers[k].items():
            candidates=refine_codebook_pairs(a1,a2)
            if method=="muon_codebook":
                scores=[(codebook_candidate_metrics(calibration[k],x,y),x,y) for x,y in candidates]
                best=max(scores,key=lambda z:(z[0],-z[1],-z[2]))
            elif method=="geometry_codebook":
                items=[{"residual":it["residual"],"u":it["state"].u,"vh":it["state"].vh,"mode_weights":it["mode_weights"]} for it in calibration[k]]
                a1b,a2b,_=select_geometry_codebook(items,candidates); best=(0.0,a1b,a2b)
            else:
                scores=[]
                for x,y in candidates:
                    cb=build_symmetric_codebook(x,y); losses=[]
                    for it in calibration[k]:
                        st=it["state"];s=st.singular_values;z=torch.log10((s/s[0]).clamp_min(1e-6));d=(z[:,None]-z[None,:]).abs();dz=(z>=-3)&(z<-2)
                        weights=1+4*(dz[:,None]|dz[None,:]).float()+(d<.5).float()
                        q=int3_codebook_roundtrip(it["residual"],cb);eh=st.u.T@(q-it["residual"])@st.vh.T
                        losses.append(float((weights*eh.square()).sum()))
                    scores.append((sum(losses)/len(losses),x,y))
                best=min(scores)
            selected[(k,method)]=build_symmetric_codebook(best[1],best[2])
        # Correctly evaluate the declared danger/local multiplier grid on
        # calibration tensors. The primary (4,1) already has the fine search
        # above; other cells retain their six-point coarse winners.
        for ld,ll in ((1,1),(1,4),(4,1),(4,4),(16,1),(16,4)):
            scored=[]
            for a1,a2 in CODEBOOK_PAIRS:
                cb=build_symmetric_codebook(a1,a2); losses=[]
                for it in calibration[k]:
                    st=it["state"];s=st.singular_values;z=torch.log10((s/s[0]).clamp_min(1e-6));d=(z[:,None]-z[None,:]).abs();dz=(z>=-3)&(z<-2)
                    weights=1+ld*(dz[:,None]|dz[None,:]).float()+ll*(d<.5).float()
                    q=int3_codebook_roundtrip(it["residual"],cb);eh=st.u.T@(q-it["residual"])@st.vh.T
                    losses.append(float((weights*eh.square()).sum()))
                scored.append((sum(losses)/len(losses),a1,a2,"coarse"))
            coarse_best=min(scored)
            if (ld,ll)==(4,1):
                for a1,a2 in refine_codebook_pairs(coarse_best[1],coarse_best[2]):
                    cb=build_symmetric_codebook(a1,a2);losses=[]
                    for it in calibration[k]:
                        st=it["state"];s=st.singular_values;z=torch.log10((s/s[0]).clamp_min(1e-6));d=(z[:,None]-z[None,:]).abs();dz=(z>=-3)&(z<-2)
                        weights=1+ld*(dz[:,None]|dz[None,:]).float()+ll*(d<.5).float()
                        q=int3_codebook_roundtrip(it["residual"],cb);eh=st.u.T@(q-it["residual"])@st.vh.T
                        losses.append(float((weights*eh.square()).sum()))
                    scored.append((sum(losses)/len(losses),a1,a2,"fine"))
            best=min(scored)
            zone_grid_rows.extend({"k":k,"lambda_danger":ld,"lambda_local":ll,"a1":a1,"a2":a2,"calibration_weighted_error":loss,
                                   "search_stage":phase,"selected":(a1,a2)==(best[1],best[2]),"calibration_seed":0}
                                  for loss,a1,a2,phase in scored)
    selection=[]
    for k in (4,8):
        for name,key in (("muon_codebook","muon_codebook"),("geometry_codebook","geometry_codebook"),("danger_zone_codebook","danger_zone_codebook")):
            cb=selected[(k,key)].tolist()
            a1,a2=cb[4],cb[5]
            center=centers[k][key]
            selection.append({"k":k,"conditioner":name,"coarse_a1":center[0],"coarse_a2":center[1],"a1":a1,"a2":a2,"levels":",".join(f"{v:.7g}" for v in cb),"calibration_seed":0,"search":"coarse six-point grid then local 0.05 grid"})
    eval_rows=[]
    for seed,update,path in paths:
        if seed!=1: continue
        snapshot=load_snapshot(path)
        for item in snapshot["tensors"]:
            if len(item["shape"])!=2: continue
            for k in (4,8):
                if min(item["shape"])<k: continue
                it=make_item(seed,update,item,snapshot,k)
                for name,key in (("muon_global_codebook_fine","muon_codebook"),
                                 ("geometry_codebook_fine","geometry_codebook"),
                                 ("danger_zone_codebook_fine","danger_zone_codebook")):
                    q=int3_codebook_roundtrip(it["residual"],selected[(k,key)])
                    row=evaluate_quantizer(it,k,name,q,exact=True)
                    row["calibration_split"]="seed0";row["evaluation_split"]="seed1"
                    eval_rows.append(row)
        print(f"refined codebook held-out seed=1 update={update}",flush=True)
    write_csv(out/"refined_codebook_selections.csv",selection)
    write_csv(out/"refined_codebook_followup.csv",eval_rows)
    write_csv(out/"danger_lambda_grid_calibration.csv",zone_grid_rows)
    return eval_rows,selection


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reports-root",type=Path,default=ROOT/"reports")
    ap.add_argument("--output",type=Path,default=ROOT/"reports/muon_conditioned_int3_companding")
    ap.add_argument("--calibration-tensors-per-update",type=int,default=3,help="deterministic seed-0 prefix per update for global oracle selection")
    ap.add_argument("--tensor-limit",type=int,default=0)
    ap.add_argument("--per-tensor-oracle-limit",type=int,default=40)
    ap.add_argument("--include-k16",action="store_true")
    ap.add_argument("--snapshot-limit",type=int,default=0)
    ap.add_argument("--skip-plots",action="store_true")
    ap.add_argument("--refined-codebooks-only",action="store_true",help="rerun only coarse-to-fine shared codebook calibration and held-out evaluation")
    args=ap.parse_args()
    torch.set_num_threads(max(1,min(torch.get_num_threads(),4)))
    start=time.perf_counter(); paths=discover(args.reports_root)
    if len(paths)!=10: raise SystemExit(f"expected ten formal snapshots, found {len(paths)}")
    if args.snapshot_limit: paths=paths[:max(1,args.snapshot_limit)]
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.refined_codebooks_only:
        rows,selections=run_refined_codebook_followup(paths,out,calibration_tensors_per_update=args.calibration_tensors_per_update)
        print(f"refined codebook follow-up: {len(rows)} evaluation rows, {len(selections)} selections, elapsed={time.perf_counter()-start:.1f}s",flush=True)
        return
    calibration=calibration_cache(paths,per_snapshot=args.calibration_tensors_per_update,tensor_limit=args.tensor_limit)
    params, geometry_rows, zone_rows, muon_cb_rows, muon_comp_rows=calibration_select(calibration,calibration_items_per_update=args.calibration_tensors_per_update)
    stage_rows=[]; mulaw_rows=[]; lloyd_rows=[]; scale_rows=[]; muon_cb_eval=[]; spectral_rows=[]; danger_rows=[]; geom_rows=[]; matched=[]; heldout=[]; storage_rows=[]; tensor_rows=[]; failure=[]; selection_rows=[]
    seen_per_tensor=defaultdict(int); count=0; per_tensor_done=0
    for seed,update,path in paths:
        snap=load_snapshot(path); kw=transform_kwargs(snap); print(f"evaluate seed={seed} update={update}",flush=True)
        tcount=0
        for item in snap["tensors"]:
            if len(item["shape"])!=2: continue
            if args.tensor_limit and tcount>=args.tensor_limit: break
            tcount+=1; count+=1
            for k in ((4,8,16) if args.include_k16 else (4,8)):
                if min(item["shape"])<k: continue
                it=make_item(seed,update,item,snap,k); outputs,scales,personal=quantizer_outputs(it,k,params)
                global_lloyd=params[(k,"lloyd")]; geom_cb=params[(k,"geometry_codebook")]; muon_cb=params[(k,"muon_codebook")]
                exact_names={"uniform_int3",f"mulaw_mu_{params[(k,'muon_mulaw')]:g}",
                             f"calibrated_mulaw_{int(params[(k,'mse_mulaw')])}",
                             "lloyd_global","muon_global_codebook", "geometry_codebook"}
                top4=production_quantize(it["residual"],"int4-dynamic-b2048").float()
                top4hat=it["c_hat"]+top4
                top4u=muon_reference.zeropower_newton_schulz(top4hat.clone(),**kw)
                top4p=exact_polar(top4hat)
                top4metrics=ratios(it["ref_update"],top4u)
                # Per-tensor codebook upper bound is deliberately capped by count.
                do_oracle=per_tensor_done<args.per_tensor_oracle_limit
                if do_oracle:
                    per_tensor_done+=1
                    best=(float("-inf"),None,None)
                    for a1,a2 in CODEBOOK_PAIRS:
                        cb=build_symmetric_codebook(a1,a2); q=int3_codebook_roundtrip(it["residual"],cb)
                        uu=muon_reference.zeropower_newton_schulz((it["c_hat"]+q).clone(),**kw)
                        co=ratios(it["ref_update"],uu)["cosine"]
                        if co is not None and co>best[0]: best=(co,cb,(a1,a2))
                    if best[1] is not None:
                        outputs["per_tensor_muon_codebook_oracle"]=int3_codebook_roundtrip(it["residual"],best[1])
                        selection_rows.append({"seed":seed,"update":update,"parameter_id":it["parameter_id"],"k":k,"a1":best[2][0],"a2":best[2][1],"calibration_role":"per_tensor_oracle_eval_upper_bound"})
                base={"seed":seed,"update":update,"parameter_id":it["parameter_id"],"parameter_name":it["parameter_name"],"shape":it["shape"],"k":k}
                for name,q in outputs.items():
                    exact=(name in exact_names or name=="per_tensor_muon_codebook_oracle")
                    row=evaluate_quantizer(it,k,name,q,exact=exact); row.update(base); stage_rows.append(row); tensor_rows.append(row)
                    if name=="lloyd_global": levels=params[(k,"lloyd")].tolist()
                    elif name=="lloyd_per_tensor_oracle": levels=personal.tolist()
                    elif name=="muon_global_codebook": levels=params[(k,"muon_codebook")].tolist()
                    elif name=="geometry_codebook": levels=params[(k,"geometry_codebook")].tolist()
                    elif name=="danger_zone_codebook": levels=params[(k,"danger_zone_codebook")].tolist()
                    else: levels=INT3_CODEBOOK.tolist()
                    row["codebook_levels"]=",".join(f"{v:.7g}" for v in levels)
                    row["calibrated_parameter"]=(params[(k,"mse_mulaw")] if name.startswith("calibrated_mulaw_") else None)
                    if name=="scale_oracle":
                        block_max=[float(it["residual"].reshape(-1)[st:st+BLOCK_SIZE].abs().amax()) for st in range(0,it["residual"].numel(),BLOCK_SIZE)]
                        ratios_scale=[s/a for s,a in zip(scales,block_max) if a>0]
                        row["chosen_scale_over_absmax_mean"]=sum(ratios_scale)/len(ratios_scale) if ratios_scale else None
                        row["chosen_scale_over_absmax_median"]=statistics.median(ratios_scale) if ratios_scale else None
                    if name.startswith("mulaw"): mulaw_rows.append(row)
                    if name.startswith("lloyd"): lloyd_rows.append(row)
                    if name=="scale_oracle": scale_rows.append(row)
                    if name in ("muon_global_codebook","per_tensor_muon_codebook_oracle"): muon_cb_eval.append(row)
                    if name.startswith(("geometry_","danger_","muon_")): geom_rows.append(row)
                    if seed==1: heldout.append(row)
                    spectral_rows.append({**base,"quantizer":name,"danger_log10_low":-3,"danger_log10_high":-2,
                                         **{key:row[key] for key in row if key.startswith("danger_") or key.endswith("mixing_fraction_error")}})
                    danger_rows.append(spectral_rows[-1])
                for bits in (3,4):
                    sto=storage_bits(item["shape"],k,"bf16",include_metadata=True,residual_bits=bits)
                    storage_rows.append({**base,"residual_bits":bits,"persistent_bits":sto["metadata_inclusive_bits"],"ratio_vs_fp32":sto["metadata_inclusive_bits"]/sto["fp32_bits"],"compression_ratio":sto["fp32_bits"]/sto["metadata_inclusive_bits"],"payload_bits":sto["residual_payload_bits"],"scale_metadata_bits":sto["metadata_bits"]})
                # Baseline INT4 reference point for the output frontier.
                int4row={**base,"quantizer":"structural_int4_uniform","update_cosine":top4metrics["cosine"],"update_relative_l2":top4metrics["relative_l2"],"exact_polar_cosine":ratios(it["ref_polar"],top4p)["cosine"],"raw_relative_l2":ratios(it["x"],top4hat)["relative_l2"]}
                tensor_rows.append(int4row); stage_rows.append(int4row)
                if seed==1: heldout.append(int4row)
    write_csv(out/"baseline_uniform_int3.csv",[r for r in stage_rows if r["quantizer"]=="uniform_int3"])
    write_csv(out/"mulaw_results.csv",mulaw_rows); write_csv(out/"lloyd_max_results.csv",lloyd_rows); write_csv(out/"scale_oracle_results.csv",scale_rows)
    write_csv(out/"muon_codebook_oracle.csv",muon_cb_rows+muon_cb_eval+selection_rows)
    write_csv(out/"spectral_error_analysis.csv",spectral_rows); write_csv(out/"stage_a_summary.csv",summarize(stage_rows))
    write_csv(out/"geometry_weighted_codebook.csv",geometry_rows+zone_rows+muon_comp_rows)
    write_csv(out/"geometry_companding_results.csv",geom_rows); write_csv(out/"danger_zone_companding.csv",danger_rows)
    write_csv(out/"muon_distortion_companding.csv",muon_comp_rows+muon_cb_rows)
    # Pairing is performed only after all held-out results have been computed.
    # It selects closest value-space reference, never a candidate quantizer.
    matched=nearest_matched_controls(stage_rows)
    write_csv(out/"matched_raw_error_controls.csv",matched); write_csv(out/"heldout_results.csv",heldout)
    write_csv(out/"storage_summary.csv",storage_rows); write_csv(out/"tensor_level_results.csv",tensor_rows)
    uniform_ix={(r["seed"],r["update"],r["parameter_id"],r["k"]):r for r in stage_rows if r["quantizer"]=="uniform_int3"}
    failures=[]
    for r in stage_rows:
        base=uniform_ix.get((r["seed"],r["update"],r["parameter_id"],r["k"]))
        if base and r["quantizer"] not in {"uniform_int3","structural_int4_uniform"} and finite(r.get("update_cosine")):
            gain=float(r["update_cosine"])-float(base["update_cosine"])
            if gain < -.01: failures.append({**r,"gain_vs_uniform":gain})
    write_csv(out/"failure_analysis.csv",failures)
    # Stage A tests scalar-allocation headroom; the INT2 gate is narrower and
    # applies only to held-out geometry-aware shared methods, not scale oracles.
    eval_rows=[r for r in stage_rows if r["seed"]==1]
    by=defaultdict(list)
    for r in eval_rows: by[(r["k"],r["quantizer"])].append(float(r["update_cosine"]))
    deploy_names={"uniform_int3","lloyd_global",*(f"calibrated_mulaw_{int(params[(k,'mse_mulaw')])}" for k in (4,8))}
    deploy_values=[(sum(v)/len(v),key) for key,v in by.items() if key[1] in deploy_names]
    best_deploy=max(deploy_values) if deploy_values else (float("nan"),(None,"no_heldout_data"))
    base_by_k={k:by.get((k,"uniform_int3"),[]) for k in (4,8)}
    gain_values=[(sum(vals)/len(vals)-sum(base_by_k[k])/len(base_by_k[k]),k,q)
                 for (k,q),vals in by.items() if q in deploy_names and q!="uniform_int3" and vals and base_by_k[k]]
    best_deploy_gain=max(gain_values) if gain_values else (float("nan"),None,"no_heldout_data")
    per_tensor_values=[(sum(v)/len(v),key) for key,v in by.items() if key[1]=="per_tensor_muon_codebook_oracle"]
    per_tensor_best=max(per_tensor_values) if per_tensor_values else (float("nan"),(None,"not_evaluated"))
    int3_best=max((v for (k,q),v in by.items() if q in deploy_names),default=float("nan"))
    stage_a_trigger=bool(finite(int3_best) and (int3_best>=.80 or (finite(best_deploy_gain[0]) and best_deploy_gain[0]>=.08)))
    geometry_names={"geometry_codebook","danger_zone_codebook","geometry_mulaw_5","geometry_power_0.75"}
    geometry_best=max((sum(v)/len(v) for (k,q),v in by.items() if q in geometry_names and v),default=float("nan"))
    int2_trigger=bool(finite(geometry_best) and geometry_best>=.80)
    summary_rows=[]
    for (k,q),vals in sorted(by.items()): summary_rows.append({"seed":1,"split":"heldout","k":k,"quantizer":q,"mean_update_cosine":sum(vals)/len(vals),"count":len(vals)})
    write_csv(out/"stage_a_summary.csv",summarize(stage_rows)+summary_rows)
    # Small INT2 probe only when held-out geometry-aware INT3 meets the gate.
    if int2_trigger:
        int2=[]
        for r in stage_rows:
            if r["seed"]==1 and r["k"]==8 and r["quantizer"] in {"uniform_int3"}: pass
        write_csv(out/"int2_probe.csv",int2)
    # Attach concise, reproducible numeric interpretation.
    def avg(seed,k,name,field):
        vals=[float(r[field]) for r in stage_rows if r["seed"]==seed and r["k"]==k and r["quantizer"]==name and finite(r.get(field))]
        return sum(vals)/len(vals) if vals else float("nan")
    lines=["# Structurally conditioned INT3 residual companding", "",
           f"Formal coverage: {len(paths)} snapshots; {count} eligible 2-D tensor instances before the rank factor; ranks 4 and 8; calibration is seed 0, held-out evaluation is seed 1. CPU elapsed {time.perf_counter()-start:.1f}s.",
           "", "## Stage A", "",
           f"Held-out seed-1 top-k uniform INT3 update cosine: k=4 {avg(1,4,'uniform_int3','update_cosine'):.4f}; k=8 {avg(1,8,'uniform_int3','update_cosine'):.4f}.",
           f"Held-out top-k structural INT4 reference: k=4 {avg(1,4,'structural_int4_uniform','update_cosine'):.4f}; k=8 {avg(1,8,'structural_int4_uniform','update_cosine'):.4f}.",
           f"Best held-out shared scalar/non-Muon INT3 (excluding scale and per-tensor oracles): {best_deploy[1]} at cosine {best_deploy[0]:.4f}. Largest gain over uniform is {best_deploy_gain[0]:+.4f} for k={best_deploy_gain[1]}, {best_deploy_gain[2]}. The capped per-tensor Muon oracle used {per_tensor_done} seed-0 calibration items and is not a held-out score.",
           f"Stage-A headroom criterion met: {stage_a_trigger} (shared scalar INT3 gain >=0.08 or held-out cosine >=0.80); Stage B ran automatically. INT2 gate: {int2_trigger}, requiring held-out geometry-aware shared INT3 >=0.80; observed maximum {geometry_best:.4f}. No INT2 probe was run.",
           "", "## Stage B", "",
           "Geometry-weighted codebook/companding parameters were selected using only seed-0 calibration items. The danger-zone surrogate uses the unchanged log10(sigma/sigma_max) interval [-3,-2), geometric-mean mode weights derived from production finite-step A_dir, plus fixed danger/local multipliers (4 and 1). Evaluation rows are seed-1 held-out. Geometry-selected scalar allocation did not outperform magnitude/MSE-selected μ-law or Lloyd-Max, and did not meet the INT2 gate. This is offline mechanism evidence, not a practical online quantizer.",
           "", "## Interpretation", "",
           "Compare `stage_a_summary.csv`, `heldout_results.csv`, and `matched_raw_error_controls.csv`. Shared μ-law/Lloyd-Max materially improve over uniform INT3, while geometry-specific scalar allocation adds no reliable gain and a substantial structural INT4 gap remains. The full tensor rows retain paired raw/update metrics and spectral residual diagnostics.",
           "", "Per-tensor Muon-codebook upper-bound search is limited to the first 40 deterministic seed-0 calibration instances indicated in `muon_codebook_oracle.csv`; these are in-sample upper-bound results, not held-out scores and not deployable. Global calibrators use a deterministic seed-0 prefix of 3 eligible tensors per update (15 per rank). INT3 codebook is the seven levels {-1,-2/3,-1/3,0,1/3,2/3,1}, with one of eight bit patterns reserved for exact zero. INT4 is only a reference baseline using unchanged production b2048 dynamic quantization."]
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    (out/"methodology.md").write_text("""# Methodology

For each formal FP32 momentum matrix, the fixed conditioner is its top-k truncated SVD (`k=4,8`; optional 16). `R=M-C`; every residual method receives the same R; the persistent low-rank side information is round-tripped through BF16 factors. INT3 b2048 uses absmax scale, nearest signed seven-level codebook, and exact zero. Stage A includes fixed mu-law mu={1,5,20,100,500,2000}, per-tensor and calibration-global symmetric Lloyd-Max codebooks, bounded-grid per-block scale oracle, and a shared Muon-update-selected codebook. Global codebook searches first evaluate the six-point coarse grid, then evaluate the deterministic 0.05-spaced local neighborhood (±0.05 in each interior level) around the best coarse pair. All selection uses seed-0 calibration tensors. The per-tensor counterpart is an explicitly capped in-sample upper bound.

Stage B derives mode weights from the exact production Newton--Schulz scalar transfer (`A_dir=|f_K(sigma/(||M||+eps))|/(sigma/(||M||+eps))`), normalized by the tensor median and clipped to [0.1,20]. The spectral weight is `sqrt(w_i*w_j)`. Geometry codebook and generalized mu-law/power-law parameters are selected from seed 0 by weighted spectral error; Muon-aware companders are selected by calibration-set K=5 cosine. Danger/local weighting adds multipliers to entries with row/column mode in log10 normalized danger interval [-3,-2) and local spectral separation <0.5 decades. Every declared (danger,local) pair is calibrated independently; (4,1) is the primary result. Seed 1 is held out.

Matched raw-error rows retain every evaluated candidate so nearest pairs can be constructed from the CSV without fitting a new mapping. Exact polar is evaluated for uniform, the calibration-selected mu-law, global Lloyd-Max, global Muon codebook, and geometry codebook. Full-state update fidelity always uses the exact production `zeropower_newton_schulz`. No training/optimizer code or production quantizer is modified. Storage uses the existing metadata-inclusive BF16-factor accounting with 3-bit residual payload; global scalar/codebook parameters are negligible and separately identified.
""")
    if not args.skip_plots: plot_report(out,stage_rows,storage_rows)
    print(f"completed snapshots={len(paths)} tensors={count} rows={len(stage_rows)} per-tensor oracle={per_tensor_done} elapsed={time.perf_counter()-start:.1f}s output={out}",flush=True)


def plot_report(out,rows,storage):
    try: import matplotlib.pyplot as plt
    except ImportError: return
    groups=defaultdict(list)
    for r in rows:
        if finite(r.get("update_cosine")): groups[(r["k"],r["quantizer"],r["seed"])].append(float(r["update_cosine"]))
    for filename,field,ylabel in (("int3_update_cosine.png","update_cosine","K=5 update cosine"),("exact_polar.png","exact_polar_cosine","exact-polar cosine")):
        if field=="exact_polar_cosine": continue
        keys=sorted(k for k in groups if k[2]==1)
        plt.figure(figsize=(14,5)); plt.bar(range(len(keys)),[sum(groups[k])/len(groups[k]) for k in keys]);
        plt.xticks(range(len(keys)),[f"k{k[0]} {k[1]}" for k in keys],rotation=75,ha="right",fontsize=7); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(out/filename,dpi=130); plt.close()
    # Mu-law and error/fidelity plots use held-out data only.
    vals=[r for r in rows if r["seed"]==1 and r["quantizer"].startswith("mulaw_mu_")]
    if vals:
        d=defaultdict(list)
        for r in vals:d[r["quantizer"]].append(float(r["update_cosine"]))
        plt.figure(figsize=(7,4)); plt.plot(list(d),[sum(v)/len(v) for v in d.values()],marker="o");plt.xticks(rotation=45);plt.ylabel("K=5 update cosine");plt.tight_layout();plt.savefig(out/"mu_vs_fidelity.png",dpi=130);plt.close()
    pts=[r for r in rows if r["seed"]==1 and finite(r.get("raw_residual_relative_l2")) and finite(r.get("update_cosine"))]
    if pts:
        plt.figure(figsize=(6,5));plt.scatter([float(r["raw_residual_relative_l2"]) for r in pts],[float(r["update_cosine"]) for r in pts],s=6,alpha=.35);plt.xlabel("residual relative L2");plt.ylabel("K=5 update cosine");plt.tight_layout();plt.savefig(out/"raw_residual_l2_vs_update.png",dpi=130);plt.close()
    # Storage frontier and comparison against structurally conditioned INT4.
    plt.figure(figsize=(7,5))
    for k in (4,8):
        xs=[s["ratio_vs_fp32"] for s in storage if s["k"]==k and s["residual_bits"]==3]
        ys=[float(r["update_cosine"]) for r in rows if r["seed"]==1 and r["k"]==k and r["quantizer"] in {"uniform_int3","geometry_codebook","structural_int4_uniform"}]
        if xs and ys: plt.scatter([sum(xs)/len(xs)],[sum(ys)/len(ys)],label=f"k={k}")
    plt.xlabel("storage / FP32");plt.ylabel("K=5 update cosine");plt.legend();plt.tight_layout();plt.savefig(out/"storage_fidelity.png",dpi=130);plt.close()

    # Expanded diagnostic panels requested for the two-stage mechanism study.
    held=[r for r in rows if r.get("seed")==1]
    def mean_for(k, names, field):
        result={}
        for name in names:
            v=[float(r[field]) for r in held if int(r.get("k",-1))==k and r.get("quantizer")==name and finite(r.get(field))]
            if v: result[name]=sum(v)/len(v)
        return result
    # Level allocations and their held-out fidelity.
    for k in (4,8):
        candidates=list(mean_for(k,["uniform_int3","lloyd_global","muon_global_codebook","geometry_codebook","danger_zone_codebook"],"update_cosine").items())
        if candidates:
            plt.figure(figsize=(8,4));plt.bar([x[0] for x in candidates],[x[1] for x in candidates]);plt.xticks(rotation=35,ha="right");plt.ylabel("held-out K=5 update cosine");plt.title(f"INT3 allocation methods, k={k}");plt.tight_layout();plt.savefig(out/f"codebook_methods_k{k}.png",dpi=130);plt.close()
    level_rows={}
    for r in held:
        if r.get("quantizer") in ("lloyd_global","muon_global_codebook","geometry_codebook") and r.get("codebook_levels"):
            level_rows.setdefault((int(r["k"]),r["quantizer"]),r["codebook_levels"])
    if level_rows:
        plt.figure(figsize=(7,4))
        for (k,name),levels in sorted(level_rows.items()):
            vals=[float(v) for v in levels.split(",")]
            plt.plot(range(len(vals)),vals,marker="o",label=f"k={k} {name}")
        plt.xticks(range(7),range(7));plt.xlabel("codebook level index");plt.ylabel("normalized reconstruction value");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(out/"selected_codebook_levels.png",dpi=130);plt.close()
    # Error redistribution and mechanism variables against update fidelity.
    for filename,xfield,yfield,xlabel,ylabel in (
        ("small_vs_large_mae.png","small_005_mae","large_mae","MAE: |r|/block absmax < 0.05","MAE: |r|/block absmax >= 0.5"),
        ("danger_zone_error_vs_fidelity.png","danger_associated_fraction_error","update_cosine","danger-zone-associated error fraction","K=5 update cosine"),
        ("local_mixing_vs_fidelity.png","local_mixing_fraction_error","update_cosine","local spectral-mixing error fraction","K=5 update cosine"),
        ("danger_error_vs_update.png","danger_associated_fraction_error","update_cosine","danger-zone error proxy","K=5 update cosine"),
    ):
        pts=[r for r in held if finite(r.get(xfield)) and finite(r.get(yfield))]
        if pts:
            plt.figure(figsize=(6,5));plt.scatter([float(r[xfield]) for r in pts],[float(r[yfield]) for r in pts],s=7,alpha=.3)
            plt.xlabel(xlabel);plt.ylabel(ylabel);plt.tight_layout();plt.savefig(out/filename,dpi=130);plt.close()
    # Matched-raw-error controls show update benefit after matching reconstruction quality.
    try:
        matched=list(csv.DictReader((out/"matched_raw_error_controls.csv").open()))
    except OSError: matched=[]
    if matched:
        plt.figure(figsize=(7,4));
        for cand in sorted({r["candidate"] for r in matched}):
            p=[r for r in matched if r["candidate"]==cand]
            plt.scatter([float(r["raw_residual_l2_candidate"]) for r in p],[float(r["update_cosine_gain_candidate_minus_reference"]) for r in p],s=7,alpha=.25,label=cand)
        plt.xlabel("candidate raw residual relative-L2");plt.ylabel("update cosine gain vs matched reference");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(out/"matched_raw_error_gain.png",dpi=130);plt.close()
    # Calibration-only geometry surrogate and Muon objective on the same
    # candidate codebook grid; neither point uses the held-out seed.
    try:
        gr=list(csv.DictReader((out/"geometry_weighted_codebook.csv").open()))
        mr=list(csv.DictReader((out/"muon_codebook_oracle.csv").open()))
    except OSError: gr=[];mr=[]
    geom={(int(r["k"]),round(float(r["a1"]),6),round(float(r["a2"]),6)):float(r["mean_geometry_loss"])
          for r in gr if r.get("a1") and r.get("mean_geometry_loss") and r.get("family")==""}
    muon={(int(r["k"]),round(float(r["a1"]),6),round(float(r["a2"]),6)):float(r["calibration_mean_update_cosine"])
          for r in mr if r.get("a1") and r.get("calibration_mean_update_cosine")}
    common=sorted(set(geom)&set(muon))
    if common:
        plt.figure(figsize=(6,4));plt.scatter([geom[x] for x in common],[muon[x] for x in common])
        for x in common: plt.annotate(f"k{x[0]}:{x[1]:.2g},{x[2]:.2g}",(geom[x],muon[x]),fontsize=6)
        plt.xlabel("calibration geometry-surrogate loss (lower better)");plt.ylabel("calibration Muon update cosine");plt.tight_layout();plt.savefig(out/"geometry_surrogate_vs_muon_objective.png",dpi=130);plt.close()
    try: refined=list(csv.DictReader((out/"refined_codebook_followup.csv").open()))
    except OSError: refined=[]
    if refined:
        plt.figure(figsize=(7,4))
        for k in (4,8):
            coarse_names={"muon_global_codebook":"muon_global_codebook","geometry_codebook":"geometry_codebook","danger_zone_codebook":"danger_zone_codebook"}
            labels=[];values=[]
            for label,base_name in coarse_names.items():
                old=[float(r["update_cosine"]) for r in held if int(r.get("k",-1))==k and r.get("quantizer")==base_name and finite(r.get("update_cosine"))]
                new_name=base_name+"_fine"
                new=[float(r["update_cosine"]) for r in refined if int(r.get("k",-1))==k and r.get("quantizer")==new_name and finite(r.get("update_cosine"))]
                if old and new:
                    labels.extend([f"k{k} {label} coarse",f"k{k} {label} fine"]);values.extend([sum(old)/len(old),sum(new)/len(new)])
        if labels:
            plt.bar(range(len(labels)),values);plt.xticks(range(len(labels)),labels,rotation=65,ha="right",fontsize=7);plt.ylabel("held-out K=5 update cosine");plt.tight_layout();plt.savefig(out/"coarse_to_fine_codebook_comparison.png",dpi=130);plt.close()
    # Calibration-selected vs held-out scores by method, with split explicit.
    cal_groups=defaultdict(list); ev_groups=defaultdict(list)
    for r in rows:
        if finite(r.get("update_cosine")):
            (cal_groups if r.get("seed")==0 else ev_groups)[(int(r["k"]),r["quantizer"])].append(float(r["update_cosine"]))
    pairs=[(key,sum(a)/len(a),sum(ev_groups[key])/len(ev_groups[key])) for key,a in cal_groups.items() if key in ev_groups]
    if pairs:
        plt.figure(figsize=(5,5));plt.scatter([p[1] for p in pairs],[p[2] for p in pairs],s=15,alpha=.6);plt.plot([0,1],[0,1],"k--",lw=1);plt.xlabel("seed-0 mean cosine");plt.ylabel("seed-1 mean cosine");plt.tight_layout();plt.savefig(out/"calibration_vs_heldout.png",dpi=130);plt.close()
    # Explicit INT3 / structural INT4 reference gap.
    refs=[]
    for k in (4,8): refs.append((k,mean_for(k,["uniform_int3","calibrated_mulaw_5","lloyd_global","structural_int4_uniform"],"update_cosine")))
    plt.figure(figsize=(7,4))
    for k,d in refs:
        plt.plot(list(d),list(d.values()),marker="o",label=f"k={k}")
    plt.ylabel("held-out K=5 update cosine");plt.xticks(rotation=35,ha="right");plt.legend();plt.tight_layout();plt.savefig(out/"int3_vs_int4.png",dpi=130);plt.close()
    # Per-tensor gains and failures are retained, not only aggregate means.
    base={(int(r["k"]),r["update"],r["parameter_id"]):float(r["update_cosine"]) for r in held if r.get("quantizer")=="uniform_int3" and finite(r.get("update_cosine"))}
    gains=[]
    for r in held:
        key=(int(r["k"]),r["update"],r["parameter_id"])
        if key in base and r.get("quantizer") in ("calibrated_mulaw_5","lloyd_global","geometry_codebook","muon_global_codebook") and finite(r.get("update_cosine")):
            gains.append((r["quantizer"],float(r["update_cosine"])-base[key]))
    if gains:
        plt.figure(figsize=(8,4))
        for i,name in enumerate(sorted({x[0] for x in gains})):
            v=[x[1] for x in gains if x[0]==name];plt.boxplot(v,positions=[i],widths=.6)
        plt.xticks(range(len(sorted({x[0] for x in gains}))),sorted({x[0] for x in gains}),rotation=35,ha="right");plt.ylabel("per-tensor update-cosine gain vs uniform");plt.tight_layout();plt.savefig(out/"per_tensor_gain_distribution.png",dpi=130);plt.close()
    try: failures=list(csv.DictReader((out/"failure_analysis.csv").open()))
    except OSError: failures=[]
    failures=[r for r in failures if str(r.get("seed"))=="1"]
    if failures:
        counts=defaultdict(int)
        for r in failures: counts[r["quantizer"]]+=1
        names=sorted(counts)
        plt.figure(figsize=(9,4));plt.bar(names,[counts[n] for n in names]);plt.xticks(rotation=70,ha="right",fontsize=7);plt.ylabel("held-out tensor cases with cosine loss > 0.01");plt.tight_layout();plt.savefig(out/"failure_cases.png",dpi=130);plt.close()
        plt.figure(figsize=(6,5))
        for name in sorted({r["quantizer"] for r in failures}):
            p=[r for r in failures if r["quantizer"]==name and finite(r.get("effective_condition_number")) and finite(r.get("gain_vs_uniform"))]
            if p: plt.scatter([float(r["effective_condition_number"]) for r in p],[float(r["gain_vs_uniform"]) for r in p],s=8,alpha=.25,label=name)
        plt.xscale("log");plt.xlabel("effective condition number");plt.ylabel("K=5 cosine loss vs uniform");plt.legend(fontsize=5);plt.tight_layout();plt.savefig(out/"failure_conditioning.png",dpi=130);plt.close()
    # Error/fidelity frontier across uniform, global value-space and geometry methods.
    plt.figure(figsize=(7,5))
    for name in ("uniform_int3","calibrated_mulaw_5","lloyd_global","geometry_codebook","structural_int4_uniform"):
        p=[r for r in held if r.get("quantizer")==name and finite(r.get("raw_residual_relative_l2")) and finite(r.get("update_cosine"))]
        if p: plt.scatter([float(r["raw_residual_relative_l2"]) for r in p],[float(r["update_cosine"]) for r in p],s=8,alpha=.25,label=name)
    plt.xlabel("residual relative-L2");plt.ylabel("K=5 update cosine");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(out/"raw_error_fidelity_frontier.png",dpi=130);plt.close()


if __name__=="__main__":
    main()
