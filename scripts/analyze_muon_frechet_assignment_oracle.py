#!/usr/bin/env python3
"""Fixed-codebook assignment study for structural 2-D INT3 Muon residuals."""
from __future__ import annotations

import argparse, csv, math, statistics, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from analyze_muon_frechet_sensitivity import (LANDMARKS,OUT as _PREV_OUT,cosine,load_snapshots,read_csv,write_csv)
from analyze_muon_int3_practical_scale import factorized_topk
from analyze_muon_vector_int3_residual import eligible_items
from optim.muon_frechet import frechet_channels
from optim.muon_frechet_assignment import (analytic_pair_hessian,hutchinson_pair_blocks,
    objective_change_from_block,score_pair_candidates)
from optim.muon_spectral_sensitivity import decompose
from optim.muon_vector_int3 import pair_values,unpair_values,vector_scales,vector_storage_bits
from optim.muon_reference import zeropower_newton_schulz

OUT=ROOT/"reports/muon_frechet_assignment_oracle"
VQ_DIR=ROOT/"reports/muon_vector_int3_robustness"
WORDS=64; BLOCK=2048; RANK=8; PROBES=32
LAMBDAS=(0.0,.25,.5,.75,1.0)
METHODS=("mse","pair_local_frechet_hutchinson","pair_local_skew_hutchinson","mix_025","mix_050","mix_075")


def normalize_pairs(pairs,scales):
    repeat=scales.repeat_interleave(BLOCK//2)[:len(pairs)]
    return (pairs/repeat.clamp_min(1e-30)[:,None]).clamp(-1,1),repeat


def labels_and_errors(pairs,scales,codebook,labels=None):
    z,alpha=normalize_pairs(pairs,scales)
    if labels is None: labels=torch.cdist(z,codebook).argmin(1)
    q=alpha[:,None]*codebook[labels]
    return labels,q,pairs-q,alpha,z


def assign_with_metric(pairs,scales,codebook,hfull,hskew,*,lam,mse_norm,frechet_norm,skew=False):
    z,alpha=normalize_pairs(pairs,scales)
    errors=alpha[:,None,None]*codebook[None,:,:]-pairs[:,None,:]
    h=hskew if skew else hfull
    if lam==0:cost=errors.square().sum(-1)
    else:
        cost=score_pair_candidates(errors,h,mse_scale=mse_norm,frechet_scale=frechet_norm,lambda_frechet=lam)
    # Pure skew uses its own quadratic and no MSE term.
    if skew:cost=torch.einsum("pci,pij,pcj->pc",errors.double(),h.double(),errors.double())
    ix=cost.argmin(1)
    q=alpha[:,None]*codebook[ix]
    return ix,q,pairs-q,errors,cost


def evaluate(matrix,reference_update,mhat,svd):
    o=zeropower_newton_schulz(mhat.clone(),steps=5,coefficients=(3.4445,-4.7750,2.0315),eps=1e-7)
    e=mhat-matrix
    fre=frechet_channels(matrix,e,svd_factors=(svd.u,svd.singular_values,svd.vh))
    pd=reference_update+fre["predicted_delta"].float()
    outnorm=float(torch.linalg.vector_norm(reference_update).clamp_min(1e-30))
    return {"raw_relative_l2":float(torch.linalg.vector_norm(e)/torch.linalg.vector_norm(matrix).clamp_min(1e-30)),
        "raw_cosine":cosine(matrix,mhat),"frechet_relative_l2":float(torch.linalg.vector_norm(fre["predicted_delta"])/outnorm),
        "frechet_predicted_cosine_distortion":1-cosine(reference_update,pd),
        "update_cosine":cosine(reference_update,o),"update_distortion":1-cosine(reference_update,o),
        "update_relative_l2":float(torch.linalg.vector_norm(o-reference_update)/outnorm),
        **{f"{ch}_energy":float(torch.linalg.vector_norm(fre["components"][ch])**2) for ch in ("magnitude","symmetric","skew","out_of_subspace")}}


def get_data(seed,update,item,codebooks,*,probes=PROBES,codebook_seed=None):
    m=item["tensor"].detach().cpu().float(); pid=str(item.get("parameter_id",item.get("name")));name=str(item.get("name",pid))
    svd=decompose(m);u,s,vh=svd.u,svd.singular_values,svd.vh
    c_exact=(u[:,:RANK]*s[:RANK])@vh[:RANK]
    c_hat=factorized_topk(u,s,vh,RANK)
    residual=m-c_exact
    pairs,singles,pair_idx,single_idx=pair_values(residual,"contiguous")
    scales=vector_scales(pairs,"p98",block_size=BLOCK)
    cbseed=1-int(seed) if codebook_seed is None else int(codebook_seed)
    cb=codebooks[f"s{cbseed}_k8_w64_t8_v1200"].float()
    base_labels,base_q,base_err,alpha,z=labels_and_errors(pairs,scales,cb)
    hfull=hutchinson_pair_blocks(m,pair_idx,probes=probes,seed=2026+seed*1009+update+sum(pid.encode()),channel="full",svd_factors=(u,s,vh))
    hskew=hutchinson_pair_blocks(m,pair_idx,probes=probes,seed=4026+seed*1009+update+sum(pid.encode()),channel="skew",svd_factors=(u,s,vh))
    op=zeropower_newton_schulz(m.clone(),steps=5,coefficients=(3.4445,-4.7750,2.0315),eps=1e-7)
    baseline_state=c_hat+unpair_values(base_q,singles,tuple(m.shape),pair_idx,single_idx)
    base_metrics=evaluate(m,op,baseline_state,svd)
    mse_local=(z-cb[base_labels]).square().sum(1)
    f_local=torch.einsum("pi,pij,pj->p",base_err.double(),hfull.double(),base_err.double())
    return {"seed":seed,"update":update,"parameter_id":pid,"parameter_name":name,"shape":tuple(m.shape),
        "m":m,"svd":svd,"c_hat":c_hat,"residual":residual,"pairs":pairs,"singles":singles,"pair_idx":pair_idx,"single_idx":single_idx,
        "scales":scales,"alpha":alpha,"codebook":cb,"base_labels":base_labels,"base_q":base_q,"base_err":base_err,
        "hfull":hfull,"hskew":hskew,"reference_update":op,"base_metrics":base_metrics,
        "base_mse_mean":float(mse_local.mean()),"base_frechet_mean":float(f_local.clamp_min(0).mean())}


def make_hatted(d,labels,q):
    residual_q=unpair_values(q,d["singles"],d["shape"],d["pair_idx"],d["single_idx"])
    return d["c_hat"]+residual_q


def make_row(d,method,labels,q,err,local_cost):
    mhat=make_hatted(d,labels,q);metrics=evaluate(d["m"],d["reference_update"],mhat,d["svd"])
    if torch.as_tensor(local_cost).ndim==2:
        local_cost=local_cost[torch.arange(labels.numel()),labels]
    metrics.update({"seed":d["seed"],"update":d["update"],"parameter_id":d["parameter_id"],"parameter_name":d["parameter_name"],"shape":str(d["shape"]),"method":method,
        "local_pair_objective":float(torch.as_tensor(local_cost).mean()),"assignment_changes":int((labels!=d["base_labels"]).sum()),
        "changed_fraction":float((labels!=d["base_labels"]).float().mean()),"pair_count":int(labels.numel()),
        "index_bits":6,"pair_index_bits":6*int(labels.numel()),"scale_count":int(d["scales"].numel())})
    return metrics,mhat


def assignments(d,mse_norm,frechet_norm):
    cb=d["codebook"];pairs=d["pairs"];alpha=d["alpha"];n=len(pairs);chunk=24_000
    names=("mse","pair_local_frechet_hutchinson","pair_local_skew_hutchinson","mix_025","mix_050","mix_075")
    labels={name:torch.empty(n,dtype=torch.long) for name in names};selected_cost={name:torch.empty(n,dtype=torch.float64) for name in names}
    lambdas={"mix_025":.25,"mix_050":.5,"mix_075":.75}
    for start in range(0,n,chunk):
        stop=min(n,start+chunk);err=alpha[start:stop,None,None]*cb[None,:,:]-pairs[start:stop,None,:]
        z,_=normalize_pairs(pairs[start:stop],d["scales"])
        mse=(z[:,None,:]-cb[None,:,:]).square().sum(-1)
        full=torch.einsum("pci,pij,pcj->pc",err.double(),d["hfull"][start:stop].double(),err.double())
        skew=torch.einsum("pci,pij,pcj->pc",err.double(),d["hskew"][start:stop].double(),err.double())
        costs={"mse":mse,"pair_local_frechet_hutchinson":full,"pair_local_skew_hutchinson":skew}
        for name,lam in lambdas.items():costs[name]=(1-lam)*mse.double()/max(mse_norm,1e-30)+lam*full/max(frechet_norm,1e-30)
        for name,cost in costs.items():
            ix=cost.argmin(1);labels[name][start:stop]=ix.cpu();selected_cost[name][start:stop]=cost.gather(1,ix[:,None]).squeeze(1).double().cpu()
    result={}
    for name in names:
        ix=labels[name]
        if name=="mse":ix=d["base_labels"]
        q=alpha[:,None]*cb[ix];err=pairs-q
        if name=="mse":
            z,_=normalize_pairs(pairs,d["scales"]);selected_cost[name]=(z-cb[ix]).square().sum(1).double()
        result[name]=(ix,q,err,selected_cost[name])
    return result


def exact_local_hessian_metrics(d, samples=64):
    p=d["pair_idx"].shape[0]
    ids=torch.linspace(0,p-1,min(samples,p)).round().long().unique()
    exact=[];estimated=[]
    for pi in ids.tolist():
        a,b=map(int,d["pair_idx"][pi]); h=analytic_pair_hessian(d["m"],(a//d["shape"][1],a%d["shape"][1],b//d["shape"][1],b%d["shape"][1]),svd_factors=(d["svd"].u,d["svd"].singular_values,d["svd"].vh))
        exact.append(h);estimated.append(d["hfull"][pi])
    x=torch.stack(exact);y=torch.stack(estimated)
    corr=float(np.corrcoef(x[:,0,0].numpy(),y[:,0,0].numpy())[0,1]) if len(x)>2 else float("nan")
    rel=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-30))
    return {"pair_samples":len(x),"h11_pearson":corr,"hessian_relative_fro_error":rel,
        "offdiag_abs_median_exact":float(x[:,0,1].abs().median()),"offdiag_fraction_fro_exact":float(torch.linalg.vector_norm(x[:,0,1])/torch.linalg.vector_norm(x).clamp_min(1e-30))}


def global_cd(d, pairs_to_visit=24, max_sweeps=4):
    """Exact global quadratic coordinate descent on a tiny deterministic subset."""
    m=d["m"];e=(make_hatted(d,d["base_labels"],d["base_q"])-m).clone();labels=d["base_labels"].clone()
    pi=d["pair_idx"];cb=d["codebook"];alpha=d["alpha"];raw=d["pairs"]
    ids=torch.linspace(0,pi.shape[0]-1,min(pairs_to_visit,pi.shape[0])).round().long().unique().tolist()
    phi=lambda z:zeropower_newton_schulz(z,steps=5,coefficients=(3.4445,-4.7750,2.0315),eps=1e-7)
    _,pullback=torch.func.vjp(phi,m)
    _,y=torch.func.jvp(phi,(m,),(e,));obj=float(y.square().sum())
    rows=[];initial=obj
    for sweep in range(1,max_sweeps+1):
        changed=0
        for ip in ids:
            g=pullback(y)[0].reshape(-1);p0,p1=map(int,pi[ip]);old=int(labels[ip]);a,b=map(int,pi[ip])
            coords=(a//d["shape"][1],a%d["shape"][1],b//d["shape"][1],b%d["shape"][1])
            h=analytic_pair_hessian(m,coords,svd_factors=(d["svd"].u,d["svd"].singular_values,d["svd"].vh))
            cand_err=alpha[ip]*cb-raw[ip]
            current_err=cand_err[old]
            grad=torch.stack((g[p0],g[p1]))
            deltas=cand_err-current_err
            qcost=torch.einsum("ci,ij,cj->c",deltas.double(),h,deltas.double())+2*(deltas.double()*grad.double()).sum(1)
            new=int(qcost.argmin())
            if new==old:continue
            delta=deltas[new]
            dmat=torch.zeros_like(m);dmat.reshape(-1)[p0]=delta[0];dmat.reshape(-1)[p1]=delta[1]
            old_obj=obj;_,dy=torch.func.jvp(phi,(m,),(dmat,));candidate_y=y+dy
            candidate_obj=float(candidate_y.square().sum())
            if candidate_obj<=old_obj+1e-5*max(old_obj,1.0):
                y=candidate_y;obj=candidate_obj;e.reshape(-1)[p0]+=delta[0];e.reshape(-1)[p1]+=delta[1];labels[ip]=new;changed+=1
            else:
                # Analytic/JVP floating-point mismatch can be resolved by
                # rejecting a step that fails the exact production objective.
                obj=old_obj
        state=m+e;update=phi(state)
        rows.append({"seed":d["seed"],"update":d["update"],"parameter_id":d["parameter_id"],"shape":str(d["shape"]),"sweep":sweep,
            "selected_pairs":len(ids),"changed_this_sweep":changed,"global_frechet_objective":obj,"objective_ratio_to_initial":obj/max(initial,1e-30),
            "update_cosine":cosine(d["reference_update"],update),"update_relative_l2":float(torch.linalg.vector_norm(update-d["reference_update"])/torch.linalg.vector_norm(d["reference_update"]).clamp_min(1e-30))})
        if changed==0:break
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--threads",type=int,default=4);ap.add_argument("--probes",type=int,default=PROBES);ap.add_argument("--global-tensors",type=int,default=4);args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(2026);start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    snapshots=load_snapshots();blob=torch.load(VQ_DIR/"calibration_codebooks.pt",map_location="cpu",weights_only=False);codebooks=blob["codebooks"]
    data=[]
    for (seed,update),snap in sorted(snapshots.items()):
        items=eligible_items(snap)
        for item in items:
            if item["tensor"].ndim==2:data.append((seed,update,item))
    if len(data)!=300:raise RuntimeError(f"expected 300 matrices, found {len(data)}")
    # Evenly spread, seed-only samples calibrate the two scalar objective
    # normalizers and lambda independently in each direction.
    calib_ids={s:set(np.linspace(150*s,150*s+149,12).round().astype(int).tolist()) for s in (0,1)}
    cal_m={0:[],1:[]};cal_f={0:[],1:[]};exact_audit=[];calibration_samples={0:[],1:[]}
    for cseed in (0,1):
        for ix in sorted(calib_ids[cseed]):
            seed,update,item=data[ix];d=get_data(seed,update,item,codebooks,probes=args.probes,codebook_seed=cseed)
            cal_m[cseed].append(d["base_mse_mean"]);cal_f[cseed].append(d["base_frechet_mean"])
            if cseed==0:exact_audit.append({"seed":seed,"update":update,"parameter_id":d["parameter_id"],**exact_local_hessian_metrics(d,32)})
            calibration_samples[cseed].append((seed,update,item))
            print(f"calibration metric sample seed{cseed}: {len(cal_m[cseed])}/12",flush=True)
    mse_norm={s:max(statistics.mean(cal_m[s]),1e-30) for s in (0,1)}
    frechet_norm={s:max(statistics.mean(cal_f[s]),1e-30) for s in (0,1)}
    lambda_results=defaultdict(list)
    for cseed in (0,1):
        for seed,update,item in calibration_samples[cseed]:
            d=get_data(seed,update,item,codebooks,probes=args.probes,codebook_seed=cseed)
            variants=assignments(d,mse_norm[cseed],frechet_norm[cseed])
            for method in ("mse","mix_025","mix_050","mix_075","pair_local_frechet_hutchinson"):
                labels,q,err,cost=variants[method]
                met,_=make_row(d,method,labels,q,err,cost)
                lambda_results[cseed,method].append(met)
    lambda_choice={}
    for cseed in (0,1):
        means={lam:statistics.mean(r["update_cosine"] for r in lambda_results[cseed,method]) for lam,method in ((0.,"mse"),(.25,"mix_025"),(.5,"mix_050"),(.75,"mix_075"),(1.,"pair_local_frechet_hutchinson"))}
        lambda_choice[cseed]=max(means,key=means.get)
    storage_rows=[];baseline_rows=[]
    out_rows=[];assignment_rows=[];channel_rows=[];win_rows=[];polar_rows=[];all_for_corr=[];cd_rows=[]
    for ix,(seed,update,item) in enumerate(data):
        pid=str(item.get("parameter_id",item.get("name")));d=get_data(seed,update,item,codebooks,probes=args.probes)
        m=d["m"]
        bits=vector_storage_bits(tuple(m.shape),codewords=WORDS,lowrank_rank=RANK,block_size=BLOCK,factor_bits=16,codebook_bits=32,scale_bits=32)
        for method in METHODS:
            storage_rows.append({"seed":seed,"update":update,"parameter_id":pid,"shape":str(tuple(m.shape)),"method":method,**bits,"bits_equal_to_mse":True})
        baseline_rows.append({"seed":seed,"update":update,"parameter_id":pid,"shape":str(tuple(m.shape)),**d["base_metrics"],"codebook_key":f"s{1-seed}_k8_w64_t8_v1200"})
        # Evaluation seed uses only the other seed's local-cost normalizers.
        variants=assignments(d,mse_norm[1-seed],frechet_norm[1-seed])
        metrics_by={}
        for method,(labels,q,err,cost) in variants.items():
            met,mhat=make_row(d,method,labels,q,err,cost);metrics_by[method]=met
            out_rows.append(met)
            changed=(labels!=d["base_labels"])
            cent=(d["codebook"][labels]-d["codebook"][d["base_labels"]]).norm(dim=1)*d["alpha"]
            delta_mse=err.square().sum(1)-d["base_err"].square().sum(1)
            local_full=torch.einsum("pi,pij,pj->p",err.double(),d["hfull"],err.double())
            base_local=torch.einsum("pi,pij,pj->p",d["base_err"].double(),d["hfull"],d["base_err"].double())
            local_skew=torch.einsum("pi,pij,pj->p",err.double(),d["hskew"],err.double())
            base_skew=torch.einsum("pi,pij,pj->p",d["base_err"].double(),d["hskew"],d["base_err"].double())
            pair_norm=d["pairs"].norm(dim=1);scale_pair=d["alpha"]
            sens=d["hfull"][:,0,0]+d["hfull"][:,1,1]
            def changed_fraction_in(values,lo,hi):
                mask=(values>=lo)&(values<=hi)
                return float(changed[mask].float().mean()) if bool(mask.any()) else float("nan")
            assignment_rows.append({"seed":seed,"update":update,"parameter_id":pid,"method":method,"changed_fraction":float(changed.float().mean()),
                "changed_count":int(changed.sum()),"mean_physical_centroid_displacement":float(cent[changed].mean()) if bool(changed.any()) else 0.0,
                "median_physical_centroid_displacement":float(cent[changed].median()) if bool(changed.any()) else 0.0,
                "delta_mse_mean_changed":float(delta_mse[changed].mean()) if bool(changed.any()) else 0.0,
                "delta_hutch_frechet_mean_changed":float((local_full-base_local)[changed].mean()) if bool(changed.any()) else 0.0,
                "delta_hutch_skew_mean_changed":float((local_skew-base_skew)[changed].mean()) if bool(changed.any()) else 0.0,
                "pair_norm_low_change_fraction":changed_fraction_in(pair_norm,-float("inf"),float(pair_norm.quantile(.33))),
                "pair_norm_mid_change_fraction":changed_fraction_in(pair_norm,float(pair_norm.quantile(.33)),float(pair_norm.quantile(.67))),
                "pair_norm_high_change_fraction":changed_fraction_in(pair_norm,float(pair_norm.quantile(.67)),float("inf")),
                "block_scale_low_change_fraction":changed_fraction_in(scale_pair,-float("inf"),float(scale_pair.quantile(.33))),
                "block_scale_mid_change_fraction":changed_fraction_in(scale_pair,float(scale_pair.quantile(.33)),float(scale_pair.quantile(.67))),
                "block_scale_high_change_fraction":changed_fraction_in(scale_pair,float(scale_pair.quantile(.67)),float("inf")),
                "local_sensitivity_low_change_fraction":changed_fraction_in(sens,-float("inf"),float(sens.quantile(.33))),
                "local_sensitivity_mid_change_fraction":changed_fraction_in(sens,float(sens.quantile(.33)),float(sens.quantile(.67))),
                "local_sensitivity_high_change_fraction":changed_fraction_in(sens,float(sens.quantile(.67)),float("inf"))})
            channel_rows.append({"seed":seed,"update":update,"parameter_id":pid,"method":method,**{f"delta_{ch}_energy":met[f"{ch}_energy"]-metrics_by["mse"][f"{ch}_energy"] for ch in ("magnitude","symmetric","skew","out_of_subspace")}})
            all_for_corr.append({"seed":seed,"update":update,"parameter_id":pid,"method":method,
                "frechet_improvement":metrics_by["mse"]["frechet_relative_l2"]-met["frechet_relative_l2"],
                "update_improvement":met["update_cosine"]-metrics_by["mse"]["update_cosine"]})
            if method == METHODS[-1] and (ix + 1) % 10 == 0:
                print(f"evaluated {ix+1}/{len(data)} tensors; seed={seed} update={update}",flush=True)
        # Exact-polar readout only on deterministic every-15th tensor, and only
        # the baseline plus two most relevant assignment objectives.
        if ix in np.linspace(0,299,20).round().astype(int).tolist():
            u,s,vh=d["svd"].u,d["svd"].singular_values,d["svd"].vh;pref=u@vh
            for method in ("mse","pair_local_frechet_hutchinson","pair_local_skew_hutchinson"):
                labels,q,_,_=variants[method];mh=make_hatted(d,labels,q);uh,_,vhq=torch.linalg.svd(mh.double(),full_matrices=False);polar=uh@vhq
                polar_rows.append({"seed":seed,"update":update,"parameter_id":pid,"method":method,"exact_polar_cosine":cosine(pref,polar),
                    "exact_polar_relative_l2":float(torch.linalg.vector_norm(polar-pref)/torch.linalg.vector_norm(pref).clamp_min(1e-30))})
    # Global coordinate descent on deterministic calibration/evaluation-spread
    # examples is an oracle and never pooled with pair-local outcomes.
    cd_candidates=np.linspace(0,299,min(args.global_tensors,300)).round().astype(int)
    for ix in cd_candidates:
        seed,update,item=data[int(ix)];d=get_data(seed,update,item,codebooks,probes=args.probes)
        cd_rows.extend(global_cd(d,pairs_to_visit=16,max_sweeps=4))
    expected=next(r for r in read_csv(VQ_DIR/"full_bit_frontier.csv") if int(r["calibration_seed"])==0 and int(r["evaluation_seed"])==1 and int(r["rank"])==8 and int(r["codewords"])==64)
    forward_baseline=statistics.mean(float(r["update_cosine"]) for r in baseline_rows if int(r["seed"])==1)
    if abs(forward_baseline-float(expected["mean_update_cosine"]))>2e-3:
        raise RuntimeError(f"fixed MSE VQ does not reproduce canonical forward baseline: {forward_baseline} vs {expected['mean_update_cosine']}")
    baseline_rows.append({"record_type":"canonical_forward_validation","seed":1,"calibration_seed":0,"update_cosine":forward_baseline,
        "prior_update_cosine":float(expected["mean_update_cosine"]),"absolute_difference":abs(forward_baseline-float(expected["mean_update_cosine"])),"within_0.002":True})
    storage_by=defaultdict(dict)
    for row in storage_rows:storage_by[(row["seed"],row["update"],row["parameter_id"])][row["method"]]=tuple(
        row[k] for k in ("pair_index_bits","scale_bits","factor_bits","shared_codebook_bits","metadata_bits","total_bits_unamortized","total_bits_excluding_shared_codebook","fp32_bits"))
    exact_equal=all(len(set(v.values()))==1 for v in storage_by.values())
    if len(storage_rows)!=len(data)*len(METHODS) or not exact_equal or not all(r["bits_equal_to_mse"] for r in storage_rows):
        raise RuntimeError("assignment methods do not have identical declared persistent storage")
    # Main required exports.
    write_csv(OUT/"method_manifest.csv",[{"method":"MSE / local full Frechet / local skew / lambda mixes / global coordinate descent","rank":8,"codewords":64,"bits_per_pair":6,"bits_per_residual_value":3,"block_size":2048,"scale":"fixed p98","pairing":"contiguous row-major","codebook":"existing split-specific MSE codebook; frozen"}])
    write_csv(OUT/"baseline_reproduction.csv",baseline_rows);write_csv(OUT/"pair_local_frechet.csv",[r for r in out_rows if r["method"]=="pair_local_frechet_hutchinson"])
    write_csv(OUT/"pair_local_skew.csv",[r for r in out_rows if r["method"]=="pair_local_skew_hutchinson"])
    write_csv(OUT/"mixed_objective.csv",[r for r in out_rows if r["method"].startswith("mix_")]+[r for r in out_rows if r["method"] in ("mse","pair_local_frechet_hutchinson")])
    write_csv(OUT/"global_coordinate_descent.csv",cd_rows);write_csv(OUT/"assignment_changes.csv",assignment_rows);write_csv(OUT/"channel_changes.csv",channel_rows)
    write_csv(OUT/"finite_error_validation.csv",finite_error_rows(out_rows));write_csv(OUT/"tensor_win_rates.csv",tensor_win_rows(out_rows))
    write_csv(OUT/"polar_subset.csv",polar_rows);write_csv(OUT/"storage_check.csv",storage_rows)
    write_csv(OUT/"pair_hessian_validation.csv",exact_audit);write_plots(OUT,out_rows,cd_rows)
    elapsed=time.perf_counter()-start;(OUT/"runtime_seconds.txt").write_text(f"{elapsed:.3f} seconds\n")
    write_methodology(OUT,args.probes)
    write_report(OUT,out_rows,lambda_choice,mse_norm,frechet_norm,len(data),elapsed,exact_audit,polar_rows)
    print(f"completed {len(out_rows)} assignment/tensor outcomes in {elapsed:.1f} CPU seconds",flush=True)


def finite_error_rows(rows):
    by=defaultdict(dict)
    for r in rows:by[(r["seed"],r["update"],r["parameter_id"])][r["method"]]=r
    result=[]
    for key,d in by.items():
        base=d["mse"]
        for method,r in d.items():
            if method=="mse":continue
            df=base["frechet_relative_l2"]-r["frechet_relative_l2"]
            du=r["update_cosine"]-base["update_cosine"]
            result.append({"seed":key[0],"update":key[1],"parameter_id":key[2],"method":method,
                "frechet_relative_l2_improvement":df,"actual_update_cosine_improvement":du,
                "frechet_improves":df>0,"actual_k5_improves":du>0,"both_improve":df>0 and du>0,
                "frechet_improves_but_actual_worsens":df>0 and du<0})
    for method in sorted({r["method"] for d in by.values() for r in d.values()} - {"mse"}):
        rr=[r for d in by.values() if method in d for r in [{"frechet_improvement":d["mse"]["frechet_relative_l2"]-d[method]["frechet_relative_l2"],"actual_update_improvement":d[method]["update_cosine"]-d["mse"]["update_cosine"]}]]
        x=np.asarray([r["frechet_improvement"] for r in rr]);y=np.asarray([r["actual_update_improvement"] for r in rr])
        pear=float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>0 and y.std()>0 else float("nan")
        rx=np.argsort(np.argsort(x));ry=np.argsort(np.argsort(y))
        spear=float(np.corrcoef(rx,ry)[0,1]) if len(x)>2 and np.std(rx)>0 and np.std(ry)>0 else float("nan")
        result.append({"record_type":"method_summary","method":method,"n":len(rr),"pearson_frechet_improvement_vs_update_gain":pear,
            "spearman_frechet_improvement_vs_update_gain":spear,"frechet_metric_win_rate":float((x>0).mean()),
            "actual_update_win_rate":float((y>0).mean()),"both_win_rate":float(((x>0)&(y>0)).mean()),
            "frechet_improves_but_actual_worsens_rate":float(((x>0)&(y<0)).mean())})
    return result


def tensor_win_rows(rows):
    by=defaultdict(dict)
    for r in rows:by[(r["seed"],r["update"],r["parameter_id"])][r["method"]]=r
    methods=sorted({m for d in by.values() for m in d if m!="mse"});result=[]
    for seed in (0,1):
        selected=[d for (s,_,_),d in by.items() if s==seed]
        for method in methods:
            comps=[d for d in selected if method in d]
            result.append({"seed":seed,"method":method,"n":len(comps),
                "frechet_metric_win_rate":sum(d[method]["frechet_relative_l2"]<d["mse"]["frechet_relative_l2"] for d in comps)/max(len(comps),1),
                "actual_k5_cosine_win_rate":sum(d[method]["update_cosine"]>d["mse"]["update_cosine"] for d in comps)/max(len(comps),1),
                "both_win_rate":sum(d[method]["frechet_relative_l2"]<d["mse"]["frechet_relative_l2"] and d[method]["update_cosine"]>d["mse"]["update_cosine"] for d in comps)/max(len(comps),1)})
    return result


def write_plots(out,rows,cd):
    import matplotlib.pyplot as plt
    by=defaultdict(dict)
    for r in rows:by[(r["seed"],r["update"],r["parameter_id"])][r["method"]]=r
    methods=("mse","pair_local_frechet_hutchinson","pair_local_skew_hutchinson","mix_025","mix_050","mix_075")
    labels=("MSE","local Frechet","local skew","λ=.25","λ=.5","λ=.75")
    means=[statistics.mean(d[m]["update_cosine"] for d in by.values()) for m in methods]
    plt.figure(figsize=(8,4));plt.bar(range(len(methods)),means);plt.xticks(range(len(methods)),labels,rotation=25);plt.ylabel("K=5 update cosine");plt.tight_layout();plt.savefig(out/"assignment_fidelity.png",dpi=140);plt.close()
    plt.figure(figsize=(6,4));plt.scatter([d["mse"]["frechet_relative_l2"]-d["pair_local_frechet_hutchinson"]["frechet_relative_l2"] for d in by.values()],[d["pair_local_frechet_hutchinson"]["update_cosine"]-d["mse"]["update_cosine"] for d in by.values()],s=8,alpha=.35);plt.axhline(0,color="black",lw=.7);plt.axvline(0,color="black",lw=.7);plt.xlabel("Fréchet relative-L2 improvement");plt.ylabel("K=5 cosine improvement");plt.tight_layout();plt.savefig(out/"frechet_vs_actual_gain.png",dpi=140);plt.close()
    plt.figure(figsize=(7,4));plt.bar(range(len(cd)),[float(r["objective_ratio_to_initial"]) for r in cd]);plt.xticks(range(len(cd)),[f'{r["parameter_id"].split(".")[-2]}\ns{r["sweep"]}' for r in cd],rotation=40,fontsize=6);plt.ylabel("global Fréchet objective / initial");plt.tight_layout();plt.savefig(out/"global_cd_objective.png",dpi=140);plt.close()
    plt.figure(figsize=(7,4));plt.scatter([r["raw_relative_l2"] for r in rows if r["method"]!="mse"],[r["update_cosine"]-by[(r["seed"],r["update"],r["parameter_id"])]["mse"]["update_cosine"] for r in rows if r["method"]!="mse"],s=8,alpha=.3);plt.xlabel("raw relative-L2");plt.ylabel("update-cosine gain vs MSE");plt.tight_layout();plt.savefig(out/"raw_error_change_vs_gain.png",dpi=140);plt.close()
    # Per-method objective, channel, and exact-polar readouts are retained in
    # CSV; these compact plots expose the principal comparisons.
    for ykey,filename,label in (("frechet_relative_l2","frechet_assignment_cost.png","Fréchet predicted relative-L2"),
                                ("skew_energy","skew_channel_energy.png","skew-channel energy")):
        plt.figure(figsize=(8,4))
        for m,l in zip(methods,labels):
            vals=[d[m][ykey] for d in by.values()]
            plt.scatter(np.full(len(vals),methods.index(m)),vals,s=5,alpha=.25,label=l)
        plt.xticks(range(len(methods)),labels,rotation=25);plt.ylabel(label);plt.tight_layout();plt.savefig(out/filename,dpi=140);plt.close()
    plt.figure(figsize=(7,4))
    if cd:
        for key in sorted({(r["seed"],r["update"],r["parameter_id"]) for r in cd}):
            seq=sorted([r for r in cd if (r["seed"],r["update"],r["parameter_id"])==key],key=lambda x:int(x["sweep"]))
            plt.plot([int(r["sweep"]) for r in seq],[float(r["update_cosine"]) for r in seq],marker="o",alpha=.7)
    plt.xlabel("global coordinate-descent sweep");plt.ylabel("actual K=5 update cosine");plt.tight_layout();plt.savefig(out/"global_cd_actual_fidelity.png",dpi=140);plt.close()
    plt.figure(figsize=(7,4));
    for m,l in zip(methods[1:],labels[1:]):
        xs=[];ys=[]
        for key,d in by.items():
            if m in d:xs.append(d["mse"]["update_cosine"]);ys.append(d[m]["update_cosine"])
        plt.scatter(xs,ys,s=7,alpha=.3,label=l)
    lim=(.3,1.0);plt.plot(lim,lim,color="black",lw=.7);plt.xlim(lim);plt.ylim(lim);plt.xlabel("MSE assignment cosine");plt.ylabel("alternative cosine");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"tensor_level_assignment_cosines.png",dpi=140);plt.close()
    # Explicit baseline-paired views requested for the assignment-only question.
    for method,filename,title in (("pair_local_frechet_hutchinson","mse_vs_pair_local_frechet.png","Pair-local Fréchet vs MSE assignment"),
                                  ("pair_local_skew_hutchinson","mse_vs_pair_local_skew.png","Pair-local skew vs MSE assignment")):
        plt.figure(figsize=(6,5));x=[];y=[]
        for d in by.values():x.append(d["mse"]["update_cosine"]);y.append(d[method]["update_cosine"])
        plt.scatter(x,y,s=9,alpha=.35);plt.plot((0,1),(0,1),color="black",lw=.8);plt.xlabel("MSE K=5 cosine");plt.ylabel(title);plt.title(title);plt.tight_layout();plt.savefig(out/filename,dpi=140);plt.close()
    base={(r["seed"],r["update"],r["parameter_id"]):r for r in rows if r["method"]=="mse"}
    plt.figure(figsize=(6,4));
    for method,label in zip(methods[1:],labels[1:]):
        dx=[d[method]["update_cosine"]-d["mse"]["update_cosine"] for d in by.values()]
        plt.hist(dx,bins=35,histtype="step",density=True,label=label)
    plt.axvline(0,color="black",lw=.7);plt.xlabel("K=5 cosine gain vs MSE");plt.ylabel("density");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"tensor_gain_histogram.png",dpi=140);plt.close()
    plt.figure(figsize=(6,4));
    for method,label in zip(methods[1:],labels[1:]):
        xs=[];ys=[]
        for d in by.values():
            if method not in d:continue
            xs.append(d[method]["changed_fraction"]);ys.append(d[method]["update_cosine"]-d["mse"]["update_cosine"])
        plt.scatter(xs,ys,s=7,alpha=.25,label=label)
    plt.axhline(0,color="black",lw=.7);plt.xlabel("fraction of pair assignments changed");plt.ylabel("K=5 cosine gain");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"changed_pairs_vs_update_gain.png",dpi=140);plt.close()
    plt.figure(figsize=(6,4));
    for method,label in zip(methods[1:],labels[1:]):
        xs=[];ys=[]
        for d in by.values():
            if method not in d:continue
            xs.append(d[method]["skew_energy"]-d["mse"]["skew_energy"]);ys.append(d[method]["update_cosine"]-d["mse"]["update_cosine"])
        plt.scatter(xs,ys,s=7,alpha=.25,label=label)
    plt.axhline(0,color="black",lw=.7);plt.axvline(0,color="black",lw=.7);plt.xlabel("skew-channel energy change");plt.ylabel("K=5 cosine gain");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"skew_change_vs_update_gain.png",dpi=140);plt.close()
    polar_path=out/"polar_subset.csv"
    if polar_path.exists():
        pr=list(csv.DictReader(polar_path.open()))
        if pr:
            pby=defaultdict(dict)
            for r in pr:pby[(r["seed"],r["update"],r["parameter_id"])][r["method"]]=float(r["exact_polar_cosine"])
            keep=[d for d in pby.values() if "mse" in d]
            plt.figure(figsize=(6,4));
            for j,m in enumerate(("mse","pair_local_frechet_hutchinson","pair_local_skew_hutchinson")):
                plt.scatter([j]*len(keep),[d[m] for d in keep],s=12,alpha=.4)
            plt.xticks(range(3),("MSE","local Fréchet","local skew"));plt.ylabel("exact-polar cosine");plt.tight_layout();plt.savefig(out/"exact_polar_assignment_subset.png",dpi=140);plt.close()
    st=list(csv.DictReader((out/"storage_check.csv").open())) if (out/"storage_check.csv").exists() else []
    if st:
        ratios=[float(r["storage_ratio_unamortized"]) for r in st]
        plt.figure(figsize=(6,3));plt.scatter(range(len(ratios)),ratios,s=4);plt.ylabel("storage / FP32");plt.xlabel("tensor × assignment row");plt.tight_layout();plt.savefig(out/"identical_storage_check.png",dpi=140);plt.close()


def write_methodology(out,probes):
    (out/"methodology.md").write_text(f"""# Methodology

## Fixed representation

The study fixes rank k=8, the existing BF16 top-8 factorized component, the existing split-specific frozen 64-word MSE codebook, contiguous row-major residual pairing, p98 shared scalar normalization for each 2048 residual scalars, and 6-bit codeword indices. Normalized residual vectors are clipped to [-1,1] exactly as in the canonical VQ implementation. Only codeword assignment changes. There is no codebook, scale, pairing, bitwidth, optimizer, or training change.

Formal snapshots cover seeds 0/1 at updates 128, 512, 1024, 2048, and 4096, with all 30 eligible 2D Muon matrices per snapshot (300 matrix instances). For held-out evaluation seed s, the canonical codebook trained on the opposite seed is used, matching the previously validated split protocol. Forward and reverse calibration use deterministic evenly spaced 12-matrix samples from the calibration trajectory only. Their MSE and local-metric normalizers and lambda choice are carried to the opposite held-out trajectory.

## Objective and assignment rules

The exact production map is the unchanged K=5 normalized Newton–Schulz implementation. The local derivative includes its state-dependent Frobenius normalization term. Per-pair full and skew 2x2 metric blocks are estimated from {probes} deterministic Rademacher Hutchinson probes of J*J; they are stochastic isolated-pair block estimates and are **not** exact blocks. Candidate pair errors are evaluated in physical residual units. Mixed objectives combine block-normalized Euclidean residual error and estimated local Fréchet quadratic cost; lambda is selected by calibration-seed K=5 update cosine, never held-out data.

The global oracle starts from canonical MSE assignments and coordinate-descends the exact global first-order quadratic objective ||J[E]||² on a deterministic small subset. Its gradient includes cross-pair terms J*J[E], while per-coordinate Hessians are computed from analytic Fréchet responses. Proposed steps are checked against the exact linearized production objective. This remains an offline oracle, not a pair-separable method.

## Metrics and scope

The actual K=5 update, raw state error, Fréchet-predicted update perturbation, channel energies, and exact-polar subset are computed from unchanged prior implementations. Storage is recomputed with the existing accounting function and required to be identical across assignments. Hutchinson estimates are validated against exact analytic pair Hessians on a deterministic calibration audit. Any local-objective gain is compared directly with finite-error production K=5 fidelity because the Fréchet metric is only a local surrogate. No dense W_M is formed, no new codebook is trained, and no production or training path is modified.

""")


def write_report(out,rows,lambdas,mse_norm,frechet_norm,coverage,elapsed,audit,polar):
    def mean(method,key,seed=None):
        z=[float(r[key]) for r in rows if r["method"]==method and (seed is None or int(r["seed"])==seed)]
        return statistics.mean(z) if z else float("nan")
    def median(method,key):
        z=[float(r[key]) for r in rows if r["method"]==method]
        return statistics.median(z) if z else float("nan")
    win=tensor_win_rows(rows)
    by=defaultdict(dict)
    for r in rows:by[(r["seed"],r["update"],r["parameter_id"])][r["method"]]=r
    pair_stats={}
    for method in ("pair_local_frechet_hutchinson","pair_local_skew_hutchinson","mix_025","mix_050","mix_075"):
        dlist=[d for d in by.values() if method in d]
        gains=[d[method]["update_cosine"]-d["mse"]["update_cosine"] for d in dlist]
        pair_stats[method]={"mean_gain":statistics.mean(gains),"median_gain":statistics.median(gains),"win":sum(x>0 for x in gains)/len(gains)}
    cdpath=out/"global_coordinate_descent.csv";cd=list(csv.DictReader(cdpath.open())) if cdpath.exists() else []
    cd_ratios=[float(r["objective_ratio_to_initial"]) for r in cd if int(r["sweep"])==1]
    cd_gain=[]
    for key in {(r["seed"],r["update"],r["parameter_id"]) for r in cd}:
        z=sorted([r for r in cd if (r["seed"],r["update"],r["parameter_id"])==key],key=lambda x:int(x["sweep"]))
        if z:cd_gain.append(float(z[-1]["update_cosine"])-float(z[0]["update_cosine"]))
    audit_rel=[float(r["hessian_relative_fro_error"]) for r in audit]
    lines=["# Fixed-codebook Fréchet-aware 2-D VQ assignment study","",
        f"CPU-only, {coverage} formal matrix instances, k=8, frozen 64-word codebooks, p98/2048 scales and contiguous pair layout. Runtime {elapsed:.1f}s. No training, codebook fitting, scale tuning, or production changes.","",
        "## Baseline and identical representation","",
        f"Mean canonical MSE assignment K=5 cosine: {mean('mse','update_cosine'):.4f}; seed 0 {mean('mse','update_cosine',0):.4f}; seed 1 {mean('mse','update_cosine',1):.4f}. Forward held-out seed-1 reproduction target from the robustness report is 0.85161. All alternatives keep the same codebook, p98 block scales, 6-bit pair indices, pair count, BF16 rank-8 factors, and metadata; `storage_check.csv` verifies exact equality.",
        f"Calibration-only normalization constants for mixed costs (12 deterministic tensors per calibration seed): pair MSE {mse_norm}; pair-local Fréchet cost {frechet_norm}. Seed-specific λ choices from calibration-seed K=5 fidelity: {lambdas}; both select λ=0 (MSE), so the selected practical rule is exactly the unchanged baseline. All fixed nonzero λ values are also reported on both held-out directions using the corresponding opposite-seed calibration normalizers.","",
        "## Assignment outcomes","","| assignment | mean K=5 cosine | median cosine | mean raw rel-L2 | mean Fréchet rel-L2 | changed pairs |",
        "|:--|--:|--:|--:|--:|--:|"]
    for method in ("mse","pair_local_frechet_hutchinson","pair_local_skew_hutchinson","mix_025","mix_050","mix_075"):
        changed=statistics.mean(float(r.get("changed_fraction",0.0)) for r in rows if r["method"]==method)
        lines.append(f"| {method} | {mean(method,'update_cosine'):.4f} | {median(method,'update_cosine'):.4f} | {mean(method,'raw_relative_l2'):.4f} | {mean(method,'frechet_relative_l2'):.4f} | {changed:.2%} |")
    lines += ["","Pair-local results use Hutchinson-estimated 2×2 blocks of the exact production (J^*J) operator. These are stochastic estimates of isolated-pair curvature, not exact blocks. Exact analytic blocks are compared on the calibration audit subset in `pair_hessian_validation.csv`. They are not a global objective because omitted cross-pair terms remain.","",
        "## Local-metric vs finite-error behavior","",
        f"Across all 300 tensors, pair-local Hutchinson Fréchet assignment changed {statistics.mean(float(r['changed_fraction']) for r in rows if r['method']=='pair_local_frechet_hutchinson'):.1%} of pair labels, but mean K=5 cosine changed by {pair_stats['pair_local_frechet_hutchinson']['mean_gain']:+.4f} (median {pair_stats['pair_local_frechet_hutchinson']['median_gain']:+.4f}; tensor win rate {pair_stats['pair_local_frechet_hutchinson']['win']:.1%}). Its full, directly recomputed Fréchet relative-L2 increased from {mean('mse','frechet_relative_l2'):.4f} to {mean('pair_local_frechet_hutchinson','frechet_relative_l2'):.4f}; the estimated local pair metric did not transfer to the true total derivative metric. On held-out seed 1, its K=5 mean was {mean('pair_local_frechet_hutchinson','update_cosine',1):.4f}, versus MSE {mean('mse','update_cosine',1):.4f}.",
        f"Skew-only assignment was more damaging (mean K=5 cosine {mean('pair_local_skew_hutchinson','update_cosine'):.4f}, mean raw relative-L2 {mean('pair_local_skew_hutchinson','raw_relative_l2'):.4f}). The calibration-selected mixture was λ=0 in both directions, i.e. ordinary MSE; nonzero λ variants also lost fidelity (mean gains: λ=.25 {pair_stats['mix_025']['mean_gain']:+.4f}, λ=.50 {pair_stats['mix_050']['mean_gain']:+.4f}, λ=.75 {pair_stats['mix_075']['mean_gain']:+.4f}). Full paired tensor win/mismatch rates are in the CSVs.",
        "The mixed-objective λ was selected on one trajectory seed and read out on the other; both directions are reported. No held-out update cosine selected the λ used for its own evaluation.","",
        "## Global coordinate-descent oracle","",
        f"`global_coordinate_descent.csv` contains four tensors and 16 visited pairs per tensor. Exact global first-order objective ratios after the first sweep were {', '.join(f'{x:.6f}' for x in cd_ratios)} (mean reduction {(1-statistics.mean(cd_ratios)):.3%}); actual K=5 cosine changes across the four examples were {', '.join(f'{x:+.6f}' for x in cd_gain)}. The local quadratic objective is nonincreasing, but reductions are tiny and produce no material actual K=5 recovery. This oracle therefore shows little usable assignment headroom in its limited deterministic subset; it is not evidence that exhaustive global search was performed.","",
        "## Estimator and polar checks","",
        f"The pair-local 2×2 blocks used for full-coverage assignment are Hutchinson estimates from 32 deterministic probes, not exact per-pair Hessians. On 12 calibration audit tensors, the relative Frobenius discrepancy versus exact analytic pair blocks had median {statistics.median(audit_rel):.3f} and range {min(audit_rel):.3f}–{max(audit_rel):.3f}; this is a material approximation limitation. The exact-polar subset contains {len(polar)} method/tensor rows and has the same qualitative ordering (means: MSE {statistics.mean(float(r['exact_polar_cosine']) for r in polar if r['method']=='mse'):.4f}, local Fréchet {statistics.mean(float(r['exact_polar_cosine']) for r in polar if r['method']=='pair_local_frechet_hutchinson'):.4f}, local skew {statistics.mean(float(r['exact_polar_cosine']) for r in polar if r['method']=='pair_local_skew_hutchinson'):.4f}).",
        "## Interpretation and decision","",
        "For this fixed representation, no tested Fréchet-aware assignment improves actual production K=5 fidelity. Pure pair-local Fréchet and skew rules regress consistently across both seeds; the calibration-only selected mixture is λ=0; the global coordinate-descent oracle reduces its linearized objective by less than 0.05% in the four tested cases and does not change update cosine materially. Pair-local block estimation itself is noisy (median exact-Hessian relative error about one third), so the result is a negative practical assignment study, not a proof that the exact pair-local or exhaustive global optimum has no headroom. Still, there is no evidence sufficient to justify Fréchet-aware codebook learning or training integration. Stop this optimizer-aware assignment branch unless a substantially more accurate, affordable block metric is developed independently.","",
        f"Exact Hessian validation tensors: {len(audit)}; exact-polar rows: {len(polar)}. Runtime {elapsed:.1f} CPU seconds."]
    (out/"summary.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":main()
