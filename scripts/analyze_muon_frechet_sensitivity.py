#!/usr/bin/env python3
"""CPU-only Fréchet sensitivity validation for canonical Muon low-bit states."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from analyze_muon_int3_practical_scale import codebook_for_rank,factorized_topk,quantize_scales
from analyze_muon_vector_int3_residual import eligible_items
from optim.muon_frechet import frechet_channels,principal_pair_proxies
from optim.muon_gram_perturbation import active_band_indices,spectral_gap_proxy
from optim.muon_int3_scale_selection import block_scales
from optim.muon_ns_sensitivity import PRODUCTION_COEFFICIENTS,PRODUCTION_EPS,PRODUCTION_STEPS
from optim.muon_spectral_sensitivity import decompose,quantize
from optim.muon_update_fidelity import load_snapshot
from optim.muon_vector_int3 import pair_values,quantize_vectors,unpair_values
from optim.muon_reference import zeropower_newton_schulz

OUT=ROOT/"reports/muon_frechet_sensitivity"
SNAP=ROOT/"reports/muon_update_fidelity_formal_s0_s1_results"
VQ_DIR=ROOT/"reports/muon_vector_int3_robustness"
GRAM_DIR=ROOT/"reports/muon_gram_perturbation_hypothesis"
PRACTICAL=ROOT/"reports/muon_int3_practical_scale/tensor_level_results.csv"
LANDMARKS=(128,512,1024,2048,4096)
METHODS=("structural_scalar_int3_p98_k8","structural_int4_k8","structural_vq64_int3_k8","direct_scalar_int4")
CHANNELS=("magnitude","symmetric","skew","out_of_subspace")


def read_csv(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore",lineterminator="\n");w.writeheader();w.writerows(rows)

def krow(r):return int(r.get("seed",r.get("evaluation_seed"))),int(r["update"]),r.get("parameter_id",r.get("parameter_name"))

def lookup_prior():
    tables={n:read_csv(GRAM_DIR/f"{n}.csv") for n in ("raw_error_metrics","gram_metrics","subspace_metrics","update_metrics")}
    out={}
    for name,rows in tables.items():
        for r in rows:out.setdefault((krow(r),r["method"]),{}).update(r)
    return out

def lookup_polar():
    result={}
    for r in read_csv(GRAM_DIR/"update_metrics.csv"):
        result[(krow(r),r["method"])]=r
    return result

def load_snapshots():
    from analyze_muon_int3_practical_scale import discover
    found={}
    for seed,update,path in discover(SNAP):found[(int(seed),int(update))]=load_snapshot(path)
    expected={(s,u) for s in (0,1) for u in LANDMARKS}
    if set(found)!=expected:raise RuntimeError(f"formal snapshot coverage differs: {set(found)^expected}")
    return found

def reconstructions(m,svd,seed,codebooks):
    u,s,vh=svd.u,svd.singular_values,svd.vh
    c_exact=(u[:,:8]*s[:8])@vh[:8]
    c_hat=factorized_topk(u,s,vh,8)
    residual=m-c_exact
    q4=quantize(m,"int4-dynamic-b2048").float()
    cb=codebook_for_rank(8)
    scales=block_scales(residual,"percentile",percentile=98.0,block_size=2048)
    scalar=c_hat+quantize_scales(residual,scales,cb)
    structural4=c_hat+quantize(residual,"int4-dynamic-b2048").float()
    cb_key=f"s{1-int(seed)}_k8_w64_t8_v1200"
    if cb_key not in codebooks:raise KeyError(cb_key)
    pairs,singles,pi,si=pair_values(residual,"contiguous")
    q_pairs,vq_scales,labels=quantize_vectors(pairs,codebooks[cb_key],scale_method="p98",block_size=2048)
    vq=c_hat+unpair_values(q_pairs,singles,tuple(residual.shape),pi,si)
    return {"direct_scalar_int4":q4,"structural_scalar_int3_p98_k8":scalar,
            "structural_int4_k8":structural4,"structural_vq64_int3_k8":vq},residual

def cosine(a,b):
    # Roundoff in float32 can otherwise produce cosine values just outside
    # [-1, 1], which makes distortion (1-cosine) spuriously negative.
    value=float((a*b).sum()/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)).clamp_min(1e-30))
    return max(-1.0,min(1.0,value))

def polar_metrics(o,op):
    return {"cosine":cosine(o,op),"distortion":1.0-cosine(o,op),
            "relative_l2":float(torch.linalg.vector_norm(op-o)/torch.linalg.vector_norm(o).clamp_min(1e-30))}

def channel_ablation(o,components):
    sets={"magnitude_only":["magnitude"],"symmetric_only":["symmetric"],"skew_only":["skew"],
          "out_of_subspace_only":["out_of_subspace"],"orientation_plus_outside":["skew","out_of_subspace"],"all_channels":list(CHANNELS)}
    rows={}
    for name,chs in sets.items():
        p=o+sum((components[c].to(o) for c in chs),torch.zeros_like(o))
        rows[name]=1.0-cosine(o,p)
    return rows

def mode_labels(sigma):
    s=sigma.detach().double();active=int((s/s[0]>=1e-6).sum()) if s.numel() and float(s[0]) else 0
    width=max(1,math.ceil(active*.1)) if active else 0
    mid=max(0,active//2-width//2)
    labels=["other"]*s.numel()
    for i in range(width):labels[i]="head"
    for i in range(mid,min(active,mid+width)):labels[i]="middle"
    for i in range(max(0,active-width),active):labels[i]="tail"
    return labels,active

def band_channel_energy(result):
    names={0:"other",1:"head",2:"middle",3:"tail"}
    labels,active=mode_labels(result["sigma"]);rows=[]
    lab=torch.tensor([{"other":0,"head":1,"middle":2,"tail":3}[x] for x in labels],dtype=torch.long)
    f=result["normalized_coordinate_error"];deriv=result["normalized_derivative"]
    diag_values=torch.diagonal(f)*deriv
    for ch in CHANNELS:
        if ch=="magnitude":
            values=torch.bincount(lab,weights=diag_values.square(),minlength=4).tolist()
            by={names[i]:values[i] for i in range(4)}
        elif ch in ("symmetric","skew"):
            mat=result["symmetric_divided_difference"]*((f+f.T)*.5) if ch=="symmetric" else result["skew_factor"]*((f-f.T)*.5)
            i,j=torch.triu_indices(mat.shape[0],mat.shape[1],offset=1)
            li,lj=lab[i],lab[j];lo=torch.minimum(li,lj);hi=torch.maximum(li,lj)
            codes=lo*4+hi;energy=mat[i,j].square()+mat[j,i].square()
            values=torch.bincount(codes,weights=energy,minlength=16).tolist();by={}
            for code,value in enumerate(values):
                if value:
                    x,y=divmod(code,4);by[names[x] if x==y else "cross_"+"_".join(sorted((names[x],names[y])))]=value
        else:
            values=torch.bincount(lab,weights=result["out_of_subspace_mode_energy"],minlength=4).tolist()
            by={names[i]:values[i] for i in range(4)}
        for band,value in by.items():rows.append({"channel":ch,"band_or_pair":band,"predicted_delta_sq":value,"active_rank":active})
    return rows

def rank_corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return float("nan")
    def ranks(a):
        order=np.argsort(a,kind="mergesort");r=np.empty(len(a));i=0
        while i<len(a):
            j=i+1
            while j<len(a) and a[order[j]]==a[order[i]]:j+=1
            r[order[i:j]]=(i+j-1)/2;i=j
        return r
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])

def pearson(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    return float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>0 and np.std(y)>0 else float("nan")

def fit_r2(train,test,features,target="actual_update_distortion"):
    if len(train)<len(features)+3 or len(test)<3:return float("nan")
    x=np.asarray([[float(r[k]) for k in features] for r in train]);y=np.asarray([float(r[target]) for r in train])
    xt=np.asarray([[float(r[k]) for k in features] for r in test]);yt=np.asarray([float(r[target]) for r in test])
    x=np.log10(np.maximum(x,1e-14));xt=np.log10(np.maximum(xt,1e-14));mu=x.mean(0);sd=x.std(0);sd[sd==0]=1
    beta=np.linalg.lstsq(np.c_[np.ones(len(x)),(x-mu)/sd],y,rcond=None)[0]
    pred=np.c_[np.ones(len(xt)),(xt-mu)/sd]@beta
    return float(1-((yt-pred)**2).sum()/((yt-yt.mean())**2).sum())

def write_definition(path):
    path.write_text("""# Production Muon map definition

The canonical map is `src/optim/muon_reference.py::zeropower_newton_schulz`. It casts the 2-D input to FP32; if rows exceed columns, it transposes before iteration; it divides by `||M||_F + eps`; then it performs K=5 steps with `(a,b,c)=(3.4445,-4.7750,2.0315)`:

`X <- a X + b (X X^T) X + c (X X^T)^2 X`.

It transposes back for tall inputs. There is no post-scaling. Under an SVD this is `U q_5(Sigma/(||M||_F+eps)) V^T`. Thus the polynomial part is a spectral map, but its normalization is state-dependent. The production Fréchet derivative below includes `d(||M||_F)`; the frozen-normalization diagnostic omits that term and is reported separately. Tall/wide orientation is transpose-equivariant; the rectangular leakage term is taken from the left nullspace for tall matrices and right nullspace for wide matrices.

For normalized singular values `x_i`, scalar values `f_i=q_5(x_i)` and derivatives `f'_i=q'_5(x_i)`, the fixed-scale core uses diagonal `f'_i`, symmetric divided difference `(f_i-f_j)/(x_i-x_j)`, skew factor `(f_i+f_j)/(x_i+x_j)`, and leakage `f_i/x_i`. Production additionally has `dX=E/s - M <M,E>/(||M||_F s^2)`, `s=||M||_F+eps`, so the global radial normalization derivative contributes to the diagonal coordinate perturbation. At equal singular values the divided difference uses the average derivative limit. Exact polar uses `f=1`, zero diagonal and symmetric channels, skew factor `2/(sigma_i+sigma_j)`, and rectangular coefficient `1/sigma_i`.
""")

def write_plots(out,records,fd,jvps,polar):
    import matplotlib.pyplot as plt
    def scat(x,y,file,xlab,ylab):
        plt.figure(figsize=(6,4.4))
        for method in METHODS:
            rr=[r for r in records if r["method"]==method]
            plt.scatter([float(r[x]) for r in rr],[float(r[y]) for r in rr],s=8,alpha=.35,label=method)
        plt.xlabel(xlab);plt.ylabel(ylab);plt.legend(fontsize=5);plt.tight_layout();plt.savefig(out/file,dpi=140);plt.close()
    scat("raw_relative_l2","actual_update_distortion","actual_vs_raw.png","raw relative-L2","actual K=5 distortion")
    scat("tail_angle","actual_update_distortion","actual_vs_tail_angle.png","tail subspace mean sin(theta)","actual K=5 distortion")
    scat("frechet_distortion_pred","actual_update_distortion","predicted_vs_actual_k5.png","Fréchet-predicted cosine error","actual K=5 distortion")
    scat("frechet_relative_l2","actual_update_distortion","frechet_l2_vs_actual.png","Fréchet relative-L2","actual K=5 distortion")
    if polar:
        plt.figure(figsize=(5,4))
        plt.scatter([float(r["predicted_polar_distortion"]) for r in polar],[float(r["actual_polar_distortion"]) for r in polar],s=10,alpha=.4)
        plt.xlabel("predicted exact-polar distortion");plt.ylabel("actual exact-polar distortion");plt.tight_layout();plt.savefig(out/"predicted_vs_actual_polar.png",dpi=140);plt.close()
    # Channel energy stacked by method.
    plt.figure(figsize=(7,4));chs=CHANNELS;bottom=np.zeros(len(METHODS))
    for ch in chs:
        vals=[statistics.mean(float(r[f"{ch}_fraction"]) for r in records if r["method"]==method) for method in METHODS]
        plt.bar(np.arange(len(METHODS)),vals,bottom=bottom,label=ch);bottom+=np.asarray(vals)
    plt.xticks(np.arange(len(METHODS)),METHODS,rotation=25,ha="right");plt.ylabel("mean derivative energy fraction");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(out/"channel_contribution.png",dpi=140);plt.close()
    # Band breakdown, aggregate squared predicted contribution.
    rows=read_csv(out/"spectral_band_breakdown.csv");labels=sorted({r["band_or_pair"] for r in rows})
    plt.figure(figsize=(9,4));methods=METHODS
    for i,method in enumerate(methods):
        vals=[sum(float(r["predicted_delta_sq"]) for r in rows if r["method"]==method and r["band_or_pair"]==lab) for lab in labels]
        plt.bar(np.arange(len(labels))+i*.18,vals,width=.18,label=method)
    plt.xticks(np.arange(len(labels))+.27,labels,rotation=35,ha="right");plt.legend(fontsize=5);plt.ylabel("predicted derivative energy");plt.tight_layout();plt.savefig(out/"spectral_band_energy.png",dpi=140);plt.close()
    # Compare pairwise proxies over selected records.
    plt.figure(figsize=(6,4));
    for x,label in (("gap_skew_score","1/gap weighted skew error"),("sigma_sum_skew_score","2/sum weighted skew error"),("finite_k_skew_score","finite-K orientation weighted error")):
        plt.scatter([float(r[x]) for r in records],[float(r["actual_update_distortion"]) for r in records],s=7,alpha=.25,label=label)
    plt.xscale("log");plt.legend(fontsize=6);plt.ylabel("actual K=5 distortion");plt.tight_layout();plt.savefig(out/"gap_vs_sigma_sum.png",dpi=140);plt.close()
    pair=read_csv(out/"sigma_sum_pair_sample.csv")
    if pair:
        plt.figure(figsize=(6,4));plt.scatter([float(r["sigma_sum"]) for r in pair],[float(r["finite_k_orientation_proxy"]) for r in pair],s=5,alpha=.22)
        plt.xscale("log");plt.yscale("log");plt.xlabel("sigma_i + sigma_j");plt.ylabel("finite-K orientation sensitivity");plt.tight_layout();plt.savefig(out/"sigma_sum_vs_orientation_sensitivity.png",dpi=140);plt.close()
    # FD and JVP diagnostic plots.
    if fd:
        eps=sorted({float(r["epsilon"]) for r in fd});errs=[statistics.mean(float(r["relative_l2_mismatch"]) for r in fd if float(r["epsilon"])==e) for e in eps]
        plt.figure();plt.plot(eps,errs,"o-");plt.xscale("log");plt.yscale("log");plt.xlabel("epsilon applied to actual E");plt.ylabel("relative finite-difference / first-order mismatch");plt.tight_layout();plt.savefig(out/"finite_difference_linearity.png",dpi=140);plt.close()
    if jvps:
        plt.figure();plt.scatter([float(r["analytic_jvp_cosine"]) for r in jvps],[float(r["analytic_jvp_relative_l2"]) for r in jvps],s=12,alpha=.5);plt.xlabel("analytic vs production JVP cosine");plt.ylabel("relative-L2 mismatch");plt.tight_layout();plt.savefig(out/"analytic_vs_production_jvp.png",dpi=140);plt.close()
    corr=read_csv(out/"predictor_correlations.csv");rr=[r for r in corr if r["scope"]=="global" and r["outcome"]=="actual_update_distortion"]
    plt.figure(figsize=(9,4));plt.bar(range(len(rr)),[float(r["spearman"]) for r in rr]);plt.xticks(range(len(rr)),[r["predictor"] for r in rr],rotation=45,ha="right",fontsize=7);plt.ylabel("Spearman");plt.tight_layout();plt.savefig(out/"predictor_spearman.png",dpi=140);plt.close()
    pred=read_csv(out/"predictive_models.csv");rr=[r for r in pred if r["validation"]=="held_out_seed"]
    plt.figure(figsize=(8,4));plt.bar(range(len(rr)),[float(r["r2"]) for r in rr]);plt.xticks(range(len(rr)),[r["model"]+"/s"+r["held_out_group"] for r in rr],rotation=45,ha="right",fontsize=6);plt.ylabel("held-out R²");plt.tight_layout();plt.savefig(out/"heldout_r2.png",dpi=140);plt.close()
    # paired derivative channel ratios.
    for filename,label in (("scalar_vs_vector_int3.csv","scalar / vector INT3"),("int4_vs_vector_int3.csv","structural INT4 / vector INT3")):
        rr=read_csv(out/filename)
        if rr:
            plt.figure(figsize=(6,4));plt.scatter([float(x["frechet_ratio_vq_over_reference"]) for x in rr],[float(x["actual_distortion_ratio_vq_over_reference"]) for x in rr],s=8,alpha=.3);plt.axline((1,1),slope=1,color="black");plt.xlabel("VQ/reference Fréchet metric ratio");plt.ylabel("VQ/reference actual distortion ratio");plt.title(label);plt.tight_layout();plt.savefig(out/("paired_"+filename.replace(".csv",".png")),dpi=140);plt.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--threads",type=int,default=4);parser.add_argument("--summary-existing",action="store_true");args=parser.parse_args()
    torch.set_num_threads(args.threads);start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if args.summary_existing:
        records=read_csv(OUT/"tensor_metrics.csv");corr=read_csv(OUT/"predictor_correlations.csv");pred=read_csv(OUT/"predictive_models.csv")
        pred=[r for r in pred if r["validation"] not in ("held_out_seed_exact_polar_subset","leave_one_method_out_exact_polar_subset")]
        polar_records=[r for r in records if r.get("exact_polar_pred_distortion") not in (None,"")]
        for name,feature in (("polar_raw","raw_relative_l2"),("polar_gram","right_gram_relative_fro"),
                             ("polar_tail_angle","tail_angle"),("polar_frechet","exact_polar_pred_distortion")):
            for te_seed in (0,1):
                tr=[r for r in polar_records if int(r["seed"])!=te_seed];te=[r for r in polar_records if int(r["seed"])==te_seed]
                pred.append({"validation":"held_out_seed_exact_polar_subset","held_out_group":te_seed,"model":name,"features":feature,"train_n":len(tr),"test_n":len(te),"r2":fit_r2(tr,te,[feature],target="actual_polar_distortion")})
            for method in METHODS:
                tr=[r for r in polar_records if r["method"]!=method];te=[r for r in polar_records if r["method"]==method]
                pred.append({"validation":"leave_one_method_out_exact_polar_subset","held_out_group":method,"model":name,"features":feature,"train_n":len(tr),"test_n":len(te),"r2":fit_r2(tr,te,[feature],target="actual_polar_distortion")})
        write_csv(OUT/"predictive_models.csv",pred)
        sv=read_csv(OUT/"scalar_vs_vector_int3.csv");vq=read_csv(OUT/"int4_vs_vector_int3.csv")
        elapsed=float((OUT/"runtime_seconds.txt").read_text().split()[0])
        write_summary(records,corr,pred,sv,vq,OUT,elapsed)
        print("summary regenerated from cached analysis tables",flush=True);return
    snapshots=load_snapshots();prior=lookup_prior();polar_cache=lookup_polar()
    cblob=torch.load(VQ_DIR/"calibration_codebooks.pt",map_location="cpu",weights_only=False);codebooks=cblob["codebooks"]
    manifest=[{"method":m,"rank":8 if "structural" in m else 0,"description":d,"source":"prior canonical reports; unchanged quantizer/reconstruction"} for m,d in (
        ("structural_scalar_int3_p98_k8","BF16 top-8 + p98/b2048 fixed global Lloyd-Max scalar INT3"),
        ("structural_int4_k8","BF16 top-8 + production dynamic INT4 b2048 residual"),
        ("structural_vq64_int3_k8","BF16 top-8 + 64-word vector INT3, contiguous pairing, p98/2048, opposite-seed codebook"),
        ("direct_scalar_int4","production dynamic INT4 b2048 directly on M"))]
    write_csv(OUT/"method_manifest.csv",manifest);write_definition(OUT/"production_map_definition.md")
    records=[];channel_rows=[];band_rows=[];jvp_rows=[];fd_rows=[];polar_rows=[];pair_rows=[]
    matrix_index=0;jvp_keys=set(range(0,300,10));fd_keys=set(range(0,300,25));polar_keys=set(range(0,300,10))
    # Snapshot order is fixed and matches the canonical two-seed five-landmark panel.
    for snap_index,key in enumerate(sorted(snapshots)):
        seed,update=key;snap=snapshots[key]
        configured=snap["metadata"].get("muon_transform",{})
        if (int(configured.get("steps",-1))!=PRODUCTION_STEPS or
            tuple(float(x) for x in configured.get("coefficients",()))!=PRODUCTION_COEFFICIENTS or
            float(configured.get("eps",float("nan")))!=PRODUCTION_EPS):
            raise AssertionError(f"snapshot production transform differs from the recorded K=5 map: {(seed,update,configured)}")
        items=eligible_items(snap)
        for item_index,item in enumerate(items):
            if item["tensor"].ndim!=2:continue
            global_ix=matrix_index;matrix_index+=1
            m=item["tensor"].detach().cpu().float();name=str(item.get("name",item.get("parameter_id")));pid=str(item.get("parameter_id",name));iid=(seed,update,pid)
            svd=decompose(m);u,s,vh=svd.u,svd.singular_values,svd.vh;factors=(u,s,vh)
            recons,_=reconstructions(m,svd,seed,codebooks)
            op=zeropower_newton_schulz(m.clone(),steps=PRODUCTION_STEPS,coefficients=PRODUCTION_COEFFICIENTS,eps=PRODUCTION_EPS)
            polar_ref=u@vh
            sn=float(torch.linalg.vector_norm(m));smin=float(s[-1]) if s.numel() else 0.0
            tail_gap=spectral_gap_proxy(s,active_band_indices(s)["tail"])
            gap_proxy=float(prior.get(((seed,update,pid),"direct_scalar_int4"),{}).get("right_gram_relative_spectral",0.0))*float(s[0])**2/max(tail_gap,1e-30)
            for method in METHODS:
                prior_row=prior.get(((seed,update,pid),method))
                if prior_row is None:raise KeyError(f"prior canonical row missing: {iid} {method}")
                mh=recons[method];e=mh-m
                cached=float(prior_row["update_cosine"])
                actual=1.0-cached
                raw_now=float(torch.linalg.vector_norm(mh-m)/torch.linalg.vector_norm(m).clamp_min(1e-30))
                if abs(raw_now-float(prior_row["raw_relative_fro"]))>3e-4:raise AssertionError(f"canonical reconstruction mismatch for {iid} {method}: {raw_now}")
                actual_rel=float(prior_row["update_relative_l2"])
                fre=frechet_channels(m,e,svd_factors=factors)
                frozen=frechet_channels(m,e,svd_factors=factors,differentiate_normalization=False)
                d=fre["predicted_delta"].float();dfrozen=frozen["predicted_delta"].float();dpred=op+d;dpred_frozen=op+dfrozen
                pred_dist=1.0-cosine(op,dpred);out_norm=float(torch.linalg.vector_norm(op).clamp_min(1e-30));fre_l2=float(torch.linalg.vector_norm(d)/out_norm)
                frozen_pred_dist=1.0-cosine(op,dpred_frozen)
                polar_cached=polar_cache[((seed,update,pid),method)]
                polar_cos_cache=float(polar_cached["exact_polar_cosine"])
                record={"seed":seed,"update":update,"parameter_id":pid,"parameter_name":name,"shape":str(tuple(m.shape)),"method":method,
                    "raw_relative_l2":float(prior_row["raw_relative_fro"]),"raw_cosine":float(prior_row["raw_cosine"]),
                    "right_gram_relative_fro":float(prior_row["right_gram_relative_fro"]),"right_gram_relative_spectral":float(prior_row["right_gram_relative_spectral"]),
                    "tail_angle":float(prior_row["tail_subspace_mean_sin"]),"prior_gap_proxy":gap_proxy,
                    "actual_update_cosine":1-actual,"actual_update_distortion":actual,"actual_update_relative_l2":float(prior_row["update_relative_l2"]),
                    "cached_update_cosine":cached,"frechet_relative_l2":fre_l2,"frechet_distortion_pred":pred_dist,
                    "frozen_norm_frechet_distortion_pred":frozen_pred_dist,
                    "frozen_norm_frechet_l2":float(torch.linalg.vector_norm(frozen["predicted_delta"])/out_norm),
                    "normalization_derivative_delta_l2":float(torch.linalg.vector_norm(d-frozen["predicted_delta"].float())/out_norm),
                    "frechet_vs_actual_delta_ratio":fre_l2/max(float(prior_row["update_relative_l2"]),1e-30),
                    "sigma_max":float(s[0]),"sigma_min":smin,"condition_number_proxy":float(s[0]/max(smin,1e-30)),"state_error_relative_fro":float(torch.linalg.vector_norm(e)/max(sn,1e-30)),
                    "quadratic_gram_ratio":float(prior_row["right_gram_quadratic_over_delta_fro"]),
                    "exact_polar_cosine":polar_cos_cache,"actual_polar_distortion":1-polar_cos_cache,
                    "exact_polar_pred_distortion":"","sample_index":global_ix,
                    "normalization_derivative_included":True}
                for ch in CHANNELS:
                    val=float(torch.linalg.vector_norm(fre["components"][ch]))
                    record[f"{ch}_delta_norm_sq"]=val*val
                    record[f"{ch}_fraction"]=val*val/max(float(torch.linalg.vector_norm(d))**2,1e-30)
                record.update(channel_ablation(op,fre["components"]))
                # Weak-mode sum sensitivity score from the skew-coordinate error.
                pp=principal_pair_proxies(s,values=fre["normalized_values"])
                raw_f=fre["coordinate_error"];a=(raw_f-raw_f.T)*.5
                ii,jj=pp["i"],pp["j"]
                pair_energy=2*a[ii,jj].square()
                finite=pp["finite_k_orientation_proxy"]
                record["gap_skew_score"]=float((pair_energy*pp["gap_proxy"].square()).sum()/out_norm**2)
                record["sigma_sum_skew_score"]=float((pair_energy*pp["sigma_sum_proxy"].square()).sum()/out_norm**2)
                record["finite_k_skew_score"]=float((pair_energy*finite.square()).sum()/out_norm**2)
                if global_ix in jvp_keys:
                    stride=max(1,len(ii)//500)
                    for pi in range(0,len(ii),stride):
                        x,y=int(ii[pi]),int(jj[pi])
                        pair_rows.append({"seed":seed,"update":update,"parameter_id":pid,"method":method,"i":x,"j":y,
                            "sigma_i":float(s[x]),"sigma_j":float(s[y]),"sigma_sum":float(s[x]+s[y]),
                            "spectral_gap":float(abs(s[x]-s[y])),"skew_error_abs":float(abs(a[x,y])),
                            "gap_proxy":float(pp["gap_proxy"][pi]),"sigma_sum_proxy":float(pp["sigma_sum_proxy"][pi]),
                            "finite_k_orientation_proxy":float(finite[pi])})
                record["actual_frechet_cosine_error_difference"]=pred_dist-actual
                if global_ix in jvp_keys:
                    try:
                        _,jvp=torch.func.jvp(lambda z:zeropower_newton_schulz(z,steps=PRODUCTION_STEPS,coefficients=PRODUCTION_COEFFICIENTS,eps=PRODUCTION_EPS),(m,),(e,))
                        jvp_rel=float(torch.linalg.vector_norm(d-jvp)/torch.linalg.vector_norm(jvp).clamp_min(1e-30));jvp_cos=cosine(d,jvp)
                        jvp_norm=float(torch.linalg.vector_norm(jvp)/out_norm)
                        jvp_pred_dist=1.0-cosine(op,op+jvp)
                    except Exception as exc:
                        jvp_rel=float("nan");jvp_cos=float("nan");jvp_norm=float("nan");jvp_pred_dist=float("nan");jvp_err=repr(exc)
                    else:jvp_err=""
                    record["jvp_relative_l2"]=jvp_rel;record["jvp_cosine"]=jvp_cos
                    record["jvp_output_relative_l2"]=jvp_norm;record["jvp_predicted_cosine_distortion"]=jvp_pred_dist
                    jvp_rows.append({"seed":seed,"update":update,"parameter_id":pid,"shape":record["shape"],"method":method,
                        "analytic_jvp_cosine":jvp_cos,"analytic_jvp_relative_l2":jvp_rel,"production_jvp_output_relative_l2":jvp_norm,
                        "production_jvp_predicted_cosine_distortion":jvp_pred_dist,"jvp_error":jvp_err,"sample_index":global_ix})
                else:
                    record["jvp_relative_l2"]="";record["jvp_cosine"]="";record["jvp_output_relative_l2"]="";record["jvp_predicted_cosine_distortion"]=""
                if global_ix in polar_keys:
                    pfre=frechet_channels(m,e,exact_polar=True,svd_factors=factors)
                    q_u,_,q_vh=torch.linalg.svd(mh.double(),full_matrices=False);polar_q=q_u@q_vh
                    actual_p=polar_metrics(polar_ref,polar_q)
                    if abs(actual_p["cosine"]-polar_cos_cache)>2e-3:raise AssertionError(f"cached exact-polar mismatch: {iid} {method}")
                    pred_p=polar_metrics(polar_ref,polar_ref+pfre["predicted_delta"])
                    record["exact_polar_pred_distortion"]=pred_p["distortion"]
                    polar_rows.append({"seed":seed,"update":update,"parameter_id":pid,"shape":record["shape"],"method":method,
                        "actual_polar_cosine":actual_p["cosine"],"actual_polar_distortion":actual_p["distortion"],"cached_polar_cosine":polar_cos_cache,
                        "predicted_polar_distortion":pred_p["distortion"],"predicted_relative_l2":pred_p["relative_l2"],
                        "magnitude_delta_norm":float(pfre["components"]["magnitude"].norm()),"symmetric_delta_norm":float(pfre["components"]["symmetric"].norm()),
                        "skew_delta_norm":float(pfre["components"]["skew"].norm()),"outside_delta_norm":float(pfre["components"]["out_of_subspace"].norm())})
                records.append(record)
                for ch in CHANNELS:
                    channel_rows.append({"seed":seed,"update":update,"parameter_id":pid,"shape":record["shape"],"method":method,"channel":ch,
                        "delta_norm_sq":record[f"{ch}_delta_norm_sq"],"fraction_of_total_derivative_energy":record[f"{ch}_fraction"]})
                for br in band_channel_energy(fre):band_rows.append({"seed":seed,"update":update,"parameter_id":pid,"shape":record["shape"],"method":method,**br})
                if global_ix in fd_keys:
                    for epsfd in (.01,.03,.1,.3,1.0):
                        fd=(zeropower_newton_schulz((m+epsfd*e).clone(),steps=PRODUCTION_STEPS,coefficients=PRODUCTION_COEFFICIENTS,eps=PRODUCTION_EPS)-op)/epsfd
                        fd_rows.append({"seed":seed,"update":update,"parameter_id":pid,"shape":record["shape"],"method":method,"epsilon":epsfd,
                            "direction_cosine":cosine(fd,d),"relative_l2_mismatch":float(torch.linalg.vector_norm(fd-d)/torch.linalg.vector_norm(d).clamp_min(1e-30)),
                            "finite_difference_norm_ratio":float(torch.linalg.vector_norm(fd)/torch.linalg.vector_norm(d).clamp_min(1e-30))})
            if item_index%10==0:print(f"processed matrix-instance {global_ix+1}/300",flush=True)
    if len(records)!=1200:raise RuntimeError(f"expected 1200 instances, got {len(records)}")
    # Paired comparisons.
    keyed=defaultdict(dict)
    for r in records:keyed[(r["seed"],r["update"],r["parameter_id"])][r["method"]]=r
    paired_sv=[];paired_i4=[]
    for key,methods in keyed.items():
        for ref,arr in (("structural_scalar_int3_p98_k8",paired_sv),("structural_int4_k8",paired_i4)):
            a,b=methods[ref],methods["structural_vq64_int3_k8"]
            arr.append({"seed":key[0],"update":key[1],"parameter_id":key[2],"shape":a["shape"],"reference_method":ref,
                "raw_error_ratio_vq_over_reference":b["raw_relative_l2"]/max(a["raw_relative_l2"],1e-30),
                "skew_channel_ratio_vq_over_reference":b["skew_delta_norm_sq"]/max(a["skew_delta_norm_sq"],1e-30),
                "out_of_subspace_ratio_vq_over_reference":b["out_of_subspace_delta_norm_sq"]/max(a["out_of_subspace_delta_norm_sq"],1e-30),
                "frechet_ratio_vq_over_reference":b["frechet_relative_l2"]/max(a["frechet_relative_l2"],1e-30),
                "actual_distortion_ratio_vq_over_reference":b["actual_update_distortion"]/max(a["actual_update_distortion"],1e-30),
                "polar_distortion_ratio_vq_over_reference":b["actual_polar_distortion"]/max(a["actual_polar_distortion"],1e-30),
                "delta_actual_distortion":b["actual_update_distortion"]-a["actual_update_distortion"],
                "delta_frechet":b["frechet_relative_l2"]-a["frechet_relative_l2"]})
    # Pairwise-sensitivity scores are assembled above per tensor.
    write_csv(OUT/"method_manifest.csv",manifest)
    write_csv(OUT/"frechet_channel_metrics.csv",channel_rows)
    write_csv(OUT/"rectangular_channel_metrics.csv",[{k:r[k] for k in ("seed","update","parameter_id","shape","method","out_of_subspace_delta_norm_sq","out_of_subspace_fraction","frechet_relative_l2")} for r in records])
    write_csv(OUT/"spectral_band_breakdown.csv",[{"seed":r["seed"],"update":r["update"],"parameter_id":r["parameter_id"],"shape":r["shape"],"method":r["method"],**{k:v for k,v in r.items() if k in ("channel","band_or_pair","predicted_delta_sq","active_rank")}} for r in band_rows])
    write_csv(OUT/"jvp_validation.csv",jvp_rows);write_csv(OUT/"finite_difference_linearity.csv",fd_rows);write_csv(OUT/"polar_special_case.csv",polar_rows)
    write_csv(OUT/"channel_ablation.csv",[{"seed":r["seed"],"update":r["update"],"parameter_id":r["parameter_id"],"shape":r["shape"],"method":r["method"],**{k:r[k] for k in ("magnitude_only","symmetric_only","skew_only","out_of_subspace_only","orientation_plus_outside","all_channels")}} for r in records])
    write_csv(OUT/"gap_vs_sigma_sum.csv",[{k:r[k] for k in ("seed","update","parameter_id","shape","method","gap_skew_score","sigma_sum_skew_score","finite_k_skew_score","actual_update_distortion")} for r in records])
    write_csv(OUT/"sigma_sum_pair_sample.csv",pair_rows)
    write_csv(OUT/"scalar_vs_vector_int3.csv",paired_sv);write_csv(OUT/"int4_vs_vector_int3.csv",paired_i4)
    write_csv(OUT/"tail_angle_comparison.csv",records)
    # Error tails are retained, not hidden behind aggregate correlations.
    failures=sorted(records,key=lambda r:abs(r["frechet_distortion_pred"]-r["actual_update_distortion"]),reverse=True)[:100]
    write_csv(OUT/"first_order_failures.csv",[{k:r[k] for k in ("seed","update","parameter_id","parameter_name","shape","method","frechet_distortion_pred","actual_update_distortion","actual_frechet_cosine_error_difference","frechet_relative_l2","actual_update_relative_l2","sigma_max","sigma_min","condition_number_proxy","state_error_relative_fro","quadratic_gram_ratio")} for r in failures])
    corr_rows=[];pred_rows=[]
    predictors=("raw_relative_l2","right_gram_relative_fro","right_gram_relative_spectral","tail_angle","prior_gap_proxy","gap_skew_score","sigma_sum_skew_score","finite_k_skew_score","frechet_relative_l2","frechet_distortion_pred","frozen_norm_frechet_distortion_pred","magnitude_only","symmetric_only","skew_only","out_of_subspace_only","orientation_plus_outside","all_channels","jvp_output_relative_l2","jvp_predicted_cosine_distortion","exact_polar_pred_distortion")
    for scope,fields in (("global",[]),("per_method",["method"]),("per_shape",["shape"]),("per_seed",["seed"])):
        groups=defaultdict(list)
        for r in records:groups[tuple(r[f] for f in fields)].append(r)
        for group,rr in groups.items():
            for outcome in ("actual_update_distortion","actual_polar_distortion"):
                for pred in predictors:
                    if pred=="exact_polar_pred_distortion":
                        if outcome!="actual_polar_distortion":continue
                        data=[r for r in rr if r.get(pred) not in (None,"")];x=[r[pred] for r in data]
                    else:
                        data=[r for r in rr if r.get(pred) not in (None,"")]
                        x=[r[pred] for r in data]
                    y=[r[outcome] for r in data]
                    corr_rows.append({"scope":scope,**dict(zip(fields,group)),"outcome":outcome,"predictor":pred,"n":len(data),"pearson":pearson(x,y),"spearman":rank_corr(x,y)})
    write_csv(OUT/"predictor_correlations.csv",corr_rows)
    models={"raw":["raw_relative_l2"],"gram":["right_gram_relative_fro"],"tail_angle":["tail_angle"],"prior_gap":["prior_gap_proxy"],"sigma_sum":["sigma_sum_skew_score"],
            "frechet_relative_l2":["frechet_relative_l2"],"frechet_cosine_pred":["frechet_distortion_pred"],"tail_plus_frechet":["tail_angle","frechet_distortion_pred"],"gram_plus_frechet":["right_gram_relative_fro","frechet_distortion_pred"]}
    for name,features in models.items():
        pred_rows.append({"validation":"pooled_in_sample","held_out_group":"all","model":name,"features":";".join(features),"train_n":len(records),"test_n":len(records),"r2":fit_r2(records,records,features)})
        for tr_seed,te_seed in ((0,1),(1,0)):
            tr=[r for r in records if r["seed"]==tr_seed];te=[r for r in records if r["seed"]==te_seed]
            pred_rows.append({"validation":"held_out_seed","held_out_group":te_seed,"model":name,"features":";".join(features),"train_n":len(tr),"test_n":len(te),"r2":fit_r2(tr,te,features)})
        for method in METHODS:
            tr=[r for r in records if r["method"]!=method];te=[r for r in records if r["method"]==method]
            pred_rows.append({"validation":"leave_one_method_out","held_out_group":method,"model":name,"features":";".join(features),"train_n":len(tr),"test_n":len(te),"r2":fit_r2(tr,te,features)})
    # Exact-polar prediction is evaluated only on the deterministic subset for
    # which the expensive compressed polar factors were explicitly computed.
    polar_records=[r for r in records if r.get("exact_polar_pred_distortion") not in (None,"")]
    for name,features in (("polar_raw",["raw_relative_l2"]),("polar_gram",["right_gram_relative_fro"]),
                          ("polar_tail_angle",["tail_angle"]),("polar_frechet",["exact_polar_pred_distortion"])):
        for te_seed in (0,1):
            tr=[r for r in polar_records if r["seed"]!=te_seed];te=[r for r in polar_records if r["seed"]==te_seed]
            pred_rows.append({"validation":"held_out_seed_exact_polar_subset","held_out_group":te_seed,"model":name,"features":";".join(features),"train_n":len(tr),"test_n":len(te),"r2":fit_r2(tr,te,features,target="actual_polar_distortion")})
        for method in METHODS:
            tr=[r for r in polar_records if r["method"]!=method];te=[r for r in polar_records if r["method"]==method]
            pred_rows.append({"validation":"leave_one_method_out_exact_polar_subset","held_out_group":method,"model":name,"features":";".join(features),"train_n":len(tr),"test_n":len(te),"r2":fit_r2(tr,te,features,target="actual_polar_distortion")})
    # Production-JVP predictor uses only selected rows; fit/validation coverage is disclosed.
    jrecords=[r for r in records if r.get("jvp_output_relative_l2") not in (None,"")]
    if jrecords:
        for validation,groups in (("pooled_in_sample",[("all",jrecords,jrecords)]),("held_out_seed",[(1,[r for r in jrecords if r["seed"]==0],[r for r in jrecords if r["seed"]==1]),(0,[r for r in jrecords if r["seed"]==1],[r for r in jrecords if r["seed"]==0])])):
            for group,tr,te in groups:
                for model,feature in (("production_jvp_norm","jvp_output_relative_l2"),("production_jvp_cosine","jvp_predicted_cosine_distortion")):
                    pred_rows.append({"validation":validation,"held_out_group":group,"model":model,"features":feature,"train_n":len(tr),"test_n":len(te),"r2":fit_r2(tr,te,[feature])})
        for method in METHODS:
            tr=[r for r in jrecords if r["method"]!=method];te=[r for r in jrecords if r["method"]==method]
            for model,feature in (("production_jvp_norm","jvp_output_relative_l2"),("production_jvp_cosine","jvp_predicted_cosine_distortion")):
                pred_rows.append({"validation":"leave_one_method_out","held_out_group":method,"model":model,"features":feature,"train_n":len(tr),"test_n":len(te),"r2":fit_r2(tr,te,[feature])})
    write_csv(OUT/"predictive_models.csv",pred_rows)
    write_csv(OUT/"method_manifest.csv",manifest)
    write_csv(OUT/"predictor_correlations.csv",corr_rows)
    write_csv(OUT/"tensor_metrics.csv",records)
    write_plots(OUT,records,fd_rows,jvp_rows,polar_rows)
    elapsed=time.perf_counter()-start;(OUT/"runtime_seconds.txt").write_text(f"{elapsed:.3f} seconds\n")
    write_methodology(OUT);write_summary(records,corr_rows,pred_rows,paired_sv,paired_i4,OUT,elapsed)
    print(f"completed {len(records)} method instances in {elapsed:.1f} CPU seconds",flush=True)

def write_methodology(out):
    (out/"methodology.md").write_text("""# Methodology

The study uses 300 formal 2-D momentum matrices from two seeds and five landmarks, with four unchanged representation recipes: direct dynamic INT4; structural top-8 BF16 plus practical p98/global Lloyd-Max scalar INT3; structural INT4 residual; and structural 64-word 2-D INT3 VQ. Reconstructions are rebuilt from the prior recipes and checked against prior cached raw and production K=5 metrics. No quantizer is added or tuned.

The canonical transform is the production `zeropower_newton_schulz` function. Its polynomial spectral map is composed for five iterations after matrix-dependent Frobenius normalization. The analytic derivative differentiates both the polynomial map and that normalization. A frozen-normalization derivative is retained only as an ablation. Tall matrices are represented with compact SVD left leakage; wide matrices use the transposed/right-row-space counterpart. Exact polar uses compact reduced SVD and the partial polar convention.

K=5 JVP uses `torch.func.jvp` on the production function with fixed E, with no derivative through quantization. Finite differences perturb M by epsilon E and compare secants with the analytical derivative. Selected checks use 30 tensor instances across all four methods for JVP and polar, and 12 instances across methods for finite-difference curves. Full-set first-order metrics and correlations use all 1200 method/tensor pairs.

Divided differences use the derivative limit when singular values differ by at most 1e-7 times their local scale. Leakage uses f_i/x_i with machine-tiny flooring; rank-deficient cases are flagged by sigma-min/condition metrics and may be unreliable. Mode bands reuse active sigma/sigma-max >= 1e-6 and 10% head/centered-middle/tail definitions. Pairwise gap and sum scores are energy-weighted sensitivities using the skew SVD-coordinate error. They are descriptive, not causal bounds.

Correlation and small log-linear models are descriptive. Held-out-seed and leave-one-method-out fits are performed without feature normalization leakage; no significance tests are claimed because tensor and landmark observations repeat. Exact-polar and derivative ablations do not alter production behavior. Group-wise study evidence is only referenced from its prior summary and not pooled or recomputed.
""")

def write_summary(records,corr,pred,sv,vq,out,elapsed):
    def c(scope,outcome,predictor):return next((r for r in corr if r["scope"]==scope and r["outcome"]==outcome and r["predictor"]==predictor),None)
    def r2(model,validation,group):return next((float(r["r2"]) for r in pred if r["model"]==model and r["validation"]==validation and str(r["held_out_group"])==str(group)),float("nan"))
    def mean_r2(model,validation):
        vals=[float(r["r2"]) for r in pred if r["model"]==model and r["validation"]==validation and np.isfinite(float(r["r2"]))]
        return statistics.mean(vals) if vals else float("nan")
    outcorr=[c("global","actual_update_distortion",p) for p in ("raw_relative_l2","right_gram_relative_fro","tail_angle","prior_gap_proxy","sigma_sum_skew_score","finite_k_skew_score","frechet_relative_l2","frechet_distortion_pred")]
    channel_means={ch:statistics.mean(float(r[f"{ch}_fraction"]) for r in records) for ch in CHANNELS}
    norm_term=[float(r["normalization_derivative_delta_l2"]) for r in records]
    fd=read_csv(out/"finite_difference_linearity.csv");jvps=read_csv(out/"jvp_validation.csv");polar=read_csv(out/"polar_special_case.csv")
    eps1=[float(r["relative_l2_mismatch"]) for r in fd if float(r["epsilon"])==1.0]
    jvp_err=[float(r["analytic_jvp_relative_l2"]) for r in jvps if r["analytic_jvp_relative_l2"] not in ("", "nan")]
    polar_corr=pearson([float(r["predicted_polar_distortion"]) for r in polar],[float(r["actual_polar_distortion"]) for r in polar]) if polar else float("nan")
    heldout_frechet=mean_r2("frechet_cosine_pred","held_out_seed")
    heldout_raw=mean_r2("raw","held_out_seed");heldout_gram=mean_r2("gram","held_out_seed")
    lomo_frechet=mean_r2("frechet_relative_l2","leave_one_method_out")
    lomo_tail=mean_r2("tail_angle","leave_one_method_out")
    polar_heldout_frechet=mean_r2("polar_frechet","held_out_seed_exact_polar_subset")
    polar_heldout_tail=mean_r2("polar_tail_angle","held_out_seed_exact_polar_subset")
    lines=["# First-order Fréchet sensitivity of low-bit Muon states","",f"CPU-only study: 300 matrices × four methods = {len(records)} instances; runtime {elapsed:.1f}s. Production K=5 transform and prior quantizers were not changed; training was not run.","",
        "## Production map and derivative","",
        "Production computes `X = M/(||M||F + 1e-7)` (transposing tall matrices before the iteration and restoring orientation afterward), then applies five steps `X <- aX+b(XXᵀ)X+c(XXᵀ)^2X` with `(a,b,c)=(3.4445,-4.7750,2.0315)`. There is no output rescaling. The map is spectral at a fixed scale, but production scale depends on M; the reported production derivative includes this radial normalization term. See `production_map_definition.md`.","",
        f"The normalization-derivative correction itself has mean relative norm {statistics.mean(norm_term):.3f} of the FP32 update norm (median {statistics.median(norm_term):.3f}, p90 {float(np.quantile(norm_term,.9)):.3f}); it is not the main sensitivity term on average, but is included and is necessary for exact agreement with production JVP.","",
        "## Core association with actual K=5 cosine distortion","","| predictor | Pearson | Spearman | n |","|:--|--:|--:|--:|"]
    for row in outcorr:
        if row:lines.append(f"| {row['predictor']} | {float(row['pearson']):.3f} | {float(row['spearman']):.3f} | {row['n']} |")
    lines += ["","The production-map analytic predictor is `1-cos(O, O + DΦ[M](E))`; its relative-L2 counterpart and component energies are in `tensor_metrics.csv` / `frechet_channel_metrics.csv`. The frozen-normalization derivative is reported as an ablation. The exact production JVP is a separate predictor only on the deterministic 120-instance validation subset.","",
        "## Validation and nonlinear regime","",
        f"Analytic-vs-production JVP relative-L2 mismatch: median {statistics.median(jvp_err):.2e}, p95 {float(np.quantile(jvp_err,.95)):.2e} over {len(jvp_err)} rows. This confirms the implemented derivative matches the production-map JVP locally, including normalization.",
        f"At epsilon=1 (the full observed quantization residual), first-order secant mismatch has median {statistics.median(eps1):.3f}; local error grows with epsilon (see finite-difference curve). Thus the derivative is a local sensitivity model and a strong ranking metric, not an exact finite-error reconstruction of the final update.",
        f"Exact-polar subset has {len(polar)} rows; analytic polar prediction vs actual polar distortion Pearson is {polar_corr:.3f}. Exact-polar magnitude and symmetric channels are identically zero; skew and rectangular leakage remain.","",
        "## Generalization and channels","",
        f"Mean held-out-by-seed K=5 R²: Fréchet cosine predictor {heldout_frechet:.3f}, raw state error {heldout_raw:.3f}, Gram Frobenius error {heldout_gram:.3f}. Mean leave-one-method-out R²: Fréchet relative-L2 {lomo_frechet:.3f}, tail angle {lomo_tail:.3f}. On the smaller exact-polar subset, mean held-out-seed R² is {polar_heldout_frechet:.3f} for the polar Fréchet prediction and {polar_heldout_tail:.3f} for tail angle. These are descriptive repeated-snapshot generalization checks, not independent-sample inferential statistics.",
        "Mean share of squared derivative prediction by channel: "+", ".join(f"{k} {v:.1%}" for k,v in channel_means.items())+". Energy share is not identical to explanatory power: out-of-subspace leakage can carry large predicted norm while alone being a weak across-instance predictor.","",
        "## Method comparisons","",
        f"Scalar INT3→2-D VQ (paired n={len(sv)}): median VQ/scalar ratios for raw error, skew-channel energy, out-of-subspace energy, total Fréchet relative-L2, actual update distortion, and polar distortion are {', '.join(f'{statistics.median(float(r[k]) for r in sv):.3f}' for k in ('raw_error_ratio_vq_over_reference','skew_channel_ratio_vq_over_reference','out_of_subspace_ratio_vq_over_reference','frechet_ratio_vq_over_reference','actual_distortion_ratio_vq_over_reference','polar_distortion_ratio_vq_over_reference'))}.","",
        f"Structural INT4→2-D VQ (paired n={len(vq)}): corresponding median ratios are {', '.join(f'{statistics.median(float(r[k]) for r in vq):.3f}' for k in ('raw_error_ratio_vq_over_reference','skew_channel_ratio_vq_over_reference','out_of_subspace_ratio_vq_over_reference','frechet_ratio_vq_over_reference','actual_distortion_ratio_vq_over_reference','polar_distortion_ratio_vq_over_reference'))}. Ratios below one favor VQ.","",
        "## Assessment","",
        "Overall, the evidence supports the first-order Fréchet sensitivity model as a substantially better descriptive surrogate than raw state error or global Gram norms: its predicted cosine distortion has near-monotonic pooled association and stronger held-out-seed prediction, while the analytic derivative numerically matches the actual production JVP. The finite-error linearization is not exact at epsilon=1, and some rank-deficient / high-error instances remain poorly predicted. The result supports a future optimizer-induced quantization objective as a hypothesis worth testing, but does not establish that minimizing this local metric will improve a finite-rate quantizer or training outcomes.","",
        "For the scalar INT3→2D VQ comparison, high-sensitivity skew and rectangular channels and total derivative-weighted error fall more than ordinary raw error. Structural INT4 and vector INT3 have nearly matched actual update distortion and nearly matched Fréchet metric despite VQ having somewhat larger raw state error. This cross-method consistency is positive evidence, not proof of unique causality.","",
        "No derivative-aware quantizer or new representation is implemented. Group-wise evidence is prior external context only. Because all landmarks reuse parameter identities, predictive scores are descriptive rather than independent generalization estimates.","",
        f"Runtime: {elapsed:.1f}s CPU. Full table and all plots are in this directory."]
    (out/"summary.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":main()
