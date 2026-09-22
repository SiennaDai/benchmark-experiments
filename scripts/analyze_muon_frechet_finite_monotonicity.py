#!/usr/bin/env python3
"""Finite-error monotonicity study for production Muon Fréchet sensitivity."""
from __future__ import annotations

import csv, math, statistics, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from analyze_muon_frechet_sensitivity import (LANDMARKS,OUT as PRIOR_OUT,cosine,load_snapshots,
    pearson,rank_corr,read_csv,reconstructions,write_csv)
from analyze_muon_vector_int3_residual import eligible_items
from optim.muon_finite_monotonicity import (cosine64,cosine_distortion,pairwise_order_accuracy,
    polar_factor,polar_skew_exact,projected_sensitivity)
from optim.muon_frechet import frechet_channels
from optim.muon_spectral_sensitivity import decompose
from optim.muon_reference import zeropower_newton_schulz

OUT=ROOT/"reports/muon_frechet_finite_monotonicity"
VQ_DIR=ROOT/"reports/muon_vector_int3_robustness"
T_VALUES=(.05,.1,.2,.4,.6,.8,1.,1.25,1.5)
METHODS=("direct_scalar_int4","structural_scalar_int3_p98_k8","structural_int4_k8","structural_vq64_int3_k8")
LABELS={"direct_scalar_int4":"direct INT4","structural_scalar_int3_p98_k8":"structural scalar INT3",
        "structural_int4_k8":"structural INT4","structural_vq64_int3_k8":"64-word 2D INT3 VQ"}


def read(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))


def rank_from_factors(sigma):
    if not sigma.numel() or float(sigma[0])<=0:return 0
    return int((sigma/sigma[0]>=1e-6).sum())


def exact_polar_metrics(m,e,o0,t):
    ot=polar_factor((m+t*e).double())
    return cosine_distortion(o0,ot),float(torch.linalg.vector_norm(ot-o0)/torch.linalg.vector_norm(o0).clamp_min(1e-30))


def execute_two_mode_controls():
    rows=[]
    # Skew family: all positive singular values, hence no rank/sign transition.
    for si,sj,e in ((2.,1.,.2),(2.3,.6,.4),(.9,.35,.3),(.3,.08,.15)):
        for t in T_VALUES:
            pred=polar_skew_exact(si,sj,e,t)
            mat=torch.tensor([[si,t*e],[-t*e,sj]],dtype=torch.float64)
            q=polar_factor(mat);numeric=cosine_distortion(torch.eye(2,dtype=torch.float64),q)
            rows.append({"family":"pure_skew","sigma_i":si,"sigma_j":sj,"e":e,"t":t,**pred,
                         "local_cosine_quadratic_coefficient":.5*pred["s"]**2,
                         "numeric_cosine_distortion":numeric,"formula_abs_error":abs(numeric-pred["distortion"]),
                         "rotation_angle_error":abs(math.atan2(float(q[1,0]),float(q[0,0]))-pred["angle"])})
    # Smooth controls remain positive definite for the entire requested range.
    si,sj=2.,1.
    for t in T_VALUES:
        for family,e in (("diagonal",torch.diag(torch.tensor([.15,-.08],dtype=torch.float64))),
                         ("symmetric_offdiag",torch.tensor([[0.,.2],[.2,0.]],dtype=torch.float64))):
            m=torch.diag(torch.tensor([si,sj],dtype=torch.float64));q0=polar_factor(m)
            qt=polar_factor(m+t*e)
            ev=torch.linalg.eigvalsh(m+t*e)
            rows.append({"family":family,"sigma_i":si,"sigma_j":sj,"e":float(torch.linalg.vector_norm(e)),"t":t,
                         "numeric_cosine_distortion":cosine_distortion(q0,qt),"minimum_eigenvalue":float(ev.min()),
                         "smooth_positive_definite":bool(float(ev.min())>0)})
    # Symmetric branch transition is explicitly marked rather than treated as a smooth point.
    m=torch.diag(torch.tensor([1.,1.],dtype=torch.float64));e=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)
    for t in (.8,1.,1.2):
        q=polar_factor(m+t*e)
        rows.append({"family":"symmetric_crossing_diagnostic","sigma_i":1.,"sigma_j":1.,"e":1.,"t":t,
                     "numeric_cosine_distortion":cosine_distortion(torch.eye(2,dtype=torch.float64),q),
                     "minimum_eigenvalue":float(torch.linalg.eigvalsh(m+t*e).min()),
                     "smooth_positive_definite":bool(float(torch.linalg.eigvalsh(m+t*e).min())>1e-12)})
    return rows


def model_value(name,x,p):
    x=np.maximum(np.asarray(x,dtype=float),0)
    if name=="polar_inspired":return 1-1/np.sqrt(1+p[0]*x*x)
    if name=="rational":return p[0]*x*x/(1+p[1]*x*x)
    if name=="exponential":return 1-np.exp(-p[0]*x*x)
    raise ValueError(name)


def fit_saturation(name,x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    grid=np.exp(np.linspace(-12,12,481))
    if name=="rational":
        # Deterministic coarse-to-fine Cartesian grid; this model has only two
        # positive parameters and the study needs interpretability, not a
        # high-dimensional optimizer.
        best=None
        for a in grid[::8]:
            for b in grid[::8]:
                pred=model_value(name,x,(a,b));loss=float(np.sum((pred-y)**2))
                if best is None or loss<best[0]:best=(loss,a,b)
        _,a,b=best
        fine_a=np.exp(np.linspace(math.log(a)-1.0,math.log(a)+1.0,81));fine_b=np.exp(np.linspace(math.log(b)-1.0,math.log(b)+1.0,81))
        for a2 in fine_a:
            for b2 in fine_b:
                pred=model_value(name,x,(a2,b2));loss=float(np.sum((pred-y)**2))
                if loss<best[0]:best=(loss,a2,b2)
        p=np.asarray(best[1:])
    else:
        losses=[]
        for a in grid:
            pred=model_value(name,x,(a,));losses.append(float(np.sum((pred-y)**2)))
        p=np.asarray([grid[int(np.argmin(losses))]])
    pred=model_value(name,x,p)
    r2=1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)),1e-30)
    return p,pred,r2


def eval_outputs(m,base,e,t):
    with torch.no_grad():ot=zeropower_newton_schulz((m+t*e).clone(),steps=5,coefficients=(3.4445,-4.7750,2.0315),eps=1e-7)
    if not bool(torch.isfinite(ot).all()):return float("nan"),float("nan"),ot
    return cosine_distortion(base,ot),float(torch.linalg.vector_norm(ot-base)/torch.linalg.vector_norm(base).clamp_min(1e-30)),ot


def derive_documents(out):
    (out/"local_cosine_derivation.md").write_text(r"""# Local cosine expansion

Let (x=O_0=Phi(M)), (r=|x|_F>0), and (u=x/r). Write the perturbed output as a radial and orthogonal part:

\[
O(t)=\alpha(t)u+q(t),\qquad q(t)=P_x^\perp O(t).
\]

If (Phi) is twice differentiable near (M),

\[
O(t)=x+t v+\tfrac12t^2a+O(t^3),\quad v=D\Phi_M[E].
\]

Consequently (alpha(t)=r+t\langle u,v\rangle+O(t^2)) and (q(t)=t v_\perp+\tfrac12t^2a_\perp+O(t^3)), where (v_\perp=P_x^\perp v). For positive (alpha),

\[
\cos(x,O(t))=\frac{\alpha(t)}{\sqrt{\alpha(t)^2+\|q(t)\|_F^2}}
=\left(1+\frac{\|q(t)\|_F^2}{\alpha(t)^2}\right)^{-1/2}.
\]

Since (|q(t)|^2=t^2|v_\perp|^2+t^3\langle v_\perp,a_\perp\rangle+O(t^4)), expanding ((1+z)^{-1/2}=1-z/2+O(z^2)) gives

\[
1-\cos(x,O(t))=\frac{t^2}{2r^2}\|P_x^\perp D\Phi_M[E]\|_F^2+O(t^3).
\]

The second derivative (a) cannot enter the quadratic coefficient; it first enters the cubic term through its component orthogonal to (x). A radial second-order correction changes output norm but not the leading angle. If (v_\perp=0), the quadratic coefficient vanishes and the first nonzero angular term can be fourth order. This is a local expansion only; it says nothing by itself about monotone behavior for finite (t).

For exact polar, (Q(M)^TQ(M)=I) on the active square factor, and its derivative is tangent to the orthogonal/Stiefel manifold. Thus \(\langle Q,DQ[E]\rangle_F=0\) and (v_\perp=v); with rectangular partial isometries the same identity holds on the active polar factor when rank is locally constant.
""")
    (out/"two_mode_exact_polar.md").write_text(r"""# Exact two-mode polar response

Take (Sigma=\operatorname{diag}(\sigma_i,\sigma_j)), (sigma_i,sigma_j>0), and (E=\begin{pmatrix}0&e\\-e&0\end{pmatrix}). Then

\[
A(t)=\Sigma+tE=\begin{pmatrix}\sigma_i&te\\-te&\sigma_j\end{pmatrix},\qquad \det A(t)=\sigma_i\sigma_j+t^2e^2>0.
\]

Its polar factor is

\[
Q(t)=A(A^TA)^{-1/2}=\frac1{\sqrt{(\sigma_i+\sigma_j)^2+4t^2e^2}}
\begin{pmatrix}\sigma_i+\sigma_j&2te\\-2te&\sigma_i+\sigma_j\end{pmatrix}.
\]

Writing (Q=\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix}) gives (\tan\theta=2te/(\sigma_i+\sigma_j)). With (s=2|e|/(\sigma_i+\sigma_j)), the two-dimensional Frobenius cosine to (Q(0)=I) is (cos\theta), hence

\[
d(t)=1-\cos\theta=1-\frac1{\sqrt{1+t^2s^2}},\qquad \partial_s d=\frac{t^2s}{(1+t^2s^2)^{3/2}}>0
\]

for (s,t>0). The ordering in (s) is exactly monotone at every fixed positive (t), although the true angle is (\arctan(ts)), not its unbounded linear approximation (ts).

For diagonal perturbations and symmetric off-diagonal perturbations, the matrix remains symmetric positive definite over the reported smooth interval; its polar factor remains (I), so distortion is exactly zero. At a zero eigenvalue the polar map is not differentiable; after a sign change the factor changes branch. Those crossing points are recorded separately and excluded from smooth-branch monotonicity claims.
""")


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--threads",type=int,default=4);args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(2026);start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    snapshots=load_snapshots();blob=torch.load(VQ_DIR/"calibration_codebooks.pt",map_location="cpu",weights_only=False);codebooks=blob["codebooks"]
    prior=[]
    for r in read_csv(PRIOR_OUT/"tensor_metrics.csv"):
        key=(int(r["seed"]),int(r["update"]),r["parameter_id"],r["method"]);prior.append((key,r))
    pmap=dict(prior);data=[]
    for (seed,update),snap in sorted(snapshots.items()):
        for item in eligible_items(snap):
            if item["tensor"].ndim==2:data.append((seed,update,item))
    if len(data)!=300:raise RuntimeError(f"expected 300 formal tensor snapshots, found {len(data)}")
    tensor_rows=[];sweep_rows=[];polar_rows=[];mode_rows=[];interaction_rows=[];manifest=[]
    checkpoint=OUT/".finite_sweep_checkpoint.pt";start_index=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        tensor_rows=saved["tensor_rows"];sweep_rows=saved["sweep_rows"];polar_rows=saved["polar_rows"]
        mode_rows=saved["mode_rows"];interaction_rows=saved["interaction_rows"];start_index=int(saved["next_index"])
        print(f"resuming finite sweep at matrix {start_index}/300",flush=True)
    exact_polar_ix=set(np.linspace(0,299,6).round().astype(int).tolist())
    mode_ix=set(np.linspace(0,299,3).round().astype(int).tolist())
    for ix in range(start_index,len(data)):
        seed,update,item=data[ix]
        m=item["tensor"].detach().cpu().float();pid=str(item.get("parameter_id",item.get("name")));name=str(item.get("name",pid));svd=decompose(m)
        recons,residual=reconstructions(m,svd,seed,codebooks)
        o0=zeropower_newton_schulz(m.clone(),steps=5,coefficients=(3.4445,-4.7750,2.0315),eps=1e-7).double()
        norm_m=float(torch.linalg.vector_norm(m));sigma=svd.singular_values.double();active=rank_from_factors(sigma)
        tail_mass=float(sigma[max(0,active-int(math.ceil(.1*active))):active].square().sum()/sigma[:active].square().sum().clamp_min(1e-30)) if active else 0.
        errors_by={}
        for method in METHODS:
            mh=recons[method].float();e=mh-m
            fr=frechet_channels(m,e,svd_factors=(svd.u,svd.singular_values,svd.vh))
            v=fr["predicted_delta"].double();sperp,sfull,vperp=projected_sensitivity(o0,v)
            prior_row=pmap.get((seed,update,pid,method),{})
            raw=float(torch.linalg.vector_norm(e)/max(norm_m,1e-30))
            condition=float(sigma[0]/sigma[active-1].clamp_min(1e-30)) if active else float("inf")
            row={"seed":seed,"update":update,"parameter_id":pid,"parameter_name":name,"shape":str(tuple(m.shape)),"method":method,
                 "state_error_relative_l2":raw,"s_perp":sperp,"s_full":sfull,"local_cosine_coefficient":.5*sperp*sperp,
                 "derivative_radial_fraction":float((torch.linalg.vector_norm(v-vperp)/torch.linalg.vector_norm(v).clamp_min(1e-30))),
                 "sigma_min_active_over_max":float(sigma[active-1]/sigma[0]) if active else 0.,"condition_active":condition,"tail_spectral_mass":tail_mass,
                 "active_rank":active,"matrix_norm":norm_m,"error_norm":float(torch.linalg.vector_norm(e)),
                 "tail_angle_prior":float(prior_row.get("tail_angle","nan")),
                 "raw_relative_l2_prior":float(prior_row.get("raw_relative_l2",raw)),
                 "gram_fro_prior":float(prior_row.get("right_gram_relative_fro","nan")),
                 "cached_update_distortion_prior":float(prior_row.get("actual_update_distortion","nan"))}
            errors_by[method]=(e,fr,row)
            tensor_rows.append(row)
            for t in T_VALUES:
                d,rel,ot=eval_outputs(m,o0.float(),e,t)
                delta=ot.double()-o0
                rem=delta-float(t)*v
                on=o0/torch.linalg.vector_norm(o0).clamp_min(1e-30)
                rv=v-on*torch.sum(on*v)
                rvhat=rv/torch.linalg.vector_norm(rv).clamp_min(1e-30)
                rrad=on*torch.sum(on*rem);rtan=rvhat*torch.sum(rvhat*rem)
                rrest=rem-rrad-rtan
                rn=float(torch.linalg.vector_norm(rem).clamp_min(1e-30))
                sweep_rows.append({"seed":seed,"update":update,"parameter_id":pid,"parameter_name":name,"shape":str(tuple(m.shape)),"method":method,"t":t,
                    "finite_cosine_distortion":d,"finite_update_relative_l2":rel,"t_s_perp":t*sperp,"t_s_full":t*sfull,
                    "s_perp":sperp,"s_full":sfull,"tail_angle_prior":float(prior_row.get("tail_angle","nan")),
                    "state_error_relative_l2":raw,"gram_fro_prior":float(prior_row.get("right_gram_relative_fro","nan")),
                    "first_order_cosine_prediction":.5*(t*sperp)**2,"finite_vector_mismatch":float(torch.linalg.vector_norm(rem)/torch.linalg.vector_norm(delta).clamp_min(1e-30)),
                    "remainder_radial_fraction":float(torch.linalg.vector_norm(rrad)/rn),"remainder_tangent_fraction":float(torch.linalg.vector_norm(rtan)/rn),
                    "remainder_other_fraction":float(torch.linalg.vector_norm(rrest)/rn),"stable":bool(torch.isfinite(ot).all())})
            if ix in exact_polar_ix:
                op=polar_factor(m.double())
                pfr=frechet_channels(m,e,svd_factors=(svd.u,svd.singular_values,svd.vh),exact_polar=True)
                psperp,psfull,_=projected_sensitivity(op,pfr["predicted_delta"])
                for t in T_VALUES:
                    dist,rel=exact_polar_metrics(m.double(),e.double(),op,t)
                    polar_rows.append({"seed":seed,"update":update,"parameter_id":pid,"method":method,"t":t,"exact_polar_distortion":dist,"exact_polar_relative_l2":rel,
                                       "s_perp":psperp,"s_full":psfull})
        if ix in mode_ix:
            # Isolate the top five derivative-weighted skew spectral mode pairs.
            base_method="structural_vq64_int3_k8";e,fr,_=errors_by[base_method]
            f=fr["normalized_coordinate_error"];a=(f-f.T)*.5;weight=fr["skew_factor"].abs()*a.abs()
            vals=weight.triu(diagonal=1);flat=torch.argsort(vals.flatten(),descending=True)[:5]
            pair_data=[];individual_d=[];individual_s2=[];individual_e=[]
            u,s,vh=svd.u.double(),svd.singular_values.double(),svd.vh.double()
            for rank,flatidx in enumerate(flat.tolist(),1):
                i=flatidx//vals.shape[1];j=flatidx%vals.shape[1];amp=float(a[i,j]);fc=torch.zeros_like(f);fc[i,j]=amp;fc[j,i]=-amp
                ep=u@fc@vh
                ei=frechet_channels(m,ep.float(),svd_factors=(svd.u,svd.singular_values,svd.vh))["predicted_delta"].double()
                si,_,_=projected_sensitivity(o0,ei)
                s_analytic=2*abs(amp)/max(float(s[i]+s[j]),1e-30)
                m2=torch.diag(torch.tensor([float(s[i]),float(s[j])],dtype=torch.float64));e2=torch.tensor([[0.,amp],[-amp,0.]],dtype=torch.float64)
                polarfit=polar_skew_exact(float(s[i]),float(s[j]),amp,1.)
                for t in T_VALUES:
                    d2=polar_skew_exact(float(s[i]),float(s[j]),amp,t)["distortion"]
                    with torch.no_grad():oo=zeropower_newton_schulz((m2+t*e2).float(),steps=5,coefficients=(3.4445,-4.7750,2.0315),eps=1e-7)
                    oo0=zeropower_newton_schulz(m2.float(),steps=5,coefficients=(3.4445,-4.7750,2.0315),eps=1e-7)
                    dp=cosine_distortion(oo0,oo)
                    pair_data.append({"seed":seed,"update":update,"parameter_id":pid,"method":base_method,"mode_i":i,"mode_j":j,"pair_rank":rank,
                        "sigma_i_over_max":float(s[i]/s[0]),"sigma_j_over_max":float(s[j]/s[0]),"skew_coordinate_amplitude":amp,"finite_k_weighted_skew":float(weight[i,j]),
                        "derivative_s_perp_full_tensor_isolated_pair":si,"two_mode_sensitivity_s":s_analytic,"t":t,"two_mode_exact_polar_distortion":d2,
                        "two_mode_k5_distortion":dp,"two_mode_k5_vector_relative_l2":float(torch.linalg.vector_norm(oo-oo0)/torch.linalg.vector_norm(oo0).clamp_min(1e-30))})
                individual_d.append(ep);individual_s2.append(si*si);individual_e.append(float(amp))
            # Interaction for the five selected pairs at t=1 in the original tensor.
            Etop=sum(individual_d,torch.zeros_like(m.double()))
            dtop,_,_=eval_outputs(m,o0.float(),Etop.float(),1.0)
            singles=[]
            for ep in individual_d:
                ds,_,_=eval_outputs(m,o0.float(),ep.float(),1.0);singles.append(ds)
            dfull,_,_=eval_outputs(m,o0.float(),e.float(),1.0)
            ftop=frechet_channels(m,Etop.float(),svd_factors=(svd.u,svd.singular_values,svd.vh))["predicted_delta"].double()
            s_top,_,_=projected_sensitivity(o0,ftop)
            interaction_rows.append({"seed":seed,"update":update,"parameter_id":pid,"method":base_method,"selected_mode_pairs":5,
                "full_error_distortion_t1":dfull,"top5_combined_distortion_t1":dtop,"sum_top5_isolated_distortions_t1":sum(singles),
                "actual_interaction_vs_sum":dtop-sum(singles),"full_s_perp":errors_by[base_method][2]["s_perp"],
                "top5_combined_s_perp":s_top,"sum_top5_isolated_s_perp_sq":sum(individual_s2),"frechet_cross_term_s_perp_sq":s_top*s_top-sum(individual_s2)})
            mode_rows.extend(pair_data)
        if (ix+1)%20==0:
            torch.save({"next_index":ix+1,"tensor_rows":tensor_rows,"sweep_rows":sweep_rows,"polar_rows":polar_rows,
                        "mode_rows":mode_rows,"interaction_rows":interaction_rows},checkpoint)
            print(f"finite sweep {ix+1}/300 matrices",flush=True)
    # Full t-by-tensor correlation table and pairwise order stability.
    corr_rows=[];ordering_rows=[];predictors=("s_perp","s_full","tail_angle_prior","state_error_relative_l2","gram_fro_prior")
    for t in T_VALUES:
        current=[r for r in sweep_rows if float(r["t"])==t]
        for predictor in predictors:
            x=[float(r[predictor]) for r in current];y=[float(r["finite_cosine_distortion"]) for r in current]
            corr_rows.append({"t":t,"predictor":predictor,"n":len(current),"pearson":pearson(x,y),"spearman":rank_corr(x,y)})
        sens=[float(r["s_perp"]) for r in current];dist=[float(r["finite_cosine_distortion"]) for r in current]
        for row in pairwise_order_accuracy(sens,dist,seed=2026+round(t*100),max_pairs=250000):ordering_rows.append({"t":t,**row})
    direction_rows=[];by_direction=defaultdict(list)
    for r in sweep_rows:by_direction[(r["seed"],r["update"],r["parameter_id"],r["method"])].append(r)
    for key,rr in by_direction.items():
        rr=sorted(rr,key=lambda x:float(x["t"]));d=[float(r["finite_cosine_distortion"]) for r in rr]
        direction_rows.append({"seed":key[0],"update":key[1],"parameter_id":key[2],"method":key[3],"n_t":len(d),
            "monotone_non_decreasing":all(b+1e-10>=a for a,b in zip(d,d[1:])),"monotone_step_fraction":sum(b+1e-10>=a for a,b in zip(d,d[1:]))/max(len(d)-1,1),
            "minimum_step_change":min((b-a for a,b in zip(d,d[1:])),default=float("nan")),"t1_distortion":next(float(r["finite_cosine_distortion"]) for r in rr if float(r["t"])==1.)})
    # Fit three common monotone response models, then method-specific equivalents.
    fit_rows=[];method_curve_rows=[];fit_cache={}
    for direction,cal_seed,eval_seed in (("0_to_1",0,1),("1_to_0",1,0)):
        train=[r for r in sweep_rows if int(r["seed"])==cal_seed];test=[r for r in sweep_rows if int(r["seed"])==eval_seed]
        for name in ("polar_inspired","rational","exponential"):
            pars,_,_=fit_saturation(name,[r["t_s_perp"] for r in train],[r["finite_cosine_distortion"] for r in train])
            pred=model_value(name,[r["t_s_perp"] for r in test],pars);y=np.asarray([r["finite_cosine_distortion"] for r in test]);r2=1-np.sum((y-pred)**2)/max(np.sum((y-y.mean())**2),1e-30)
            fit_rows.append({"direction":direction,"model":name,"scope":"common","calibration_seed":cal_seed,"evaluation_seed":eval_seed,
                "parameters":";".join(f"{x:.9g}" for x in pars),"heldout_r2":float(r2),"heldout_spearman":rank_corr(pred,y)})
            fit_cache[direction,name,"common"]=pars
            for method in METHODS:
                tr=[r for r in train if r["method"]==method];te=[r for r in test if r["method"]==method]
                p,_,_=fit_saturation(name,[r["t_s_perp"] for r in tr],[r["finite_cosine_distortion"] for r in tr])
                pm=model_value(name,[r["t_s_perp"] for r in te],p);ym=np.asarray([r["finite_cosine_distortion"] for r in te]);mr2=1-np.sum((ym-pm)**2)/max(np.sum((ym-ym.mean())**2),1e-30)
                method_curve_rows.append({"direction":direction,"method":method,"model":name,"calibration_seed":cal_seed,"evaluation_seed":eval_seed,
                    "parameters":";".join(f"{x:.9g}" for x in p),"heldout_r2":float(mr2),"heldout_spearman":rank_corr(pm,ym),"n_train":len(tr),"n_test":len(te)})
    # Calibration common-curve residual diagnostics versus state/spectrum.
    condition_rows=[]
    for direction,cal_seed,eval_seed in (("0_to_1",0,1),("1_to_0",1,0)):
        pars=fit_cache[direction,"polar_inspired","common"]
        te=[r for r in sweep_rows if int(r["seed"])==eval_seed]
        pred=model_value("polar_inspired",[r["t_s_perp"] for r in te],pars)
        for r,p in zip(te,pred):
            residual=float(r["finite_cosine_distortion"]-p)
            base=next(x for x in tensor_rows if x["seed"]==r["seed"] and x["update"]==r["update"] and x["parameter_id"]==r["parameter_id"] and x["method"]==r["method"])
            condition_rows.append({"seed":eval_seed,"update":r["update"],"parameter_id":r["parameter_id"],"method":r["method"],"t":r["t"],"model_residual":residual,
                "sigma_min_active_over_max":base["sigma_min_active_over_max"],"condition_active":base["condition_active"],"tail_spectral_mass":base["tail_spectral_mass"],"error_norm":base["error_norm"],"state_error_relative_l2":base["state_error_relative_l2"],"shape":base["shape"]})
    condition_corr=[]
    for key in ("sigma_min_active_over_max","condition_active","tail_spectral_mass","error_norm","state_error_relative_l2"):
        xx=[float(r[key]) for r in condition_rows];yy=[float(r["model_residual"]) for r in condition_rows]
        condition_corr.append({"predictor":key,"pearson_vs_common_curve_residual":pearson(xx,yy),"spearman_vs_common_curve_residual":rank_corr(xx,yy),"n":len(xx)})
    # Documents, exact polar controls, and all result tables.
    controls=execute_two_mode_controls();derive_documents(OUT)
    expansion_rows=[]
    for si,sj,e in ((2.,1.,.2),(2.3,.6,.4),(.9,.35,.3)):
        s=2*abs(e)/(si+sj);m2=torch.diag(torch.tensor([si,sj],dtype=torch.float64));e2=torch.tensor([[0.,e],[-e,0.]],dtype=torch.float64);o=polar_factor(m2)
        for t in (1e-2,3e-3,1e-3):
            numerical=cosine_distortion(o,polar_factor(m2+t*e2));coefficient=.5*s*s
            expansion_rows.append({"sigma_i":si,"sigma_j":sj,"e":e,"t":t,"s_perp":s,"quadratic_coefficient":coefficient,
                "distortion_over_t2":numerical/t**2,"relative_coefficient_error":abs(numerical/t**2-coefficient)/max(coefficient,1e-30)})
    polar_corr_rows=[]
    for t in T_VALUES:
        rr=[r for r in polar_rows if float(r["t"])==t]
        for predictor in ("s_perp","s_full"):
            polar_corr_rows.append({"t":t,"predictor":predictor,"n":len(rr),"pearson":pearson([float(r[predictor]) for r in rr],[float(r["exact_polar_distortion"]) for r in rr]),
                "spearman":rank_corr([float(r[predictor]) for r in rr],[float(r["exact_polar_distortion"]) for r in rr])})
    write_csv(OUT/"tensor_sensitivities.csv",tensor_rows);write_csv(OUT/"t_sweep_metrics.csv",sweep_rows)
    write_csv(OUT/"correlation_vs_t.csv",corr_rows);write_csv(OUT/"pairwise_ordering.csv",ordering_rows)
    write_csv(OUT/"response_curve_fits.csv",fit_rows);write_csv(OUT/"method_specific_curves.csv",method_curve_rows)
    write_csv(OUT/"higher_order_remainder.csv",sweep_rows);write_csv(OUT/"polar_vs_k5.csv",polar_rows)
    write_csv(OUT/"two_mode_controls.csv",controls);write_csv(OUT/"two_mode_real_tensor.csv",mode_rows)
    write_csv(OUT/"cross_mode_interactions.csv",interaction_rows);write_csv(OUT/"shape_spectrum_residuals.csv",condition_rows)
    write_csv(OUT/"shape_spectrum_correlations.csv",condition_corr);write_csv(OUT/"direction_monotonicity.csv",direction_rows)
    write_csv(OUT/"local_expansion_validation.csv",expansion_rows);write_csv(OUT/"polar_correlation_vs_t.csv",polar_corr_rows)
    make_plots(OUT,sweep_rows,corr_rows,ordering_rows,fit_rows,method_curve_rows,polar_rows,mode_rows,interaction_rows);write_methodology(OUT)
    elapsed=time.perf_counter()-start
    (OUT/"runtime_seconds.txt").write_text(f"aggregation_seconds={elapsed:.3f}\nprimary_finite_sweep=completed in two checkpointed CPU runs; end-to-end wall time was not retained by the interrupted first run\n")
    write_summary(OUT,tensor_rows,sweep_rows,corr_rows,ordering_rows,fit_rows,method_curve_rows,polar_rows,mode_rows,interaction_rows,elapsed)
    if checkpoint.exists():checkpoint.unlink()
    print(f"completed {len(tensor_rows)} sensitivity and {len(sweep_rows)} finite-response rows in {elapsed:.1f}s CPU",flush=True)


def make_plots(out,sweeps,corr,ordering,fits,method_fits,polar,modes,interactions):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    cmap={"direct_scalar_int4":"tab:blue","structural_scalar_int3_p98_k8":"tab:orange","structural_int4_k8":"tab:green","structural_vq64_int3_k8":"tab:red"}
    plt.figure(figsize=(7,4))
    for method in METHODS:
        x=[float(r["t"]) for r in corr if r["predictor"]=="s_perp"]
        y=[float(r["spearman"]) for r in corr if r["predictor"]=="s_perp"]
        plt.plot(x,y,marker="o",label=LABELS[method])
    # The pooled correlation is the same for each method label; draw it once.
    plt.clf();x=sorted({float(r["t"]) for r in corr if r["predictor"]=="s_perp"});y=[float(next(r["spearman"] for r in corr if r["predictor"]=="s_perp" and float(r["t"])==t)) for t in x]
    plt.plot(x,y,marker="o");plt.xlabel("perturbation scale t");plt.ylabel("Spearman(s_perp, finite cosine distortion)");plt.tight_layout();plt.savefig(out/"spearman_vs_t.png",dpi=150);plt.close()
    plt.figure(figsize=(7,4));
    for bucket in ("all","near_tie_lt_0.1","medium_0.1_to_0.5","large_ge_0.5"):
        rr=[r for r in ordering if r["separation_bucket"]==bucket];plt.plot([float(r["t"]) for r in rr],[float(r["ordering_accuracy"]) for r in rr],marker="o",label=bucket)
    plt.xlabel("t");plt.ylabel("pairwise ordering accuracy");plt.legend();plt.tight_layout();plt.savefig(out/"pairwise_ordering_vs_t.png",dpi=150);plt.close()
    plt.figure(figsize=(7,5));
    for method in METHODS:
        rr=[r for r in sweeps if r["method"]==method];step=max(1,len(rr)//3000);rr=rr[::step]
        plt.scatter([float(r["t_s_perp"]) for r in rr],[float(r["finite_cosine_distortion"]) for r in rr],s=5,alpha=.25,label=LABELS[method],color=cmap[method])
    plt.xlabel("t × s_perp");plt.ylabel("actual cosine distortion");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"distortion_vs_t_sperp.png",dpi=150);plt.close()
    plt.figure(figsize=(7,5));
    for method in METHODS:
        rr=[r for r in sweeps if r["method"]==method];step=max(1,len(rr)//3000);rr=rr[::step]
        plt.scatter([(float(r["t_s_perp"])**2) for r in rr],[float(r["finite_cosine_distortion"]) for r in rr],s=5,alpha=.25,label=LABELS[method],color=cmap[method])
    plt.xlabel("(t × s_perp)^2");plt.ylabel("actual cosine distortion");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"distortion_vs_squared_sensitivity.png",dpi=150);plt.close()
    plt.figure(figsize=(7,5));
    for method in METHODS:
        rr=[r for r in sweeps if r["method"]==method and float(r["t"])==1.0];step=max(1,len(rr)//2000);rr=rr[::step]
        plt.scatter([float(r["first_order_cosine_prediction"]) for r in rr],[float(r["finite_cosine_distortion"]) for r in rr],s=6,alpha=.3,label=LABELS[method])
    plt.xlabel("first-order cosine prediction");plt.ylabel("actual finite distortion (t=1)");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"linear_prediction_vs_finite.png",dpi=150);plt.close()
    plt.figure(figsize=(7,4));
    for method in METHODS:
        rr=[r for r in method_fits if r["direction"]=="0_to_1" and r["method"]==method and r["model"]=="polar_inspired"]
        if rr:plt.bar(LABELS[method],float(rr[0]["heldout_r2"]),color=cmap[method])
    plt.xticks(rotation=25,ha="right");plt.ylabel("held-out R² (seed0→seed1)");plt.tight_layout();plt.savefig(out/"method_curve_transfer.png",dpi=150);plt.close()
    plt.figure(figsize=(7,4));
    for method in METHODS:
        rr=[r for r in sweeps if r["method"]==method];groups=defaultdict(list)
        for r in rr:groups[float(r["t"])].append(float(r["finite_cosine_distortion"]))
        ts=sorted(groups);plt.plot(ts,[statistics.mean(groups[t]) for t in ts],marker="o",label=LABELS[method],color=cmap[method])
    plt.xlabel("t");plt.ylabel("mean K=5 cosine distortion");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"method_response_curves.png",dpi=150);plt.close()
    plt.figure(figsize=(7,4));
    for method in METHODS:
        rr=[r for r in sweeps if r["method"]==method];groups=defaultdict(list)
        for r in rr:groups[float(r["t"])].append(float(r["finite_vector_mismatch"]))
        ts=sorted(groups);plt.plot(ts,[statistics.median(groups[t]) for t in ts],marker="o",label=LABELS[method],color=cmap[method])
    plt.xlabel("t");plt.ylabel("median first-order vector mismatch");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"vector_mismatch_vs_t.png",dpi=150);plt.close()
    if polar:
        plt.figure(figsize=(7,4));
        for method in METHODS:
            rr=[r for r in polar if r["method"]==method];groups=defaultdict(list)
            for r in rr:groups[float(r["t"])].append(float(r["exact_polar_distortion"]))
            ts=sorted(groups);plt.plot(ts,[statistics.mean(groups[t]) for t in ts],marker="o",label=LABELS[method]+" polar")
            kg=defaultdict(list)
            for r in sweeps:
                if r["method"]==method and (int(r["seed"]),int(r["update"]),r["parameter_id"]) in {(int(p["seed"]),int(p["update"]),p["parameter_id"]) for p in polar if p["method"]==method}:
                    kg[float(r["t"])].append(float(r["finite_cosine_distortion"]))
            if kg:plt.plot(sorted(kg),[statistics.mean(kg[t]) for t in sorted(kg)],linestyle="--",alpha=.65,label=LABELS[method]+" K=5")
        plt.xlabel("t");plt.ylabel("cosine distortion");plt.legend(fontsize=6,ncol=2);plt.tight_layout();plt.savefig(out/"polar_vs_k5_monotonicity.png",dpi=150);plt.close()
    if polar:
        plt.figure(figsize=(7,4))
        for predictor,style in (("s_perp","-o"),("s_full","--s")):
            rr=[r for r in read(out/"polar_correlation_vs_t.csv") if r["predictor"]==predictor]
            plt.plot([float(r["t"]) for r in rr],[float(r["spearman"]) for r in rr],style,label="exact polar "+predictor)
        rr=[r for r in corr if r["predictor"]=="s_perp"];plt.plot([float(r["t"]) for r in rr],[float(r["spearman"]) for r in rr],"-^",label="production K=5")
        plt.xlabel("t");plt.ylabel("Spearman sensitivity vs distortion");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"polar_vs_k5_spearman.png",dpi=150);plt.close()
    plt.figure(figsize=(7,4));
    for metric in ("remainder_radial_fraction","remainder_tangent_fraction","remainder_other_fraction"):
        groups=defaultdict(list)
        for r in sweeps:groups[float(r["t"])].append(float(r[metric]))
        ts=sorted(groups);plt.plot(ts,[statistics.mean(groups[t]) for t in ts],marker="o",label=metric)
    plt.xlabel("t");plt.ylabel("remainder norm fraction");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"higher_order_remainder_orientation.png",dpi=150);plt.close()
    if modes:
        plt.figure(figsize=(7,4));
        for rank in range(1,6):
            rr=[r for r in modes if int(r["pair_rank"])==rank];groups=defaultdict(list)
            for r in rr:groups[float(r["t"])].append(float(r["two_mode_k5_distortion"]))
            ts=sorted(groups);plt.plot(ts,[statistics.mean(groups[t]) for t in ts],marker="o",label=f"mode pair rank {rank}")
        plt.xlabel("t");plt.ylabel("isolated two-mode K=5 distortion");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/"isolated_mode_pair_response.png",dpi=150);plt.close()
    if interactions:
        plt.figure(figsize=(6,4));plt.scatter([float(r["sum_top5_isolated_distortions_t1"]) for r in interactions],[float(r["top5_combined_distortion_t1"]) for r in interactions],s=30);plt.xlabel("sum of isolated top-5 distortions");plt.ylabel("combined top-5 distortion");plt.tight_layout();plt.savefig(out/"full_vs_mode_pair_combination.png",dpi=150);plt.close()
    # Exact two-mode polar finite response and its linear-angle approximation.
    ctrl=[r for r in read(out/"two_mode_controls.csv") if r["family"]=="pure_skew"]
    if ctrl:
        plt.figure(figsize=(7,4));
        for pair in sorted({(float(r["sigma_i"]),float(r["sigma_j"]),float(r["e"])) for r in ctrl}):
            rr=[r for r in ctrl if (float(r["sigma_i"]),float(r["sigma_j"]),float(r["e"]))==pair]
            plt.plot([float(r["t"])*float(r["s"]) for r in rr],[float(r["distortion"]) for r in rr],marker="o",label=str(pair))
        plt.xlabel("t × exact polar sensitivity s");plt.ylabel("exact finite polar distortion");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(out/"two_mode_polar_finite_response.png",dpi=150);plt.close()
        xs=[float(r["t"])*float(r["s"]) for r in ctrl]
        plt.figure(figsize=(6,4));plt.scatter(xs,[float(r["angle"]) for r in ctrl],s=14,label="true angle = atan(ts)");plt.scatter(xs,[float(r["linear_angle"]) for r in ctrl],s=14,label="first-order angle = ts");plt.xlabel("t × s");plt.ylabel("rotation angle");plt.legend();plt.tight_layout();plt.savefig(out/"linear_vs_nonlinear_rotation_angle.png",dpi=150);plt.close()
    fit_csv=out/"response_curve_fits.csv"
    if fit_csv.exists():
        rr=[r for r in read(fit_csv) if r["direction"]=="0_to_1" and r["scope"]=="common" and r["model"]=="polar_inspired"]
        if rr:
            pars=np.asarray([float(x) for x in rr[0]["parameters"].split(";")]);test=[r for r in sweeps if int(r["seed"])==1]
            xx=np.asarray([float(r["t_s_perp"]) for r in test]);yy=np.asarray([float(r["finite_cosine_distortion"]) for r in test]);pred=model_value("polar_inspired",xx,pars)
            plt.figure(figsize=(6,5));plt.scatter(pred,yy,s=5,alpha=.2);lims=(min(0.,float(min(pred.min(),yy.min()))),max(float(pred.max()),float(yy.max())));plt.plot(lims,lims,color="black",lw=.8);plt.xlabel("seed0-calibrated prediction");plt.ylabel("seed1 actual distortion");plt.tight_layout();plt.savefig(out/"common_saturation_fit.png",dpi=150);plt.close()


def write_summary(out,tensors,sweeps,corr,ordering,fits,mfits,polar,modes,interactions,elapsed):
    def mean_at(method,t,key,seed=None):
        rr=[r for r in sweeps if r["method"]==method and float(r["t"])==t and (seed is None or int(r["seed"])==seed)]
        return statistics.mean(float(r[key]) for r in rr)
    cr={float(r["t"]):r for r in corr if r["predictor"]=="s_perp"}
    ordall={float(r["t"]):r for r in ordering if r["separation_bucket"]=="all"}
    seedx=next((r for r in fits if r["direction"]=="0_to_1" and r["model"]=="polar_inspired" and r["scope"]=="common"),None)
    seedrev=next((r for r in fits if r["direction"]=="1_to_0" and r["model"]=="polar_inspired" and r["scope"]=="common"),None)
    direction_rows=read(out/"direction_monotonicity.csv")
    monotone_rate=sum(str(r["monotone_non_decreasing"]).lower()=="true" for r in direction_rows)/max(len(direction_rows),1)
    mean_step_rate=statistics.mean(float(r["monotone_step_fraction"]) for r in direction_rows)
    lines=["# Fréchet sensitivity as a finite-error monotonic coordinate","",
        f"CPU-only sweep of {len(tensors)} tensor-method error directions and {len(sweeps)} finite perturbations; 4 canonical errors × 300 formal matrices, t={T_VALUES}. Runtime {elapsed:.1f}s. No quantizer, codebook, optimizer, or training behavior changed.","",
        "## Local result and exact two-mode model","",
        "The derivation in `local_cosine_derivation.md` gives the leading coefficient ½ s_perp²; acceleration/second derivative enters only at O(t³), while radial response does not affect the leading angle. For exact polar, the derivative is tangent and s_perp=s_full. The analytic 2×2 skew formula matched numerical polar factors to maximum absolute cosine-distortion error "+f"{max((float(r.get('formula_abs_error','0')) for r in read(out/'two_mode_controls.csv') if r['family']=='pure_skew'),default=float('nan')):.2e}. Its sensitivity ordering is strictly monotone for t>0, even though angle=atan(ts) saturates versus the linear angle ts. Diagonal and symmetric controls stay at zero polar distortion while positive definite; branch crossing is separately documented.","",
        "## Production K=5 monotonic ordering","","| t | Spearman s_perp | Pearson s_perp | pair ordering accuracy | near-tie | medium gap | large gap |",
        "|--:|--:|--:|--:|--:|--:|--:|"]
    for t in T_VALUES:
        oo={r["separation_bucket"]:r for r in ordering if float(r["t"])==t}
        lines.append(f"| {t:g} | {float(cr[t]['spearman']):.4f} | {float(cr[t]['pearson']):.4f} | {float(ordall[t]['ordering_accuracy']):.4f} | {float(oo['near_tie_lt_0.1']['ordering_accuracy']):.4f} | {float(oo['medium_0.1_to_0.5']['ordering_accuracy']):.4f} | {float(oo['large_ge_0.5']['ordering_accuracy']):.4f} |")
    lines += ["","At t=1, mean actual K=5 cosine distortion and median first-order finite-vector mismatch by method:","",
        "| method | mean cosine distortion | mean s_perp | median vector mismatch |","|:--|--:|--:|--:|"]
    for method in METHODS:
        rr=[r for r in sweeps if r["method"]==method and float(r["t"])==1.]
        tr=[r for r in tensors if r["method"]==method]
        lines.append(f"| {LABELS[method]} | {statistics.mean(float(r['finite_cosine_distortion']) for r in rr):.4f} | {statistics.mean(float(r['s_perp']) for r in tr):.4g} | {statistics.median(float(r['finite_vector_mismatch']) for r in rr):.4f} |")
    lines += ["","## Saturation response transfer","",
        f"The polar-inspired common response curve calibrated on seed 0 and tested on seed 1: held-out R² {float(seedx['heldout_r2']):.4f}, Spearman {float(seedx['heldout_spearman']):.4f}; reverse split R² {float(seedrev['heldout_r2']):.4f}, Spearman {float(seedrev['heldout_spearman']):.4f}. All three monotonic model families and method-specific fits are in `response_curve_fits.csv` and `method_specific_curves.csv`. Fit quality is descriptive; repeated tensors/landmarks are not independent samples.","",
        "## Higher-order behavior and polar subset","",
        f"At t=1, pooled mean remainder norm fractions are radial {statistics.mean(float(r['remainder_radial_fraction']) for r in sweeps if float(r['t'])==1.):.3f}, along the tangent first-order direction {statistics.mean(float(r['remainder_tangent_fraction']) for r in sweeps if float(r['t'])==1.):.3f}, and remaining orthogonal {statistics.mean(float(r['remainder_other_fraction']) for r in sweeps if float(r['t'])==1.):.3f}. Fractions are projections onto an orthogonalized basis (O0, v_perp), not overlapping projections onto O0 and raw v. Exact polar subset: {len(polar)} rows across six deterministic matrices and four methods; see its own Spearman/ordering and curves in `polar_vs_k5.csv`.","",
        f"The real-tensor isolated-mode experiment contains {len(modes)} t/pair rows, and the selected-pair interaction table contains {len(interactions)} tensors. These are mechanism diagnostics; mode pairs are not independent additive causes. See `two_mode_real_tensor.csv` and `cross_mode_interactions.csv`.","",
        f"The exact 2×2 local expansion coefficient was numerically checked down to t=0.001; maximum reported relative coefficient error was {max(float(r['relative_coefficient_error']) for r in read(out/'local_expansion_validation.csv')):.3e}. Across the 1,200 real tensor-method directions, {monotone_rate:.1%} were nondecreasing at every tested t step and the mean fraction of nondecreasing steps was {mean_step_rate:.4f}. Per-direction results are in `direction_monotonicity.csv`; this is separate from pooled cross-direction ordering.","",
        "## Interpretation","",
        "The local theorem is unconditional only as a sufficiently small-t expansion at a twice-differentiable point. Finite monotonic-coordinate support must be judged from the measured t-sweep and held-out saturation fits above. Strong rank ordering can coexist with poor first-order vector prediction because curvature changes the displacement while preserving its severity order. Conversely, loss of pairwise accuracy away from ties, weak cross-seed fit, or strong method-specific curves would bound that interpretation. No global monotonic theorem is claimed.","",
        f"Post-sweep aggregation runtime in the final checkpoint-resume process: {elapsed:.1f} CPU seconds; the primary finite sweep was completed in two checkpointed CPU runs and its end-to-end wall time was not retained. Full output coverage: {len(tensors)} tensor-method sensitivities, {len(sweeps)} K=5 finite-response rows; exact-polar subset {len(polar)} rows."]
    (out/"summary.md").write_text("\n".join(lines)+"\n")


def write_methodology(out):
    (out/"methodology.md").write_text("""# Methodology

This CPU-only diagnostic reuses exactly the four canonical reconstruction families from `reports/muon_frechet_sensitivity`: direct scalar INT4, structural scalar INT3 with rank 8 and p98 scales, structural INT4 with rank 8, and frozen 64-word 2D INT3 VQ with rank-8 BF16 factors. It covers both formal seeds, all five landmarks, and all eligible 2D Muon matrices (300 matrices; 1,200 error directions). No quantizer, codebook, scale, structural rank, optimizer, or training path is changed.

For each error E, the existing production-map Fréchet derivative is evaluated once at M. The output is then recomputed on M+tE for t in {0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.25, 1.5}. The primary coordinate is the orthogonal derivative norm `s_perp = ||P_O0^perp D Phi[M][E]|| / ||O0||`; the full derivative norm, prior tail angle, state error, and Gram proxy are controls. Spearman/Pearson, deterministic capped pair-order comparisons, direction-wise monotonicity, and simple positive-parameter saturation curves are reported.

The exact 2x2 polar family is evaluated in float64 and includes pure skew, diagonal, symmetric-positive-definite, and explicitly marked sign/zero-eigenvalue crossing controls. Real-tensor isolated mode pairs use the existing FP32 SVD basis and are mechanism diagnostics only. The higher-order remainder is decomposed into the O0 radial direction, the orthogonal first-derivative direction, and the remaining orthogonal component; these are mutually orthogonal components.

Calibration/transfer fits use seed 0 -> seed 1 and the reverse split. Curve fitting uses deterministic NumPy grid search (no external optimizer dependency), not update-fidelity tuning of any quantizer. Repeated snapshots share parameter identities, so held-out split scores are descriptive transfer checks rather than independent-sample inference. No global monotonic theorem is claimed; all finite-error conclusions are restricted to the tested perturbation families and t range, excluding nonsmooth polar branch crossings.
""")


if __name__=="__main__":main()
