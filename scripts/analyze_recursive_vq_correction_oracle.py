#!/usr/bin/env python3
"""Offline oracle study for low-rank correction of recursive VQ momentum.

This script processes one seed/landmark at a time.  ``M_ref`` is the matching
FP32 momentum snapshot and ``M_vq`` is the decoded recursive checkpoint state;
their difference is a reference-aligned trajectory error, not an instantaneous
quantization error.
"""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optim.muon_recursive import StructuralVQCodec, StructuralINT4Codec, StructuralVQState, StructuralINT4State
from src.optim.muon_reference import zeropower_newton_schulz

LANDMARKS=(128,512,1024,2048,4096)
RANKS=(0,1,2,4,8)
BASE_BITS=3.48828125  # overwritten from actual storage below

def _snapshot(path):
    x=torch.load(path,map_location='cpu',weights_only=False)
    # Formal snapshots identify tensors by parameter name (not numeric
    # optimizer id); their serialized order is the canonical Muon ordering.
    return [t['tensor'].float() for t in x['tensors']], x.get('metadata',{})

def _state_from_entry(e):
    if e.get('_recursive_state_type')=='vq':
        return StructuralVQState.from_state_dict({k:v for k,v in e.items() if k!='_recursive_state_type'})
    return StructuralINT4State(e['u'],e['singular_values'],e['vh'],e['scales'],e['codes'],int(e['count']),tuple(e['shape']))

def _decoded_checkpoint(path, kind, codebook):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    states=[]
    for pid in range(30):
        entry=ck['optimizer']['state'][str(pid)] if str(pid) in ck['optimizer']['state'] else ck['optimizer']['state'][pid]
        enc=entry['compressed_momentum']
        st=_state_from_entry(enc)
        codec=StructuralVQCodec(codebook,rank=8,block_size=2048) if kind=='vq' else StructuralINT4Codec(rank=8,block_size=2048)
        states.append(codec.decode(st,device=torch.device('cpu')).float())
    return states

def _k5(x):
    return zeropower_newton_schulz(x,5,(3.4445,-4.7750,2.0315),1e-7).float()

def _metrics(a,b):
    d=(a-b).float(); na=a.norm(); nb=b.norm()
    return float(torch.dot(a.flatten(),b.flatten())/(na*nb).clamp_min(1e-30)), float(d.norm()/na.clamp_min(1e-30))

def _metric_sums(a,b):
    d=(a-b).float(); aa=a.float(); bb=b.float()
    return {'dot':float(torch.dot(aa.flatten(),bb.flatten())), 'a2':float(aa.square().sum()), 'b2':float(bb.square().sum()), 'd2':float(d.square().sum()), **{}}

def _finish_sums(x):
    return (x['dot']/max(math.sqrt(x['a2']*x['b2']),1e-30), math.sqrt(x['d2']/max(x['a2'],1e-30)))

def _corr_from_svd(vq,u,s,vh,r):
    if r==0: return vq
    rr=min(r,s.numel())
    # BF16 factor payload, matching the proposed correction state.
    A=(u[:,:rr]*s[:rr].sqrt()).to(torch.bfloat16)
    B=(s[:rr].sqrt()[:,None]*vh[:rr]).to(torch.bfloat16)
    return vq + A.float()@B.float()

def _svd_stats(e, singular_values=None):
    s=singular_values if singular_values is not None else torch.linalg.svdvals(e); en=float(s.square().sum())
    return {f'energy_r{r}':float(s[:min(r,s.numel())].square().sum()/max(en,1e-30)) for r in (1,2,4,8)}

def _state_count(refs): return sum(x.numel() for x in refs)

def _write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def _plots(out, agg, budget_rows, int4_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for metric, ylabel, name in [('momentum_cosine_mean','momentum cosine','momentum_cosine'),('momentum_rel_l2_mean','momentum relative-L2','momentum_rel_l2'),('k5_cosine_mean','K5-map cosine','k5_cosine')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for seed in sorted({r['seed'] for r in agg}):
            for rank in sorted({r['rank'] for r in agg}):
                x=[r['update'] for r in agg if r['seed']==seed and r['rank']==rank]; y=[r[metric] for r in agg if r['seed']==seed and r['rank']==rank]
                if x: ax.plot(x,y,marker='o',label=f's{seed} r{rank}')
        ax.set_xlabel('update'); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(out/f'{name}_fixed_rank.png',dpi=140); plt.close(fig)
    if budget_rows:
        fig,ax=plt.subplots(figsize=(7,4));
        for seed in sorted({r['seed'] for r in budget_rows}):
            x=[r['total_bits_per_value'] for r in budget_rows if r['seed']==seed and r['update']==4096]; y=[r['k5_cosine_mean'] for r in budget_rows if r['seed']==seed and r['update']==4096]; ax.plot(x,y,'o-',label=f'VQ oracle s{seed}')
        if int4_rows:
            z=[r for r in int4_rows if r['update']==4096]; x=[r['effective_bits_per_value'] for r in z]; y=[r['k5_cosine_mean'] for r in z]; ax.scatter(x,y,marker='x',label='structural INT4')
        ax.set_xlabel('total effective bits/value'); ax.set_ylabel('K5-map cosine'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/'storage_fidelity_pareto.png',dpi=140); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo-root',default='.'); ap.add_argument('--out',default='reports/recursive_vq_correction_oracle'); ap.add_argument('--seed',type=int,default=None); ap.add_argument('--no-plots',action='store_true'); args=ap.parse_args()
    root=Path(args.repo_root).resolve(); out=root/args.out
    ckroot=root/'artifacts/recursive_muon_s1_checkpoints'; snaproot=root/'reports/muon_update_fidelity_formal_s0_s1_results/muon-fidelity-formal'
    cblob=torch.load(root/'reports/muon_vector_int3_robustness/calibration_codebooks.pt',map_location='cpu',weights_only=False)
    # The seed-1 recursive run used the seed-0-calibrated codebook, while the
    # seed-0 run used the seed-1-calibrated codebook.  Select per-seed below.
    seeds=[args.seed] if args.seed is not None else [1,0]
    rows=[]; spectra=[]; budget_rows=[]; int4_rows=[]; provenance={'requested_seeds':seeds,'available':[],'missing':[],'interpretation':'E=M_ref-M_vq is reference-aligned trajectory error; not instantaneous pre-quantization error.'}
    for seed in seeds:
        croot=root/f'artifacts/recursive_muon_s{seed}_checkpoints'; sroot=snaproot/f'fp32_muon_snapshot_s{seed}'/'muon_momentum_snapshots'
        if not croot.exists(): provenance['missing'].append({'seed':seed,'reason':'recursive checkpoints absent'}); continue
        for update in LANDMARKS:
            cp=croot/f'vq_int3_update_{update:06d}.pt'; sp=sroot/f'update_{update:06d}.pt'
            if not cp.exists() or not sp.exists(): provenance['missing'].append({'seed':seed,'update':update}); continue
            refs,_=_snapshot(sp)
            key='s0_k8_w64_t8_v1200' if seed==1 else 's1_k8_w64_t8_v1200'
            codebook=cblob['codebooks'][key].float()
            vqs=_decoded_checkpoint(cp,'vq',codebook)
            n=_state_count(refs); base_bits=0
            # exact payload accounting from the decoded checkpoint states
            raw=torch.load(cp,map_location='cpu',weights_only=False)
            for pid in range(30):
                entry=raw['optimizer']['state'][str(pid)] if str(pid) in raw['optimizer']['state'] else raw['optimizer']['state'][pid]
                st=_state_from_entry(entry['compressed_momentum']); base_bits += StructuralVQCodec(codebook,rank=8,block_size=2048).storage_bits(st)
            base_bits += codebook.numel()*32; base_bpv=base_bits/n
            provenance['available'].append({'seed':seed,'update':update,'scalar_count':n,'base_vq_bits_per_value':base_bpv})
            tensor_cache=[]
            for pid,vq in enumerate(vqs):
                ref=refs[pid]
                e=ref-vq; U,S,Vh=torch.linalg.svd(e,full_matrices=False); sm=_svd_stats(e,S)
                tensor_cache.append((pid,ref,vq,U,S,Vh,sm))
                for r in RANKS:
                    corr=_corr_from_svd(vq,U,S,Vh,r); c,rl=_metrics(corr,ref); kc,krl=_metrics(_k5(corr),_k5(ref))
                    # correction payload is exact BF16 factor cost, not FP32 oracle E
                    if r: cost=2*r*(ref.shape[0]+ref.shape[1])
                    else: cost=0
                    ms=_metric_sums(corr,ref); ks=_metric_sums(_k5(corr),_k5(ref))
                    row={'seed':seed,'update':update,'parameter_id':pid,'shape':str(tuple(ref.shape)),'rank':r,'correction_bytes':cost,'correction_bits_per_value':cost*8/n,'total_bits_per_value':base_bpv+cost*8/n,'momentum_cosine':c,'momentum_rel_l2':rl,'k5_cosine':kc,'k5_rel_l2':krl,'m_dot':ms['dot'],'m_a2':ms['a2'],'m_b2':ms['b2'],'m_d2':ms['d2'],'k_dot':ks['dot'],'k_a2':ks['a2'],'k_b2':ks['b2'],'k_d2':ks['d2'],**sm}
                    rows.append(row)
                spectra.append({'seed':seed,'update':update,'parameter_id':pid,'shape':str(tuple(ref.shape)),'error_norm':float(e.norm()),**sm})
            # Greedy MSE-gain/byte allocation. Each singular component gives
            # exactly s_i^2 Frobenius-error reduction and costs BF16 A/B bytes.
            for target in (0.0,0.10,0.25,0.50,1.0):
                budget=int(math.floor(target*n/8.0)); choices=[]
                for pid,ref,vq,U,S,Vh,sm in tensor_cache:
                    cost=2*(ref.shape[0]+ref.shape[1])
                    for j,sv in enumerate(S): choices.append((float(sv*sv)/cost,float(sv*sv),pid,j,cost))
                choices.sort(reverse=True); used=0
                selected={pid:0 for pid,ref,vq,U,S,Vh,sm in tensor_cache}; gain=0
                for ratio,g,pid,j,cost in choices:
                    if used+cost>budget: continue
                    # rank components must be prefix ranks for a valid SVD
                    if j!=selected[pid]: continue
                    selected[pid]+=1; used+=cost; gain+=g
                msum={'dot':0.,'a2':0.,'b2':0.,'d2':0.}; ksum={'dot':0.,'a2':0.,'b2':0.,'d2':0.}
                for pid,ref,vq,U,S,Vh,sm in tensor_cache:
                    rr=selected[pid]; corr=_corr_from_svd(vq,U,S,Vh,rr)
                    for key,val in _metric_sums(corr,ref).items(): msum[key]+=val
                    for key,val in _metric_sums(_k5(corr),_k5(ref)).items(): ksum[key]+=val
                mc,ml=_finish_sums(msum); kc,kl=_finish_sums(ksum)
                budget_rows.append({'seed':seed,'update':update,'target_additional_bits_per_value':target,'budget_bytes':budget,'used_bytes':used,'actual_additional_bits_per_value':used*8/n,'total_bits_per_value':base_bpv+used*8/n,'mse_gain':gain,'momentum_cosine_mean':mc,'momentum_rel_l2_mean':ml,'k5_cosine_mean':kc,'k5_rel_l2_mean':kl})
    # Gap closure is defined relative to the rank-0 VQ row for the same
    # tensor/landmark (FP32 is cosine 1 and relative-L2 0).
    baselines={(r['seed'],r['update'],r['parameter_id']):r for r in rows if r['rank']==0}
    for r in rows:
        b=baselines[(r['seed'],r['update'],r['parameter_id'])]
        r['momentum_cosine_closure']=(r['momentum_cosine']-b['momentum_cosine'])/(1-b['momentum_cosine']) if b['momentum_cosine']<1 else 0.0
        r['k5_cosine_closure']=(r['k5_cosine']-b['k5_cosine'])/(1-b['k5_cosine']) if b['k5_cosine']<1 else 0.0
        r['momentum_rel_l2_closure']=(b['momentum_rel_l2']-r['momentum_rel_l2'])/b['momentum_rel_l2'] if b['momentum_rel_l2'] else 0.0
    out.mkdir(parents=True,exist_ok=True); _write_csv(out/'per_landmark_metrics.csv',rows); _write_csv(out/'error_spectrum.csv',spectra); _write_csv(out/'budget_pareto.csv',budget_rows)
    # Aggregate fixed-rank curves.
    agg=[]
    for key in sorted({(r['seed'],r['update'],r['rank']) for r in rows}):
        ss,uu,rr=key; x=[r for r in rows if (r['seed'],r['update'],r['rank'])==key];
        ms={k:sum(r['m_'+k] for r in x) for k in ('dot','a2','b2','d2')}; ks={k:sum(r['k_'+k] for r in x) for k in ('dot','a2','b2','d2')}; mc,ml=_finish_sums(ms); kc,kl=_finish_sums(ks)
        agg.append({'seed':ss,'update':uu,'rank':rr,'correction_bits_per_value':sum(r['correction_bits_per_value'] for r in x),'total_bits_per_value':sum(r['total_bits_per_value'] for r in x)/len(x),'momentum_cosine_mean':mc,'momentum_rel_l2_mean':ml,'k5_cosine_mean':kc,'k5_rel_l2_mean':kl})
    _write_csv(out/'fixed_rank_summary.csv',agg)
    # A compact comparison table for the uncorrected recursive INT4 baseline.
    for seed in seeds:
        croot=root/f'artifacts/recursive_muon_s{seed}_checkpoints'; sroot=snaproot/f'fp32_muon_snapshot_s{seed}'/'muon_momentum_snapshots'
        if not croot.exists(): continue
        for update in LANDMARKS:
            cp=croot/f'int4_update_{update:06d}.pt'; sp=sroot/f'update_{update:06d}.pt'
            if not cp.exists() or not sp.exists(): continue
            refs,_=_snapshot(sp); ck=torch.load(cp,map_location='cpu',weights_only=False); vals=[]; int4_bits=0
            for pid in range(30):
                entry=ck['optimizer']['state'][str(pid)] if str(pid) in ck['optimizer']['state'] else ck['optimizer']['state'][pid]
                st=_state_from_entry(entry['compressed_momentum']); codec4=StructuralINT4Codec(rank=8,block_size=2048); int4_bits += codec4.storage_bits(st); v=codec4.decode(st)
                vals.append((_metrics(v,refs[pid]),_metrics(_k5(v),_k5(refs[pid]))))
            int4_rows.append({'seed':seed,'update':update,'momentum_cosine_mean':sum(x[0][0] for x in vals)/len(vals),'momentum_rel_l2_mean':sum(x[0][1] for x in vals)/len(vals),'k5_cosine_mean':sum(x[1][0] for x in vals)/len(vals),'k5_rel_l2_mean':sum(x[1][1] for x in vals)/len(vals),'effective_bits_per_value':int4_bits*1.0/_state_count(refs)})
    _write_csv(out/'int4_comparison.csv',int4_rows)
    _plots(out,agg,budget_rows,int4_rows)
    json.dump(provenance,(out/'provenance.json').open('w'),indent=2)
    summary={'status':'completed','seeds_available':sorted({x['seed'] for x in provenance['available']}),'landmarks':LANDMARKS,'rows':len(rows),'base_method':'recursive structural VQ, rank 8, p98, shared 64-word codebook','limitations':['reference-aligned E, not instantaneous quantization error','K5 is momentum-only spectral-map comparison; saved gradients/pre-Nesterov candidates are unavailable','oracle correction is offline and not evidence of online error feedback'],'outputs':['per_landmark_metrics.csv','fixed_rank_summary.csv','budget_pareto.csv','error_spectrum.csv','int4_comparison.csv']}
    json.dump(summary,(out/'summary.json').open('w'),indent=2)
    (out/'comparison.md').write_text('# Recursive VQ correction oracle\n\nThis report uses one checkpoint and one FP32 momentum snapshot at a time. `E=M_ref-M_vq` is reference-aligned trajectory error, not instantaneous quantization error. K5 metrics apply the canonical 5-step Newton–Schulz spectral map to momentum only; they are not saved-gradient Nesterov update fidelity. See CSV files for fixed-rank metrics and storage.\n\n## Scope\n\nThe local dataset contains seed-1 recursive checkpoints; seed 0 is included in the script and reported as missing when unavailable. The oracle BF16 factor payload is accounted at `2*r*(m+n)` bytes per matrix. No checkpoint artifacts are written by this analysis.\n\n## Interpretation\n\nThe fixed-rank rows measure an offline upper bound. The budget rows greedily allocate SVD components by Frobenius-error gain per byte; they are not an online error-feedback result. A corrected point below structural INT4 storage is interesting only if its K5 metric also exceeds the INT4 comparison row at the same landmark. Lack of a saved pre-quantization candidate means instantaneous quantization error, causal recursive correction, validation NLL, and Nesterov update fidelity are not identifiable from these artifacts.\n')
    (out/'methodology.md').write_text('''# Methodology\n\nFor each available seed and landmark, the script loads exactly one FP32 momentum snapshot and one recursive checkpoint, decodes the 30 Muon states, computes `E=M_ref-M_vq`, and releases them before moving to the next landmark. The correction is an offline oracle: an exact SVD of E followed by BF16 factor storage `A=U sqrt(S)`, `B=sqrt(S) V^T`. Fixed ranks are evaluated independently. Budget points greedily select prefix singular components by squared singular-value gain per BF16-factor byte.\n\nThe K5 statistic is the canonical five-step Newton--Schulz spectral map applied to the momentum tensor alone. The training checkpoint did not save the contemporaneous gradient or pre-quantization candidate, so this is not Nesterov update fidelity and cannot identify instantaneous quantization error or causal online error feedback.\n\nStorage includes the decoded VQ payload and shared FP32 64x2 codebook. Correction bytes are `2*r*(m+n)` per matrix; checkpoint files are read-only inputs and are never copied to the report.\n''')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
