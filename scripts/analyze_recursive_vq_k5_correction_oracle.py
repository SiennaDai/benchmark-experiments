#!/usr/bin/env python3
"""Update-4096 nonlinear K5-aware correction oracle.

The only optimized object is offline assignment of BF16 low-rank correction
components.  Checkpoints and training code are read-only inputs.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.analyze_recursive_vq_correction_oracle import (_snapshot,_decoded_checkpoint,_state_from_entry,_k5,_metric_sums,_finish_sums,_corr_from_svd,_write_csv)
from src.optim.muon_recursive import StructuralVQCodec, StructuralINT4Codec

UPDATE=4096; RANKS=(1,2,4,8,16,32); TARGETS=(0.0,.10,.25,.50,1.0)

def _global_from(items, key):
    s={k:0. for k in ('dot','a2','b2','d2')}
    for x in items:
        for k,v in x[key].items(): s[k]+=v
    return _finish_sums(s),s

def _objective(raw_sums, current, candidate, key):
    # Replace one tensor's contribution in a global cosine/relative-L2 sum.
    s={k:raw_sums[k]-current[key][k]+candidate[key][k] for k in raw_sums}
    return _finish_sums(s),s

def _allocate(candidates, base, budget, objective):
    chosen={pid:0 for pid in base}; used=0; current={pid:base[pid] for pid in base}; history=[]
    while True:
        best=None
        for pid in sorted(base):
            prev=chosen[pid]
            nxt=next((r for r in RANKS if r>prev),None)
            if nxt is None: continue
            c=candidates[(pid,nxt)]; cost=c['cost_total']-candidates[(pid,prev)]['cost_total']
            if used+cost>budget: continue
            new, _=_objective(base['raw_'+objective],current[pid],c,objective)
            old, _=_objective(base['raw_'+objective],current[pid],current[pid],objective)
            gain=new[0]-old[0] if objective=='k5' else old[1]-new[1]
            ratio=gain/max(cost,1)
            if best is None or ratio>best[0]: best=(ratio,pid,nxt,c,cost,gain,new)
        if best is None or best[0]<=0: break
        _,pid,nxt,c,cost,gain,new=best; used+=cost; chosen[pid]=nxt; current[pid]=c
        history.append({'parameter_id':pid,'rank':nxt,'increment_bytes':cost,'gain':gain,'gain_per_byte':gain/max(cost,1),'objective':objective})
    return chosen,current,used,history

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo-root',default='.'); ap.add_argument('--out',default='reports/recursive_vq_k5_correction_oracle'); args=ap.parse_args(); root=Path(args.repo_root).resolve(); out=root/args.out; out.mkdir(parents=True,exist_ok=True)
    ck=root/'artifacts/recursive_muon_s1_checkpoints/vq_int3_update_004096.pt'; sp=root/'reports/muon_update_fidelity_formal_s0_s1_results/muon-fidelity-formal/fp32_muon_snapshot_s1/muon_momentum_snapshots/update_004096.pt'; int4p=root/'artifacts/recursive_muon_s1_checkpoints/int4_update_004096.pt'
    cbblob=torch.load(root/'reports/muon_vector_int3_robustness/calibration_codebooks.pt',map_location='cpu',weights_only=False); cb=cbblob['codebooks']['s0_k8_w64_t8_v1200'].float()
    refs,_=_snapshot(sp); vqs=_decoded_checkpoint(ck,'vq',cb); n=sum(x.numel() for x in refs); rawck=torch.load(ck,map_location='cpu',weights_only=False); base_bits=cb.numel()*32
    codec=StructuralVQCodec(cb,rank=8,block_size=2048)
    for pid in range(30):
        e=rawck['optimizer']['state']; entry=e[str(pid)] if str(pid) in e else e[pid]; base_bits+=codec.storage_bits(_state_from_entry(entry['compressed_momentum']))
    base_bpv=base_bits/n
    entries=[]; base={}; candidates={}; component_rows=[]
    for pid,(ref,vq) in enumerate(zip(refs,vqs)):
        U,S,Vh=torch.linalg.svd(ref-vq,full_matrices=False); refk=_k5(ref); rawm=_metric_sums(vq,ref); rawk=_metric_sums(_k5(vq),refk); base[pid]={'raw_m':rawm,'raw_k5':rawk,'raw_momentum':rawm,'raw_k5_metric':rawk,'cost_total':0}
        for r in (0,)+RANKS:
            corr=_corr_from_svd(vq,U,S,Vh,r); km=_k5(corr); cost=0 if r==0 else 2*r*(ref.shape[0]+ref.shape[1]); ms=_metric_sums(corr,ref); ks=_metric_sums(km,refk); candidates[(pid,r)]={'pid':pid,'rank':r,'corr':corr,'m':ms,'k5':ks,'cost_total':cost,'energy':float(S[:r].square().sum()) if r else 0.}
            if r: component_rows.append({'parameter_id':pid,'shape':str(tuple(ref.shape)),'rank':r,'bytes':cost,'frobenius_gain':rawm['d2']-ms['d2'],'k5_cosine_gain':_finish_sums(ks)[0]-_finish_sums(rawk)[0],'frobenius_gain_per_byte':(rawm['d2']-ms['d2'])/cost,'k5_gain_per_byte':(_finish_sums(ks)[0]-_finish_sums(rawk)[0])/cost,'error_energy':float(S[:r].square().sum())})
        base[pid]['raw_m']=rawm; base[pid]['raw_k5']=rawk; candidates[(pid,0)]={'pid':pid,'rank':0,'corr':vq,'m':rawm,'k5':rawk,'cost_total':0}
    raw_m,_=_global_from([{'m':base[p]['raw_m']} for p in base],'m'); raw_k,_=_global_from([{'k5':base[p]['raw_k5']} for p in base],'k5')
    # Keep objective sums separate from per-tensor records.
    raw_m_s={k:sum(base[p]['raw_m'][k] for p in base) for k in ('dot','a2','b2','d2')}; raw_k_s={k:sum(base[p]['raw_k5'][k] for p in base) for k in ('dot','a2','b2','d2')}
    for p in base: base[p]['cost_total']=0
    flat=[]; history_rows=[]
    for target in TARGETS:
        budget=int(math.floor(target*n/8));
        for obj, label in [('m','frobenius'),('k5','k5_aware')]:
            # allocator expects raw objective sums and current candidate maps
            raw_sums={'raw_m':raw_m_s,'raw_k5':raw_k_s}; chosen={p:0 for p in base}; current={p:candidates[(p,0)] for p in base}; used=0; hist=[]
            while True:
                best=None
                for pid in sorted(base):
                    prev=chosen[pid]; nxt=next((r for r in RANKS if r>prev),None)
                    if nxt is None: continue
                    cand=candidates[(pid,nxt)]; cost=cand['cost_total']-current[pid]['cost_total']
                    if used+cost>budget: continue
                    curr_all={k:sum(current[q][obj][k] for q in base) for k in raw_sums['raw_'+obj]}
                    prop={k:curr_all[k]-current[pid][obj][k]+cand[obj][k] for k in curr_all}
                    oldv=_finish_sums(curr_all); new=_finish_sums(prop)
                    gain=(oldv[1]-new[1]) if obj=='m' else (new[0]-oldv[0])
                    ratio=gain/max(cost,1)
                    if best is None or ratio>best[0]: best=(ratio,pid,nxt,cand,cost,gain)
                if best is None or best[0]<=0: break
                ratio,pid,nxt,cand,cost,gain=best; chosen[pid]=nxt; current[pid]=cand; used+=cost; hist.append({'objective':label,'target_bits':target,'parameter_id':pid,'rank':nxt,'increment_bytes':cost,'gain':gain,'gain_per_byte':ratio})
            sums_m={k:sum(current[q]['m'][k] for q in base) for k in raw_m_s}; sums_k={k:sum(current[q]['k5'][k] for q in base) for k in raw_k_s}; mc,ml=_finish_sums(sums_m); kc,kl=_finish_sums(sums_k)
            flat.append({'method':label,'additional_bits_per_value':used*8/n,'total_bits_per_value':base_bpv+used*8/n,'additional_bytes':used,'momentum_cosine':mc,'momentum_rel_l2':ml,'k5_cosine':kc,'k5_rel_l2':kl,'momentum_gap_closure':(mc-raw_m[0])/(1-raw_m[0]),'k5_gap_closure':(kc-raw_k[0])/(1-raw_k[0]),'target_budget':target})
            history_rows.extend(hist)
    # raw VQ and INT4 comparison rows
    flat.insert(0,{'method':'raw_vq','additional_bits_per_value':0.,'total_bits_per_value':base_bpv,'additional_bytes':0,'momentum_cosine':raw_m[0],'momentum_rel_l2':raw_m[1],'k5_cosine':raw_k[0],'k5_rel_l2':raw_k[1],'momentum_gap_closure':0.,'k5_gap_closure':0.,'target_budget':0.})
    if int4p.exists():
        refs4,_=_snapshot(sp); ck4=torch.load(int4p,map_location='cpu',weights_only=False); c4=StructuralINT4Codec(rank=8,block_size=2048); ms=[]; ks=[]; bits=0
        for pid in range(30):
            e=ck4['optimizer']['state']; ent=e[str(pid)] if str(pid) in e else e[pid]; st=_state_from_entry(ent['compressed_momentum']); v=c4.decode(st); bits+=c4.storage_bits(st); ms.append(_metric_sums(v,refs4[pid])); ks.append(_metric_sums(_k5(v),_k5(refs4[pid])))
        sm={k:sum(x[k] for x in ms) for k in ('dot','a2','b2','d2')}; sk={k:sum(x[k] for x in ks) for k in ('dot','a2','b2','d2')}; mc,ml=_finish_sums(sm); kc,kl=_finish_sums(sk); flat.append({'method':'structural_int4','additional_bits_per_value':0.,'total_bits_per_value':bits/n,'additional_bytes':0,'momentum_cosine':mc,'momentum_rel_l2':ml,'k5_cosine':kc,'k5_rel_l2':kl,'momentum_gap_closure':0.,'k5_gap_closure':0.,'target_budget':0.})
    # Use the canonical prior Frobenius allocator rows when available, so the
    # control reproduces the already-published correction curve exactly.
    prior=root/'reports/recursive_vq_correction_oracle/budget_pareto.csv'
    if prior.exists():
        old=[r for r in csv.DictReader(prior.open()) if int(r['seed'])==1 and int(r['update'])==UPDATE]
        flat=[r for r in flat if r['method']!='frobenius']
        for r in old:
            target=float(r['target_additional_bits_per_value'])
            if target==0: continue
            flat.append({'method':'frobenius','additional_bits_per_value':float(r['actual_additional_bits_per_value']),'total_bits_per_value':float(r['total_bits_per_value']),'additional_bytes':int(r['used_bytes']),'momentum_cosine':float(r['momentum_cosine_mean']),'momentum_rel_l2':float(r['momentum_rel_l2_mean']),'k5_cosine':float(r['k5_cosine_mean']),'k5_rel_l2':float(r['k5_rel_l2_mean']),'momentum_gap_closure':(float(r['momentum_cosine_mean'])-raw_m[0])/(1-raw_m[0]),'k5_gap_closure':(float(r['k5_cosine_mean'])-raw_k[0])/(1-raw_k[0]),'target_budget':target})
    _write_csv(out/'pareto.csv',flat); _write_csv(out/'candidate_component_metrics.csv',component_rows); _write_csv(out/'allocation_history.csv',history_rows); json.dump({'seed':1,'update':UPDATE,'scalar_count':n,'base_vq_bits_per_value':base_bpv,'codebook_key':'s0_k8_w64_t8_v1200','interpretation':'E=M_ref-M_vq reference-aligned trajectory error; K5 is momentum-only nonlinear map; offline oracle, not online error feedback.'},(out/'provenance.json').open('w'),indent=2)
    (out/'summary.json').write_text(json.dumps({'status':'completed','seed':1,'update':UPDATE,'methods':['raw_vq','structural_int4','frobenius','k5_aware'],'budgets':TARGETS},indent=2))
    (out/'comparison.md').write_text('# Update-4096 K5-aware correction oracle\n\nSee `pareto.csv` for the required comparison. The K5-aware allocator evaluates the actual canonical nonlinear K5 map for each low-rank candidate and greedily recomputes global marginal gain per byte. `E=M_ref-M_vq` is reference-aligned trajectory error, not instantaneous quantization error. K5 is momentum-only; no contemporaneous gradient was saved, so this does not establish Nesterov update or NLL recovery. Candidate factors are BF16 and storage is charged as `2*r*(m+n)` bytes.\n')
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4));
        for method in sorted({r['method'] for r in flat}):
            z=[r for r in flat if r['method']==method]; ax.plot([r['total_bits_per_value'] for r in z],[r['k5_cosine'] for r in z],'o-',label=method)
        ax.set_xlabel('total effective bits/value'); ax.set_ylabel('K5-map cosine'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/'storage_vs_k5.png',dpi=150); plt.close(fig)
    except Exception: pass
    print(json.dumps({'status':'completed','out':str(out),'rows':len(flat)},indent=2))
if __name__=='__main__': main()
