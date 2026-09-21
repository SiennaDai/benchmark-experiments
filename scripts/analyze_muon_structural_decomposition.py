#!/usr/bin/env python3
"""Offline structural-decomposition mechanism study for INT4 Muon."""
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
from optim.muon_ns_sensitivity import exact_polar as ns_exact_polar  # noqa: E402
from optim import muon_reference  # noqa: E402
from optim.muon_structural_decomposition import (  # noqa: E402
    ENERGY_TARGETS, FIXED_RANKS, decompose, deterministic_random_modes,
    danger_zone_energy, energy_rank, residual_dynamic_range,
    structural_reconstruct, top_k_posthoc_correction, truncated,
    valid_rank,
)
from optim.muon_continuous_spectral_risk import SPECTRAL_THRESHOLD  # noqa: E402

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
QUANTIZER = "int4-dynamic-b2048"
EPS = 1e-12


def discover_snapshots(root: Path):
    groups = {}
    for path in sorted(root.rglob("update_*.pt")):
        if "muon_momentum_snapshots" not in path.parts:
            continue
        try:
            snap = load_snapshot(path)
            seed = int(snap["metadata"]["seeds"]["seed"]); update = int(snap["metadata"]["update"])
        except (KeyError, ValueError, RuntimeError, OSError):
            continue
        if seed in SEEDS and update in LANDMARKS:
            groups.setdefault((seed, str(path.parent)), {})[update] = path
    out = []
    for seed in SEEDS:
        candidates = sorted((g, v) for (s, g), v in groups.items() if s == seed and set(v) == set(LANDMARKS))
        if candidates:
            values = candidates[0][1]; out.extend((seed, u, values[u]) for u in LANDMARKS)
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def read_csv(path: Path):
    if not path.exists(): return []
    with path.open(newline="") as h: return list(csv.DictReader(h))


def finite(v):
    try: return v not in (None, "") and math.isfinite(float(v))
    except (TypeError, ValueError): return False


def ident(seed, update, item):
    return {"seed": seed, "update": update, "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
            "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")), "shape": str(item["shape"])}


def kwargs(snapshot):
    c = snapshot["metadata"]["muon_transform"]
    return {"steps": int(c["steps"]), "coefficients": tuple(float(x) for x in c["coefficients"]), "eps": float(c["eps"])}


def metrics(reference, candidate, kw, reference_update=None):
    def ratios(a, b):
        a = a.float(); b = b.float(); an = a.norm(); bn = b.norm()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or an.item() == 0:
            return {"relative_l2": None, "cosine": None, "norm_ratio": None}
        return {"relative_l2": float((b-a).norm()/an), "cosine": float((a*b).sum()/(an*bn)) if bn.item() else None,
                "norm_ratio": float(bn/an) if an.item() else None}
    raw = ratios(reference, candidate)
    ref_u = reference_update if reference_update is not None else muon_reference.zeropower_newton_schulz(reference.clone(), **kw)
    cand_u = muon_reference.zeropower_newton_schulz(candidate.clone(), **kw)
    upd = ratios(ref_u, cand_u)
    return {"raw_relative_l2": raw["relative_l2"], "raw_cosine": raw["cosine"], "raw_norm_ratio": raw["norm_ratio"],
            "update_relative_l2": upd["relative_l2"], "update_cosine": upd["cosine"], "update_norm_ratio": upd["norm_ratio"]}


def polar_metrics(reference, candidate, reference_polar=None):
    a = reference_polar if reference_polar is not None else ns_exact_polar(reference)
    b = ns_exact_polar(candidate); an = a.norm(); bn = b.norm()
    return {"exact_polar_relative_l2": float((b-a).norm()/an) if an.item() else None,
            "exact_polar_cosine": float((a*b).sum()/(an*bn)) if an.item() and bn.item() else None,
            "exact_polar_norm_ratio": float(bn/an) if an.item() else None}


def block_scales(x, block=2048):
    flat = x.float().reshape(-1); vals = []
    for start in range(0, flat.numel(), block):
        vals.append(float(flat[start:start+block].abs().amax()))
    return torch.tensor(vals, dtype=torch.float32) if vals else torch.zeros(0)


def cross_groups(state, error):
    u, s, vh = state.u, state.singular_values, state.vh
    ehat = u.T @ error.float() @ vh.T
    z = torch.log10((s / s[0]).clamp_min(1e-6)) if s.numel() and s[0] > 0 else torch.zeros_like(s)
    active = (s / s[0] >= SPECTRAL_THRESHOLD) if s.numel() and s[0] > 0 else torch.zeros_like(s, dtype=torch.bool)
    rows = []
    for name, lo, hi in (("local", 0.0, 0.5), ("medium", 0.5, 1.5), ("distant", 1.5, float("inf"))):
        d = (z[:, None] - z[None, :]).abs(); mask = (d >= lo) & (d < hi) & ~torch.eye(len(s), dtype=torch.bool)
        # Keep all off-diagonal spectral coordinates; danger filtering is in a separate file.
        value = ehat[mask].square().sum()
        rows.append((name, float(value), float(value / error.float().square().sum()) if error.norm().item() else None))
    return rows


def pearson_spearman(rows, feature, target):
    vals = [(float(r[feature]), float(r[target])) for r in rows if finite(r.get(feature)) and finite(r.get(target))]
    if len(vals) < 2: return {"feature": feature, "target": target, "sample_count": len(vals), "pearson": None, "spearman": None}
    x = torch.tensor([a for a, _ in vals], dtype=torch.float64); y = torch.tensor([b for _, b in vals], dtype=torch.float64)
    def corr(a, b):
        ac, bc = a-a.mean(), b-b.mean(); d = ac.norm()*bc.norm(); return float((ac*bc).sum()/d) if d.item() else None
    rx = torch.argsort(torch.argsort(x, stable=True), stable=True).double(); ry = torch.argsort(torch.argsort(y, stable=True), stable=True).double()
    return {"feature": feature, "target": target, "sample_count": len(vals), "pearson": corr(x,y), "spearman": corr(rx,ry)}


def make_plots(out: Path, rows: list[dict]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs are complete.\n"); return
    def grouped(field, y, subset=None):
        d = defaultdict(list)
        for r in rows:
            if subset and not subset(r): continue
            if finite(r.get(y)): d[str(r[field])].append(float(r[y]))
        return [(k, sum(v)/len(v)) for k,v in d.items()]
    for y, name in (("update_cosine", "rank_vs_update_cosine.png"), ("exact_polar_cosine", "rank_vs_exact_polar_cosine.png"),
                    ("residual_over_matrix_norm", "rank_vs_residual_dynamic_range.png"), ("danger_fraction_of_error", "rank_vs_danger_zone_error.png")):
        vals = grouped("rank_label", y)
        if vals:
            plt.figure(figsize=(10,4)); plt.bar([x for x,_ in vals], [v for _,v in vals]); plt.xticks(rotation=60,ha="right"); plt.ylabel(y); plt.tight_layout(); plt.savefig(out/name,dpi=140); plt.close()
    vals = grouped("rank_label", "update_cosine", lambda r: r.get("kind") == "structural")
    if vals:
            plt.figure(figsize=(10,4)); plt.plot([x for x,_ in vals],[v for _,v in vals],marker="o"); plt.xticks(rotation=60,ha="right"); plt.ylabel("update cosine"); plt.tight_layout(); plt.savefig(out/"structural_update_cosine.png",dpi=140); plt.close()


def make_extended_plots(out: Path):
    base = read_csv(out / "baseline_vs_structural.csv")
    post = read_csv(out / "posthoc_rank_control.csv")
    random = read_csv(out / "random_rank_control.csv")
    mixing = read_csv(out / "cross_scale_mixing.csv")
    resolution = read_csv(out / "resolution_analysis.csv")
    dynamic = read_csv(out / "residual_dynamic_range.csv")
    danger = read_csv(out / "danger_zone_error.csv")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    def grouped(rows, key, value, pred=lambda r: True):
        d=defaultdict(list)
        for r in rows:
            if pred(r) and finite(r.get(value)): d[str(r.get(key))].append(float(r[value]))
        return [(k, sum(v)/len(v)) for k,v in d.items()]
    for rows, value, name, title in [
        (base, "raw_relative_l2", "baseline_vs_structural_quant_error.png", "Raw quantization error by rank"),
        (mixing, "energy", "cross_mode_mixing_by_rank.png", "Cross-scale quantization mixing by rank"),
        (post, "update_cosine_gain_vs_direct", "structural_vs_posthoc_rank.png", "Structural versus post-hoc rank-k gain"),
        (random, "update_cosine_gain_vs_direct", "top_k_vs_random_k.png", "Top-k versus deterministic random-k gain"),
        (dynamic, "residual_over_matrix_norm", "rank_vs_residual_dynamic_range.png", "Residual norm versus rank"),
        (danger, "struct_danger_fraction_of_matrix", "rank_vs_danger_zone_error.png", "Danger-zone error versus rank"),
    ]:
        vals = grouped(rows, "rank_label", value, lambda r: r.get("kind") == "structural" if value == "raw_relative_l2" else True)
        if vals:
            plt.figure(figsize=(10,4)); plt.plot([x for x,_ in vals],[y for _,y in vals],marker="o"); plt.xticks(rotation=60,ha="right"); plt.ylabel(value); plt.title(title); plt.tight_layout(); plt.savefig(out/name,dpi=140); plt.close()
    if resolution:
        for x, name, title in [("scale_reduction_ratio", "block_scale_reduction_vs_gain.png", "Block-scale reduction versus update gain"),
                               ("danger_error_fraction", "danger_residual_vs_gain.png", "Danger-zone residual versus update gain")]:
            pts=[(float(r[x]),float(r["update_cosine_gain"])) for r in resolution if finite(r.get(x)) and finite(r.get("update_cosine_gain"))]
            if pts:
                plt.figure(figsize=(6,4)); plt.scatter([a for a,_ in pts],[b for _,b in pts],s=5,alpha=.35); plt.xlabel(x); plt.ylabel("update cosine gain"); plt.title(title); plt.tight_layout(); plt.savefig(out/name,dpi=140); plt.close()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reports-root", type=Path, default=ROOT/"reports"); ap.add_argument("--output", type=Path, default=ROOT/"reports/muon_structural_decomposition_mechanism"); ap.add_argument("--skip-plots", action="store_true"); ap.add_argument("--plots-only", action="store_true"); ap.add_argument("--refresh-correlations", action="store_true"); ap.add_argument("--snapshot-limit", type=int, default=0); args=ap.parse_args()
    if args.plots_only:
        make_extended_plots(args.output)
        print(f"plots refreshed in {args.output}", flush=True)
        return
    if args.refresh_correlations:
        resolution = read_csv(args.output / "resolution_analysis.csv")
        rows = [pearson_spearman(resolution, feature, "update_cosine_gain")
                for feature in ("scale_reduction_ratio", "danger_error_fraction", "rank")]
        write_csv(args.output / "mechanism_correlations.csv", rows)
        print(f"correlations refreshed in {args.output}", flush=True)
        return
    started=time.perf_counter(); paths=discover_snapshots(args.reports_root)
    if len(paths)!=10: raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    if args.snapshot_limit: paths = paths[:args.snapshot_limit]
    args.output.mkdir(parents=True, exist_ok=True)
    base_rows=[]; manifest=[]; dynamic=[]; errors=[]; danger_rows=[]; mixing=[]; posthoc=[]; random_rows=[]; resolution=[]; tensor_count=0
    for seed, update, path in paths:
        snap=load_snapshot(path); kw=kwargs(snap); print(f"processing seed={seed} update={update}", flush=True)
        for item in snap["tensors"]:
            if len(item["shape"])!=2: continue
            tensor_count+=1; x=item["tensor"].detach().float(); st=decompose(x); q=quantize(x,QUANTIZER).float(); ident0=ident(seed,update,item)
            ref_u=muon_reference.zeropower_newton_schulz(x.clone(),**kw); q_u=muon_reference.zeropower_newton_schulz(q.clone(),**kw); ref_p=ns_exact_polar(x); q_p=ns_exact_polar(q)
            base=metrics(x,q,kw,ref_u); base.update(polar_metrics(x,q,ref_p)); base["kind"]="direct"; base["rank_label"]="direct"; base_rows.append(ident0|base|{"raw_error_norm":float((q-x).norm()),"update_error":1.0-(base.get("update_cosine") or 0.0),"danger_zone_residual_fraction":danger_zone_energy(st,q-x)["danger_fraction_of_error"],"cross_mode_residual_fraction":sum(z[2] or 0 for z in cross_groups(st,q-x))})
            rank_specs=[]
            for k in FIXED_RANKS:
                if k<=st.singular_values.numel(): rank_specs.append((f"fixed_{k}",k,"fixed",None))
            for target in ENERGY_TARGETS:
                k=energy_rank(st.singular_values,target); rank_specs.append((f"energy_{int(target*100)}",k,"energy",target))
            for label,k,kind,target in rank_specs:
                low,res=truncated(st,k); qr=quantize(res,QUANTIZER).float(); recon=low+qr; mm=metrics(x,recon,kw,ref_u); mm.update(polar_metrics(x,recon,ref_p)); mm.update({"kind":"structural","rank_label":label,"rank":k,"rank_kind":kind,"energy_target":target,"low_rank_energy_fraction":float(st.singular_values[:k].square().sum()/st.singular_values.square().sum()) if st.singular_values.numel() and k else 0.0,"update_cosine_gain_vs_direct":(mm.get("update_cosine") or 0)-(base.get("update_cosine") or 0),"update_l2_reduction_vs_direct":(base.get("update_relative_l2") or 0)-(mm.get("update_relative_l2") or 0),"exact_polar_cosine_gain_vs_direct":(mm.get("exact_polar_cosine") or 0)-(base.get("exact_polar_cosine") or 0),"recovered_cosine_error_fraction":((mm.get("update_cosine") or 0)-(base.get("update_cosine") or 0))/max(1-(base.get("update_cosine") or 0),EPS)})
                base_rows.append(ident0|mm); manifest.append(ident0|{"rank_label":label,"rank":k,"rank_kind":kind,"energy_target":target,"active_rank":int(st.singular_values.numel())})
                rd=residual_dynamic_range(st,res); dynamic.append(ident0|{"rank_label":label,"kind":"residual"}|rd|{"direct_max_abs":float((q-x).abs().max()),"direct_block_absmax_mean":float(block_scales(q-x).mean())})
                qe=qr-res; errors.append(ident0|{"rank_label":label,"kind":"structural","residual_error_norm":float(qe.norm()),"residual_relative_to_R":float(qe.norm()/res.norm()) if res.norm().item() else None,"error_over_M":float(qe.norm()/x.norm()) if x.norm().item() else None,"error_spectral_norm":float(torch.linalg.matrix_norm(qe,2))})
                dz=danger_zone_energy(st,recon-x); dz_q=danger_zone_energy(st,qe); danger_rows.append(ident0|{"rank_label":label,"kind":"structural"}|{f"struct_{key}":value for key,value in dz.items()}|{"quant_error_danger_fraction":dz_q["danger_fraction_of_error"],"quant_error_danger_cross_energy":dz_q["danger_cross_energy"]})
                for g,e,f in cross_groups(st,qe): mixing.append(ident0|{"rank_label":label,"kind":"structural","distance_group":g,"energy":e,"fraction_total_error":f})
                # Post-hoc top-k correction with exactly the same rank budget.
                ph=top_k_posthoc_correction(st,q,k); pm=metrics(x,ph,kw,ref_u); pm.update(polar_metrics(x,ph,ref_p)); posthoc.append(ident0|{"rank_label":label,"rank":k}|pm|{"update_cosine_gain_vs_direct":(pm.get("update_cosine") or 0)-(base.get("update_cosine") or 0)})
                modes = deterministic_random_modes(st.singular_values.numel(), k)
                if modes.numel():
                    lowr = (st.u[:, modes] * st.singular_values[modes]) @ st.vh[modes]
                else:
                    lowr = torch.zeros_like(x)
                resr = x - lowr
                rr=lowr+quantize(resr,QUANTIZER).float(); rm=metrics(x,rr,kw,ref_u); rm.update(polar_metrics(x,rr,ref_p)); random_rows.append(ident0|{"rank_label":label,"rank":k}|rm|{"random_seed":2026,"random_modes":str(modes.tolist()),"update_cosine_gain_vs_direct":(rm.get("update_cosine") or 0)-(base.get("update_cosine") or 0)})
                resolution.append(ident0|{"rank_label":label,"rank":k,"matrix_block_absmax_mean":float(block_scales(x).mean()),"residual_block_absmax_mean":float(block_scales(res).mean()),"scale_reduction_ratio":float(block_scales(res).mean()/block_scales(x).mean()) if block_scales(x).mean().item() else None,"update_cosine_gain":(mm.get("update_cosine") or 0)-(base.get("update_cosine") or 0),"danger_error_fraction":dz_q["danger_fraction_of_error"]})
    write_csv(args.output/"rank_manifest.csv",manifest); write_csv(args.output/"baseline_vs_structural.csv",base_rows); write_csv(args.output/"update_fidelity_by_rank.csv",[r for r in base_rows if r.get("kind")=="structural"]); write_csv(args.output/"residual_dynamic_range.csv",dynamic); write_csv(args.output/"residual_quantization_error.csv",errors); write_csv(args.output/"danger_zone_error.csv",danger_rows); write_csv(args.output/"cross_scale_mixing.csv",mixing); write_csv(args.output/"posthoc_rank_control.csv",posthoc); write_csv(args.output/"random_rank_control.csv",random_rows); write_csv(args.output/"resolution_analysis.csv",resolution)
    corr=[]
    structural=[r for r in base_rows if r.get("kind")=="structural"]
    for f in ("update_cosine_gain_vs_direct","danger_zone_residual_fraction","cross_mode_residual_fraction","low_rank_energy_fraction"):
        if f in structural[0]: corr.append(pearson_spearman(structural,f,"update_cosine_gain_vs_direct"))
    write_csv(args.output/"mechanism_correlations.csv",corr)
    if not args.skip_plots: make_plots(args.output,base_rows)
    def avg(field, rows=structural):
        a=[float(r[field]) for r in rows if finite(r.get(field))]; return sum(a)/len(a) if a else None
    summary=f"""# Structural-decomposition mechanism study

Coverage: {tensor_count} eligible 2D Muon tensors from 10 formal FP32 snapshots. Production `int4-dynamic-b2048` and the exact production Muon transform are reused unchanged. No training was launched. The exact-SVD oracle stores `M_k` in FP32 side information and quantizes only `R_k=M-M_k`; it is not a deployable quantizer or reproduction of an external method.

Fixed ranks: {FIXED_RANKS}. Energy ranks: {ENERGY_TARGETS}, selecting the smallest deterministic k explaining each target squared-Frobenius energy. Direct INT4 rows and all rank rows are in `baseline_vs_structural.csv`; post-hoc and deterministic random-mode controls are separate.

Mean structural K=5 cosine gain versus direct INT4 across rank rows: `{avg('update_cosine_gain_vs_direct')}`. Mean recovered cosine-error fraction: `{avg('recovered_cosine_error_fraction')}`. The direct-to-structural comparison must be read together with `residual_dynamic_range.csv`, `danger_zone_error.csv`, and `cross_scale_mixing.csv`; lower absolute residual norm is not by itself an efficiency claim.

Runtime: `{time.perf_counter()-started:.1f}` CPU seconds. This is an offline oracle/headroom study; correlations do not establish causality.
"""
    (args.output/"summary.md").write_text(summary)
    (args.output/"methodology.md").write_text("""# Methodology

For each FP32 matrix M, a reduced FP32 SVD defines exact truncated components M_k and residual R_k. The structural oracle is M_k + Q4(R_k), using the unchanged production INT4 blockwise-dynamic b2048 persistence path. The direct baseline is Q4(M). All matrices are detached diagnostic copies.

Fixed ranks are 1,2,4,8,16 when valid. Energy ranks are the smallest k reaching 25%, 50%, 75%, and 90% of squared Frobenius spectral energy. The original FP32 U,V basis is used for danger-zone and cross-scale measurements. Danger zone is the prior fixed [-3,-2) log10(sigma/sigma_max) interval. Post-hoc controls project the actual direct residual onto the same top-k two-sided FP32 subspace. Random controls use a fixed local seed 2026 without changing global RNG state.

Dynamic-range diagnostics use flattened b2048 absmax blocks and report matrix/residual block-scale statistics. Structural quantization error is Q4(R_k)-R_k; M_k is exact oracle side information. Production K=5 and exact-polar metrics are retained separately. This study measures representation headroom, not a practical storage design.
""")
    print(f"wrote {args.output}; tensors={tensor_count}; runtime={time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__": main()
