#!/usr/bin/env python3
"""Read-only spectral-gap mechanism study over formal Muon snapshots."""
from __future__ import annotations

import argparse, csv, math, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optim.muon_update_fidelity import load_snapshot, _ratios  # noqa: E402
from optim.muon_spectral_sensitivity import decompose, quantize, subspace_metrics  # noqa: E402
from optim.muon_ns_sensitivity import exact_polar, scalar_map_and_derivative, transform  # noqa: E402
from optim.muon_spectral_gap_theory import (  # noqa: E402
    active_indices, band_indices, band_separation, controlled_gap_spectrum,
    deterministic_tail_error, gap_proxies, mode_gaps, principal_subspace_rows,
    safe_correlation, tail_reweighting,
)

LANDMARKS = (128, 512, 1024, 2048, 4096)
SEEDS = (0, 1)
GAP_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
# Small enough to remain in a useful perturbative regime for the empirical
# tail gaps, while still above FP32 SVD noise.  The same norm is used for all
# requested gap variants and is recorded in the report.
EPSILON = 0.001


def discover(root: Path):
    groups = {}
    for path in sorted(root.rglob("update_*.pt")):
        if "muon_momentum_snapshots" not in path.parts: continue
        try:
            snap = load_snapshot(path); meta = snap["metadata"]
            seed, update = int(meta["seeds"]["seed"]), int(meta["update"])
        except Exception:
            continue
        if seed in SEEDS and update in LANDMARKS:
            groups.setdefault((seed, str(path.parent)), {})[update] = path
    out = []
    for seed in SEEDS:
        choices = [(g, d) for (s, g), d in groups.items() if s == seed and set(d) == set(LANDMARKS)]
        if not choices: continue
        _, d = sorted(choices)[0]
        out += [(seed, u, d[u]) for u in LANDMARKS]
    return out


def finite(x): return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    if not keys: return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def identity(item, seed, update):
    return {"seed": seed, "update": update, "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
            "parameter_name": item.get("name", item.get("parameter_id", "<unknown>")), "shape": str(item["shape"])}


def transform_kwargs(snapshot):
    c = snapshot["metadata"]["muon_transform"]
    return {"steps": int(c["steps"]), "coefficients": tuple(float(x) for x in c["coefficients"]), "eps": float(c["eps"])}


def metric_pair(source, observed, kwargs):
    ref = transform(source, **kwargs); out = transform(observed, **kwargs)
    m = _ratios(ref, out, "update"); m.update({"update_ref": ref, "update_observed": out})
    raw = _ratios(source, observed, "raw"); m.update(raw)
    return m


def select_representatives(rows):
    """Predeclare reps from baseline features only, before interventions."""
    if not rows: return []
    by = sorted(rows, key=lambda r: (float(r.get("effective_condition_number") or float("inf")), r["seed"], r["update"], r["parameter_id"]))
    err = sorted(rows, key=lambda r: (float(r.get("update_direction_error") or float("inf")), r["seed"], r["update"], r["parameter_id"]))
    chosen = [(by[0], "low_condition"), (by[len(by)//2], "medium_condition"), (by[-1], "high_condition"),
              (err[0], "low_update_error"), (err[-1], "high_update_error")]
    seen = set(); out = []
    for row, reason in chosen:
        key = (row["seed"], row["update"], row["parameter_id"])
        if key not in seen: seen.add(key); out.append((key, reason))
    return out


def make_plots(out, gap_rows, sub_rows, intervention, cluster_rows):
    try: import matplotlib.pyplot as plt
    except ImportError:
        (out / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV outputs remain complete.\n"); return
    def scatter(x, y, name, xlabel, ylabel):
        rows = [r for r in gap_rows if finite(r.get(x)) and finite(r.get(y))]
        plt.figure(figsize=(7,5)); plt.scatter([r[x] for r in rows], [r[y] for r in rows], s=7, alpha=.45)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(out/name, dpi=140); plt.close()
    scatter("tail_sensitivity_2", "tail_projection_distance", "gap_over_delta_vs_tail_subspace.png", "||E||2 / delta_tail", "tail projection distance")
    scatter("tail_sensitivity_2", "update_error", "gap_over_delta_vs_update_error.png", "||E||2 / delta_tail", "1 - Muon update cosine")
    scatter("tail_projection_distance", "update_error", "tail_subspace_vs_update_error.png", "tail projection distance", "1 - Muon update cosine")
    scatter("raw_relative_l2", "update_error", "raw_l2_vs_update_error.png", "raw relative L2", "1 - Muon update cosine")
    scatter("combined_risk", "update_error", "combined_risk_vs_update_error.png", "gap × Muon reweighting score", "1 - Muon update cosine")
    if intervention:
        plt.figure(figsize=(7,5))
        for key in sorted({(r["seed"],r["update"],r["parameter_id"]) for r in intervention}):
            rr = [r for r in intervention if (r["seed"],r["update"],r["parameter_id"]) == key]; rr.sort(key=lambda r:r["requested_gap_ratio"])
            plt.plot([r["requested_gap_ratio"] for r in rr], [r["tail_projection_distance"] for r in rr], marker=".", alpha=.7)
        plt.xscale("log"); plt.xlabel("requested tail gap ratio"); plt.ylabel("tail projection distance"); plt.tight_layout(); plt.savefig(out/"controlled_gap_vs_subspace.png", dpi=140); plt.close()
        plt.figure(figsize=(7,5))
        for key in sorted({(r["seed"],r["update"],r["parameter_id"]) for r in intervention}):
            rr = [r for r in intervention if (r["seed"],r["update"],r["parameter_id"]) == key]; rr.sort(key=lambda r:r["requested_gap_ratio"])
            plt.plot([r["requested_gap_ratio"] for r in rr], [r["update_error_production"] for r in rr], marker=".", alpha=.7)
        plt.xscale("log"); plt.xlabel("requested tail gap ratio"); plt.ylabel("production Muon update error"); plt.tight_layout(); plt.savefig(out/"controlled_gap_vs_update_error.png", dpi=140); plt.close()
    if cluster_rows:
        plt.figure(figsize=(7,5)); data = [[r["update_error"] for r in cluster_rows if r["cluster_class"] == c and finite(r.get("update_error"))] for c in ("clustered", "separated")]
        plt.boxplot(data, labels=["clustered", "separated"]); plt.ylabel("1 - update cosine"); plt.tight_layout(); plt.savefig(out/"clustered_vs_separated.png", dpi=140); plt.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--reports-root", type=Path, default=ROOT/"reports"); p.add_argument("--output", type=Path, default=ROOT/"reports/muon_spectral_gap_theory"); p.add_argument("--skip-plots", action="store_true"); args = p.parse_args()
    start = time.perf_counter(); paths = discover(args.reports_root)
    if len(paths) != 10: raise SystemExit(f"expected 10 formal snapshots, found {len(paths)}")
    args.output.mkdir(parents=True, exist_ok=True)
    gap_rows=[]; sub_rows=[]; mode_rows=[]; reps={}; baseline_cache={}
    for seed, update, path in paths:
        print(f"analyzing seed={seed} update={update}", flush=True); snap=load_snapshot(path); kwargs=transform_kwargs(snap)
        for item in snap["tensors"]:
            if len(item["shape"]) != 2: continue
            rowid=identity(item,seed,update); M=item["tensor"].float(); d=decompose(M); bands=band_indices(d.singular_values)
            Mq=quantize(M,"int4-dynamic-b2048"); dq=decompose(Mq); E=Mq-M
            mm=metric_pair(M,Mq,kwargs); mp=gap_proxies(E,d.singular_values,bands)
            update_error = 1-float(mm["update_cosine"]) if finite(mm.get("update_cosine")) else None
            tail_sub = subspace_metrics(d.u[:,bands["tail"]],dq.u[:,bands["tail"]]) if bands["tail"].numel() else {}
            right_sub = subspace_metrics(d.vh.T[:,bands["tail"]],dq.vh.T[:,bands["tail"]]) if bands["tail"].numel() else {}
            left_dist = tail_sub.get("projection_distance_normalized") if tail_sub else None
            right_dist = right_sub.get("projection_distance_normalized") if right_sub else None
            tail_dist = ((left_dist + right_dist) / 2 if finite(left_dist) and finite(right_dist)
                         else left_dist if finite(left_dist) else right_dist)
            eff_cond = float(d.singular_values[0]/d.singular_values[bands["active"][-1]]) if bands["active"].numel() else None
            relgap=mode_gaps(d.singular_values)["relative_gap"]
            tail_gap = float(relgap[bands["tail"][0]]) if bands["tail"].numel() and int(bands["tail"][0]) < len(relgap) else None
            vals, deriv=scalar_map_and_derivative(d.singular_values, matrix_norm=float(M.norm()), steps=kwargs["steps"], coefficients=kwargs["coefficients"], eps=kwargs["eps"])
            combined = (mp.get("tail_sensitivity_2") or 0.0) * (tail_reweighting(d.singular_values, vals, bands["tail"]) or 0.0)
            row=rowid | {"raw_relative_l2":mm.get("raw_relative_l2"),"raw_cosine":mm.get("raw_cosine"),"update_error":update_error,"update_cosine":mm.get("update_cosine"),"effective_condition_number":eff_cond,"tail_relative_gap":tail_gap,"tail_projection_distance":tail_dist,"tail_left_projection_distance":tail_sub.get("projection_distance_normalized") if tail_sub else None,"tail_right_projection_distance":right_sub.get("projection_distance_normalized") if right_sub else None,"combined_risk":combined,"tail_reweighting":tail_reweighting(d.singular_values,vals,bands["tail"]),"norm_e2_over_m2":mp["error_spectral_norm"]/float(d.singular_values[0]) if float(d.singular_values[0]) else None,"norm_f_over_mf":mp["error_frobenius_norm"]/float(M.norm()) if float(M.norm()) else None}
            for k,v in mp.items(): row[k]=v
            gaps=mode_gaps(d.singular_values)
            row["mode_gap_min"] = float(gaps["gap"][bands["tail"]].min()) if bands["tail"].numel() else None
            gap_rows.append(row); baseline_cache[(seed,update,rowid["parameter_id"])] = (M,d,bands,kwargs)
            if bands["tail"].numel(): sub_rows.extend(rowid | {"band":x["band"],"side":x["side"]} | x for x in principal_subspace_rows(d.u,d.vh,dq.u,dq.vh,bands))
            for i in range(len(d.singular_values)):
                band="head" if i in bands["head"].tolist() else "tail" if i in bands["tail"].tolist() else "middle"
                mode_rows.append(rowid | {"mode":i,"band":band,"sigma":float(d.singular_values[i]),"normalized_sigma":float(d.singular_values[i]/(M.norm()+kwargs["eps"])),"gap":float(gaps["gap"][i]),"relative_gap":float(gaps["relative_gap"][i]),"a_mag":float(deriv[i]),"a_dir":float(vals[i].abs()/d.singular_values[i].abs().clamp_min(1e-12))})
    threshold=float(torch.tensor([r["tail_relative_gap"] for r in gap_rows if finite(r.get("tail_relative_gap"))]).quantile(.25))
    for r in gap_rows: r["cluster_class"] = "clustered" if finite(r.get("tail_relative_gap")) and r["tail_relative_gap"] <= threshold else "separated"
    # Controlled gap intervention is selected solely from baseline features.
    selected=select_representatives(gap_rows); intervention=[]
    for key,reason in selected:
        M,d,bands,kwargs=baseline_cache[key]; tail=bands["tail"]
        if not tail.numel() or int(tail[0])==0: continue
        target=float(M.norm())*EPSILON; E=deterministic_tail_error(d.u,d.vh,tail,target)
        for ratio in GAP_RATIOS:
            sv, actual=controlled_gap_spectrum(d.singular_values,tail,ratio); variant=(d.u*sv)@d.vh; observed=variant+E
            dv=decompose(variant); do=decompose(observed); ts=subspace_metrics(dv.u[:,tail],do.u[:,tail]); ru=transform(variant,**kwargs); ou=transform(observed,**kwargs); polar_ref=exact_polar(variant); polar_obs=exact_polar(observed)
            intervention.append({"seed":key[0],"update":key[1],"parameter_id":key[2],"selection_reason":reason,"requested_gap_ratio":ratio,"actual_gap_ratio":actual/float((d.singular_values[int(tail[0])-1]-d.singular_values[int(tail[0])]).abs()),"relative_gap":actual/float(sv[int(tail[0])]),"perturbation_relative_frobenius":float(E.norm()/variant.norm()),"tail_projection_distance":ts["projection_distance_normalized"],"update_error_production":1-float(_ratios(ru,ou,"x")["x_cosine"]),"update_error_exact_polar":1-float(_ratios(polar_ref,polar_obs,"x")["x_cosine"]),"gap_threshold":threshold})
    cluster_rows=gap_rows
    corr=[]
    for x,y in (("norm_e2_over_m2","tail_projection_distance"),("norm_f_over_mf","tail_projection_distance"),("tail_sensitivity_2","tail_projection_distance"),("tail_sensitivity_f","tail_projection_distance"),("tail_projection_distance","update_error"),("tail_sensitivity_2","update_error"),("tail_sensitivity_f","update_error"),("raw_relative_l2","update_error"),("effective_condition_number","update_error"),("combined_risk","update_error")):
        corr.append(safe_correlation(gap_rows,x,y))
    write_csv(args.output/"tensor_gap_metrics.csv",gap_rows); write_csv(args.output/"subspace_gap_sensitivity.csv",sub_rows); write_csv(args.output/"gap_update_correlations.csv",corr); write_csv(args.output/"controlled_gap_intervention.csv",intervention); write_csv(args.output/"clustered_vs_separated.csv",cluster_rows); write_csv(args.output/"combined_risk_score.csv",gap_rows); write_csv(args.output/"mode_sensitivity.csv",mode_rows)
    if not args.skip_plots: make_plots(args.output,gap_rows,sub_rows,intervention,cluster_rows)
    runtime=time.perf_counter()-start
    strongest=sorted([r for r in corr if finite(r.get("spearman"))],key=lambda r:abs(r["spearman"]),reverse=True)
    methodology=f"""# Methodology

This read-only CPU study uses {len(paths)} formal FP32 snapshots and {len(gap_rows)} eligible 2D tensors. Runtime: {runtime:.2f} seconds. It does not train or modify existing artifacts.

Singular values are sorted descending. For mode i, `gap_i` is the minimum absolute difference to its available immediate neighbors (one neighbor at boundaries); `relative_gap_i=gap_i/max(sigma_i,1e-12)`. Active modes satisfy `sigma_i/sigma_max >= 1e-6`. Head, middle, and tail are the leading, centered, and trailing 10% of active index modes (at least one). A band's external separation is the minimum pairwise absolute singular-value difference to its complement; for a contiguous tail this is its boundary gap.

For E=Mq-M, `S_B2=||E||2/max(delta_B,1e-12)` and `S_BF=||E||F/max(delta_B,1e-12)` are bound-inspired Wedin/Davis--Kahan sensitivity proxies, not exact bounds. Subspace distortion is the normalized Frobenius distance between orthogonal projectors, supplemented by principal-angle summaries in `subspace_gap_sensitivity.csv`; clustered singular values make individual vectors non-identifiable, so projector metrics are primary.

The combined heuristic risk is `(||E||2/delta_tail) * median_tail(|f_K(sigma)|/sigma)`, where `f_K` is the production Newton--Schulz scalar transfer map. It is descriptive, not a theorem or causal model. Clustered tails are defined by `tail_relative_gap <= empirical 25th percentile ({threshold:.6g})`.

Controlled intervention selects the low/median/high effective-condition and low/high update-error baseline tensors before inspecting intervention results (deduplicated). It preserves U,V and overall spectrum except for the tail/complement boundary, requests gap ratios {GAP_RATIOS}, then adds a fixed deterministic cross-boundary spectral-coordinate perturbation with relative Frobenius norm {EPSILON}. Actual ratios are recorded because monotonicity/positivity can clip extreme targets. Production K and exact-polar readouts are separate.

Results are descriptive associations, not statistical significance or causal mediation. Non-2D states are excluded because production Muon only transforms eligible 2D matrices.
"""
    (args.output/"methodology.md").write_text(methodology)
    top=[(r["feature"],r["pearson"],r["spearman"],r["sample_count"]) for r in strongest[:5]]
    (args.output/"summary.md").write_text(f"""# Summary

Analyzed {len(paths)} snapshots and {len(gap_rows)} eligible 2D Muon tensors. The production INT4 dynamic b2048 reconstruction and production Muon transform were reused unchanged. Runtime was {runtime:.2f}s CPU.

Empirical clustered-tail threshold: relative boundary gap <= {threshold:.6g} (lower quartile). Representative controlled-gap tensors: {len(selected)}. Requested ratios: {GAP_RATIOS}; matched perturbation epsilon: {EPSILON}.

Strongest descriptive associations (feature, Pearson, Spearman, n): {top}

The report tests the chain quantization perturbation -> perturbation/gap instability -> tail subspace distortion -> Muon update error. Correlations do not prove causality; see `theory.md` for the mathematical mechanism and caveats.
""")
    print(f"analyzed {len(paths)} snapshots / {len(gap_rows)} tensors in {runtime:.2f}s; reps={len(selected)} threshold={threshold:g}")


if __name__ == "__main__": main()
