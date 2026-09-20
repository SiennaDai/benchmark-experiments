#!/usr/bin/env python3
"""Offline INT4 dynamic rounding headroom study for Muon snapshots."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optim.muon_rounding_oracle import analyze_oracle_tensor
from optim.muon_update_fidelity import _aggregate, load_snapshot

GLOB = "reports/muon_update_fidelity_formal_s0_s1_results/muon-fidelity-formal/fp32_muon_snapshot_s*/muon_momentum_snapshots/update_*.pt"
MODES = ("nearest", "raw_direction_oracle", "muon_update_direction_oracle")

def discover(root):
    out = []
    for p in sorted(root.glob(GLOB)):
        seed = int(p.parent.parent.name.rsplit("_s", 1)[1])
        update = int(p.stem.rsplit("_", 1)[1])
        out.append((seed, update, p))
    return sorted(out)

def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if not k.startswith("_") and k not in keys: keys.append(k)
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def aggregate(reductions):
    raw, update = _aggregate(reductions, "raw"), _aggregate(reductions, "update")
    out = {}
    for k, v in raw.items(): out[k] = v; out["raw_momentum_" + k[4:]] = v
    for k, v in update.items(): out[k] = v; out["muon_" + k] = v
    return out

def plot(output, aggregates, tensors, stats):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (output / "plots_unavailable.txt").write_text("matplotlib unavailable; CSV artifacts are complete.\n"); return
    labels = {"nearest":"nearest", "raw_direction_oracle":"raw-direction oracle", "muon_update_direction_oracle":"Muon-update oracle"}
    colors = {"nearest":"#444444", "raw_direction_oracle":"#0072B2", "muon_update_direction_oracle":"#D55E00"}
    for metric, title, filename in (("muon_update_cosine", "Post-Muon update cosine", "update_cosine_across_landmarks"), ("muon_update_relative_l2", "Post-Muon update relative L2", "update_relative_l2_across_landmarks")):
        fig, ax = plt.subplots(figsize=(7,4))
        for mode in MODES:
            rows = sorted((r for r in aggregates if r["rounding_mode"] == mode and isinstance(r.get(metric),(int,float))), key=lambda r:r["update"])
            ax.plot([r["update"] for r in rows], [r[metric] for r in rows], marker="o", label=labels[mode], color=colors[mode])
        ax.set(xlabel="Update", ylabel=metric, title=title); ax.legend(); fig.tight_layout(); fig.savefig(output/(filename+".png"), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5))
    for mode in MODES:
        rows = [r for r in aggregates if r["rounding_mode"] == mode and isinstance(r.get("raw_momentum_cosine"),(int,float)) and isinstance(r.get("muon_update_cosine"),(int,float))]
        ax.scatter([r["raw_momentum_cosine"] for r in rows], [r["muon_update_cosine"] for r in rows], label=labels[mode], color=colors[mode])
    ax.set(xlabel="Raw momentum cosine", ylabel="Post-Muon update cosine", title="Raw vs post-Muon direction"); ax.legend(); fig.tight_layout(); fig.savefig(output/"raw_vs_update_cosine.png", dpi=160); plt.close(fig)
    improvements = sorted((r for r in tensors if isinstance(r.get("delta_muon_update_cosine"),(int,float))), key=lambda r:r["delta_muon_update_cosine"], reverse=True)[:10]
    if improvements:
        fig, ax = plt.subplots(figsize=(8,4)); ax.barh([r["parameter_name"] for r in improvements[::-1]], [r["delta_muon_update_cosine"] for r in improvements[::-1]]); ax.set(xlabel="Muon-update cosine improvement", title="Largest tensor-level oracle improvements"); fig.tight_layout(); fig.savefig(output/"oracle_headroom_by_tensor.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for mode in ("raw_direction_oracle", "muon_update_direction_oracle"):
        rows=sorted((r for r in stats if r["rounding_mode"]==mode), key=lambda r:r["update"])
        ax.plot([r["update"] for r in rows],[r["candidate_fraction"] for r in rows],marker="o",label=labels[mode],color=colors[mode])
    ax.set(xlabel="Update",ylabel="Candidate fraction",title="Oracle candidate screening"); ax.legend(); fig.tight_layout(); fig.savefig(output/"candidate_fraction.png",dpi=160); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--input-root", type=Path, default=ROOT); ap.add_argument("--output", type=Path, default=ROOT/"reports/muon_direction_rounding_oracle"); ap.add_argument("--max-snapshots", type=int, default=0); ap.add_argument("--midpoint-margin", type=float, default=.25); ap.add_argument("--update-max-candidates", type=int, default=32); ap.add_argument("--update-max-groups", type=int, default=4); args = ap.parse_args()
    paths = discover(args.input_root)
    if args.max_snapshots: paths = paths[:args.max_snapshots]
    if not paths: raise SystemExit("no formal snapshots found")
    args.output.mkdir(parents=True, exist_ok=True); aggregate_rows=[]; tensor_rows=[]; stat_rows=[]; runtimes=[]
    for seed, update, path in paths:
        items = load_snapshot(path)["tensors"]; by_mode={m:[] for m in MODES}; by_stats=[]; started=time.perf_counter()
        for item in items:
            rows, stats = analyze_oracle_tensor(item, midpoint_margin=args.midpoint_margin, update_max_candidates=args.update_max_candidates, update_max_groups=args.update_max_groups)
            for row in rows:
                reduction=row.pop("_reduction"); by_mode[row["rounding_mode"]].append(reduction); row.update(seed=seed, update=update, snapshot=str(path)); tensor_rows.append(row)
            for stat in stats: stat.update(seed=seed, update=update, snapshot=str(path)); by_stats.append(stat)
        elapsed=time.perf_counter()-started; runtimes.append(elapsed)
        for mode, reductions in by_mode.items(): aggregate_rows.append(dict(seed=seed, update=update, rounding_mode=mode, snapshot=str(path), **aggregate(reductions)))
        for mode in ("raw_direction_oracle","muon_update_direction_oracle"):
            rows=[s for s in by_stats if s["rounding_mode"]==mode]; total=max(1,sum(s.get("total_count",0) for s in rows)); considered=max(1,sum(s["considered_count"] for s in rows)); accepted=sum(s["accepted_flips"] for s in rows)
            stat_rows.append(dict(seed=seed, update=update, rounding_mode=mode, snapshot=str(path), total_count=total, eligible_count=sum(s["eligible_count"] for s in rows), considered_count=sum(s["considered_count"] for s in rows), accepted_flips=accepted, exact_evaluations=sum(s["exact_evaluations"] for s in rows), candidate_fraction=sum(s["eligible_count"] for s in rows)/total, eligible_fraction=sum(s["eligible_count"] for s in rows)/total, accepted_flip_fraction=accepted/considered, runtime_seconds=elapsed))
    nearest={(r["seed"],r["update"]):r for r in aggregate_rows if r["rounding_mode"]=="nearest"}
    for r in aggregate_rows:
        ref=nearest.get((r["seed"],r["update"]));
        if ref and r["rounding_mode"]!="nearest":
            for m in ("raw_momentum_cosine","raw_momentum_relative_l2","muon_update_cosine","muon_update_relative_l2"):
                r["delta_"+m]=r[m]-ref[m] if isinstance(r.get(m),(int,float)) and isinstance(ref.get(m),(int,float)) else None
            if isinstance(r.get("muon_update_relative_l2"),(int,float)) and ref.get("muon_update_relative_l2"):
                r["update_cosine_headroom"] = r.get("muon_update_cosine") - ref.get("muon_update_cosine")
                r["update_l2_reduction_fraction"] = (ref["muon_update_relative_l2"] - r["muon_update_relative_l2"]) / ref["muon_update_relative_l2"]
    nearest_tensor={(r["seed"],r["update"],r["parameter_id"]):r for r in tensor_rows if r["rounding_mode"]=="nearest"}
    for r in tensor_rows:
        ref=nearest_tensor.get((r["seed"],r["update"],r["parameter_id"]));
        if ref and r["rounding_mode"]!="nearest":
            for m in ("raw_momentum_cosine","raw_momentum_relative_l2","muon_update_cosine","muon_update_relative_l2"): r["delta_"+m]=r[m]-ref[m] if isinstance(r.get(m),(int,float)) and isinstance(ref.get(m),(int,float)) else None
    highlights=[]
    groups=(
        ("largest_nearest_update_distortion", sorted((r for r in tensor_rows if r["rounding_mode"]=="nearest" and isinstance(r.get("muon_update_relative_l2"),(int,float))), key=lambda r:r["muon_update_relative_l2"], reverse=True)[:20]),
        ("largest_muon_oracle_improvement", sorted((r for r in tensor_rows if r["rounding_mode"]=="muon_update_direction_oracle" and isinstance(r.get("delta_muon_update_cosine"),(int,float))), key=lambda r:r["delta_muon_update_cosine"], reverse=True)[:20]),
        ("raw_helps_raw_not_update", sorted((r for r in tensor_rows if r["rounding_mode"]=="raw_direction_oracle" and isinstance(r.get("delta_raw_momentum_cosine"),(int,float)) and r["delta_raw_momentum_cosine"]>0 and isinstance(r.get("delta_muon_update_cosine"),(int,float)) and r["delta_muon_update_cosine"]<=0), key=lambda r:r["delta_raw_momentum_cosine"], reverse=True)[:20]),
        ("oracle_mode_difference", sorted((r for r in tensor_rows if r["rounding_mode"]=="muon_update_direction_oracle" and isinstance(r.get("delta_muon_update_cosine"),(int,float))), key=lambda r:abs(r["delta_muon_update_cosine"]), reverse=True)[:20]),
    )
    for label, rows in groups:
        for row in rows:
            highlights.append({"highlight":label,"seed":row["seed"],"update":row["update"],"parameter_id":row["parameter_id"],"parameter_name":row["parameter_name"],"rounding_mode":row["rounding_mode"],"muon_update_relative_l2":row.get("muon_update_relative_l2"),"muon_update_cosine":row.get("muon_update_cosine"),"delta_raw_momentum_cosine":row.get("delta_raw_momentum_cosine"),"delta_muon_update_cosine":row.get("delta_muon_update_cosine")})
    summary=[]
    for scope, selected in (("seed0",[r for r in aggregate_rows if r["seed"]==0]),("seed1",[r for r in aggregate_rows if r["seed"]==1]),("all10",aggregate_rows)):
        means={}
        for mode in MODES:
            rows=[r for r in selected if r["rounding_mode"]==mode];
            if not rows: continue
            means[mode]=dict(scope=scope,rounding_mode=mode,snapshots=len(rows),**{m:sum(r[m] for r in rows)/len(rows) for m in ("raw_momentum_cosine","raw_momentum_relative_l2","muon_update_cosine","muon_update_relative_l2")})
            summary.append(means[mode])
        ref=means.get("nearest")
        for mode in MODES[1:]:
            if ref and mode in means:
                means[mode]["update_cosine_headroom"]=means[mode]["muon_update_cosine"]-ref["muon_update_cosine"]; means[mode]["update_l2_reduction_fraction"]=(ref["muon_update_relative_l2"]-means[mode]["muon_update_relative_l2"])/ref["muon_update_relative_l2"]
    write_csv(args.output/"summary.csv",summary); write_csv(args.output/"snapshot_aggregate_metrics.csv",aggregate_rows); write_csv(args.output/"tensor_metrics.csv",tensor_rows); write_csv(args.output/"oracle_search_stats.csv",stat_rows); write_csv(args.output/"tensor_highlights.csv",highlights)
    (args.output/"methodology.md").write_text("""# Muon INT4 direction-aware rounding oracle\n\nThe fixed quantizer is the existing signed dynamic-map INT4 blockwise roundtrip with block size 2048 and absmax scale. Production nearest is delegated to `persist_state`; lower/upper candidates use the same codebook, clamp, scale, and block partition.\n\nThe raw-direction oracle starts at nearest and performs deterministic block-local grouped coordinate descent. Candidates are eligible when normalized distance to the neighboring-level midpoint is at most the configured midpoint margin (default 0.25). The Muon-update oracle uses the same eligible set, ranks candidates by a deterministic first-order raw-direction proxy, then tests bounded groups with the exact production `zeropower_newton_schulz`; a group is accepted only on strict post-Muon cosine improvement. Candidate, considered, accepted, and exact-evaluation counts are recorded. This is an offline oracle/headroom study with access to FP32 M and is not a deployable quantizer.\n\nThe search is deterministic and bounded; only final metrics use exact production Muon transforms.\n""")
    plot(args.output,aggregate_rows,tensor_rows,stat_rows); print(json.dumps({"snapshots":len(paths),"runtime_seconds":runtimes,"output":str(args.output)},indent=2))

if __name__ == "__main__": main()
