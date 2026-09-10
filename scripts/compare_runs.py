#!/usr/bin/env python3
"""Scientifically guarded, deterministic comparison of run artifacts."""
import argparse, csv, json, sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from reporting import MIB, paired_trajectory_summary, replication_summary, scientific_differences, summarize_run

FIELDS=["run","run_id","status","optimizer_name","seed","recipe_fingerprint","data_fingerprint","completed_updates","processed_target_tokens","initial_validation_nll","final_validation_nll","best_validation_nll","best_validation_update","total_elapsed_seconds","median_update_seconds","tokens_per_second","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes","optimizer_state_bytes"]
def cell(v): return "NA" if v is None else str(v)
def label(s): return s["optimizer_name"] or s["run_name"]
def table(rows):
    headers=["Run","Optimizer","Status","Final val NLL","Best val NLL","Tokens/s","Peak GPU MiB","Opt state MiB"]
    values=[]
    for s in rows:
        values.append([s["run_name"], cell(s["optimizer_name"]), cell(s["status"]), cell(s["final_validation_nll"]), cell(s["best_validation_nll"]), cell(s["tokens_per_second"]), cell(s["cuda_peak_allocated_bytes"] / MIB if s["cuda_peak_allocated_bytes"] is not None else None), cell(s["optimizer_state_bytes"] / MIB if s["optimizer_state_bytes"] is not None else None)])
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join("| " + " | ".join(row) + " |" for row in values) + "\n"
def plot(rows, output, x, filename, xlabel):
    plt.figure()
    for s in rows:
        points=s["events"]["eval"]
        if points: plt.plot([p[x] for p in points], [p["nll"] for p in points], marker="o", label=label(s))
    plt.xlabel(xlabel); plt.ylabel("validation NLL (nats/token)"); plt.legend(); plt.tight_layout(); plt.savefig(output/filename); plt.close()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--vary", nargs="*", default=[]); p.add_argument("--output", required=True); p.add_argument("--replication-json"); p.add_argument("--trajectory-json"); a=p.parse_args()
    # Argument order is meaningful: suite definitions deliberately prescribe it.
    runs=[Path(x).resolve() for x in a.runs]; cfgs=[json.loads((r/"resolved_config.json").read_text()) for r in runs]
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); differences=scientific_differences(cfgs,runs,a.vary); (out/"differences.json").write_text(json.dumps(differences,indent=2)+"\n")
    if differences: raise SystemExit("scientific conditions differ; see differences.json")
    rows=[summarize_run(r) for r in runs]
    replication = replication_summary(rows, cfgs, json.loads(a.replication_json)) if a.replication_json else None
    if replication: (out/"replication.json").write_text(json.dumps(replication, indent=2)+"\n")
    trajectory = paired_trajectory_summary(rows, cfgs, json.loads(a.trajectory_json)) if a.trajectory_json else None
    if trajectory: (out/"paired_trajectory.json").write_text(json.dumps(trajectory, indent=2)+"\n")
    with (out/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader(); writer.writerows([{k:s.get(k) for k in FIELDS} for s in rows])
    (out/"comparison.md").write_text(table(rows))
    plot(rows,out,"processed_target_tokens","loss_vs_tokens.png","processed target tokens")
    if all(all("wall_clock_elapsed_seconds" in p for p in s["events"]["eval"]) for s in rows):
        plot(rows,out,"wall_clock_elapsed_seconds","loss_vs_time.png","cumulative wall-clock seconds")
    else:
        (out/"loss_vs_time.unavailable.txt").write_text("Legacy event logs do not contain cumulative wall-clock timestamps; no potentially misleading time plot was generated.\n")
    plt.figure(); names=[s["run_name"] for s in rows]; values=[s["optimizer_state_bytes"] / MIB if s["optimizer_state_bytes"] is not None else 0 for s in rows]; plt.bar(names,values); plt.ylabel("optimizer state storage (MiB)"); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(out/"memory_comparison.png"); plt.close()
    reference=rows[0]; deltas=[]
    for s in rows[1:]:
        for key,title in (("final_validation_nll","final val NLL"),("best_validation_nll","best val NLL"),("optimizer_state_bytes","optimizer state bytes"),("tokens_per_second","tokens/s")):
            if s[key] is not None and reference[key] is not None: deltas.append(f"- {s['run_name']} vs {reference['run_name']}: {title} Δ {s[key]-reference[key]:.6g}")
    caveats=[]
    if any(s["status"] != "completed" for s in rows): caveats.append("One or more runs are not completed.")
    if len({s["seed"] for s in rows}) == 1: caveats.append("Single seed; values are descriptive, not a multi-seed estimate.")
    if not (out/"loss_vs_time.png").exists(): caveats.append("Legacy artifacts do not contain cumulative wall-clock timestamps; loss-vs-time is intentionally unavailable.")
    meta=cfgs[0]
    replication_text = ""
    if replication:
        replication_text = "\n## Replication aggregation\n\n" + "Treatment aggregation is descriptive; no significance tests, confidence intervals, or significance claims are computed.\n\n"
        for treatment, values in replication["aggregates"].items():
            f, b = values["final_validation_nll"], values["best_validation_nll"]
            d = values.get("paired_delta_final_validation_nll", {})
            replication_text += f"- `{treatment}`: n={values['n']}; final val NLL mean={f['mean']}, sample std={f['sample_std']}; best val NLL mean={b['mean']}, sample std={b['sample_std']}; paired final Δ mean={d.get('mean')}, sample std={d.get('sample_std')}, n={d.get('n', 0)}\n"
        replication_text += "\nPaired final validation NLL deltas (treatment − control):\n\n"
        for pair in replication["pairs"]:
            replication_text += f"- group `{pair['group']}`, treatment `{pair['treatment']}`: Δ={pair['delta_final_validation_nll']:.6g}\n"
    trajectory_text = ""
    if trajectory:
        trajectory_text = "\n## Paired trajectory\n\nObserved validation-NLL deltas are treatment − control; null denotes a missing evaluation event and no interpolation or interpretation is performed.\n\n"
        for pair in trajectory["pairs"]:
            trajectory_text += f"Group `{pair['group']}`, treatment `{pair['treatment']}`:\n\n| Update | Control NLL | Treatment NLL | Paired Δ |\n|---|---|---|---|\n"
            for point in pair["landmarks"]:
                trajectory_text += f"| {point['update']} | {cell(point['control_validation_nll'])} | {cell(point['treatment_validation_nll'])} | {cell(point['paired_delta_validation_nll'])} |\n"
    report="# Benchmark summary\n\n## Benchmark\n\n"+f"- Path: `{out}`\n- Protocol: `{meta['experiment']['protocol_id']}`\n- Varied fields: {', '.join(a.vary)}\n- Runs: {len(rows)}\n\n## Scientific consistency\n\nAll non-varied scientific conditions matched.\n\n## Results\n\n"+table(rows)+replication_text+trajectory_text+"\n## Key deltas\n\n"+("\n".join(deltas) if deltas else "- Reference run only.")+"\n\n## Artifacts\n\n- `comparison.csv`, `comparison.md`, `replication.json`, `paired_trajectory.json`, `loss_vs_tokens.png`, `memory_comparison.png`\n\n## Caveats\n\n"+"\n".join(f"- {x}" for x in caveats)+"\n"
    (out/"benchmark_summary.md").write_text(report); print(out)
if __name__=="__main__": main()
