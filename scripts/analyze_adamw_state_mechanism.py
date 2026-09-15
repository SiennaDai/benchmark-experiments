#!/usr/bin/env python3
"""Report-only analysis of short AdamW state-persistence diagnostic artifacts."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt

def events(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def finite(v): return isinstance(v, (int,float))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--runs", nargs=3, required=True); p.add_argument("--output", required=True, type=Path); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    rows=[]; tensors=[]
    for directory in map(Path,a.runs):
        cfg=json.loads((directory/"resolved_config.json").read_text()); label=cfg["experiment"]["name"]
        for e in events(directory/"state_diagnostics.jsonl"):
            if e["event_type"] == "state_diagnostics":
                v=e["second_moment"]; u=e["actual_update"]
                rows.append({"condition":label,"update":e["completed_updates"],"processed_target_tokens":e["processed_target_tokens"], **{f"v_{k}":x for k,x in v.items()}, **u})
            elif e["event_type"] == "state_diagnostics_tensor":
                tensors.append({"condition":label,"update":e["completed_updates"], **e})
    keys=sorted({k for r in rows for k in r});
    with (a.output/"state_diagnostics.csv").open("w",newline="") as f: w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    keys=sorted({k for r in tensors for k in r});
    with (a.output/"state_tensor_rankings.csv").open("w",newline="") as f: w=csv.DictWriter(f,keys);w.writeheader();w.writerows(tensors)
    def figure(field, filename, ylabel, log=False):
        plt.figure(figsize=(7,4.5))
        for c in sorted(set(r["condition"] for r in rows)):
            x=[r["update"] for r in rows if r["condition"]==c and finite(r.get(field))]; y=[r[field] for r in rows if r["condition"]==c and finite(r.get(field))]
            plt.plot(x,y,label=c,marker=".")
        if log: plt.yscale("log")
        plt.xlabel("Update");plt.ylabel(ylabel);plt.legend(fontsize=7);plt.tight_layout();plt.savefig(a.output/filename,dpi=180);plt.close()
    figure("v_post_quant_zero_fraction","second_moment_zero_fraction.png","Post-persistence exp_avg_sq zero fraction")
    figure("v_global_relative_l2_quantization_error","second_moment_quant_error.png","Relative L2 quantization error",True)
    figure("v_amplification_p99","inverse_denom_amplification.png","Inverse-denominator amplification p99",True)
    figure("global_parameter_update_l2_norm","parameter_update_norm.png","Parameter update L2 norm",True)
    plt.figure(figsize=(7,4.5))
    for c in sorted(set(r["condition"] for r in rows)):
        rs=[r for r in rows if r["condition"]==c and finite(r.get("train_nll")) and finite(r.get("pre_clip_grad_norm"))]
        plt.plot([r["update"] for r in rs],[r["train_nll"] for r in rs],label=f"{c} train NLL")
        plt.plot([r["update"] for r in rs],[r["pre_clip_grad_norm"] for r in rs],linestyle="--",label=f"{c} grad norm")
    plt.yscale("log");plt.xlabel("Update");plt.ylabel("Value (log)");plt.legend(fontsize=6,ncol=2);plt.tight_layout();plt.savefig(a.output/"loss_and_grad_precursors.png",dpi=180);plt.close()
    with (a.output/"mechanism_summary.md").open("w") as f:
        f.write("# AdamW INT8 state mechanism diagnostic\n\nThis report is descriptive. State events observe detached FP32 moments after their update and compare them with the post-persistence (next-step) state. The denominator amplification is a persistence diagnostic, not the current-step update denominator.\n\n")
        f.write("## Direct measurements\n\nSee `state_diagnostics.csv` for exact per-update aggregates and `state_tensor_rankings.csv` for selected landmark tensor summaries. Figures show zeroing, quantization error, denominator amplification, parameter-update magnitude, and loss/gradient trajectories.\n\n")
        f.write("## Interpretation boundary\n\nObserved state distortion and subsequent trajectory changes can support a mechanism hypothesis, but this report does not establish causality or attribute behavior to a specific parameter group without further analysis.\n")
if __name__ == "__main__": main()
