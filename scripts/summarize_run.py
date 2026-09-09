#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from reporting import MIB, summarize_run

def val(value, digits=3): return "NA" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)
def text(s):
    return "\n".join([f"Run: {s['run_id']}", f"Status: {s['status']}", f"Optimizer: {s['optimizer_name']}",
        f"Updates: {s['completed_updates']}/{s['total_updates']}", f"Tokens: {s['processed_target_tokens']}",
        f"Initial val NLL: {val(s['initial_validation_nll'])}", f"Final val NLL: {val(s['final_validation_nll'])}",
        f"Best val NLL: {val(s['best_validation_nll'])} @ {val(s['best_validation_update'])}",
        f"Elapsed: {val(s['total_elapsed_seconds'])} s", f"Median update: {val(s['median_update_seconds'] * 1000 if s['median_update_seconds'] else None)} ms",
        f"Tokens/s: {val(s['tokens_per_second'])}", f"Peak GPU: {val(s['cuda_peak_allocated_bytes'] / MIB if s['cuda_peak_allocated_bytes'] is not None else None)} MiB",
        f"Optimizer state: {val(s['optimizer_state_bytes'] / MIB if s['optimizer_state_bytes'] is not None else None)} MiB"]) + "\n"
def main():
    p=argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--output"); a=p.parse_args(); report=text(summarize_run(a.run)); print(report, end="")
    if a.output: Path(a.output).write_text("# Run summary\n\n```text\n" + report + "```\n")
if __name__ == "__main__": main()
