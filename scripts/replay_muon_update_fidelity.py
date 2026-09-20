#!/usr/bin/env python3
"""Offline, read-only replay of supported Muon momentum quantizers."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optim.muon_update_fidelity import QUANTIZERS, analyze_tensors, load_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantizers", nargs="+", choices=sorted(QUANTIZERS), default=sorted(QUANTIZERS))
    args = parser.parse_args()
    snapshot = load_snapshot(args.snapshot)
    metadata = snapshot["metadata"]
    ns = metadata.get("muon_transform", {})
    rows = analyze_tensors(snapshot["tensors"], quantizers=args.quantizers,
                           ns_steps=ns.get("steps", 5), ns_coefficients=ns.get("coefficients", (3.4445, -4.7750, 2.0315)),
                           ns_eps=ns.get("eps", 1e-7))
    with args.output.open("w") as output:
        for row in rows:
            row.update({"snapshot_update": metadata.get("update"), "snapshot": str(args.snapshot)})
            output.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
