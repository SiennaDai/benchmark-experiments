#!/usr/bin/env python3
"""Estimate raw mechanism snapshot storage without loading the dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config.recipe import load_recipe  # noqa: E402
from models.llama import Llama  # noqa: E402
from train_platform import model_namespace, parameter_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--recipe", required=True); args = ap.parse_args()
    cfg = load_recipe(args.recipe); model = Llama(model_namespace(cfg)).to(dtype=torch.float32)
    _, records = parameter_groups(model, cfg["optimizer"]["weight_decay"], cfg["optimizer"]["name"])
    scalars = sum(x["numel"] for x in records if x["group"] == "muon")
    # FP32 reference omits persisted_decoded because it equals candidate;
    # recursive VQ retains it to measure persistence error.
    fp_bytes = scalars * 4 * 4
    vq_bytes = scalars * 4 * 5
    raw_count = len(cfg["logging"].get("muon_mechanism_snapshot_updates", []))
    print(json.dumps({"muon_scalars": scalars, "fp32_bytes_per_raw_snapshot": fp_bytes,
                      "vq_bytes_per_raw_snapshot": vq_bytes, "raw_landmarks": raw_count,
                      "fp32_raw_total_bytes": fp_bytes * raw_count,
                      "vq_raw_total_bytes": vq_bytes * raw_count,
                      "paired_raw_total_bytes": (fp_bytes + vq_bytes) * raw_count}, indent=2))


if __name__ == "__main__":
    main()
