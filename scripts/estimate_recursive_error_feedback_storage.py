#!/usr/bin/env python3
"""Estimate persistent recursive-VQ and FP32 error-feedback payload bytes."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config.recipe import load_recipe  # noqa: E402
from models.llama import Llama  # noqa: E402
from train_platform import model_namespace, parameter_groups  # noqa: E402


def estimate(cfg: dict) -> dict:
    model = Llama(model_namespace(cfg)).to(dtype=torch.float32)
    _, records = parameter_groups(model, cfg["optimizer"]["weight_decay"], "recursive_muon")
    rank = int(cfg["optimizer"].get("recursive_rank", 8))
    block = int(cfg["optimizer"].get("recursive_block_size", 2048))
    tensors = [row for row in records if row["group"] == "muon"]
    rows = []
    for row in tensors:
        m, n = row["shape"]
        k = min(rank, m, n)
        values = int(row["numel"])
        if values % 2:
            raise ValueError(f"recursive VQ requires even-sized tensor: {row['name']}")
        factors = 2 * (m * k + k + k * n)  # BF16 U, sigma, Vh
        scales = 4 * math.ceil(values / block)  # FP32 p98 scales
        indices = math.ceil(values * 3 / 8)  # exactly 3 bits per residual scalar
        rows.append({"name": row["name"], "shape": row["shape"], "numel": values,
                     "factor_bytes": factors, "scale_bytes": scales,
                     "index_bytes": indices,
                     "vq_payload_bytes": factors + scales + indices})
    scalar_count = sum(row["numel"] for row in tensors)
    vq_payload = sum(row["vq_payload_bytes"] for row in rows) + 64 * 2 * 4
    mode = cfg["optimizer"].get("recursive_error_feedback_mode", "fractional" if cfg["optimizer"].get("recursive_error_feedback_alpha", 0.0) > 0 else "none")
    error_bytes = scalar_count * 4 if mode == "fractional" else 0
    periodic_accumulator_bytes = scalar_count * 4 if mode == "periodic" else 0
    feedback_bytes = error_bytes + periodic_accumulator_bytes
    return {"recipe": cfg["experiment"]["name"], "seed": cfg["experiment"]["seed"],
            "muon_tensor_count": len(rows), "muon_scalar_count": scalar_count,
            "fp32_muon_state_bytes": scalar_count * 4,
            "vq_payload_bytes_including_shared_codebook": vq_payload,
            "vq_effective_bits_per_value": 8 * vq_payload / scalar_count,
            "fp32_error_buffer_bytes": error_bytes,
            "periodic_accumulator_bytes": periodic_accumulator_bytes,
            "periodic_interval": cfg["optimizer"].get("recursive_error_feedback_interval"),
            "error_buffer_bits_per_value": 8 * feedback_bytes / scalar_count,
            "oracle_total_bytes": vq_payload + feedback_bytes,
            "oracle_total_effective_bits_per_value": 8 * (vq_payload + feedback_bytes) / scalar_count,
            "serialization_note": "payload estimate excludes checkpoint/container overhead; runtime summary reports actual logical payload",
            "tensors": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    args = parser.parse_args()
    print(json.dumps(estimate(load_recipe(args.recipe)), indent=2))


if __name__ == "__main__":
    main()
