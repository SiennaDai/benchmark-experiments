#!/usr/bin/env python3
"""Microbenchmark the recursive structural codecs without training.

The benchmark uses synthetic matrices with the formal Muon shapes and reports
stage timings.  CUDA timings synchronize around every measurement; CPU runs
remain supported for correctness/development environments without a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optim.muon_recursive import (  # noqa: E402
    StructuralINT4Codec,
    StructuralVQCodec,
    _block_stat_scales,
    _repeat_block_scales,
    pack_indices,
    unpack_indices,
)


SHAPES = ((1152, 384), (384, 384), (1024, 384), (384, 1024))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device: torch.device, repeats: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(device)
    return (time.perf_counter() - started) * 1000.0 / repeats


@torch.no_grad()
def vq_stages(matrix: torch.Tensor, codec: StructuralVQCodec) -> dict[str, callable]:
    value = matrix.float()
    holder: dict[str, torch.Tensor] = {}

    def svd():
        holder["u"], holder["s"], holder["vh"] = torch.linalg.svd(value, full_matrices=False)

    def low_rank():
        u, s, vh = holder["u"], holder["s"], holder["vh"]
        k = min(codec.rank, int(s.numel()))
        holder["residual"] = value - ((u[:, :k] * s[:k]) @ vh[:k] if k else torch.zeros_like(value))

    def scales():
        holder["scales"] = _block_stat_scales(holder["residual"].reshape(-1), codec.block_size, percentile=.98)

    def assignment():
        pairs = holder["residual"].reshape(-1, 2)
        pair_scales = _repeat_block_scales(holder["scales"], pairs.shape[0], codec.block_size // 2)
        normalized = (pairs / pair_scales.clamp_min(torch.finfo(torch.float32).tiny)[:, None]).clamp(-1, 1)
        codebook = codec._codebook_on(value.device)
        chunks = []
        for start in range(0, pairs.shape[0], 65536):
            chunks.append(torch.cdist(normalized[start:start + 65536], codebook).argmin(dim=1))
        holder["indices"] = torch.cat(chunks)

    def packing():
        holder["packed"] = pack_indices(holder["indices"], 6)

    def unpacking():
        holder["unpacked"] = unpack_indices(holder["packed"], holder["indices"].numel())

    return {"svd": svd, "low_rank": low_rank, "p98": scales,
            "assignment": assignment, "pack": packing, "unpack": unpacking}


@torch.no_grad()
def int4_stages(matrix: torch.Tensor, codec: StructuralINT4Codec) -> dict[str, callable]:
    value = matrix.float()
    holder: dict[str, torch.Tensor] = {}

    def svd():
        holder["u"], holder["s"], holder["vh"] = torch.linalg.svd(value, full_matrices=False)

    def low_rank():
        u, s, vh = holder["u"], holder["s"], holder["vh"]
        k = min(codec.rank, int(s.numel()))
        holder["residual"] = value - ((u[:, :k] * s[:k]) @ vh[:k] if k else torch.zeros_like(value))

    def scales():
        holder["scales"] = _block_stat_scales(holder["residual"].reshape(-1), codec.block_size)

    def assignment():
        flat = holder["residual"].reshape(-1)
        scales = _repeat_block_scales(holder["scales"], flat.numel(), codec.block_size)
        normalized = (flat / scales.clamp_min(torch.finfo(torch.float32).tiny)).reshape(-1, 1)
        codebook = codec._map_on(value.device).reshape(-1, 1)
        chunks = []
        for start in range(0, flat.numel(), 65536):
            chunks.append(torch.cdist(normalized[start:start + 65536], codebook).argmin(dim=1))
        holder["codes"] = torch.cat(chunks)

    def packing():
        raw = holder["codes"].to(torch.int64)
        packed = torch.empty(((raw.numel() + 1) // 2,), dtype=torch.uint8, device=value.device)
        pairs = raw.numel() // 2
        if pairs:
            packed[:pairs] = (raw[0::2][:pairs] | (raw[1::2][:pairs] << 4)).to(torch.uint8)
        if raw.numel() % 2:
            packed[-1] = raw[-1].to(torch.uint8)
        holder["packed"] = packed

    def unpacking():
        packed = holder["packed"].to(torch.int64)
        count = holder["codes"].numel()
        raw = torch.empty(count, dtype=torch.long, device=value.device)
        pairs = count // 2
        if pairs:
            raw[0:2 * pairs:2] = packed[:pairs] & 15
            raw[1:2 * pairs:2] = (packed[:pairs] >> 4) & 15
        if count % 2:
            raw[-1] = packed[-1] & 15
        holder["unpacked"] = raw

    return {"svd": svd, "low_rank": low_rank, "absmax": scales,
            "assignment": assignment, "pack": packing, "unpack": unpacking}


@torch.no_grad()
def run_method(name: str, codec, shape: tuple[int, int], device: torch.device, repeats: int, seed: int):
    torch.manual_seed(seed)
    matrix = torch.randn(shape, device=device, dtype=torch.float32)
    row = {"method": name, "shape": list(shape), "device": str(device)}
    row["total_encode_ms"] = timed(lambda: codec.encode(matrix), device, repeats)
    stages = vq_stages(matrix, codec) if name == "vq_int3" else int4_stages(matrix, codec)
    for key in ("svd", "low_rank", "p98" if name == "vq_int3" else "absmax", "assignment", "pack", "unpack"):
        row[f"{key}_ms"] = timed(stages[key], device, repeats)
    encoded = codec.encode(matrix)
    row["total_decode_ms"] = timed(lambda: codec.decode(encoded, device=device), device, repeats)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--codebook", type=Path, default=ROOT / "reports/muon_vector_int3_robustness/calibration_codebooks.pt")
    parser.add_argument("--codebook-key", default="s1_k8_w64_t8_v1200")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    blob = torch.load(args.codebook, map_location="cpu", weights_only=False)
    codebook = blob["codebooks"][args.codebook_key]
    codecs = {"int4": StructuralINT4Codec(rank=8, block_size=2048),
              "vq_int3": StructuralVQCodec(codebook, rank=8, block_size=2048, codebook_key=args.codebook_key)}
    rows = []
    for shape in SHAPES:
        for name, codec in codecs.items():
            rows.append(run_method(name, codec, shape, device, args.repeats, args.seed))
    print(json.dumps(rows, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
