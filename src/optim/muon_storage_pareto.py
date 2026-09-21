"""Storage accounting and Pareto helpers for the structural Muon study.

This module is analysis-only.  It contains no optimizer or persistence hooks;
all numbers are idealized state-storage estimates and deliberately distinguish
payload bits from the small amount of metadata needed to decode a tensor.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


BLOCK_SIZE = 2048
INT4_BITS = 4
FP32_BITS = 32
BF16_BITS = 16
FP16_BITS = 16

# U/V precision and sigma precision.  BF16 and FP16 have identical idealized
# bit counts but remain separate labels because their numerical round trips do
# differ.
FACTOR_VARIANTS = {
    "fp32": (32, 32, 32),
    "bf16": (16, 16, 16),
    "fp16": (16, 16, 16),
    "bf16_uv_fp32_sigma": (16, 32, 16),
    "fp16_uv_fp32_sigma": (16, 32, 16),
}


def factor_bits(variant: str) -> tuple[int, int, int]:
    if variant not in FACTOR_VARIANTS:
        raise ValueError(f"unknown factor precision variant: {variant}")
    return FACTOR_VARIANTS[variant]


def block_count(numel: int, block_size: int = BLOCK_SIZE) -> int:
    if numel < 0 or block_size <= 0:
        raise ValueError("numel must be non-negative and block_size positive")
    return (int(numel) + int(block_size) - 1) // int(block_size)


def metadata_bits(shape: Iterable[int], rank: int, variant: str, *,
                  include: bool = True, block_size: int = BLOCK_SIZE) -> int:
    """Return explicit, conservative decoder metadata bits.

    This is not a kernel layout.  It accounts only for dimensions (two uint32s),
    rank (uint32), factor precision labels (three uint8s), quantizer family and
    block size (two uint16s), plus one FP32 scale per dynamic INT4 block.
    """
    if not include:
        return 0
    shape = tuple(int(x) for x in shape)
    if len(shape) != 2:
        raise ValueError("storage accounting requires a matrix shape")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    # dimensions/rank=96, three precision ids=24, quantizer+block ids=32.
    fixed = 96 + 24 + 32
    scales = 32 * block_count(shape[0] * shape[1], block_size)
    return fixed + scales


def storage_bits(shape: Iterable[int], rank: int, variant: str = "fp32", *,
                 include_metadata: bool = False, block_size: int = BLOCK_SIZE,
                 residual_bits: int = INT4_BITS) -> dict[str, int | float]:
    """Compute idealized and metadata-inclusive structural storage.

    The residual always occupies a dense INT4 tensor and uses one FP32 absmax
    scale per b2048 block only in the metadata-inclusive estimate.  Low-rank
    side information stores U (m*k), sigma (k), and V (n*k).
    """
    shape = tuple(int(x) for x in shape)
    if len(shape) != 2:
        raise ValueError("storage accounting requires a 2D shape")
    m, n = shape
    k = int(rank)
    if k < 0 or k > min(m, n):
        raise ValueError("rank outside matrix dimensions")
    bu, bs, bv = factor_bits(variant)
    residual_payload = int(residual_bits) * m * n
    factor_payload = bu * m * k + bs * k + bv * n * k
    idealized = residual_payload + factor_payload
    meta = metadata_bits(shape, k, variant, include=include_metadata,
                         block_size=block_size)
    # metadata_bits includes scales only for the structural residual.
    total = idealized + meta
    return {
        "m": m, "n": n, "numel": m * n, "rank": k,
        "residual_payload_bits": residual_payload,
        "factor_payload_bits": factor_payload,
        "metadata_bits": meta,
        "idealized_bits": idealized,
        "metadata_inclusive_bits": total,
        "fp32_bits": FP32_BITS * m * n,
        "direct_int4_idealized_bits": INT4_BITS * m * n,
        "direct_int4_metadata_bits": INT4_BITS * m * n + metadata_bits(
            shape, 0, variant, include=include_metadata, block_size=block_size),
    }


def direct_storage_bits(shape: Iterable[int], *, include_metadata: bool = False,
                        block_size: int = BLOCK_SIZE) -> dict[str, int]:
    shape = tuple(int(x) for x in shape)
    if len(shape) != 2:
        raise ValueError("storage accounting requires a 2D shape")
    n = shape[0] * shape[1]
    meta = metadata_bits(shape, 0, "fp32", include=include_metadata,
                         block_size=block_size)
    return {"fp32_bits": FP32_BITS * n,
            "int4_idealized_bits": INT4_BITS * n,
            "int4_metadata_inclusive_bits": INT4_BITS * n + meta}


def cast_factors(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor,
                 variant: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-trip retained factors at the requested storage precision."""
    bu, bs, bv = factor_bits(variant)
    du = torch.float32 if bu == 32 else (torch.bfloat16 if variant.startswith("bf16") else torch.float16)
    dv = torch.float32 if bv == 32 else (torch.bfloat16 if variant.startswith("bf16") else torch.float16)
    ds = torch.float32 if bs == 32 else (torch.bfloat16 if variant.startswith("bf16") else torch.float16)
    return u.to(du).float(), s.to(ds).float(), vh.to(dv).float()


def pareto_mask(points: list[dict], x: str = "storage_ratio_vs_fp32",
                y: str = "update_cosine", *, higher_is_better: bool = True) -> list[bool]:
    """Return non-dominated points, minimizing ``x`` and optimizing ``y``."""
    out = [False] * len(points)
    valid = []
    for i, p in enumerate(points):
        try:
            xv, yv = float(p[x]), float(p[y])
            if math.isfinite(xv) and math.isfinite(yv): valid.append((i, xv, yv))
        except (KeyError, TypeError, ValueError):
            pass
    for i, xi, yi in valid:
        dominated = any(
            j != i and xj <= xi
            and ((yj >= yi) if higher_is_better else (yj <= yi))
            and (xj < xi or ((yj > yi) if higher_is_better else (yj < yi)))
            for j, xj, yj in valid
        )
        out[i] = not dominated
    return out


def minimum_rank_for_target(rows: list[dict], target: float,
                            *, rank_key: str = "rank", fidelity_key: str = "update_cosine") -> dict | None:
    valid = []
    for row in rows:
        try:
            rank, fidelity = int(row[rank_key]), float(row[fidelity_key])
            if math.isfinite(fidelity) and fidelity >= target:
                valid.append((rank, row))
        except (KeyError, TypeError, ValueError):
            continue
    return min(valid, key=lambda x: x[0])[1] if valid else None
