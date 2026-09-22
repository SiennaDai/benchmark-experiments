"""Offline block-scale rules for structurally conditioned INT3 Muon studies.

These helpers are analysis-only.  They quantize detached FP32 values with a
fixed scalar codebook and do not connect to optimizer state or persistence.
The scale is a positive block-local clipping bound; values are clipped to the
outer codebook levels before nearest-level assignment and dequantization.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from .muon_conditioned_int3_companding import INT3_CODEBOOK, nearest_levels
from .muon_quantization_aware_conditioner import BLOCK_SIZE


def _blocks(value: torch.Tensor, block_size: int):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    flat = value.detach().float().reshape(-1)
    return flat, [flat[i:i + block_size] for i in range(0, flat.numel(), block_size)]


def _positive_scale(scale: torch.Tensor) -> torch.Tensor:
    # Zero blocks are explicitly handled by the caller. Clamp only nonzero
    # blocks to the smallest positive representable FP32 value.
    return scale.clamp_min(torch.finfo(torch.float32).tiny)


@torch.no_grad()
def block_scales(value: torch.Tensor, method: str, *, multiplier: float = 1.0,
                 percentile: float = 100.0, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Estimate one deterministic positive scale for each flattened block.

    Supported methods: ``absmax``, ``percentile``, ``rms``, ``std`` (population
    standard deviation), ``median_abs``, and normal-consistent ``mad``.
    Returned scales are FP32 on the input device; a zero block has scale 0.
    """
    if multiplier <= 0 or not math.isfinite(multiplier):
        raise ValueError("multiplier must be finite and positive")
    if not (0.0 < percentile <= 100.0):
        raise ValueError("percentile must be in (0, 100]")
    _, blocks = _blocks(value, block_size)
    out = []
    for block in blocks:
        if not block.numel() or not bool(block.abs().any()):
            out.append(block.new_zeros(())); continue
        if method == "absmax":
            scale = block.abs().amax()
        elif method == "percentile":
            scale = torch.quantile(block.abs(), percentile / 100.0)
        elif method == "rms":
            scale = block.square().mean().sqrt()
        elif method == "std":
            scale = block.std(unbiased=False)
        elif method == "median_abs":
            scale = block.abs().median()
        elif method == "mad":
            med = block.median()
            scale = (block - med).abs().median() * 1.482602218505602
        else:
            raise ValueError(f"unsupported block scale method: {method}")
        out.append(_positive_scale(scale * float(multiplier)))
    return torch.stack(out) if out else torch.empty(0, dtype=torch.float32, device=value.device)


@torch.no_grad()
def fixed_codebook_roundtrip(value: torch.Tensor, scales: torch.Tensor,
                             codebook: torch.Tensor = INT3_CODEBOOK, *,
                             block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Quantize each flattened block using supplied alpha values.

    The normalized value is clipped to ``[codebook[0], codebook[-1]]``;
    nearest ties select the lower level, matching existing INT3 helpers.
    All-zero blocks map exactly to zero regardless of their supplied scale.
    """
    flat, blocks = _blocks(value, block_size)
    scales = scales.detach().to(device=flat.device, dtype=torch.float32).reshape(-1)
    if scales.numel() != len(blocks):
        raise ValueError(f"expected {len(blocks)} block scales, got {scales.numel()}")
    if bool((~torch.isfinite(scales)).any()) or bool((scales < 0).any()):
        raise ValueError("scales must be finite and nonnegative")
    cb = codebook.detach().to(device=flat.device, dtype=torch.float32).reshape(-1)
    if cb.numel() < 2 or not bool((cb[1:] > cb[:-1]).all()):
        raise ValueError("codebook must be strictly increasing")
    out = torch.empty_like(flat)
    start = 0
    for block, alpha in zip(blocks, scales):
        n = block.numel()
        if not n or not bool(block.abs().any()):
            out[start:start+n] = 0
        else:
            if alpha.item() <= 0:
                raise ValueError("nonzero blocks require positive scales")
            out[start:start+n] = nearest_levels(block / alpha, cb) * alpha
        start += n
    return out.reshape_as(value).to(dtype=torch.float32)


@torch.no_grad()
def select_scales(value: torch.Tensor, codebook: torch.Tensor = INT3_CODEBOOK, *,
                  method: str = "local_mse", grid: Iterable[float] = (0.6, 0.75, 0.9, 1.0),
                  block_size: int = BLOCK_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Select per-block scales by deterministic residual-MSE grid search.

    ``grid`` contains positive multipliers of each block's absmax. Ties choose
    the smaller alpha. The objective is block-local scalar reconstruction MSE;
    no optimizer transform or update-fidelity quantity is used.
    """
    if method != "local_mse":
        raise ValueError("only local_mse scale selection is supported")
    fractions = tuple(float(x) for x in grid)
    if not fractions or any(not math.isfinite(x) or x <= 0 for x in fractions):
        raise ValueError("grid must contain finite positive scale multipliers")
    flat, blocks = _blocks(value, block_size)
    cb = codebook.detach().to(device=flat.device, dtype=torch.float32)
    out = torch.empty_like(flat); chosen = []
    start = 0
    for block in blocks:
        n = block.numel()
        maximum = block.abs().amax() if n else block.new_zeros(())
        if not n or maximum.item() == 0:
            chosen.append(maximum); out[start:start+n] = 0; start += n; continue
        scales = sorted(set(float(f) * float(maximum) for f in fractions))
        best_error = math.inf; best_alpha = scales[0]; best_q = None
        for alpha in scales:
            q = nearest_levels(block / alpha, cb) * alpha
            error = float((q - block).square().sum())
            if error < best_error - 1e-12 or (abs(error - best_error) <= 1e-12 and alpha < best_alpha):
                best_error, best_alpha, best_q = error, alpha, q
        out[start:start+n] = best_q
        chosen.append(block.new_tensor(best_alpha)); start += n
    scales = torch.stack(chosen) if chosen else torch.empty(0, dtype=torch.float32, device=flat.device)
    return out.reshape_as(value), scales


@torch.no_grad()
def grid_select_scales(value: torch.Tensor, codebook: torch.Tensor = INT3_CODEBOOK, *,
                       grid: Iterable[float], block_size: int = BLOCK_SIZE,
                       return_block_sse: bool = False):
    """Vectorized block-local MSE grid selection.

    Each candidate is a multiplier of that block's absmax. Candidate/block
    SSEs are accumulated tensorwise, then the selected reconstruction is
    recomputed once at the chosen scale. Ties deterministically prefer the
    smaller scale.
    """
    fractions = tuple(float(x) for x in grid)
    if not fractions or any(not math.isfinite(x) or x <= 0 for x in fractions):
        raise ValueError("grid must contain finite positive multipliers")
    x = value.detach().float()
    flat = x.reshape(-1)
    nblocks = (flat.numel() + block_size - 1) // block_size
    if nblocks == 0:
        empty = torch.empty(0, dtype=torch.float32, device=x.device)
        return x.clone(), empty, empty if return_block_sse else None
    padded = torch.zeros((nblocks, block_size), dtype=torch.float32, device=x.device)
    for i, start in enumerate(range(0, flat.numel(), block_size)):
        b = flat[start:start + block_size]
        padded[i, :b.numel()] = b
    maxima = padded.abs().amax(dim=1)
    cb = codebook.detach().to(device=x.device, dtype=torch.float32).reshape(-1)
    # sorted(set()) makes both tie-breaking and output independent of input
    # ordering or duplicated candidates.
    scales = torch.tensor(sorted(set(float(f) for f in fractions)), dtype=torch.float32, device=x.device)[:, None] * maxima[None, :]
    sse = torch.empty_like(scales)
    for ci in range(scales.shape[0]):
        alpha = scales[ci].clamp_min(torch.finfo(torch.float32).tiny)[:, None]
        z = (padded / alpha).clamp(cb[0], cb[-1])
        q = nearest_levels(z, cb) * alpha
        sse[ci] = (q - padded).square().sum(dim=1)
    # torch.argmin selects the first (smallest-scale) tied candidate.
    best = sse.argmin(dim=0)
    chosen = scales.gather(0, best[None, :]).squeeze(0)
    chosen = torch.where(maxima == 0, torch.zeros_like(chosen), chosen)
    out = fixed_codebook_roundtrip(x, chosen, cb, block_size=block_size)
    selected_sse = sse.gather(0, best[None, :]).squeeze(0)
    selected_sse = torch.where(maxima == 0, torch.zeros_like(selected_sse), selected_sse)
    return out, chosen, selected_sse if return_block_sse else None


def block_diagnostics(value: torch.Tensor, reconstruction: torch.Tensor,
                      scales: torch.Tensor, *, block_size: int = BLOCK_SIZE) -> dict[str, float | int]:
    """Aggregate zero/clipping/error diagnostics for a fixed-scale result."""
    x, blocks = _blocks(value, block_size)
    q = reconstruction.detach().float().reshape(-1)
    if q.numel() != x.numel():
        raise ValueError("reconstruction shape mismatch")
    scales = scales.detach().float().reshape(-1)
    zero = clipped = total = 0
    bucket_error = {"small_lt_001": [], "small_001_005": [], "middle_005_025": [], "large_ge_025": []}
    lt_005 = []
    abs_error = (q - x).abs()
    for i, block in enumerate(blocks):
        alpha = scales[i]
        n = block.numel(); total += n
        zero += int((q[sum(b.numel() for b in blocks[:i]):sum(b.numel() for b in blocks[:i+1])] == 0).sum())
        if alpha.item() > 0:
            clipped += int((block.abs() > alpha).sum())
        ratio = block.abs() / block.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        start = sum(b.numel() for b in blocks[:i]); err = abs_error[start:start+n]
        if bool((ratio < .05).any()): lt_005.append(err[ratio < .05])
        for name, mask in (("small_lt_001", ratio < .01), ("small_001_005", (ratio >= .01) & (ratio < .05)),
                           ("middle_005_025", (ratio >= .05) & (ratio < .25)), ("large_ge_025", ratio >= .25)):
            if bool(mask.any()): bucket_error[name].append(err[mask])
    return {"zero_fraction": zero / total if total else 0.0,
            "clipping_fraction": clipped / total if total else 0.0,
            "mae": float(abs_error.mean()) if total else 0.0,
            "median_ae": float(abs_error.median()) if total else 0.0,
            "mse": float(abs_error.square().mean()) if total else 0.0,
            **{f"{name}_mae": float(torch.cat(vals).mean()) if vals else None
               for name, vals in bucket_error.items()},
            "small_lt_005_mae": float(torch.cat(lt_005).mean()) if lt_005 else None,
            "block_count": len(blocks)}
