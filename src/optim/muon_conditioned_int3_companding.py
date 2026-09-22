"""Offline residual quantizers for structurally conditioned INT3 Muon studies.

Nothing in this module is connected to training or optimizer persistence.  The
production-style uniform INT3 reference delegates to the earlier offline
implementation; alternative quantizers share b2048 block boundaries and
absmax normalization, but remain experimental.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from .muon_quantization_aware_conditioner import (
    BLOCK_SIZE, INT3_CODEBOOK, int3_dynamic_roundtrip as int3_uniform_roundtrip,
)

DANGER_LOG10_RANGE = (-3.0, -2.0)


def mulaw_transform(x: torch.Tensor, mu: float) -> torch.Tensor:
    """Odd, monotone mu-law map from [-1,1] to [-1,1]."""
    if mu <= 0:
        raise ValueError("mu must be positive")
    x = x.float().clamp(-1, 1)
    return x.sign() * torch.log1p(float(mu) * x.abs()) / math.log1p(float(mu))


def mulaw_inverse(y: torch.Tensor, mu: float) -> torch.Tensor:
    """Numerically stable inverse of :func:`mulaw_transform`."""
    if mu <= 0:
        raise ValueError("mu must be positive")
    y = y.float().clamp(-1, 1)
    return y.sign() * torch.expm1(y.abs() * math.log1p(float(mu))) / float(mu)


def build_symmetric_codebook(a1: float, a2: float, *, outer: float = 1.0,
                             device=None) -> torch.Tensor:
    """Return sorted seven-level symmetric INT3 codebook with exact zero."""
    if not (0 < a1 < a2 < outer <= 1.0 + 1e-12):
        raise ValueError("require 0 < a1 < a2 < outer <= 1")
    vals = [-outer, -a2, -a1, 0.0, a1, a2, outer]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def nearest_levels(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    cb = codebook.to(normalized)
    z = normalized.clamp(cb[0], cb[-1]).reshape(-1)
    hi = torch.searchsorted(cb, z).clamp(max=cb.numel() - 1)
    lo = (hi - 1).clamp_min(0)
    # Match production deterministic ties: choose the lower level.
    use_hi = (z - cb[lo]).abs() > (cb[hi] - z).abs()
    return torch.where(use_hi, cb[hi], cb[lo]).reshape_as(normalized)


def _block_ranges(numel: int, block_size: int):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return range(0, numel, block_size)


def int3_codebook_roundtrip(value: torch.Tensor, codebook: torch.Tensor,
                            *, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Blockwise dynamic quantize with absmax scale and an explicit codebook."""
    x = value.detach().float()
    flat = x.reshape(-1)
    out = torch.empty_like(flat)
    for start in _block_ranges(flat.numel(), block_size):
        block = flat[start:start + block_size]
        scale = block.abs().amax()
        if scale.item() == 0:
            out[start:start + block.numel()] = block
        else:
            out[start:start + block.numel()] = nearest_levels(block / scale, codebook) * scale
    return out.reshape_as(x)


def int3_mulaw_roundtrip(value: torch.Tensor, mu: float, *,
                         block_size: int = BLOCK_SIZE,
                         codebook: torch.Tensor = INT3_CODEBOOK) -> torch.Tensor:
    """Absmax-normalized mu-law INT3; invert after codebook dequantization."""
    x = value.detach().float()
    flat = x.reshape(-1)
    out = torch.empty_like(flat)
    for start in _block_ranges(flat.numel(), block_size):
        block = flat[start:start + block_size]
        scale = block.abs().amax()
        if scale.item() == 0:
            out[start:start + block.numel()] = block
        else:
            normalized = (block / scale).clamp(-1, 1)
            yq = nearest_levels(mulaw_transform(normalized, mu), codebook)
            out[start:start + block.numel()] = mulaw_inverse(yq, mu) * scale
    return out.reshape_as(x)


def int3_power_roundtrip(value: torch.Tensor, gamma: float, *,
                         block_size: int = BLOCK_SIZE,
                         codebook: torch.Tensor = INT3_CODEBOOK) -> torch.Tensor:
    """Signed power-law companding, quantization, and exact inverse."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    x = value.detach().float()
    flat, out = x.reshape(-1), torch.empty_like(x.reshape(-1))
    for start in _block_ranges(flat.numel(), block_size):
        block = flat[start:start + block_size]
        scale = block.abs().amax()
        if scale.item() == 0:
            out[start:start + block.numel()] = block
        else:
            z = (block / scale).clamp(-1, 1)
            y = z.sign() * z.abs().pow(float(gamma))
            q = nearest_levels(y, codebook)
            out[start:start + block.numel()] = q.sign() * q.abs().pow(1 / float(gamma)) * scale
    return out.reshape_as(x)


def lloyd_max_codebook(samples: torch.Tensor, *, iterations: int = 40,
                       levels: Iterable[float] | None = None) -> torch.Tensor:
    """Deterministic symmetric seven-level Lloyd-Max fit with exact zero.

    Input samples are expected to be normalized to their block absmax.  The
    symmetric distribution is fit over absolute magnitudes; outer level is
    fixed at one and the two interior positive centroids are iterated.
    """
    x = samples.detach().float().abs().reshape(-1)
    x = x[torch.isfinite(x)].clamp(0, 1)
    if not x.numel():
        return build_symmetric_codebook(1 / 3, 2 / 3)
    a1, a2 = (list(levels) if levels is not None else [1 / 3, 2 / 3])
    a1, a2 = float(a1), float(a2)
    for _ in range(int(iterations)):
        t0, t1, t2 = a1 / 2, (a1 + a2) / 2, (a2 + 1) / 2
        # Zero and the outer endpoint are fixed by design; Lloyd updates only
        # the two interior positive reconstruction levels.
        bins = ((x >= t0) & (x < t1), (x >= t1) & (x < t2))
        new_a1 = float(x[bins[0]].mean()) if bool(bins[0].any()) else a1
        new_a2 = float(x[bins[1]].mean()) if bool(bins[1].any()) else a2
        new_a1 = min(max(new_a1, 1e-4), 0.999)
        new_a2 = min(max(new_a2, new_a1 + 1e-4), 0.9999)
        if abs(new_a1-a1) + abs(new_a2-a2) < 1e-7:
            a1, a2 = new_a1, new_a2
            break
        a1, a2 = new_a1, new_a2
    return build_symmetric_codebook(a1, a2)


def normalized_block_samples(residuals: Iterable[torch.Tensor], *,
                             max_samples: int = 2_000_000,
                             block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Concatenate deterministic evenly-spaced normalized residual samples."""
    chunks = []
    for residual in residuals:
        flat = residual.detach().float().reshape(-1)
        for start in _block_ranges(flat.numel(), block_size):
            block = flat[start:start + block_size]
            scale = block.abs().amax()
            if scale.item(): chunks.append(block / scale)
    if not chunks:
        return torch.zeros(0, dtype=torch.float32)
    values = torch.cat(chunks)
    if values.numel() > max_samples:
        # deterministic systematic subsampling, no random generator required.
        idx = torch.linspace(0, values.numel()-1, max_samples).round().long()
        values = values[idx]
    return values


def optimize_block_scale(block: torch.Tensor, codebook: torch.Tensor = INT3_CODEBOOK,
                         grid: torch.Tensor | None = None) -> tuple[torch.Tensor, float, float]:
    """MSE-optimal scalar scale on a deterministic bounded alpha grid.

    Returns (dequantized block, scale, squared error).  The grid is bounded to
    [0.5, 1.5] times absmax by default; absmax itself is explicitly included.
    """
    x = block.detach().float().reshape(-1)
    maximum = float(x.abs().amax()) if x.numel() else 0.0
    if maximum == 0:
        return block.detach().float().clone(), 0.0, 0.0
    if grid is None:
        frac = torch.tensor([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20, 1.35, 1.50])
    else:
        frac = grid.detach().float().reshape(-1)
    scales = (frac * maximum).tolist() + [maximum]
    best_scale, best_error, best = maximum, math.inf, None
    for scale in scales:
        if scale <= 0:
            continue
        q = nearest_levels(x / scale, codebook) * scale
        error = float((q - x).square().sum())
        if error < best_error - 1e-12 or (abs(error-best_error) <= 1e-12 and scale < best_scale):
            best_scale, best_error, best = float(scale), error, q
    return best.reshape_as(block), best_scale, best_error


def int3_scale_oracle_roundtrip(value: torch.Tensor, *, block_size: int = BLOCK_SIZE,
                                codebook: torch.Tensor = INT3_CODEBOOK) -> tuple[torch.Tensor, list[float]]:
    x = value.detach().float(); flat = x.reshape(-1); out = torch.empty_like(flat); scales=[]
    for start in _block_ranges(flat.numel(), block_size):
        block = flat[start:start+block_size]
        q, scale, _ = optimize_block_scale(block, codebook)
        out[start:start+block.numel()] = q.reshape(-1); scales.append(scale)
    return out.reshape_as(x), scales


def danger_mode_mask(singular_values: torch.Tensor) -> torch.Tensor:
    s = singular_values.detach().float()
    if not s.numel() or s[0].item() <= 0:
        return torch.zeros_like(s, dtype=torch.bool)
    logx = torch.log10((s / s[0]).clamp_min(1e-30))
    return (logx >= DANGER_LOG10_RANGE[0]) & (logx < DANGER_LOG10_RANGE[1])


def spectral_error_metrics(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor,
                           residual_error: torch.Tensor,
                           *, block_size: int = BLOCK_SIZE) -> dict[str, float]:
    """Danger-zone diagonal/cross/associated and local/medium mixing energy."""
    err = residual_error.detach().float()
    eh = u.float().T @ err @ vh.float().T
    total = eh.square().sum().clamp_min(torch.finfo(torch.float32).tiny)
    danger = danger_mode_mask(s).to(eh.device)
    ii = torch.eye(eh.shape[0], dtype=torch.bool, device=eh.device)
    diag = eh.diagonal().square()
    danger_diag = diag[danger].sum()
    rows = danger[:, None].expand_as(eh); cols = danger[None, :].expand_as(eh)
    associated = rows | cols
    cross = associated & ~ii
    dz_cross = eh.square()[cross].sum()
    coords = torch.log10((s.float() / s.float()[0]).clamp_min(1e-6)) if s.numel() else torch.zeros(0)
    dist = (coords[:, None] - coords[None, :]).abs()
    off = ~ii
    local = eh.square()[(dist < .5) & off].sum()
    medium = eh.square()[(dist >= .5) & (dist < 1.5) & off].sum()
    distant = eh.square()[(dist >= 1.5) & off].sum()
    return {"danger_diagonal_fraction_error": float(danger_diag / total),
            "danger_cross_fraction_error": float(dz_cross / total),
            "danger_associated_fraction_error": float((danger_diag + dz_cross) / total),
            "local_mixing_fraction_error": float(local / total),
            "medium_mixing_fraction_error": float(medium / total),
            "distant_mixing_fraction_error": float(distant / total),
            "danger_diagonal_energy": float(danger_diag), "danger_cross_energy": float(dz_cross)}


def geometry_weight_matrix(mode_weights: torch.Tensor, *, combination: str = "geometric") -> torch.Tensor:
    w = mode_weights.detach().float().clamp_min(0)
    if combination == "geometric":
        return torch.sqrt(w[:, None] * w[None, :])
    if combination == "arithmetic":
        return (w[:, None] + w[None, :]) / 2
    raise ValueError("combination must be geometric or arithmetic")


def geometry_surrogate_loss(u: torch.Tensor, vh: torch.Tensor, error: torch.Tensor,
                            mode_weights: torch.Tensor, *,
                            combination: str = "geometric") -> torch.Tensor:
    eh = u.detach().float().T @ error.detach().float() @ vh.detach().float().T
    weights = geometry_weight_matrix(mode_weights.to(eh.device), combination=combination)
    return (weights * eh.square()).sum()


def select_geometry_codebook(calibration_items: list[dict], candidate_pairs: Iterable[tuple[float,float]],
                             *, combination: str = "geometric") -> tuple[float, float, list[dict]]:
    """Select (a1,a2) using only supplied calibration residuals and geometry."""
    scored=[]
    for a1,a2 in candidate_pairs:
        cb=build_symmetric_codebook(a1,a2)
        losses=[]
        for item in calibration_items:
            q=int3_codebook_roundtrip(item["residual"],cb)
            losses.append(float(geometry_surrogate_loss(item["u"],item["vh"],q-item["residual"],item["mode_weights"],combination=combination)))
        scored.append({"a1":float(a1),"a2":float(a2),"mean_geometry_loss":sum(losses)/max(1,len(losses)),"calibration_items":len(losses)})
    if not scored:
        raise ValueError("at least one candidate codebook is required")
    best=min(scored,key=lambda r:(r["mean_geometry_loss"],r["a1"],r["a2"]))
    return best["a1"],best["a2"],scored
