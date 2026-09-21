"""Continuous log-spectrum risk analysis for offline INT4 Muon studies.

This module is deliberately outside the optimizer path.  It provides small,
deterministic helpers used by the report generator: active-mode coordinates,
fixed log-spectrum bins, non-overlapping diagonal residual attribution, and
row/column-associated restoration components.  Quantization and the Muon
transform are delegated to the production implementations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .muon_spectral_sensitivity import SPECTRAL_THRESHOLD

BIN_EDGES = (-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5,
             -2.0, -1.5, -1.0, -0.5, 0.0)
BIN_LABELS = tuple(f"[{BIN_EDGES[i]:g},{BIN_EDGES[i + 1]:g})" for i in range(len(BIN_EDGES) - 1))
MIN_BIN_MODE_SUPPORT = 10
EPS = 1e-12


@dataclass(frozen=True)
class ContinuousSpectrum:
    singular_values: torch.Tensor
    active_indices: torch.Tensor
    normalized: torch.Tensor
    log10_normalized: torch.Tensor
    rank_percentile: torch.Tensor
    local_gap: torch.Tensor
    relative_gap: torch.Tensor


def active_spectrum(singular_values: torch.Tensor, *, threshold: float = SPECTRAL_THRESHOLD) -> ContinuousSpectrum:
    """Return deterministic continuous coordinates for active singular modes."""
    s = singular_values.detach().float()
    if s.numel() == 0 or float(s[0]) <= 0:
        empty = torch.empty(0, dtype=s.dtype, device=s.device)
        return ContinuousSpectrum(s, torch.empty(0, dtype=torch.long, device=s.device), empty, empty, empty, empty, empty)
    active = torch.nonzero(s / s[0] >= float(threshold), as_tuple=False).flatten()
    values = s[active]
    x = values / s[0]
    logx = torch.log10(x.clamp_min(float(threshold)))
    n = max(1, int(values.numel()))
    percentile = torch.arange(values.numel(), dtype=s.dtype, device=s.device) / n
    if values.numel() <= 1:
        gap = torch.zeros_like(values)
    else:
        gap = torch.empty_like(values)
        gap[0] = values[0] - values[1]
        gap[-1] = values[-2] - values[-1]
        if values.numel() > 2:
            gap[1:-1] = torch.minimum(values[:-2] - values[1:-1], values[1:-1] - values[2:])
    return ContinuousSpectrum(s, active, x, logx, percentile, gap, gap / values.clamp_min(float(EPS)))


def fixed_bin_ids(log10_values: torch.Tensor) -> torch.Tensor:
    """Map log10 coordinates to exhaustive fixed bins; -1 is below range."""
    z = log10_values.detach().float()
    edges = torch.tensor(BIN_EDGES, dtype=z.dtype, device=z.device)
    # bucketize(right=False) makes exact upper edge belong to the next bin.
    ids = torch.bucketize(z, edges[1:-1], right=False)
    ids = ids.to(torch.long)
    return torch.where((z >= BIN_EDGES[0]) & (z <= BIN_EDGES[-1]), ids,
                       torch.full_like(ids, -1))


def merge_bin_ids(ids: torch.Tensor, *, min_support: int = MIN_BIN_MODE_SUPPORT) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Deterministically merge globally sparse adjacent bins.

    The report generator applies this once after counting all active modes.
    Merging proceeds left-to-right, combining a sparse bin with its right
    neighbour when possible; a final sparse bin is merged into its left.
    """
    counts = torch.bincount(ids[ids >= 0], minlength=len(BIN_LABELS)).tolist()
    groups: list[list[int]] = []
    current: list[int] = []
    for index, count in enumerate(counts):
        current.append(index)
        if count >= min_support:
            groups.append(current); current = []
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    mapping = torch.full((len(BIN_LABELS),), -1, dtype=torch.long, device=ids.device)
    labels: list[str] = []
    for new, group in enumerate(groups):
        mapping[group] = new
        labels.append(f"[{BIN_EDGES[group[0]]:g},{BIN_EDGES[group[-1] + 1]:g})")
    out = torch.full_like(ids, -1)
    valid = ids >= 0
    out[valid] = mapping[ids[valid]]
    return out, tuple(labels)


def coordinate_error(u: torch.Tensor, matrix: torch.Tensor, quantized: torch.Tensor, vh: torch.Tensor) -> torch.Tensor:
    """Project Q(M)-M into the FP32 reduced SVD coordinates."""
    return u.T @ (quantized.detach().float() - matrix.detach().float()) @ vh.T


def diagonal_component(u: torch.Tensor, ehat: torch.Tensor, vh: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    """Restore only E_hat[i,i] for selected active modes."""
    diagonal = torch.diag(ehat).clone()
    diagonal[~mode_mask] = 0
    return u @ torch.diag(diagonal) @ vh


def associated_component(u: torch.Tensor, ehat: torch.Tensor, vh: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    """Restore entries whose row OR column is selected.

    This is a geometric attribution component.  Components for different bins
    overlap on cross-bin entries, so their direct restoration gains are not
    expected to add; the report states this explicitly.
    """
    mask = mode_mask[:, None] | mode_mask[None, :]
    selected = torch.where(mask, ehat, torch.zeros_like(ehat))
    return u @ selected @ vh


def pair_component(u: torch.Tensor, ehat: torch.Tensor, vh: torch.Tensor,
                   row_mask: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
    mask = row_mask[:, None] & col_mask[None, :]
    return u @ torch.where(mask, ehat, torch.zeros_like(ehat)) @ vh


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | None]:
    ref, obs = reference.detach().float(), candidate.detach().float()
    rn, on = ref.norm(), obs.norm()
    if not torch.isfinite(ref).all() or not torch.isfinite(obs).all() or rn.item() == 0:
        return {"relative_l2": None, "cosine": None, "norm_ratio": None}
    return {"relative_l2": float((obs - ref).norm() / rn),
            "cosine": float((obs * ref).sum() / (rn * on)) if on.item() else None,
            "norm_ratio": float(on / rn)}


def safe_ratio(numerator: float | None, denominator: float | None, *, minimum: float = 1e-12) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) < minimum:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


__all__ = ["BIN_EDGES", "BIN_LABELS", "MIN_BIN_MODE_SUPPORT", "ContinuousSpectrum",
           "active_spectrum", "associated_component", "coordinate_error", "diagonal_component",
           "fixed_bin_ids", "merge_bin_ids", "metrics", "pair_component", "safe_ratio"]
