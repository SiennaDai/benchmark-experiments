"""Offline danger-zone cross-mode restoration helpers.

This module is intentionally outside the optimizer path.  It operates on a
detached FP32 matrix and on the production quantizer reconstruction supplied
by the caller.  The danger interval is the interval selected by the prior
continuous-spectrum report (``[-3, -2)`` in log10 normalized singular value);
it is a fixed analysis input here, not a threshold fitted by this study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .muon_continuous_spectral_risk import EPS, active_spectrum

DANGER_LOG10_START = -3.0
DANGER_LOG10_END = -2.0
DISTANCE_LOCAL = 0.5
DISTANCE_DISTANT = 1.5


@dataclass(frozen=True)
class DangerComponents:
    """Disjoint coordinate masks for the danger-zone residual."""

    danger: torch.Tensor
    diagonal: torch.Tensor
    cross: torch.Tensor
    internal_cross: torch.Tensor
    outside_cross: torch.Tensor
    local_cross: torch.Tensor
    medium_cross: torch.Tensor
    distant_cross: torch.Tensor


def danger_mode_mask(singular_values: torch.Tensor, *, start: float = DANGER_LOG10_START,
                     end: float = DANGER_LOG10_END, threshold: float = 1e-6) -> torch.Tensor:
    """Return active singular modes in the stored prior danger interval.

    The upper endpoint is exclusive, matching the report's ``[-3,-2)`` bin
    convention.  Non-active modes are always false.
    """
    s = singular_values.detach().float()
    if s.numel() == 0 or not torch.isfinite(s).all() or s[0].item() <= 0:
        return torch.zeros_like(s, dtype=torch.bool)
    normalized = s / s[0]
    logx = torch.log10(normalized.clamp_min(float(threshold)))
    return (normalized >= float(threshold)) & (logx >= float(start)) & (logx < float(end))


def coordinate_masks(singular_values: torch.Tensor, danger: torch.Tensor | None = None,
                     *, start: float = DANGER_LOG10_START,
                     end: float = DANGER_LOG10_END,
                     threshold: float = 1e-6) -> DangerComponents:
    """Build mutually disjoint danger diagonal/cross-mode coordinate masks."""
    s = singular_values.detach().float()
    n = int(s.numel())
    if danger is None:
        danger = danger_mode_mask(s, start=start, end=end, threshold=threshold)
    danger = danger.to(dtype=torch.bool, device=s.device)
    row = danger[:, None]
    col = danger[None, :]
    offdiag = ~torch.eye(n, dtype=torch.bool, device=s.device)
    diagonal = torch.diag(danger)
    cross = (row | col) & offdiag
    internal = row & col & offdiag
    outside = cross & ~internal

    if n and s[0].item() > 0:
        z = torch.log10((s / s[0]).clamp_min(float(threshold)))
        distance = (z[:, None] - z[None, :]).abs()
    else:
        distance = torch.full((n, n), float("inf"), dtype=s.dtype, device=s.device)
    local = cross & (distance < DISTANCE_LOCAL)
    medium = cross & (distance >= DISTANCE_LOCAL) & (distance < DISTANCE_DISTANT)
    distant = cross & (distance >= DISTANCE_DISTANT)
    return DangerComponents(danger, diagonal, cross, internal, outside, local, medium, distant)


def component_from_mask(error_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a coordinate-space component with all other coefficients zero."""
    if error_hat.shape != mask.shape:
        raise ValueError("error_hat and coordinate mask must have the same shape")
    return torch.where(mask, error_hat, torch.zeros_like(error_hat))


def matrix_from_coordinates(u: torch.Tensor, vh: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Map reduced SVD coordinates back to matrix space."""
    return u @ coordinates @ vh


def restore_from_coordinates(quantized: torch.Tensor, u: torch.Tensor, vh: torch.Tensor,
                             coordinates: torch.Tensor) -> torch.Tensor:
    """Remove a selected residual component from a quantized reconstruction."""
    return quantized - matrix_from_coordinates(u, vh, coordinates)


def select_top_coefficients(coordinates: torch.Tensor, count: int) -> torch.Tensor:
    """Keep at most ``count`` largest coefficients, with stable tie ordering."""
    if count <= 0 or coordinates.numel() == 0:
        return torch.zeros_like(coordinates)
    flat = coordinates.abs().flatten()
    count = min(int(count), flat.numel())
    order = torch.argsort(flat, descending=True, stable=True)[:count]
    selected = torch.zeros_like(coordinates).flatten()
    selected[order] = coordinates.flatten()[order]
    return selected.view_as(coordinates)


def select_energy_budget(coordinates: torch.Tensor, target_norm: float) -> tuple[torch.Tensor, int, bool]:
    """Select largest actual residual coefficients up to an exact norm.

    The last selected coefficient is reduced, never increased, so the returned
    correction remains a subset of the actual residual support.  The boolean
    says whether the requested norm was clipped because the allowed residual
    contained insufficient energy.
    """
    target = float(target_norm)
    if not math.isfinite(target) or target <= 0 or coordinates.numel() == 0:
        return torch.zeros_like(coordinates), 0, False
    available = float(coordinates.norm())
    if not math.isfinite(available) or available <= 0:
        return torch.zeros_like(coordinates), 0, target > 0
    clipped = target > available
    target = min(target, available)
    if target == available:
        return coordinates.clone(), int((coordinates != 0).sum()), clipped
    flat_abs = coordinates.abs().flatten()
    order = torch.argsort(flat_abs, descending=True, stable=True)
    positive = flat_abs[order] > 0
    order = order[positive]
    if order.numel() == 0:
        return torch.zeros_like(coordinates), 0, clipped
    cumulative = torch.cumsum(flat_abs[order].square(), dim=0)
    index = int(torch.searchsorted(cumulative, torch.tensor(target * target, dtype=cumulative.dtype, device=cumulative.device), right=False).item())
    index = min(index, int(order.numel()) - 1)
    chosen = order[: index + 1]
    out = torch.zeros_like(coordinates).flatten()
    before = float(cumulative[index - 1]) if index else 0.0
    remaining = max(0.0, target * target - before)
    last = int(chosen[-1])
    sign = torch.sign(coordinates.flatten()[last])
    out[chosen[:-1]] = coordinates.flatten()[chosen[:-1]]
    out[last] = sign * math.sqrt(remaining)
    return out.view_as(coordinates), int(chosen.numel()), clipped


def correction_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | None]:
    """Finite-safe update metrics used by the oracle report."""
    ref = reference.detach().float(); obs = candidate.detach().float()
    if not torch.isfinite(ref).all() or not torch.isfinite(obs).all():
        return {"relative_l2": None, "cosine": None, "norm_ratio": None, "status": "nonfinite"}
    ref_norm = ref.norm(); obs_norm = obs.norm()
    if ref_norm.item() == 0:
        return {"relative_l2": None, "cosine": None, "norm_ratio": None, "status": "zero_reference_norm"}
    return {
        "relative_l2": float((obs - ref).norm() / ref_norm),
        "cosine": float((obs * ref).sum() / (obs_norm * ref_norm)) if obs_norm.item() else None,
        "norm_ratio": float(obs_norm / ref_norm),
        "status": "ok" if obs_norm.item() else "zero_observed_norm",
    }


def restoration_gain(baseline: dict, restored: dict) -> dict[str, float | None]:
    """Return direct cosine/L2 recovery relative to a common baseline."""
    cosine = (restored.get("cosine") - baseline.get("cosine")
              if restored.get("cosine") is not None and baseline.get("cosine") is not None else None)
    l2 = (baseline.get("relative_l2") - restored.get("relative_l2")
          if restored.get("relative_l2") is not None and baseline.get("relative_l2") is not None else None)
    return {"update_cosine_gain": cosine, "update_l2_reduction": l2}


__all__ = ["DANGER_LOG10_START", "DANGER_LOG10_END", "DISTANCE_LOCAL", "DISTANCE_DISTANT",
           "DangerComponents", "danger_mode_mask", "coordinate_masks", "component_from_mask",
           "matrix_from_coordinates", "restore_from_coordinates", "select_top_coefficients",
           "select_energy_budget", "correction_metrics", "restoration_gain"]
