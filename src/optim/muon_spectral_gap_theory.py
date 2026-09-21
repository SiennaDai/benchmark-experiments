"""Offline spectral-gap and subspace-perturbation diagnostics for Muon.

This module is intentionally outside the optimizer path.  It contains only
deterministic, read-only helpers; quantization and Muon evaluation are
delegated to the production implementations imported from sibling modules.
"""
from __future__ import annotations

import math

import torch

from .muon_spectral_sensitivity import band_slices, subspace_metrics, transform


def active_indices(singular_values: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    s = singular_values.detach().float()
    if not s.numel() or float(s[0]) <= 0:
        return torch.empty(0, dtype=torch.long, device=s.device)
    return torch.nonzero(s / s[0] >= threshold, as_tuple=False).flatten()


def mode_gaps(singular_values: torch.Tensor, *, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    """Return adjacent-neighbour gaps and relative gaps.

    At the first/last mode the only available neighbour is used.  A singleton
    spectrum has gap zero.  The gap is a local separation proxy, not a claim
    that individual vectors are stable when values are repeated.
    """
    s = singular_values.detach().float()
    n = int(s.numel())
    if n == 0:
        empty = torch.empty(0, dtype=s.dtype, device=s.device)
        return {"gap": empty, "relative_gap": empty}
    if n == 1:
        gap = torch.zeros_like(s)
    else:
        left = torch.cat((torch.full((1,), float("inf"), dtype=s.dtype, device=s.device), (s[:-1] - s[1:]).abs()))
        right = torch.cat(((s[:-1] - s[1:]).abs(), torch.full((1,), float("inf"), dtype=s.dtype, device=s.device)))
        gap = torch.minimum(left, right)
        gap[0] = (s[0] - s[1]).abs()
        gap[-1] = (s[-2] - s[-1]).abs()
    relative = gap / s.abs().clamp_min(float(eps))
    return {"gap": gap, "relative_gap": relative}


def band_indices(singular_values: torch.Tensor, *, threshold: float = 1e-6) -> dict[str, torch.Tensor]:
    """Use the established 10%-index head/middle/tail convention on active modes."""
    active = active_indices(singular_values, threshold)
    n = int(active.numel())
    if n == 0:
        empty = active
        return {"head": empty, "middle": empty, "tail": empty, "active": active}
    width = max(1, math.ceil(n * 0.10))
    middle_start = max(0, n // 2 - width // 2)
    return {
        "head": active[:width],
        "middle": active[middle_start:min(n, middle_start + width)],
        "tail": active[max(0, n - width):],
        "active": active,
    }


def band_separation(singular_values: torch.Tensor, indices: torch.Tensor,
                    *, eps: float = 1e-12) -> float | None:
    """External separation from a band to its complement.

    For a contiguous tail/head this is the boundary singular-value gap.  A
    band with no complement has undefined separation, returned as ``None``.
    """
    s = singular_values.detach().float()
    if not indices.numel() or indices.numel() == s.numel():
        return None
    mask = torch.zeros(s.numel(), dtype=torch.bool, device=s.device)
    mask[indices] = True
    inside, outside = s[mask], s[~mask]
    if not inside.numel() or not outside.numel():
        return None
    return float(torch.abs(inside[:, None] - outside[None, :]).min().clamp_min(eps))


def gap_proxies(error: torch.Tensor, singular_values: torch.Tensor,
                bands: dict[str, torch.Tensor], *, eps: float = 1e-12) -> dict:
    """Compute bound-inspired perturbation/gap ratios using spectral norms."""
    e = error.detach().float()
    e2, ef = float(torch.linalg.matrix_norm(e, ord=2)), float(torch.linalg.vector_norm(e))
    out = {"error_spectral_norm": e2, "error_frobenius_norm": ef}
    for name, indices in bands.items():
        if name == "active":
            continue
        delta = band_separation(singular_values, indices, eps=eps)
        out[f"{name}_spectral_separation"] = delta
        out[f"{name}_sensitivity_2"] = e2 / max(delta or 0.0, eps) if delta is not None else None
        out[f"{name}_sensitivity_f"] = ef / max(delta or 0.0, eps) if delta is not None else None
    return out


def tail_reweighting(singular_values: torch.Tensor, transformed_values: torch.Tensor,
                     tail: torch.Tensor, *, eps: float = 1e-12) -> float | None:
    if not tail.numel():
        return None
    s = singular_values.detach().float(); f = transformed_values.detach().float()
    return float((f[tail].abs() / s[tail].abs().clamp_min(eps)).median())


def principal_subspace_rows(u: torch.Tensor, vh: torch.Tensor, uq: torch.Tensor,
                            vhq: torch.Tensor, bands: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for band, indices in bands.items():
        if band == "active" or not indices.numel():
            continue
        for side, a, b in (("left", u[:, indices], uq[:, indices]),
                           ("right", vh.T[:, indices], vhq.T[:, indices])):
            rows.append({"band": band, "side": side, **subspace_metrics(a, b)})
    return rows


def controlled_gap_spectrum(singular_values: torch.Tensor, tail: torch.Tensor,
                            ratio: float, *, eps: float = 1e-12) -> tuple[torch.Tensor, float | None]:
    """Change only the tail/complement boundary gap around its midpoint."""
    s = singular_values.detach().float().clone()
    if not tail.numel() or tail[0].item() == 0 or tail[0].item() >= s.numel():
        return s, None
    boundary = int(tail[0])
    old = float((s[boundary - 1] - s[boundary]).abs())
    center = (s[boundary - 1] + s[boundary]) / 2
    # Keep order and positivity; extreme requested ratios are clipped and the
    # actual gap is reported so the intervention is auditable.
    target = max(old * float(ratio), eps)
    upper_limit = float(s[boundary - 1]) * 2.0
    target = min(target, max(eps, upper_limit))
    s[boundary - 1] = center + target / 2
    s[boundary] = max(center - target / 2, 0.0)
    return s, float((s[boundary - 1] - s[boundary]).abs())


def deterministic_tail_error(u: torch.Tensor, vh: torch.Tensor, tail: torch.Tensor,
                             magnitude: float) -> torch.Tensor:
    """Create a fixed-norm cross-boundary error in FP32 spectral coordinates."""
    n = u.shape[1]
    ehat = torch.zeros((n, vh.shape[0]), dtype=u.dtype, device=u.device)
    if tail.numel() and int(tail[0]) > 0:
        j = int(tail[0])
        ehat[j - 1, j] = 1.0
    elif tail.numel() >= 2:
        j = int(tail[0]); ehat[j, j + 1] = 1.0
    norm = ehat.norm()
    if norm.item() == 0:
        return torch.zeros_like(u @ ehat @ vh)
    return (u @ ehat @ vh) * (float(magnitude) / norm)


def safe_correlation(rows: list[dict], x_key: str, y_key: str) -> dict:
    """Small dependency-free Pearson/Spearman helper with explicit sample N."""
    values = [(float(r[x_key]), float(r[y_key])) for r in rows
              if r.get(x_key) is not None and r.get(y_key) is not None
              and math.isfinite(float(r[x_key])) and math.isfinite(float(r[y_key]))]
    if len(values) < 2:
        return {"feature": x_key, "target": y_key, "sample_count": len(values), "pearson": None, "spearman": None}
    x = torch.tensor([a for a, _ in values], dtype=torch.float64); y = torch.tensor([b for _, b in values], dtype=torch.float64)
    def corr(a, b):
        ac, bc = a - a.mean(), b - b.mean(); den = ac.norm() * bc.norm()
        return float((ac * bc).sum() / den) if den.item() else None
    def rank(a):
        order = torch.argsort(a, stable=True); out = torch.empty_like(a); out[order] = torch.arange(len(a), dtype=a.dtype)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]: j += 1
            if j - i > 1: out[order[i:j]] = (i + j - 1) / 2
            i = j
        return out
    return {"feature": x_key, "target": y_key, "sample_count": len(values), "pearson": corr(x, y), "spearman": corr(rank(x), rank(y))}
