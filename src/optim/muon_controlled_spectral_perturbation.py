"""Controlled, offline spectral perturbations for Muon matrices.

This module contains no optimizer or quantizer implementation.  It constructs
deterministic diagnostic matrices from a cached FP32 SVD and delegates Muon
evaluation to the production ``zeropower_newton_schulz`` implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .muon_spectral_sensitivity import (
    SpectralDecomposition,
    decompose,
    quantize,
    transform,
)
from .muon_update_fidelity import _ratios


@dataclass(frozen=True)
class SpectralBands:
    """Index sets used by the controlled experiments."""

    head: torch.Tensor
    tail: torch.Tensor


def spectral_bands(singular_values: torch.Tensor, *, head_energy: float = 0.90) -> SpectralBands:
    """Return deterministic head/tail mode indices.

    ``head`` is the smallest leading set explaining ``head_energy`` of
    squared singular-value energy, with at least two modes when possible so a
    non-trivial rotation can be constructed.  ``tail`` is its complement.
    This is an index-space definition, not a post-hoc selection based on
    perturbation results.
    """
    s = singular_values.detach().float()
    n = int(s.numel())
    if n == 0:
        empty = torch.empty(0, dtype=torch.long, device=s.device)
        return SpectralBands(empty, empty)
    total = s.square().sum()
    if total.item() == 0:
        head_count = min(n, max(1, math.ceil(n * 0.10)))
    else:
        cumulative = torch.cumsum(s.square(), dim=0) / total
        head_count = int(torch.searchsorted(cumulative, torch.tensor(head_energy, device=s.device)).item()) + 1
        head_count = min(n, max(1, head_count))
    if n > 1:
        head_count = max(2, head_count)
        head_count = min(n - 1, head_count)
    head = torch.arange(head_count, device=s.device, dtype=torch.long)
    all_indices = torch.arange(n, device=s.device, dtype=torch.long)
    mask = torch.ones(n, dtype=torch.bool, device=s.device)
    mask[head] = False
    return SpectralBands(head, all_indices[mask])


def _generator(size: int, *, device, dtype) -> torch.Tensor:
    """Build a deterministic skew generator from indices only."""
    g = torch.zeros((size, size), device=device, dtype=dtype)
    if size < 2:
        return g
    # Pair adjacent modes with alternating signs.  No RNG is consumed.
    for i in range(size - 1):
        value = 1.0 if i % 2 == 0 else -1.0
        g[i, i + 1] = value
        g[i + 1, i] = -value
    norm = torch.linalg.matrix_norm(g)
    return g / norm if norm.item() else g


def _rotated_matrix(source: SpectralDecomposition, indices: torch.Tensor, theta: float) -> torch.Tensor:
    if indices.numel() < 2:
        return source.matrix.clone()
    g = _generator(int(indices.numel()), device=source.matrix.device, dtype=source.matrix.dtype)
    rotation = torch.linalg.matrix_exp(g * float(theta))
    rotated_u = source.u.clone()
    rotated_u[:, indices] = source.u[:, indices] @ rotation
    return (rotated_u * source.singular_values) @ source.vh


def _target_norm(source: SpectralDecomposition, epsilon: float) -> torch.Tensor:
    return source.matrix.norm() * float(epsilon)


@torch.no_grad()
def rotation_only(source: SpectralDecomposition, indices: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, float, bool]:
    """Rotate U within one spectral band and solve for exact target norm."""
    target = _target_norm(source, epsilon)
    if target.item() == 0 or indices.numel() < 2:
        return source.matrix.clone(), 0.0, True
    lo, hi = 0.0, math.pi
    # The first small interval is monotonic for this construction.  Expand
    # only if required, then use deterministic bisection.
    for _ in range(8):
        candidate = _rotated_matrix(source, indices, hi)
        if (candidate - source.matrix).norm().item() >= target.item():
            break
        hi *= 2.0
    if (candidate - source.matrix).norm().item() < target.item():
        return source.matrix.clone(), 0.0, True
    for _ in range(48):
        mid = (lo + hi) / 2.0
        value = (_rotated_matrix(source, indices, mid) - source.matrix).norm().item()
        if value < target.item():
            lo = mid
        else:
            hi = mid
    theta = (lo + hi) / 2.0
    return _rotated_matrix(source, indices, theta), theta, False


@torch.no_grad()
def singular_value_only(source: SpectralDecomposition, indices: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, bool]:
    """Perturb only singular values with a positive deterministic increment."""
    target = _target_norm(source, epsilon)
    if target.item() == 0 or indices.numel() == 0:
        return source.matrix.clone(), True
    # Positive increments avoid introducing sign changes or an artificial
    # extra clipping operation.  U and V are kept exactly as supplied.
    weights = torch.arange(1, indices.numel() + 1, device=source.matrix.device, dtype=source.matrix.dtype)
    delta = torch.zeros_like(source.singular_values)
    delta[indices] = target * weights / weights.norm()
    return (source.u * (source.singular_values + delta)) @ source.vh, False


@torch.no_grad()
def compare_updates(source: torch.Tensor, observed: torch.Tensor, *, transform_kwargs: dict | None = None) -> dict:
    """Compare matrices after invoking the exact production Muon transform."""
    kwargs = transform_kwargs or {}
    source_update = transform(source, **kwargs)
    observed_update = transform(observed, **kwargs)
    result = _ratios(source, observed, "raw")
    result.update(_ratios(source_update, observed_update, "update"))
    return result


@torch.no_grad()
def controlled_row(source: SpectralDecomposition, observed: torch.Tensor, *, kind: str,
                  band: str, epsilon: float, transform_kwargs: dict | None = None,
                  theta: float | None = None, invalid: bool = False) -> dict:
    """Build metrics and construction checks for one controlled matrix."""
    metrics = compare_updates(source.matrix, observed, transform_kwargs=transform_kwargs)
    error = observed - source.matrix
    source_norm = source.matrix.norm()
    row = {
        "kind": kind, "band": band, "epsilon": float(epsilon),
        "actual_relative_frobenius": float(error.norm() / source_norm) if source_norm.item() else None,
        "target_relative_frobenius": float(epsilon),
        "theta": theta, "invalid_construction": bool(invalid),
        "raw_relative_l2": metrics.get("raw_relative_l2"), "raw_cosine": metrics.get("raw_cosine"),
        "raw_norm_ratio": metrics.get("raw_norm_ratio"),
        "update_relative_l2": metrics.get("update_relative_l2"), "update_cosine": metrics.get("update_cosine"),
        "update_norm_ratio": metrics.get("update_norm_ratio"),
    }
    row["singular_value_max_abs_error"] = float((torch.linalg.svdvals(observed) - source.singular_values).abs().max())
    return row


@torch.no_grad()
def actual_error_decomposition(source: SpectralDecomposition, quantized: torch.Tensor,
                               *, head: torch.Tensor, tail: torch.Tensor) -> dict:
    """Partition FP32 spectral-coordinate error into magnitude/mixing blocks."""
    error = quantized.detach().float() - source.matrix
    ehat = source.u.T @ error @ source.vh.T
    n = ehat.shape[0]
    head_mask = torch.zeros(n, dtype=torch.bool, device=ehat.device)
    head_mask[head] = True
    tail_mask = ~head_mask
    diag_mask = torch.eye(n, dtype=torch.bool, device=ehat.device)
    head_diag = diag_mask & head_mask[:, None]
    tail_diag = diag_mask & tail_mask[:, None]
    offdiag = ~diag_mask
    head_internal = offdiag & head_mask[:, None] & head_mask[None, :]
    tail_internal = offdiag & tail_mask[:, None] & tail_mask[None, :]
    cross = offdiag & ~(head_internal | tail_internal)
    energy = ehat.square()
    total = energy.sum()
    values = {
        "total_projected_error_energy": total,
        "head_singular_value_error_energy": energy[head_diag].sum(),
        "tail_singular_value_error_energy": energy[tail_diag].sum(),
        "head_subspace_mixing_energy": energy[head_internal].sum(),
        "tail_subspace_mixing_energy": energy[tail_internal].sum(),
        "cross_subspace_mixing_energy": energy[cross].sum(),
        "unresolved_error_energy": (error.square().sum() - total).clamp_min(0),
    }
    row = {}
    for key, value in values.items():
        row[key] = float(value) if torch.isfinite(value) else None
        row[key.replace("energy", "fraction")] = float(value / total) if total.item() and torch.isfinite(value) else None
    row["diagonal_spectral_error_energy"] = row["head_singular_value_error_energy"] + row["tail_singular_value_error_energy"]
    row["offdiagonal_spectral_error_energy"] = (row["head_subspace_mixing_energy"] + row["tail_subspace_mixing_energy"] + row["cross_subspace_mixing_energy"])
    return row
