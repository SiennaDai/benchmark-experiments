"""Offline spectral-band contribution diagnostics for INT4 Muon.

This module separates the amount of production INT4 residual in each FP32
spectral band from the measured Muon sensitivity of that band.  It is kept
outside the optimizer path: production quantization and orthogonalization are
delegated to the existing implementations and every input is detached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import muon_reference
from .muon_spectral_gap_theory import band_indices
from .muon_spectral_sensitivity import SPECTRAL_THRESHOLD, quantize
from .muon_update_fidelity import _ratios

EPSILON = 0.001
BANDS = ("head", "middle", "tail")
ACCOUNTING_BANDS = BANDS + ("other",)
KINDS = ("magnitude", "orientation")


@dataclass(frozen=True)
class BandDecomposition:
    matrix: torch.Tensor
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    bands: dict[str, torch.Tensor]
    effective_rank: int


def _established_bands(singular_values: torch.Tensor, *, threshold: float) -> dict[str, torch.Tensor]:
    """Reuse the prior top/center/tail 10%-of-active convention exactly.

    The three named slices are intentionally not exhaustive.  ``other`` is
    added only for orthogonal residual accounting so the nine requested named
    blocks are never silently treated as the whole matrix.
    """
    selected = band_indices(singular_values, threshold=threshold)
    active = selected["active"]
    used = torch.zeros(singular_values.numel(), dtype=torch.bool, device=singular_values.device)
    for name in BANDS:
        used[selected[name]] = True
    selected["other"] = active[~used[active]]
    return selected


@torch.no_grad()
def decompose(matrix: torch.Tensor, *, threshold: float = SPECTRAL_THRESHOLD) -> BandDecomposition:
    value = matrix.detach().float()
    if value.ndim != 2:
        raise ValueError("spectral contribution analysis requires a 2D matrix")
    u, s, vh = torch.linalg.svd(value, full_matrices=False)
    bands = _established_bands(s, threshold=threshold)
    return BandDecomposition(value, u, s, vh, bands, int(bands["active"].numel()))


def _finite(x: torch.Tensor | float | None) -> float | None:
    if x is None:
        return None
    value = float(x.detach().item() if isinstance(x, torch.Tensor) else x)
    return value if math.isfinite(value) else None


def _indices(d: BandDecomposition, name: str) -> torch.Tensor:
    if name not in ACCOUNTING_BANDS:
        raise ValueError(f"unknown spectral band: {name}")
    return d.bands[name]


@torch.no_grad()
def coordinate_error(d: BandDecomposition, quantized: torch.Tensor) -> torch.Tensor:
    """Return ``U.T @ (Q(M)-M) @ V`` in the FP32 reduced SVD basis."""
    return d.u.T @ (quantized.detach().float() - d.matrix) @ d.vh.T


@torch.no_grad()
def block_energy(d: BandDecomposition, quantized: torch.Tensor) -> dict:
    """Return an orthogonal spectral-coordinate energy decomposition.

    The nine named row/column blocks, together with explicit ``other`` blocks,
    partition the reduced FP32 SVD coordinates, so their squared energies sum
    to ``projected_error_energy``.  ``row`` band
    energies are used as the non-overlapping high-level band attribution:
    head-associated = H,*; middle-associated = M,*; tail-associated = T,*.
    Cross-band energy is reported separately as all off-diagonal band blocks.
    Any error outside the reduced SVD coordinates is explicit and is not
    silently assigned to a band.
    """
    ehat = coordinate_error(d, quantized)
    total = (quantized.detach().float() - d.matrix).square().sum()
    projected = ehat.square().sum()
    row: dict = {
        "total_residual_energy": _finite(total),
        "projected_residual_energy": _finite(projected),
        "unresolved_residual_energy": _finite((total - projected).clamp_min(0)),
    }
    masks = {name: torch.zeros(ehat.shape[0], dtype=torch.bool, device=ehat.device) for name in ACCOUNTING_BANDS}
    for name in ACCOUNTING_BANDS:
        masks[name][_indices(d, name)] = True
    energies: dict[str, torch.Tensor] = {}
    for rname in ACCOUNTING_BANDS:
        for cname in ACCOUNTING_BANDS:
            value = ehat[masks[rname]][:, masks[cname]].square().sum()
            energies[f"{rname}_{cname}"] = value
            row[f"{rname}_{cname}_energy"] = _finite(value)
            row[f"{rname}_{cname}_fraction"] = _finite(value / total) if total.item() else None
    cross = sum((energies[f"{r}_{c}"] for r in ACCOUNTING_BANDS for c in ACCOUNTING_BANDS if r != c), torch.zeros_like(projected))
    row["cross_band_mixing_energy"] = _finite(cross)
    row["cross_band_mixing_fraction"] = _finite(cross / total) if total.item() else None
    for name in ACCOUNTING_BANDS:
        associated = sum((energies[f"{name}_{c}"] for c in ACCOUNTING_BANDS), torch.zeros_like(projected))
        row[f"{name}_associated_energy"] = _finite(associated)
        row[f"{name}_associated_fraction"] = _finite(associated / total) if total.item() else None
    row["projected_energy_fraction"] = _finite(projected / total) if total.item() else None
    return row


@torch.no_grad()
def band_component(d: BandDecomposition, quantized: torch.Tensor, band: str) -> torch.Tensor:
    """Return the disjoint row-band component ``U_B U_B.T E``."""
    ehat = coordinate_error(d, quantized)
    indices = _indices(d, band)
    if not indices.numel():
        return torch.zeros_like(d.matrix)
    return d.u[:, indices] @ ehat[indices, :] @ d.vh


@torch.no_grad()
def restoration(d: BandDecomposition, quantized: torch.Tensor, bands: tuple[str, ...]) -> torch.Tensor:
    """Restore the selected actual residual row-band components."""
    corrected = quantized.detach().float().clone()
    for band in bands:
        corrected = corrected - band_component(d, quantized, band)
    return corrected


def _skew(size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    value = torch.zeros((size, size), dtype=dtype, device=device)
    if size < 2:
        return value
    idx = torch.arange(size - 1, device=device)
    signs = torch.where(idx.remainder(2) == 0, 1.0, -1.0).to(dtype)
    value[idx, idx + 1] = signs
    value[idx + 1, idx] = -signs
    norm = value.norm()
    return value / norm if norm.item() else value


@torch.no_grad()
def controlled_perturbation(d: BandDecomposition, band: str, kind: str,
                           epsilon: float = EPSILON) -> tuple[torch.Tensor, bool]:
    """Construct a deterministic equal-energy local perturbation.

    ``magnitude`` changes only selected singular values with a deterministic
    signed coefficient vector.  ``orientation`` rotates the left singular
    vectors within the selected band.  Both target exactly
    ``epsilon * ||M||_F`` up to FP32/bisection tolerance.  A one-mode or empty
    band cannot support the requested orientation rotation and is marked
    invalid rather than silently substituting another perturbation.
    """
    indices = _indices(d, band)
    target = d.matrix.norm() * float(epsilon)
    if not indices.numel() or target.item() == 0:
        return d.matrix.clone(), True
    if kind == "magnitude":
        weights = torch.arange(1, indices.numel() + 1, dtype=d.matrix.dtype, device=d.matrix.device)
        delta = torch.zeros_like(d.singular_values)
        delta[indices] = target * weights / weights.norm()
        return (d.u * (d.singular_values + delta)) @ d.vh, False
    if kind != "orientation":
        raise ValueError(f"unknown perturbation kind: {kind}")
    if indices.numel() < 2:
        return d.matrix.clone(), True
    generator = _skew(int(indices.numel()), dtype=d.matrix.dtype, device=d.matrix.device)

    def build(theta: float) -> torch.Tensor:
        rotation = torch.linalg.matrix_exp(generator * float(theta))
        rotated = d.u.clone()
        rotated[:, indices] = d.u[:, indices] @ rotation
        return (rotated * d.singular_values) @ d.vh

    lo, hi = 0.0, math.pi
    candidate = build(hi)
    for _ in range(12):
        if (candidate - d.matrix).norm().item() >= target.item():
            break
        hi *= 2.0
        candidate = build(hi)
    if (candidate - d.matrix).norm().item() < target.item():
        return d.matrix.clone(), True
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if (build(mid) - d.matrix).norm().item() < target.item():
            lo = mid
        else:
            hi = mid
    return build((lo + hi) / 2.0), False


@torch.no_grad()
def controlled_sensitivity(d: BandDecomposition, band: str, kind: str, *, epsilon: float,
                          transform_kwargs: dict) -> dict:
    perturbed, invalid = controlled_perturbation(d, band, kind, epsilon)
    reference = muon_reference.zeropower_newton_schulz(d.matrix.clone(), **transform_kwargs)
    observed = muon_reference.zeropower_newton_schulz(perturbed.clone(), **transform_kwargs)
    raw = _ratios(d.matrix, perturbed, "raw")
    update = _ratios(reference, observed, "update")
    actual = (perturbed - d.matrix).norm() / d.matrix.norm() if d.matrix.norm().item() else None
    return {
        "band": band, "perturbation_kind": kind, "epsilon": float(epsilon),
        "invalid_construction": bool(invalid),
        "actual_relative_frobenius": _finite(actual),
        "raw_relative_l2": raw.get("raw_relative_l2"), "raw_cosine": raw.get("raw_cosine"),
        "update_relative_l2": update.get("update_relative_l2"), "update_cosine": update.get("update_cosine"),
        "update_cosine_error": (1.0 - update["update_cosine"] if isinstance(update.get("update_cosine"), float) else None),
        "local_sensitivity": (_finite(update.get("update_relative_l2") / epsilon)
                              if isinstance(update.get("update_relative_l2"), (int, float)) and epsilon else None),
    }


@torch.no_grad()
def metrics(source: torch.Tensor, candidate: torch.Tensor, *, reference_update: torch.Tensor,
            transform_kwargs: dict) -> dict:
    observed_update = muon_reference.zeropower_newton_schulz(candidate.detach().float().clone(), **transform_kwargs)
    raw = _ratios(source, candidate, "raw")
    update = _ratios(reference_update, observed_update, "update")
    return {
        "raw_relative_l2": raw.get("raw_relative_l2"), "raw_cosine": raw.get("raw_cosine"),
        "raw_norm_ratio": raw.get("raw_norm_ratio"),
        "update_relative_l2": update.get("update_relative_l2"), "update_cosine": update.get("update_cosine"),
        "update_norm_ratio": update.get("update_norm_ratio"),
    }


def contribution_proxy(residual_energy: float | None, sensitivity: float | None) -> float | None:
    if residual_energy is None or sensitivity is None:
        return None
    return math.sqrt(max(0.0, residual_energy)) * sensitivity


__all__ = ["BANDS", "EPSILON", "KINDS", "BandDecomposition", "band_component", "block_energy",
           "controlled_perturbation", "controlled_sensitivity", "contribution_proxy", "decompose",
           "metrics", "quantize", "restoration"]
