"""Offline tail-spectral residual corrections for Muon INT4 studies.

All quantization and Muon semantics are delegated to the production helpers.
The functions here only construct detached diagnostic corrections from a
cached FP32 SVD; they are not part of training and are not a quantizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import muon_reference
from .muon_spectral_sensitivity import SPECTRAL_THRESHOLD, quantize
from .muon_update_fidelity import _ratios

BUDGETS = (0, 1, 2, 4, 8, 16)
RANDOM_SEED = 1729


@dataclass(frozen=True)
class TailDecomposition:
    matrix: torch.Tensor
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    effective_rank: int


@torch.no_grad()
def decompose(matrix: torch.Tensor, threshold: float = SPECTRAL_THRESHOLD) -> TailDecomposition:
    value = matrix.detach().float()
    u, s, vh = torch.linalg.svd(value, full_matrices=False)
    active = int((s / s.max().clamp_min(torch.finfo(s.dtype).tiny) >= threshold).sum()) if s.numel() and s.max().item() else 0
    return TailDecomposition(value, u, s, vh, active)


def mode_indices(decomposition: TailDecomposition, k: int, band: str, *, seed: int = RANDOM_SEED) -> torch.Tensor:
    """Return nested deterministic active-mode sets for head/tail/random controls."""
    n = min(max(int(k), 0), decomposition.effective_rank)
    device = decomposition.singular_values.device
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    active = decomposition.effective_rank
    if band == "head":
        return torch.arange(n, dtype=torch.long, device=device)
    if band == "tail":
        return torch.arange(active - n, active, dtype=torch.long, device=device)
    if band == "random":
        generator = torch.Generator(device=device).manual_seed(int(seed) + active * 1009 + n * 9176)
        return torch.sort(torch.randperm(active, generator=generator, device=device)[:n]).values
    raise ValueError(f"unknown spectral band: {band}")


@torch.no_grad()
def correction(decomposition: TailDecomposition, quantized: torch.Tensor, indices: torch.Tensor,
               kind: str) -> torch.Tensor:
    """Project production residual M-Q(M) into selected SVD modes."""
    residual = decomposition.matrix - quantized.detach().float()
    if indices.numel() == 0:
        return torch.zeros_like(decomposition.matrix)
    rhat = decomposition.u.T @ residual @ decomposition.vh.T
    if kind == "diagonal":
        values = torch.diag(rhat)[indices]
        return (decomposition.u[:, indices] * values) @ decomposition.vh[indices, :]
    if kind == "full":
        return decomposition.u[:, indices] @ rhat[indices][:, indices] @ decomposition.vh[indices, :]
    raise ValueError(f"unknown correction kind: {kind}")


@torch.no_grad()
def metrics(source: torch.Tensor, baseline: torch.Tensor, corrected: torch.Tensor,
            *, transform_kwargs: dict | None = None, reference_update: torch.Tensor | None = None,
            baseline_update: torch.Tensor | None = None) -> dict:
    kwargs = transform_kwargs or {}
    reference_update = reference_update if reference_update is not None else muon_reference.zeropower_newton_schulz(source.clone(), **kwargs)
    corrected_update = muon_reference.zeropower_newton_schulz(corrected.clone(), **kwargs)
    result = {}
    result.update({f"raw_{key}": value for key, value in _ratios(source, corrected, "x").items()
                   if key.startswith("x_")})
    result.update({f"update_{key}": value for key, value in _ratios(reference_update, corrected_update, "x").items()
                   if key.startswith("x_")})
    # Baseline is included in every row so callers can compute gains without
    # relying on row ordering.
    baseline_update = baseline_update if baseline_update is not None else muon_reference.zeropower_newton_schulz(baseline.clone(), **kwargs)
    result.update({f"baseline_{key}": value for key, value in _ratios(reference_update, baseline_update, "x").items()
                   if key.startswith("x_")})
    return {key.replace("x_", "", 1): value for key, value in result.items()}


@torch.no_grad()
def row_for_budget(decomposition: TailDecomposition, quantized: torch.Tensor, *, band: str,
                   kind: str, budget: int, transform_kwargs: dict | None = None,
                   reference_update: torch.Tensor | None = None,
                   baseline_update: torch.Tensor | None = None) -> dict:
    indices = mode_indices(decomposition, budget, band)
    corr = correction(decomposition, quantized, indices, kind)
    corrected = quantized + corr
    result = metrics(decomposition.matrix, quantized, corrected, transform_kwargs=transform_kwargs,
                     reference_update=reference_update, baseline_update=baseline_update)
    matrix_norm = decomposition.matrix.norm(); residual_norm = (decomposition.matrix - quantized).norm()
    tail_energy = decomposition.singular_values[indices].square().sum() / decomposition.singular_values.square().sum() if decomposition.singular_values.numel() and decomposition.singular_values.square().sum().item() else None
    corr_norm = corr.norm()
    result.update({
        "band": band, "correction_type": kind, "budget": int(budget),
        "correction_rank": int(indices.numel()), "effective_rank": decomposition.effective_rank,
        "number_of_scalar_coefficients": int(indices.numel() if kind == "diagonal" else indices.numel() ** 2),
        "correction_fraction_elements": float(indices.numel() / decomposition.matrix.numel()) if decomposition.matrix.numel() else None,
        "explicit_storage_scalars": int(indices.numel() * (decomposition.matrix.shape[0] + decomposition.matrix.shape[1] + (1 if kind == "diagonal" else indices.numel()))),
        "tail_energy_fraction": float(tail_energy) if tail_energy is not None else None,
        "correction_relative_matrix_norm": float(corr_norm / matrix_norm) if matrix_norm.item() else None,
        "correction_relative_residual_norm": float(corr_norm / residual_norm) if residual_norm.item() else None,
        "update_cosine_gain": (result.get("update_cosine") - result.get("baseline_cosine")
                               if result.get("update_cosine") is not None and result.get("baseline_cosine") is not None else None),
        "update_l2_reduction_fraction": ((result.get("baseline_relative_l2") - result.get("update_relative_l2")) / result.get("baseline_relative_l2")
                                         if result.get("baseline_relative_l2") not in (None, 0) and result.get("update_relative_l2") is not None else None),
    })
    return result


def validate_projection(decomposition: TailDecomposition, correction_value: torch.Tensor,
                        indices: torch.Tensor, kind: str, atol: float = 1e-4) -> bool:
    """Check that a correction has no spectral components outside its budget."""
    projected = decomposition.u.T @ correction_value @ decomposition.vh.T
    mask = torch.zeros_like(projected, dtype=torch.bool)
    if kind == "diagonal":
        mask[indices, indices] = True
    elif kind == "full":
        mask[indices[:, None], indices[None, :]] = True
    else:
        raise ValueError(kind)
    outside = projected.masked_fill(mask, 0)
    return bool(outside.norm().item() <= atol * max(1.0, projected.norm().item()))
