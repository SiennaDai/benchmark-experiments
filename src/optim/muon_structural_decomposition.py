"""Offline structural-decomposition diagnostics for INT4 Muon.

This module is deliberately outside the optimizer.  It keeps a dominant
truncated SVD component in FP32 (an oracle side channel) and applies the
unchanged production INT4 dynamic roundtrip only to the residual.  All
functions operate on detached copies and never alter optimizer state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import muon_reference
from .muon_spectral_sensitivity import SPECTRAL_THRESHOLD, quantize
from .muon_update_fidelity import _ratios


@dataclass(frozen=True)
class SVDState:
    matrix: torch.Tensor
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor


FIXED_RANKS = (1, 2, 4, 8, 16)
ENERGY_TARGETS = (0.25, 0.50, 0.75, 0.90)


@torch.no_grad()
def decompose(matrix: torch.Tensor) -> SVDState:
    value = matrix.detach().float()
    if value.ndim != 2:
        raise ValueError("structural decomposition requires a 2D matrix")
    u, s, vh = torch.linalg.svd(value, full_matrices=False)
    return SVDState(value, u, s, vh)


def valid_rank(s: torch.Tensor, k: int) -> int:
    return max(0, min(int(k), int(s.numel())))


def energy_rank(s: torch.Tensor, target: float) -> int:
    """Smallest rank explaining target squared-Frobenius energy."""
    if not 0 < target <= 1:
        raise ValueError("energy target must be in (0, 1]")
    total = s.float().square().sum()
    if total.item() == 0:
        return 0
    cumulative = s.float().square().cumsum(0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(float(target), dtype=cumulative.dtype)).item()) + 1


@torch.no_grad()
def truncated(state: SVDState, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact rank-k component and residual, both FP32."""
    k = valid_rank(state.singular_values, k)
    if k == 0:
        low = torch.zeros_like(state.matrix)
    else:
        low = (state.u[:, :k] * state.singular_values[:k]) @ state.vh[:k]
    return low, state.matrix - low


@torch.no_grad()
def structural_reconstruct(state: SVDState, k: int, *, quantizer: str = "int4-dynamic-b2048") -> dict:
    low, residual = truncated(state, k)
    quantized_residual = quantize(residual, quantizer)
    return {"low_rank": low, "residual": residual,
            "quantized_residual": quantized_residual,
            "reconstruction": low + quantized_residual}


def _safe_ratio(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if b.item() == 0 or not torch.isfinite(a) or not torch.isfinite(b):
        return None
    value = a / b
    return float(value.item()) if math.isfinite(float(value.item())) else None


def _stats(value: torch.Tensor, reference: torch.Tensor) -> dict:
    value, reference = value.float(), reference.float()
    return {
        "norm": float(value.norm().item()),
        "relative_to_reference": _safe_ratio(value.norm(), reference.norm()),
        "max_abs": float(value.abs().max().item()) if value.numel() else 0.0,
        "rms": float(value.square().mean().sqrt().item()) if value.numel() else 0.0,
    }


@torch.no_grad()
def residual_dynamic_range(state: SVDState, residual: torch.Tensor) -> dict:
    """Summarize dynamic range and block absmax without duplicating quantization."""
    value, source = residual.detach().float(), state.matrix
    flat, source_flat = value.reshape(-1), source.reshape(-1)
    block = 2048
    if flat.numel():
        padded = flat[: (flat.numel() // block) * block]
        scales = padded.reshape(-1, block).abs().amax(1) if padded.numel() else flat.abs().amax().reshape(1)
        final = flat[(flat.numel() // block) * block:]
        if final.numel():
            scales = torch.cat((scales, final.abs().amax().reshape(1)))
    else:
        scales = torch.zeros(1)
    source_scales = source_flat[: (source_flat.numel() // block) * block]
    source_scales = source_scales.reshape(-1, block).abs().amax(1) if source_scales.numel() else torch.empty(0)
    source_tail = source_flat[(source_flat.numel() // block) * block:]
    if source_tail.numel(): source_scales = torch.cat((source_scales, source_tail.abs().amax().reshape(1)))
    out = {"residual_norm": float(value.norm().item()),
           "residual_over_matrix_norm": _safe_ratio(value.norm(), source.norm()),
           "max_abs_residual": float(value.abs().max().item()) if value.numel() else 0.0,
           "max_abs_matrix": float(source.abs().max().item()) if source.numel() else 0.0,
           "rms_residual": float(value.square().mean().sqrt().item()) if value.numel() else 0.0,
           "rms_matrix": float(source.square().mean().sqrt().item()) if source.numel() else 0.0,
           "block_absmax_mean": float(scales.mean().item()) if scales.numel() else 0.0,
           "block_absmax_median": float(scales.median().item()) if scales.numel() else 0.0,
           "block_absmax_max": float(scales.max().item()) if scales.numel() else 0.0,
           "matrix_block_absmax_mean": float(source_scales.mean().item()) if source_scales.numel() else 0.0,
           "matrix_block_absmax_median": float(source_scales.median().item()) if source_scales.numel() else 0.0}
    out["max_abs_ratio_to_matrix"] = _safe_ratio(value.abs().max(), source.abs().max()) if value.numel() else None
    return out


@torch.no_grad()
def metrics(reference: torch.Tensor, candidate: torch.Tensor, *, transform_kwargs: dict | None = None) -> dict:
    """Raw and exact production Muon metrics for a reconstruction."""
    row = {}
    row.update(_ratios(reference, candidate, "raw"))
    if reference.ndim != 2:
        row.update({"update_metric_status": "excluded_not_2d_muon_matrix",
                    "update_relative_l2": None, "update_cosine": None, "update_norm_ratio": None})
        return row
    kwargs = transform_kwargs or {}
    ref_update = muon_reference.zeropower_newton_schulz(reference.detach().clone(), **kwargs)
    cand_update = muon_reference.zeropower_newton_schulz(candidate.detach().clone(), **kwargs)
    row.update(_ratios(ref_update, cand_update, "update"))
    row["muon_update_relative_l2"] = row.get("update_relative_l2")
    row["muon_update_cosine"] = row.get("update_cosine")
    row["muon_update_norm_ratio"] = row.get("update_norm_ratio")
    return row


@torch.no_grad()
def exact_polar(matrix: torch.Tensor) -> torch.Tensor:
    value = matrix.detach().float()
    u, _, vh = torch.linalg.svd(value, full_matrices=False)
    return u @ vh


@torch.no_grad()
def danger_zone_energy(state: SVDState, error: torch.Tensor, *, low: float = -3.0, high: float = -2.0) -> dict:
    """Energy in the stored continuous danger-zone, using FP32 U/V coordinates."""
    ehat = state.u.T @ error.detach().float() @ state.vh.T
    s = state.singular_values
    active = s / s.max().clamp_min(torch.finfo(s.dtype).tiny) >= SPECTRAL_THRESHOLD if s.numel() else torch.zeros(0, dtype=torch.bool)
    z = torch.log10((s / s.max().clamp_min(torch.finfo(s.dtype).tiny)).clamp_min(torch.finfo(s.dtype).tiny)) if s.numel() else s
    danger = active & (z >= low) & (z < high)
    diag = torch.diag(ehat)
    diag_e = diag[danger].square().sum() if danger.numel() else torch.tensor(0.0)
    cross = ehat[danger, :].square().sum() + ehat[:, danger].square().sum() - diag_e * 2
    total = error.detach().float().square().sum()
    return {"danger_mode_count": int(danger.sum().item()),
            "danger_active_fraction": float(danger.float().mean().item()) if danger.numel() else 0.0,
            "danger_diagonal_energy": float(diag_e.item()),
            "danger_cross_energy": float(cross.clamp_min(0).item()),
            "danger_total_associated_energy": float((diag_e + cross.clamp_min(0)).item()),
            "danger_fraction_of_error": _safe_ratio(diag_e + cross.clamp_min(0), total),
            "danger_fraction_of_matrix": _safe_ratio(diag_e + cross.clamp_min(0), state.matrix.square().sum())}


@torch.no_grad()
def top_k_posthoc_correction(state: SVDState, direct: torch.Tensor, k: int) -> torch.Tensor:
    """Two-sided top-k projection of actual direct residual, applied post hoc."""
    k = valid_rank(state.singular_values, k)
    if k == 0:
        return direct.detach().float().clone()
    error = state.matrix - direct.detach().float()
    return direct.detach().float() + state.u[:, :k] @ (state.u[:, :k].T @ error @ state.vh[:k].T) @ state.vh[:k]


@torch.no_grad()
def deterministic_random_modes(rank: int, k: int, seed: int = 2026) -> torch.Tensor:
    """Return a deterministic mode set without touching global RNG state."""
    k = max(0, min(int(k), int(rank)))
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    return torch.randperm(rank, generator=generator)[:k].sort().values


@torch.no_grad()
def selected_mode_decomposition(state: SVDState, modes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    modes = modes.to(device=state.matrix.device, dtype=torch.long)
    low = (state.u[:, modes] * state.singular_values[modes]) @ state.vh[modes] if modes.numel() else torch.zeros_like(state.matrix)
    return low, state.matrix - low
