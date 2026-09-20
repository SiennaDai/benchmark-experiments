"""Offline spectral diagnostics for Muon momentum quantization.

The functions in this module are deliberately outside the optimizer path. A
caller supplies a detached FP32 snapshot tensor; quantization and
orthogonalization are delegated to the production implementations. No
optimizer state is mutated and no random numbers are consumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import muon_reference
from .muon_update_fidelity import QUANTIZERS, _ratios
from .state_simulation import persist_state

BLOCK_SIZE = 2048
SPECTRAL_THRESHOLD = 1e-6
QUANTIZER_TO_SIMULATION = {
    "int8-linear-b2048": "int8_linear_momentum",
    "int4-linear-b2048": "int4_linear_momentum",
    "int4-dynamic-b2048": "int4_dynamic_momentum",
}


@dataclass
class SpectralDecomposition:
    matrix: torch.Tensor
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor


@torch.no_grad()
def decompose(matrix: torch.Tensor) -> SpectralDecomposition:
    """Return a reduced FP32 SVD, retaining the decomposition for reuse."""
    matrix = matrix.detach().float()
    if matrix.ndim != 2:
        raise ValueError("spectral analysis requires a 2D matrix")
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return SpectralDecomposition(matrix=matrix, u=u, singular_values=s, vh=vh)


def _finite_number(value: torch.Tensor | float) -> float | None:
    value = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
    return value if math.isfinite(value) else None


def _quantile(s: torch.Tensor, q: float) -> float | None:
    return _finite_number(torch.quantile(s, q)) if s.numel() else None


def spectral_metrics(s: torch.Tensor, *, threshold: float = SPECTRAL_THRESHOLD) -> dict:
    """Compute spectrum summaries with explicit effective-rank semantics.

    Singular values are assumed sorted descending. ``effective_rank`` counts
    values with ``sigma_i / sigma_max >= threshold`` and the effective
    condition number divides sigma_max by the last value in that rank. The
    naive condition is unavailable (``None``) for a zero minimum, rather than
    presenting a meaningless machine-infinity. Entropy uses
    ``exp(-sum p log p)`` for ``p=sigma/sum(sigma)``; stable rank is
    ``sum(sigma**2)/sigma_max**2``. Tail energy is the squared-energy share of
    the last ceil(10%) or ceil(25%) index modes.
    """
    s = s.detach().float().abs()
    if not s.numel():
        return {
            "sigma_max": None, "sigma_min": None, "naive_condition_number": None,
            "effective_rank": 0, "effective_condition_number": None,
            "effective_rank_entropy": None, "stable_rank": None,
            "sigma_p10": None, "sigma_p25": None, "sigma_median": None,
            "sigma_p75": None, "sigma_p90": None,
            "tail_energy_fraction_10pct": None, "tail_energy_fraction_25pct": None,
        }
    smax, smin = s[0], s[-1]
    total_sq = s.square().sum()
    rank = int((s / smax >= threshold).sum().item()) if smax.item() > 0 else 0
    rank_values = s[:rank]
    p = s / s.sum().clamp_min(torch.finfo(s.dtype).tiny)
    entropy = torch.exp(-(p * p.clamp_min(torch.finfo(s.dtype).tiny).log()).sum())
    n10 = max(1, math.ceil(s.numel() * 0.10))
    n25 = max(1, math.ceil(s.numel() * 0.25))
    return {
        "sigma_max": _finite_number(smax),
        "sigma_min": _finite_number(smin),
        "naive_condition_number": _finite_number(smax / smin) if smin.item() > 0 else None,
        "effective_rank": rank,
        "effective_condition_number": _finite_number(smax / rank_values[-1]) if rank else None,
        "effective_rank_entropy": _finite_number(entropy),
        "stable_rank": _finite_number(total_sq / smax.square()) if smax.item() > 0 else None,
        "sigma_p10": _quantile(s, 0.10), "sigma_p25": _quantile(s, 0.25),
        "sigma_median": _quantile(s, 0.50), "sigma_p75": _quantile(s, 0.75),
        "sigma_p90": _quantile(s, 0.90),
        "tail_energy_fraction_10pct": _finite_number(s[-n10:].square().sum() / total_sq) if total_sq.item() else None,
        "tail_energy_fraction_25pct": _finite_number(s[-n25:].square().sum() / total_sq) if total_sq.item() else None,
    }


@torch.no_grad()
def quantize(matrix: torch.Tensor, quantizer: str) -> torch.Tensor:
    """Apply an existing production quantizer to a diagnostic copy."""
    if quantizer not in QUANTIZER_TO_SIMULATION:
        raise ValueError(f"unsupported quantizer: {quantizer}")
    return persist_state(
        matrix.detach().clone(), QUANTIZER_TO_SIMULATION[quantizer], "muon_momentum",
        quantization_granularity="blockwise", quantization_block_size=BLOCK_SIZE,
    )


@torch.no_grad()
def transform(matrix: torch.Tensor, *, steps=5,
              coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7) -> torch.Tensor:
    """Call the exact production Muon transform on a diagnostic copy."""
    return muon_reference.zeropower_newton_schulz(
        matrix.detach().clone(), steps, coefficients, eps,
    )


@torch.no_grad()
def update_metrics(source: torch.Tensor, quantized: torch.Tensor, *, transform_kwargs=None) -> dict:
    """Compare exact production Muon transforms of two matrices."""
    kwargs = transform_kwargs or {}
    reference = transform(source, **kwargs)
    observed = transform(quantized, **kwargs)
    metrics = _ratios(reference, observed, "update")
    metrics.update({f"muon_{key}": metrics.get(f"update_{key}")
                    for key in ("relative_l2", "cosine", "norm_ratio")})
    return metrics


def band_slices(rank: int) -> dict[str, slice]:
    """Top, centered-middle, and tail bands by deterministic index fraction."""
    if rank <= 0:
        return {"top": slice(0, 0), "middle": slice(0, 0), "tail": slice(0, 0)}
    width = max(1, math.ceil(rank * 0.10))
    middle_start = max(0, rank // 2 - width // 2)
    return {
        "top": slice(0, width),
        "middle": slice(middle_start, min(rank, middle_start + width)),
        "tail": slice(max(0, rank - width), rank),
    }


def subspace_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Summarize principal angles and projection distance for two bases."""
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("subspaces must have equal 2D basis shapes")
    width = a.shape[1]
    if width == 0:
        return {"principal_angle_mean": None, "principal_angle_max": None,
                "projection_distance": None, "projection_distance_normalized": None}
    # SVD outputs are orthonormal; use singular values of A^T B so repeated
    # singular vectors are treated as a subspace rather than paired vectors.
    cosines = torch.linalg.svdvals(a.T @ b).clamp(0.0, 1.0)
    angles = torch.arccos(cosines)
    projection_distance = torch.linalg.norm(a @ a.T - b @ b.T)
    return {
        "principal_angle_mean": _finite_number(angles.mean()),
        "principal_angle_max": _finite_number(angles.max()),
        "projection_distance": _finite_number(projection_distance),
        "projection_distance_normalized": _finite_number(projection_distance / math.sqrt(width)),
    }


@torch.no_grad()
def quantized_spectral_metrics(source: SpectralDecomposition, quantized: torch.Tensor,
                               *, threshold: float = SPECTRAL_THRESHOLD) -> dict:
    """Compute Q(M)'s spectrum and error against a cached M decomposition."""
    q = decompose(quantized)
    source_metrics = spectral_metrics(source.singular_values, threshold=threshold)
    quantized_metrics = spectral_metrics(q.singular_values, threshold=threshold)
    error = q.matrix - source.matrix
    source_norm = source.singular_values.norm()
    return {
        "quantized": q,
        "spectral_norm_error": _finite_number(torch.linalg.matrix_norm(error, ord=2)),
        "frobenius_error": _finite_number(torch.linalg.vector_norm(error)),
        "relative_singular_value_l2": _finite_number(torch.linalg.vector_norm(q.singular_values - source.singular_values) / source_norm) if source_norm.item() else None,
        "effective_rank_change": quantized_metrics["effective_rank"] - source_metrics["effective_rank"],
        "condition_number_change": (
            quantized_metrics["effective_condition_number"] - source_metrics["effective_condition_number"]
            if quantized_metrics["effective_condition_number"] is not None
            and source_metrics["effective_condition_number"] is not None else None
        ),
        "source_metrics": source_metrics,
        "quantized_metrics": quantized_metrics,
    }


@torch.no_grad()
def spectral_error_decomposition(source: SpectralDecomposition, quantized: torch.Tensor,
                                 *, threshold: float = SPECTRAL_THRESHOLD) -> dict:
    """Project E=Q(M)-M into the cached FP32 U,V coordinates.

    For rectangular matrices with more columns than rows, the reduced SVD
    basis does not span the right null space. ``unresolved_error_energy``
    explicitly records that residual instead of silently treating projected
    energy as total error.
    """
    error = quantized.detach().float() - source.matrix
    error_hat = source.u.T @ error @ source.vh.T
    projected_error = source.u @ error_hat @ source.vh
    total_energy = error.square().sum()
    projected_energy = error_hat.square().sum()
    diagonal = error_hat.diagonal()
    s = source.singular_values
    denom = torch.where(
        s / s.max().clamp_min(torch.finfo(s.dtype).tiny) >= threshold,
        s.abs(), torch.full_like(s, float("nan")),
    )
    relative_mode = diagonal.abs() / denom
    bands = band_slices(s.numel())
    out = {
        "diagonal_error_energy": _finite_number(diagonal.square().sum()),
        "off_diagonal_error_energy": _finite_number(error_hat.square().sum() - diagonal.square().sum()),
        "total_error_energy": _finite_number(total_energy),
        "projected_error_energy": _finite_number(projected_energy),
        "unresolved_error_energy": _finite_number((total_energy - projected_energy).clamp_min(0)),
    }
    for name, sl in bands.items():
        mode_values = relative_mode[sl]
        mode_energy = error_hat[sl, :].square().sum()
        diag_energy = diagonal[sl].square().sum()
        out[f"{name}_relative_mode_perturbation_mean"] = (
            _finite_number(torch.nanmean(mode_values)) if torch.isfinite(mode_values).any() else None
        )
        finite_values = mode_values[torch.isfinite(mode_values)]
        out[f"{name}_relative_mode_perturbation_max"] = (
            _finite_number(finite_values.max()) if finite_values.numel() else None
        )
        out[f"{name}_error_energy_fraction"] = _finite_number(mode_energy / total_energy) if total_energy.item() else None
        out[f"{name}_diagonal_error_energy_fraction"] = _finite_number(diag_energy / total_energy) if total_energy.item() else None
    return out


@torch.no_grad()
def conditioning_intervention(source: SpectralDecomposition, tau_ratio: float, *, quantizer: str,
                              transform_kwargs=None) -> dict:
    """Construct M_tau=U diag(max(sigma,tau)) V^T and analyze it."""
    tau = source.singular_values.max() * float(tau_ratio)
    singular_values = torch.maximum(source.singular_values, tau)
    intervened = (source.u * singular_values) @ source.vh
    quantized = quantize(intervened, quantizer)
    result = {
        "tau_ratio": float(tau_ratio),
        "conditioned_matrix": intervened,
        "conditioned": spectral_metrics(torch.linalg.svdvals(intervened)),
        "quantized": quantized,
    }
    result.update(_ratios(intervened, quantized, "raw"))
    result.update(update_metrics(intervened, quantized, transform_kwargs=transform_kwargs))
    return result
