"""Offline quantization-aware spectral conditioner helpers.

This module is intentionally outside the optimizer.  It searches subsets of
the *existing* FP32 singular components and quantizes only the resulting
residual.  The INT4 path delegates to the production diagnostic quantizer;
the INT3 path is an explicitly documented, deterministic analogue used only
by this offline study.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from . import muon_reference
from .muon_spectral_sensitivity import quantize as production_quantize

BLOCK_SIZE = 2048
SPECTRAL_THRESHOLD = 1e-6
INT3_CODEBOOK = torch.tensor([-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0,
                              1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=torch.float32)


@dataclass(frozen=True)
class ConditionerSelection:
    method: str
    bits: int
    rank: int
    modes: tuple[int, ...]
    candidate_pool: int
    objective_start: float
    objective_final: float
    candidate_evaluations: int


def int3_dynamic_roundtrip(value: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Three-bit signed dynamic roundtrip used by this offline oracle.

    Each block uses its absmax as scale and nearest lookup in the symmetric
    seven-level codebook ``{-1,-2/3,-1/3,0,1/3,2/3,1}``.  The eighth binary
    code is reserved, so zero is represented exactly and the codebook remains
    symmetric.  This is not a production quantizer and is never used by the
    optimizer.
    """
    x = value.detach().float()
    flat = x.reshape(-1)
    if flat.numel() == 0:
        return x.clone()
    out = torch.empty_like(flat)
    cb = INT3_CODEBOOK.to(flat)
    for start in range(0, flat.numel(), int(block_size)):
        block = flat[start:start + int(block_size)]
        scale = block.abs().amax()
        if scale.item() == 0:
            out[start:start + block.numel()] = block
        else:
            z = (block / scale).clamp(-1, 1)
            idx = torch.searchsorted(cb, z).clamp(0, cb.numel() - 1)
            lo = (idx - 1).clamp_min(0)
            choose_hi = (z - cb[lo]).abs() > (cb[idx] - z).abs()
            q = torch.where(choose_hi, cb[idx], cb[lo])
            out[start:start + block.numel()] = q * scale
    return out.reshape_as(x)


def quantize_residual(value: torch.Tensor, bits: int) -> torch.Tensor:
    if int(bits) == 4:
        return production_quantize(value, "int4-dynamic-b2048").float()
    if int(bits) == 3:
        return int3_dynamic_roundtrip(value)
    raise ValueError("offline conditioner supports only INT4 and INT3")


def active_modes(singular_values: torch.Tensor, threshold: float = SPECTRAL_THRESHOLD) -> list[int]:
    s = singular_values.detach().float()
    if not s.numel() or s[0].item() == 0:
        return []
    return [int(i) for i in torch.nonzero(s / s[0] >= threshold, as_tuple=False).flatten()]


def mode_component(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor, i: int) -> torch.Tensor:
    return torch.outer(u[:, int(i)] * s[int(i)], vh[int(i)])


def subset_component(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor, modes: list[int] | tuple[int, ...]) -> torch.Tensor:
    if not modes:
        return torch.zeros((u.shape[0], vh.shape[1]), dtype=torch.float32, device=u.device)
    idx = torch.tensor(list(modes), dtype=torch.long, device=u.device)
    return (u[:, idx] * s[idx]) @ vh[idx]


def block_absmax_mean(value: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    flat = value.detach().float().reshape(-1)
    if not flat.numel():
        return torch.tensor(0.0)
    return torch.stack([flat[i:i + block_size].abs().amax() for i in range(0, flat.numel(), block_size)]).mean()


def _objective(value: torch.Tensor, method: str, bits: int, *, reference_matrix: torch.Tensor | None = None,
               reference_update: torch.Tensor | None = None, transform_kwargs: dict | None = None) -> float:
    if method == "range_aware":
        return float(block_absmax_mean(value).item())
    q = quantize_residual(value, bits)
    if method == "quant_error_aware":
        return float((q - value).norm().item())
    if method == "muon_update_aware":
        if reference_update is None:
            raise ValueError("reference_update is required for Muon-aware selection")
        kw = transform_kwargs or {}
        if reference_matrix is None:
            raise ValueError("reference_matrix is required for Muon-aware selection")
        # ``value`` is the residual after the currently selected components;
        # put those components back before evaluating the actual reconstructed
        # state.  Evaluating O(Q(value)) alone would optimize the wrong object.
        update = muon_reference.zeropower_newton_schulz((reference_matrix - value + q).clone(), **kw)
        rn, un = reference_update.norm(), update.norm()
        cosine = (reference_update * update).sum() / (rn * un).clamp_min(torch.finfo(torch.float32).tiny)
        return float((1 - cosine).item())
    raise ValueError(f"unsupported objective: {method}")


@torch.no_grad()
def greedy_select(matrix: torch.Tensor, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor,
                  rank: int, bits: int, method: str, *, candidate_pool: int = 32,
                  reference_update: torch.Tensor | None = None,
                  transform_kwargs: dict | None = None) -> ConditionerSelection:
    """Greedy spectral-mode selection using only the declared objective."""
    if method not in {"range_aware", "quant_error_aware", "muon_update_aware"}:
        raise ValueError("greedy method must be range_aware, quant_error_aware, or muon_update_aware")
    n = min(int(candidate_pool), int(s.numel()))
    candidates = list(range(n))
    current = matrix.detach().float().clone()
    start = _objective(current, method, bits, reference_matrix=matrix, reference_update=reference_update, transform_kwargs=transform_kwargs)
    selected: list[int] = []
    evaluations = 0
    for _ in range(min(int(rank), n)):
        best = None
        best_value = math.inf
        for i in candidates:
            if i in selected:
                continue
            value = current - mode_component(u, s, vh, i)
            score = _objective(value, method, bits, reference_matrix=matrix, reference_update=reference_update, transform_kwargs=transform_kwargs)
            evaluations += 1
            if score < best_value - 1e-12 or (abs(score - best_value) <= 1e-12 and (best is None or i < best)):
                best, best_value = i, score
        if best is None:
            break
        selected.append(best)
        current = current - mode_component(u, s, vh, best)
    return ConditionerSelection(method, int(bits), int(rank), tuple(selected), n, start, best_value if selected else start, evaluations)


def selected_reconstruction(matrix: torch.Tensor, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor,
                            modes: list[int] | tuple[int, ...], bits: int, *, factor_dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (FP32 conditioner, quantized residual, reconstructed state)."""
    c = subset_component(u, s, vh, modes)
    residual = matrix.detach().float() - c
    qres = quantize_residual(residual, bits)
    idx = torch.tensor(list(modes), dtype=torch.long, device=u.device)
    if idx.numel():
        uf = u[:, idx].to(factor_dtype).float()
        sf = s[idx].to(factor_dtype).float()
        vhf = vh[idx].to(factor_dtype).float()
        c_stored = (uf * sf) @ vhf
    else:
        c_stored = torch.zeros_like(matrix, dtype=torch.float32)
    return c, qres, c_stored + qres


def cross_scale_fractions(state, error: torch.Tensor) -> dict[str, float]:
    ehat = state.u.T @ error.detach().float() @ state.vh.T
    total = float(ehat.square().sum().item())
    s = state.singular_values.float()
    if not total or not s.numel() or s[0].item() == 0:
        return {"local": 0.0, "medium": 0.0, "distant": 0.0}
    z = torch.log10((s / s[0]).clamp_min(1e-6))
    d = (z[:, None] - z[None, :]).abs()
    off = ~torch.eye(len(s), dtype=torch.bool, device=s.device)
    out = {}
    for name, lo, hi in (("local", 0.0, 0.5), ("medium", 0.5, 1.5), ("distant", 1.5, float("inf"))):
        mask = (d >= lo) & (d < hi) & off
        out[name] = float(ehat[mask].square().sum().item() / total)
    return out
