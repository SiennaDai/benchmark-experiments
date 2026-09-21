"""Offline 3x3 spectral-block contribution analysis for INT4 Muon.

The implementation is deliberately outside the optimizer path. It uses the
FP32 SVD basis only as an analysis coordinate system, delegates quantization
to the existing persist_state wrapper and delegates Muon updates to the
production reference Newton--Schulz implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from . import muon_reference
from .muon_spectral_contribution import ACCOUNTING_BANDS, BANDS, BandDecomposition, decompose
from .muon_spectral_sensitivity import quantize

PRIMARY_BLOCKS = tuple(f"{row}{col}" for row in "HMT" for col in "HMT")
_SYMBOL = {"head": "H", "middle": "M", "tail": "T"}
GROUPS = {
    "none": (),
    "HH": ("HH",), "HM": ("HM",), "HT": ("HT",),
    "MH": ("MH",), "MM": ("MM",), "MT": ("MT",),
    "TH": ("TH",), "TM": ("TM",), "TT": ("TT",),
    "all_diag": ("HH", "MM", "TT"),
    "all_offdiag": ("HM", "HT", "MH", "MT", "TH", "TM"),
    "H_to_M": ("HM", "MH"), "M_to_T": ("MT", "TM"), "H_to_T": ("HT", "TH"),
    "U_head_rows": ("HH", "HM", "HT"),
    "U_middle_rows": ("MH", "MM", "MT"),
    "U_tail_rows": ("TH", "TM", "TT"),
    "V_head_cols": ("HH", "MH", "TH"),
    "V_middle_cols": ("HM", "MM", "TM"),
    "V_tail_cols": ("HT", "MT", "TT"),
}


@dataclass(frozen=True)
class BlockEnergy:
    """Squared Frobenius energies in primary and accounting blocks."""

    total: float
    projected: float
    unresolved: float
    blocks: dict[str, float]


def _indices(d: BandDecomposition, band: str) -> torch.Tensor:
    if band not in ACCOUNTING_BANDS:
        raise ValueError(f"unknown accounting band: {band}")
    return d.bands[band]


def _block_name(row: str, col: str) -> str:
    return (_SYMBOL[row] + _SYMBOL[col]) if row in _SYMBOL and col in _SYMBOL else f"{row}_{col}"


@torch.no_grad()
def spectral_coordinates(d: BandDecomposition, quantized: torch.Tensor) -> torch.Tensor:
    """Return U.T @ (Q(M)-M) @ V for the reduced FP32 SVD."""
    return d.u.T @ (quantized.detach().float() - d.matrix) @ d.vh.T


@torch.no_grad()
def block_component(d: BandDecomposition, quantized: torch.Tensor,
                    row_band: str, col_band: str) -> torch.Tensor:
    """Return one orthogonal row/column spectral block of the residual."""
    rows, cols = _indices(d, row_band), _indices(d, col_band)
    result = torch.zeros_like(d.matrix)
    if rows.numel() and cols.numel():
        coords = spectral_coordinates(d, quantized)
        result = d.u[:, rows] @ coords[rows][:, cols] @ d.vh[cols, :]
    return result


@torch.no_grad()
def all_components(d: BandDecomposition, quantized: torch.Tensor) -> dict[str, torch.Tensor]:
    components = {_block_name(row, col): block_component(d, quantized, row, col)
                  for row in ACCOUNTING_BANDS for col in ACCOUNTING_BANDS}
    residual = quantized.detach().float() - d.matrix
    projected = d.u @ spectral_coordinates(d, quantized) @ d.vh
    components["unresolved"] = residual - projected
    return components


@torch.no_grad()
def block_energies(d: BandDecomposition, quantized: torch.Tensor) -> BlockEnergy:
    coords = spectral_coordinates(d, quantized)
    residual = quantized.detach().float() - d.matrix
    blocks: dict[str, float] = {}
    for row in ACCOUNTING_BANDS:
        for col in ACCOUNTING_BANDS:
            rows, cols = _indices(d, row), _indices(d, col)
            value = coords[rows][:, cols].square().sum() if rows.numel() and cols.numel() else coords.new_zeros(())
            blocks[_block_name(row, col)] = float(value)
    projected = float(coords.square().sum())
    total = float(residual.square().sum())
    return BlockEnergy(total, projected, max(0.0, total - projected), blocks)


@torch.no_grad()
def restore_blocks(d: BandDecomposition, quantized: torch.Tensor,
                   blocks: Iterable[str]) -> torch.Tensor:
    """Remove exactly the selected actual residual blocks from Q(M)."""
    corrected = quantized.detach().float().clone()
    reverse = {"H": "head", "M": "middle", "T": "tail"}
    for name in blocks:
        if name == "unresolved":
            residual = quantized.detach().float() - d.matrix
            projected = d.u @ spectral_coordinates(d, quantized) @ d.vh
            corrected = corrected - (residual - projected)
            continue
        if len(name) == 2 and name[0] in reverse and name[1] in reverse:
            row_band, col_band = reverse[name[0]], reverse[name[1]]
        elif "_" in name:
            row_band, col_band = name.split("_", 1)
        else:
            raise ValueError(f"invalid spectral block {name!r}")
        corrected = corrected - block_component(d, quantized, row_band, col_band)
    return corrected


@torch.no_grad()
def exact_polar(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the reduced-SVD polar factor, preserving rectangular shapes."""
    value = matrix.detach().float()
    u, _, vh = torch.linalg.svd(value, full_matrices=False)
    return u @ vh


def _safe(value: torch.Tensor | float | None) -> float | None:
    if value is None:
        return None
    result = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
    return result if math.isfinite(result) else None


@torch.no_grad()
def metric_pair(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | None]:
    ref, obs = reference.detach().float(), candidate.detach().float()
    ref_norm, obs_norm = ref.norm(), obs.norm()
    error = (obs - ref).norm()
    denom = ref_norm * obs_norm
    return {
        "relative_l2": _safe(error / ref_norm) if ref_norm.item() else None,
        "cosine": _safe((obs * ref).sum() / denom) if denom.item() else None,
        "norm_ratio": _safe(obs_norm / ref_norm) if ref_norm.item() else None,
    }


@torch.no_grad()
def paired_restore_metrics(source: torch.Tensor, quantized: torch.Tensor,
                           corrected: torch.Tensor, *, transform_kwargs: dict,
                           reference_update: torch.Tensor | None = None,
                           baseline_update: torch.Tensor | None = None,
                           polar: bool = False) -> dict[str, float | None]:
    """Return raw and update metrics against the FP32 reference."""
    if reference_update is None:
        reference_update = (exact_polar(source) if polar else
                            muon_reference.zeropower_newton_schulz(source.clone(), **transform_kwargs))
    if baseline_update is None:
        baseline_update = (exact_polar(quantized) if polar else
                           muon_reference.zeropower_newton_schulz(quantized.clone(), **transform_kwargs))
    candidate_update = exact_polar(corrected) if polar else muon_reference.zeropower_newton_schulz(corrected.clone(), **transform_kwargs)
    raw = metric_pair(source, corrected)
    update = metric_pair(reference_update, candidate_update)
    baseline = metric_pair(reference_update, baseline_update)
    return {
        "raw_relative_l2": raw["relative_l2"], "raw_cosine": raw["cosine"], "raw_norm_ratio": raw["norm_ratio"],
        "update_relative_l2": update["relative_l2"], "update_cosine": update["cosine"], "update_norm_ratio": update["norm_ratio"],
        "baseline_update_relative_l2": baseline["relative_l2"], "baseline_update_cosine": baseline["cosine"],
        "update_cosine_gain": (update["cosine"] - baseline["cosine"] if update["cosine"] is not None and baseline["cosine"] is not None else None),
        "update_l2_reduction": (baseline["relative_l2"] - update["relative_l2"] if baseline["relative_l2"] is not None and update["relative_l2"] is not None else None),
    }


def primary_energy_fractions(energy: BlockEnergy) -> dict[str, float | None]:
    denom = energy.total
    out = {name: (energy.blocks.get(name, 0.0) / denom if denom else None) for name in PRIMARY_BLOCKS}
    out["diagonal"] = sum(out.get(name, 0.0) or 0.0 for name in ("HH", "MM", "TT"))
    out["offdiagonal"] = sum(out.get(name, 0.0) or 0.0 for name in ("HM", "HT", "MH", "MT", "TH", "TM"))
    out["other_associated"] = sum(value for name, value in out.items() if len(name) == 2 and name.startswith("other") and value is not None)
    return out


__all__ = ["ACCOUNTING_BANDS", "BANDS", "GROUPS", "PRIMARY_BLOCKS", "BlockEnergy", "BandDecomposition",
           "all_components", "block_component", "block_energies", "decompose", "exact_polar",
           "metric_pair", "paired_restore_metrics", "primary_energy_fractions", "restore_blocks", "spectral_coordinates"]
