"""Deterministic approximate low-rank extraction for offline Muon studies.

The approximate path uses alternating subspace iteration and a small projected
SVD.  It deliberately never calls a full matrix SVD; exact SVD belongs only to
the evaluation/reference code in the analysis script.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ApproxFactors:
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    rank: int
    working_rank: int
    iterations: int
    oversampling: int
    initialization: str
    m_multiplies: int
    mt_multiplies: int

    # Compatibility names used by the earlier offline prototype.
    @property
    def q(self) -> int:
        return self.iterations

    @property
    def matmul_mx_count(self) -> int:
        return self.m_multiplies

    @property
    def matmul_mtx_count(self) -> int:
        return self.mt_multiplies

    @property
    def qr_count(self) -> int:
        return self.iterations + 1 if self.rank else 0


ApproximateFactors = ApproxFactors


def _basis(dim: int, width: int, initialization: str, seed: int) -> torch.Tensor:
    if width > dim:
        raise ValueError("working rank cannot exceed the smaller matrix dimension")
    if initialization == "canonical":
        return torch.eye(dim, width, dtype=torch.float32)
    if initialization != "randomized":
        raise ValueError(f"unknown initialization: {initialization}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(dim, width, generator=generator, dtype=torch.float32)


def _qr(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(value, mode="reduced").Q


@torch.no_grad()
def approximate_svd(matrix: torch.Tensor, rank: int, *, iterations: int = 0,
                    oversampling: int = 0, initialization: str = "canonical",
                    seed: int = 2026) -> ApproxFactors:
    """Return rank-k factors from subspace iteration and a small projected SVD.

    ``iterations=0`` means a single deterministic range projection followed by
    the projected solve; each positive iteration adds one alternating M/M^T
    refinement.  Only the projected matrix (at most ``k+p`` square) is SVD'd.
    """
    value = matrix.detach().float()
    if value.ndim != 2:
        raise ValueError("approximate extraction requires a 2D matrix")
    m, n = value.shape
    limit = min(m, n)
    k = max(0, min(int(rank), limit))
    if k == 0:
        return ApproxFactors(torch.zeros((m, 0)), torch.zeros(0),
                             torch.zeros((0, n)), 0, 0, int(iterations),
                             int(oversampling), initialization, 0, 0)
    if iterations < 0 or oversampling < 0:
        raise ValueError("iterations and oversampling must be non-negative")
    width = min(limit, k + int(oversampling))
    m_multiplies = 0
    mt_multiplies = 0
    if m >= n:
        right = _qr(_basis(n, width, initialization, seed))
        left = _qr(value @ right); m_multiplies += 1
        for _ in range(int(iterations)):
            right = _qr(value.T @ left); mt_multiplies += 1
            left = _qr(value @ right); m_multiplies += 1
    else:
        left = _qr(_basis(m, width, initialization, seed))
        right = _qr(value.T @ left); mt_multiplies += 1
        for _ in range(int(iterations)):
            left = _qr(value @ right); m_multiplies += 1
            right = _qr(value.T @ left); mt_multiplies += 1
    projected = left.T @ value @ right
    # Solve only the small projected problem.  Eigh avoids any full-matrix SVD
    # call in the approximate path and is stable for the PSD Gram matrix.
    gram = projected @ projected.T
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values, vectors = values[order].clamp_min(0), vectors[:, order]
    s = values[:k].sqrt()
    u = left @ vectors[:, :k]
    vh_small = vectors[:, :k].T @ projected
    vh_small = vh_small / s.clamp_min(torch.finfo(value.dtype).tiny).unsqueeze(1)
    vh = vh_small @ right.T
    return ApproxFactors(u, s, vh, k, width, int(iterations),
                         int(oversampling), initialization,
                         m_multiplies, mt_multiplies)


@torch.no_grad()
def reconstruct(factors: ApproxFactors) -> torch.Tensor:
    if factors.rank == 0:
        return torch.zeros((factors.u.shape[0], factors.vh.shape[1]), dtype=torch.float32)
    return (factors.u * factors.singular_values) @ factors.vh


@torch.no_grad()
def extract(matrix: torch.Tensor, rank: int, *, q: int = 0,
            oversampling: int = 0, initialization: str = "canonical",
            seed: int = 2026) -> ApproxFactors:
    """Backward-compatible alias for :func:`approximate_svd`."""
    return approximate_svd(matrix, rank, iterations=q, oversampling=oversampling,
                           initialization=initialization, seed=seed)


@torch.no_grad()
def subspace_projection_distance(exact: torch.Tensor, approx: torch.Tensor) -> float:
    if exact.numel() == 0 or approx.numel() == 0:
        return 0.0
    q1 = torch.linalg.qr(exact.float(), mode="reduced").Q
    q2 = torch.linalg.qr(approx.float(), mode="reduced").Q
    return float((q1 @ q1.T - q2 @ q2.T).norm().item())


@torch.no_grad()
def principal_angle_max(exact: torch.Tensor, approx: torch.Tensor) -> float | None:
    if exact.numel() == 0 or approx.numel() == 0:
        return None
    q1 = torch.linalg.qr(exact.float(), mode="reduced").Q
    q2 = torch.linalg.qr(approx.float(), mode="reduced").Q
    values = torch.linalg.svdvals(q1.T @ q2).clamp(0, 1)
    return float(torch.acos(values.min()).item()) if values.numel() else None


def factor_storage_bits(shape: tuple[int, int], rank: int, bits: int = 16) -> int:
    m, n = (int(x) for x in shape)
    return int(bits) * (m * int(rank) + n * int(rank) + int(rank))
