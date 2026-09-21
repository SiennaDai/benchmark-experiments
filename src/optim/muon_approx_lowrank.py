"""CPU-only approximate low-rank extraction for offline Muon studies.

The routines in this module are deliberately separate from the optimizer.  The
approximate path uses range/subspace iteration and an eigendecomposition of the
small projected Gram matrix; it never calls a full matrix SVD.  Full SVDs are
the responsibility of the caller's reference/evaluation path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ApproximateFactors:
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    q: int
    oversampling: int
    initialization: str
    matmul_mx_count: int
    matmul_mtx_count: int
    qr_count: int

    @property
    def rank(self) -> int:
        return int(self.singular_values.numel())


def _generator(seed: int, device: torch.device) -> torch.Generator:
    # A private generator keeps diagnostics from consuming experiment RNG.
    g = torch.Generator(device=device.type if device.type != "cuda" else device)
    g.manual_seed(int(seed))
    return g


def _initial_omega(n: int, l: int, initialization: str, seed: int,
                   *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if initialization == "canonical":
        omega = torch.zeros((n, l), dtype=dtype, device=device)
        omega[torch.arange(l, device=device), torch.arange(l, device=device)] = 1
        return omega
    if initialization == "randomized":
        g = _generator(seed, device)
        return torch.randn((n, l), generator=g, dtype=dtype, device=device)
    raise ValueError(f"unknown initialization: {initialization}")


@torch.no_grad()
def extract(matrix: torch.Tensor, rank: int, *, q: int = 0,
            oversampling: int = 0, initialization: str = "canonical",
            seed: int = 2026) -> ApproximateFactors:
    """Extract approximate rank-``rank`` factors without a full SVD.

    ``q=0`` is a one-pass randomized/canonical range finder.  Each positive
    q adds one subspace iteration ``Y <- M(M^T Q)``.  The final projected
    problem is solved by ``eigh(B B^T)`` (dimension at most k+p), not by a
    full SVD.  Factors are returned in FP32 and are detached copies.
    """
    x = matrix.detach().float()
    if x.ndim != 2:
        raise ValueError("approximate extraction requires a 2D matrix")
    if int(rank) < 0 or int(q) < 0 or int(oversampling) < 0:
        raise ValueError("rank, q, and oversampling must be non-negative")
    m, n = map(int, x.shape)
    max_rank = min(m, n)
    k = min(int(rank), max_rank)
    if k == 0:
        return ApproximateFactors(torch.zeros((m, 0), dtype=x.dtype),
                                  torch.zeros((0,), dtype=x.dtype),
                                  torch.zeros((0, n), dtype=x.dtype), int(q),
                                  int(oversampling), initialization, 0, 0, 0)
    l = min(max_rank, k + int(oversampling))
    omega = _initial_omega(n, l, initialization, seed, dtype=x.dtype, device=x.device)
    y = x @ omega
    mx, mtx, qr = 1, 0, 0
    qbasis, _ = torch.linalg.qr(y, mode="reduced"); qr += 1
    for _ in range(int(q)):
        z = x.T @ qbasis; mtx += 1
        y = x @ z; mx += 1
        qbasis, _ = torch.linalg.qr(y, mode="reduced"); qr += 1
    # Solve the small projected problem without torch.linalg.svd.  Eigenvalues
    # are nonnegative up to roundoff; clamp only that roundoff.
    b = qbasis.T @ x
    gram = b @ b.T
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], vectors[:, order]
    values = values.clamp_min(0)
    uk = qbasis @ vectors[:, :k]
    sk = values[:k].sqrt()
    vh = vectors[:, :k].T @ b
    vh = vh / sk.clamp_min(torch.finfo(x.dtype).tiny).unsqueeze(1)
    # Canonical sign convention makes repeated offline runs byte-stable.
    # Avoid sign-dependent comparisons while retaining deterministic factors.
    piv = uk.abs().argmax(dim=0)
    signs = torch.where(uk[piv, torch.arange(k, device=x.device)] < 0, -uk.new_ones(k), uk.new_ones(k))
    uk, vh = uk * signs, signs.unsqueeze(1) * vh
    return ApproximateFactors(uk.contiguous(), sk.contiguous(), vh.contiguous(),
                              int(q), int(oversampling), initialization,
                              mx, mtx, qr)


@torch.no_grad()
def reconstruct(factors: ApproximateFactors) -> torch.Tensor:
    return (factors.u * factors.singular_values) @ factors.vh


@torch.no_grad()
def subspace_projection_distance(exact: torch.Tensor, approx: torch.Tensor) -> float:
    """Frobenius distance between equal-rank orthogonal projectors."""
    if exact.numel() == 0 or approx.numel() == 0:
        return 0.0
    q1, _ = torch.linalg.qr(exact.float(), mode="reduced")
    q2, _ = torch.linalg.qr(approx.float(), mode="reduced")
    return float((q1 @ q1.T - q2 @ q2.T).norm().item())


def principal_angle_max(exact: torch.Tensor, approx: torch.Tensor) -> float | None:
    if exact.numel() == 0 or approx.numel() == 0:
        return None
    q1, _ = torch.linalg.qr(exact.float(), mode="reduced")
    q2, _ = torch.linalg.qr(approx.float(), mode="reduced")
    overlap = q1.T @ q2
    eig = torch.linalg.eigvalsh(overlap @ overlap.T).clamp(0, 1)
    return float(torch.acos(eig.min().sqrt()).item()) if eig.numel() else None
