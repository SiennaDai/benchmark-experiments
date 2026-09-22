"""Offline Gram and singular-subspace diagnostics for Muon state error."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def gram_perturbation_metrics(matrix: torch.Tensor, reconstruction: torch.Tensor,
                             reference_spectral_norm: float | None = None) -> dict[str, float]:
    """Measure both Gram perturbations and verify their exact residual identity.

    All inputs are detached FP32 copies. Spectral norms are computed from the
    symmetric Gram perturbations' eigenvalues; no production path is involved.
    """
    m = matrix.detach().float()
    mh = reconstruction.detach().float()
    if m.ndim != 2 or mh.shape != m.shape:
        raise ValueError("Gram diagnostics require same-shape 2-D matrices")
    e = mh - m

    def side(left: bool) -> dict[str, float]:
        if left:
            g = m @ m.T
            gh = mh @ mh.T
            linear = m @ e.T + e @ m.T
            quadratic = e @ e.T
            label = "left"
        else:
            g = m.T @ m
            gh = mh.T @ mh
            linear = m.T @ e + e.T @ m
            quadratic = e.T @ e
            label = "right"
        delta = gh - g
        # Symmetrization removes only round-off asymmetry; the defining matrices
        # are symmetric by construction.
        delta = (delta + delta.T) * 0.5
        g = (g + g.T) * 0.5
        gh = (gh + gh.T) * 0.5
        linear = (linear + linear.T) * 0.5
        quadratic = (quadratic + quadratic.T) * 0.5
        g_f = torch.linalg.matrix_norm(g, ord="fro")
        d_f = torch.linalg.matrix_norm(delta, ord="fro")
        l_f = torch.linalg.matrix_norm(linear, ord="fro")
        q_f = torch.linalg.matrix_norm(quadratic, ord="fro")
        eig = torch.linalg.eigvalsh(delta)
        d_2 = eig.abs().max() if eig.numel() else torch.zeros(())
        if reference_spectral_norm is None:
            g_eig = torch.linalg.eigvalsh(g)
            g_2 = g_eig[-1].clamp_min(0) if g_eig.numel() else torch.zeros(())
        else:
            g_2 = torch.as_tensor(float(reference_spectral_norm) ** 2, dtype=g.dtype, device=g.device)
        gh_f = torch.linalg.matrix_norm(gh, ord="fro")
        cosine = (g * gh).sum() / (g_f * gh_f).clamp_min(torch.finfo(torch.float32).tiny)
        tr = torch.trace(g).abs()
        return {
            f"{label}_gram_relative_fro": float(d_f / g_f.clamp_min(torch.finfo(torch.float32).tiny)),
            f"{label}_gram_relative_spectral": float(d_2 / g_2.clamp_min(torch.finfo(torch.float32).tiny)),
            f"{label}_gram_fro_cosine": float(cosine),
            f"{label}_gram_trace_relative": float(torch.trace(delta).abs() / tr.clamp_min(torch.finfo(torch.float32).tiny)),
            f"{label}_gram_linear_relative_fro": float(l_f / g_f.clamp_min(torch.finfo(torch.float32).tiny)),
            f"{label}_gram_quadratic_relative_fro": float(q_f / g_f.clamp_min(torch.finfo(torch.float32).tiny)),
            f"{label}_gram_quadratic_over_delta_fro": float(q_f / d_f.clamp_min(torch.finfo(torch.float32).tiny)),
            f"{label}_gram_delta_fro": float(d_f),
            f"{label}_gram_identity_residual": float(torch.linalg.matrix_norm(delta - linear - quadratic, ord="fro")),
        }

    return {**side(False), **side(True)}


def active_band_indices(singular_values: torch.Tensor, threshold: float = 1e-6) -> dict[str, list[int]]:
    """Prior deterministic convention: top/middle/tail 10% of active modes."""
    s = singular_values.detach().float().flatten()
    if not s.numel() or float(s[0]) <= 0:
        return {"head": [], "middle": [], "tail": []}
    active = int((s / s[0] >= float(threshold)).sum())
    if active == 0:
        return {"head": [], "middle": [], "tail": []}
    width = max(1, math.ceil(active * 0.10))
    middle_start = max(0, active // 2 - width // 2)
    return {
        "head": list(range(0, width)),
        "middle": list(range(middle_start, min(active, middle_start + width))),
        "tail": list(range(active - width, active)),
    }


def subspace_angle_metrics(u: torch.Tensor, u_hat: torch.Tensor,
                           indices: Sequence[int]) -> dict[str, float | int]:
    """Principal-angle sine summaries for equal-index spectral subspaces."""
    idx = list(map(int, indices))
    if not idx:
        return {"dimension": 0, "mean_sin": float("nan"), "max_sin": float("nan"),
                "fro_sin": float("nan")}
    if u.ndim != 2 or u_hat.ndim != 2 or u.shape[0] != u_hat.shape[0]:
        raise ValueError("subspace bases need the same ambient dimension")
    if max(idx) >= min(u.shape[1], u_hat.shape[1]):
        raise ValueError("requested subspace exceeds available basis vectors")
    # Accumulate overlap in FP64 so a perfectly repeated basis does not get a
    # spurious O(sqrt(FP32 epsilon)) angle from dot-product rounding.
    left = u[:, idx].double()
    right = u_hat[:, idx].double()
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    sin = (1 - singular.square()).clamp_min(0).sqrt()
    return {"dimension": len(idx), "mean_sin": float(sin.mean()),
            "max_sin": float(sin.max()), "fro_sin": float(sin.norm())}


def spectral_gap_proxy(singular_values: torch.Tensor, indices: Sequence[int]) -> float:
    """Minimum external eigengap for a selected contiguous/disjoint band.

    Eigenvalues are sigma squared. This denominator is a descriptive local
    separation, not a theorem-level perturbation bound.
    """
    s = singular_values.detach().double().flatten()
    idx = sorted(set(int(i) for i in indices))
    comp = [i for i in range(s.numel()) if i not in set(idx)]
    if not idx or not comp:
        return float("nan")
    lam = s.square()
    a = lam[idx]
    b = lam[comp]
    return float((a[:, None] - b[None, :]).abs().min())
