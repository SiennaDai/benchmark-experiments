"""Offline finite-response diagnostics for Muon's Frechet sensitivity coordinate."""
from __future__ import annotations

import math

import torch


def cosine64(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(den) == 0:
        return float("nan")
    return float(torch.sum(a * b) / den)


def projected_sensitivity(output: torch.Tensor, derivative: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """Return (s_perp, s_full, derivative component orthogonal to output)."""
    o, v = output.detach().double(), derivative.detach().double()
    onorm = torch.linalg.vector_norm(o).clamp_min(torch.finfo(o.dtype).tiny)
    vperp = v - o * (torch.sum(o * v) / onorm.square())
    return (float(torch.linalg.vector_norm(vperp) / onorm),
            float(torch.linalg.vector_norm(v) / onorm), vperp)


def local_cosine_coefficient(output: torch.Tensor, derivative: torch.Tensor) -> float:
    """Coefficient A in 1-cos(O,Phi(M+tE)) = A t^2 + O(t^3)."""
    s_perp, _, _ = projected_sensitivity(output, derivative)
    return 0.5 * s_perp * s_perp


def polar_skew_exact(sigma_i: float, sigma_j: float, e: float, t: float) -> dict[str, float]:
    """Exact 2x2 polar response for diag(sigma_i,sigma_j)+t[[0,e],[-e,0]]."""
    den = float(sigma_i) + float(sigma_j)
    s = 2.0 * abs(float(e)) / den
    angle = math.atan(float(t) * s)
    predicted_cos = 1.0 / math.sqrt(1.0 + (float(t) * s) ** 2)
    return {"s": s, "angle": angle, "linear_angle": float(t) * s,
            "cosine": predicted_cos, "distortion": 1.0 - predicted_cos}


def polar_factor(matrix: torch.Tensor) -> torch.Tensor:
    """Numerical orthogonal polar factor for a nonsingular square matrix."""
    u, _, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    return u @ vh


def cosine_distortion(output: torch.Tensor, perturbed_output: torch.Tensor) -> float:
    return 1.0 - cosine64(output, perturbed_output)


def pairwise_order_accuracy(sensitivity, distortion, *, seed: int = 2026,
                            max_pairs: int = 250_000) -> list[dict]:
    """Deterministically estimate pair ordering, stratified by sensitivity gap."""
    import numpy as np

    s = np.asarray(sensitivity, dtype=np.float64)
    d = np.asarray(distortion, dtype=np.float64)
    n = min(len(s), len(d))
    i, j = np.triu_indices(n, 1)
    if len(i) > max_pairs:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(i), size=max_pairs, replace=False))
        i, j = i[chosen], j[chosen]
    denom = np.maximum(np.maximum(np.abs(s[i]), np.abs(s[j])), 1e-30)
    sep = np.abs(s[i] - s[j]) / denom
    pred = np.sign(s[j] - s[i])
    obs = np.sign(d[j] - d[i])
    valid = np.isfinite(pred) & np.isfinite(obs) & (pred != 0) & (obs != 0)
    buckets = {"all": np.ones_like(valid, dtype=bool),
               "near_tie_lt_0.1": sep < 0.1,
               "medium_0.1_to_0.5": (sep >= 0.1) & (sep < 0.5),
               "large_ge_0.5": sep >= 0.5}
    result = []
    for name, mask in buckets.items():
        keep = valid & mask
        result.append({"separation_bucket": name, "n_pairs": int(keep.sum()),
                       "ordering_accuracy": float(np.mean(pred[keep] == obs[keep])) if keep.any() else float("nan"),
                       "median_relative_sensitivity_separation": float(np.median(sep[keep])) if keep.any() else float("nan")})
    return result
