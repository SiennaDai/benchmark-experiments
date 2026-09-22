"""Offline Fréchet derivative of Muon's finite-step spectral map.

This diagnostic module does not alter or wrap the optimizer implementation.
It differentiates the existing normalized Newton--Schulz spectral map, while
the production matrix function remains the authority for JVP comparisons.
"""
from __future__ import annotations

import torch

from .muon_ns_sensitivity import PRODUCTION_COEFFICIENTS, PRODUCTION_EPS, PRODUCTION_STEPS


def normalized_spectral_map(sigma: torch.Tensor, matrix_norm: torch.Tensor | float,
                            *, steps: int = PRODUCTION_STEPS,
                            coefficients=PRODUCTION_COEFFICIENTS,
                            eps: float = PRODUCTION_EPS):
    """Finite-step scalar map and derivative in the normalized singular scale."""
    sigma = sigma.detach().double()
    norm = torch.as_tensor(matrix_norm, dtype=torch.float64, device=sigma.device)
    x = sigma / (norm + float(eps))
    # Derivative here is with respect to normalized input; the caller supplies
    # dX including the separate derivative of production's global scale.
    dx = torch.ones_like(x)
    a, b, c = map(float, coefficients)
    for _ in range(int(steps)):
        x2 = x.square()
        dx = dx * (a + 3.0*b*x2 + 5.0*c*x2.square())
        x = a*x + b*x*x2 + c*x*x2.square()
    return x, dx


def frechet_channels(matrix: torch.Tensor, perturbation: torch.Tensor, *,
                     steps: int = PRODUCTION_STEPS,
                     coefficients=PRODUCTION_COEFFICIENTS,
                     eps: float = PRODUCTION_EPS,
                     exact_polar: bool = False,
                     differentiate_normalization: bool = True,
                     svd_factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                     relative_equal_tolerance: float = 1e-7) -> dict:
    """Return channel-wise first-order perturbation of a rectangular matrix map.

    For finite K, production first applies ``X=M/(||M||_F+eps)``. Its scalar
    norm derivative is included in ``dX`` before applying the SVD divided-
    difference formula. The tall-matrix transpose used by production is
    mathematically equivalent to this transpose-equivariant compact-SVD
    formula; wide matrices use the row-space leakage counterpart.

    For exact polar, scale invariance means no Frobenius-normalization term is
    applied. Rank-deficient leakage coefficients use sigma floored at the
    dtype tiny value and callers must regard those components as unstable.
    """
    m = matrix.detach().to(torch.float64)
    e = perturbation.detach().to(torch.float64)
    if m.ndim != 2 or e.shape != m.shape:
        raise ValueError("matrix and perturbation must be matching 2-D tensors")
    if svd_factors is None:
        u, sigma, vh = torch.linalg.svd(m, full_matrices=False)
    else:
        u, sigma, vh = (part.detach().to(torch.float64) for part in svd_factors)
    v = vh.T
    norm = torch.linalg.vector_norm(m)
    if exact_polar:
        values = torch.ones_like(sigma)
        derivative = torch.zeros_like(sigma)
        dx = e
        spectral_sigma = sigma
    else:
        scale = norm + float(eps)
        # Production: d(M / (||M||_F+eps)) = E/scale -
        # M <M,E>/(||M||_F scale^2). The frozen-scale option is the
        # idealized spectral-map derivative at the same operating point.
        if differentiate_normalization:
            radial = torch.sum(m * e)
            dx = e / scale - m * radial / (norm.clamp_min(torch.finfo(m.dtype).tiny) * scale.square())
        else:
            dx = e / scale
        values, derivative = normalized_spectral_map(sigma, norm, steps=steps,
                                                      coefficients=coefficients, eps=eps)
        spectral_sigma = sigma / scale
    raw_f = u.T @ e @ v
    f = u.T @ dx @ v
    symmetric = (f + f.T) * 0.5
    skew = (f - f.T) * 0.5
    diagonal = torch.diag(torch.diagonal(f) * derivative)
    delta = spectral_sigma[:, None] - spectral_sigma[None, :]
    numerator = values[:, None] - values[None, :]
    divided = torch.empty_like(delta)
    threshold = float(relative_equal_tolerance) * torch.maximum(
        torch.maximum(spectral_sigma[:, None].abs(), spectral_sigma[None, :].abs()),
        torch.full_like(delta, torch.finfo(sigma.dtype).tiny))
    regular = delta.abs() > threshold
    divided[regular] = numerator[regular] / delta[regular]
    limit = (derivative[:, None] + derivative[None, :]) * 0.5
    divided[~regular] = limit[~regular]
    divided.fill_diagonal_(0.0)
    sym_part = divided * symmetric
    sum_denominator = spectral_sigma[:, None] + spectral_sigma[None, :]
    sum_threshold = torch.finfo(sigma.dtype).tiny
    skew_factor = (values[:, None] + values[None, :]) / sum_denominator.clamp_min(sum_threshold)
    if exact_polar:
        # Exact continuous limit for f=1: 2/(sigma_i+sigma_j).
        skew_factor = 2.0 / sum_denominator.clamp_min(sum_threshold)
    skew_factor.fill_diagonal_(0.0)
    skew_part = skew_factor * skew
    compact = diagonal + sym_part + skew_part
    components = {
        "magnitude": u @ diagonal @ vh,
        "symmetric": u @ sym_part @ vh,
        "skew": u @ skew_part @ vh,
    }
    rank = sigma.numel()
    leakage = values / spectral_sigma.clamp_min(torch.finfo(sigma.dtype).tiny)
    outside_mode_energy = torch.zeros_like(sigma)
    if m.shape[0] > m.shape[1]:
        outside = dx @ v - u @ f
        components["out_of_subspace"] = (outside * leakage.unsqueeze(0)) @ vh
        outside_mode_energy = outside.square().sum(0) * leakage.square()
    elif m.shape[1] > m.shape[0]:
        outside = dx.T @ u - v @ f.T
        components["out_of_subspace"] = u @ torch.diag(leakage) @ outside.T
        outside_mode_energy = outside.square().sum(0) * leakage.square()
    else:
        components["out_of_subspace"] = torch.zeros_like(m)
    total = sum(components.values())
    return {"u": u, "sigma": sigma, "vh": vh, "normalized_values": values,
            "normalized_derivative": derivative, "coordinate_error": raw_f,
            "normalized_coordinate_error": f, "normalized_perturbation": dx,
            "symmetric_coordinate_error": symmetric, "skew_coordinate_error": skew,
            "symmetric_divided_difference": divided, "skew_factor": skew_factor,
            "leakage_coefficients": leakage, "out_of_subspace_mode_energy": outside_mode_energy,
            "components": components, "predicted_delta": total,
            "matrix_norm": norm, "rank": rank,
            "normalization_derivative_included": bool(differentiate_normalization and not exact_polar)}


def polar_channel_matrices(sigma: torch.Tensor):
    """Expose exact-polar diagonal/symmetric/skew scalar coefficients."""
    s = sigma.detach().double()
    den = s[:, None] + s[None, :]
    skew = 2.0 / den.clamp_min(torch.finfo(s.dtype).tiny)
    skew.fill_diagonal_(0.0)
    return {"magnitude_derivative": torch.zeros_like(s),
            "symmetric_divided_difference": torch.zeros_like(den),
            "skew_orientation_coefficient": skew,
            "rectangular_leakage_coefficient": 1.0/s.clamp_min(torch.finfo(s.dtype).tiny)}


def principal_pair_proxies(sigma: torch.Tensor, *, values: torch.Tensor | None = None,
                           floor: float = 1e-12):
    """Upper-triangle pair coordinates and gap/sum/finite-K sensitivities."""
    s = sigma.detach().double()
    i, j = torch.triu_indices(s.numel(), s.numel(), offset=1, device=s.device)
    gap = (s[i] - s[j]).abs().clamp_min(float(floor))
    sum_sigma = (s[i] + s[j]).clamp_min(float(floor))
    result = {"i": i, "j": j, "gap_proxy": 1.0/gap,
              "sigma_sum_proxy": 2.0/sum_sigma}
    if values is not None:
        f = values.detach().double()
        result["finite_k_orientation_proxy"] = (f[i] + f[j]).abs()/sum_sigma
        numerator = (f[i] - f[j]).abs()
        result["finite_k_symmetric_proxy"] = numerator/gap
    return result
