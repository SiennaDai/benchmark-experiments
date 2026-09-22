"""Assignment-only Fréchet metric utilities for fixed 2-D residual VQ.

All functions are offline diagnostics.  They do not learn centroids or alter
the production optimizer/quantizer.  Pair blocks estimated with Hutchinson
are explicitly approximate; exact pair blocks are available for validation
and the small global coordinate-descent oracle.
"""
from __future__ import annotations

import torch

from .muon_frechet import frechet_channels, normalized_spectral_map
from .muon_ns_sensitivity import PRODUCTION_COEFFICIENTS, PRODUCTION_EPS, PRODUCTION_STEPS


def analytic_pair_hessian(matrix: torch.Tensor, coordinates: tuple[int, int, int, int], *,
                          svd_factors=None) -> torch.Tensor:
    """Exact 2x2 Gram block of production Fréchet responses for two entries."""
    r0, c0, r1, c1 = map(int, coordinates)
    basis = []
    for r, c in ((r0, c0), (r1, c1)):
        e = torch.zeros_like(matrix, dtype=torch.float64)
        e[r, c] = 1.0
        basis.append(frechet_channels(matrix, e, svd_factors=svd_factors)["predicted_delta"])
    h = torch.stack((torch.stack(((basis[0] * basis[0]).sum(), (basis[0] * basis[1]).sum())),
                     torch.stack(((basis[1] * basis[0]).sum(), (basis[1] * basis[1]).sum())))).double()
    return h


def _skew_map(matrix: torch.Tensor, u: torch.Tensor, sigma: torch.Tensor,
              vh: torch.Tensor, eps: float, steps: int, coefficients):
    """Linear skew/orientation channel; normalization's radial term is diagonal."""
    scale = torch.linalg.vector_norm(matrix) + float(eps)
    values, _ = normalized_spectral_map(sigma, torch.linalg.vector_norm(matrix),
                                        steps=steps, coefficients=coefficients, eps=eps)
    x = sigma / scale
    den = x[:, None] + x[None, :]
    factor = (values[:, None] + values[None, :]) / den.clamp_min(torch.finfo(x.dtype).tiny)
    factor.fill_diagonal_(0.0)
    v = vh.T

    def apply(error):
        error = error.to(torch.float64)
        f = (u.T @ (error / scale) @ v)
        a = (f - f.T) * 0.5
        return u @ (factor * a) @ vh
    return apply


def hutchinson_pair_blocks(matrix: torch.Tensor, pair_indices: torch.Tensor, *,
                           probes: int = 16, seed: int = 2026, channel: str = "full",
                           svd_factors=None, steps: int = PRODUCTION_STEPS,
                           coefficients=PRODUCTION_COEFFICIENTS,
                           eps: float = PRODUCTION_EPS) -> torch.Tensor:
    """Estimate pair-local blocks of J*J with deterministic Rademacher probes.

    For each probe z, compute J*z then J*J*z.  This estimates diagonal and
    selected off-diagonal entries of the input-space metric.  It is a
    stochastic block-diagonal approximation to the full metric, not the exact
    global objective and not an exact pair-local block.
    """
    if probes < 2:
        raise ValueError("at least two probes are required")
    m = matrix.detach().float()
    pairs = pair_indices.detach().long().to(m.device)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must have shape (P,2)")
    if channel == "full":
        transform = lambda z: _production_map(z, steps, coefficients, eps)
    elif channel == "skew":
        if svd_factors is None:
            u, s, vh = torch.linalg.svd(m.double(), full_matrices=False)
        else:
            u, s, vh = (x.detach().double() for x in svd_factors)
        transform = _skew_map(m.double(), u, s, vh, eps, steps, coefficients)
    else:
        raise ValueError("channel must be 'full' or 'skew'")
    _, pullback = torch.func.vjp(transform, m)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    diag0 = torch.zeros(pairs.shape[0], dtype=torch.float64)
    diag1 = torch.zeros_like(diag0)
    cross = torch.zeros_like(diag0)
    p0, p1 = pairs[:, 0], pairs[:, 1]
    # Batch the fixed probes.  This preserves the estimator exactly while
    # replacing 2*probes Python/transform dispatches with two vectorized maps.
    # Chunk only for unusually large inputs to cap temporary workspace.
    probe_chunk = min(int(probes), max(1, 8_000_000 // m.numel()))
    accumulated = 0
    while accumulated < int(probes):
        count = min(probe_chunk, int(probes) - accumulated)
        z = torch.randint(0, 2, (count, *m.shape), generator=g, dtype=torch.int8).to(m.device, dtype=m.dtype).mul_(2).sub_(1)
        jz = torch.func.vmap(lambda direction: torch.func.jvp(transform, (m,), (direction,))[1])(z)
        wz = torch.func.vmap(lambda tangent: pullback(tangent)[0])(jz)
        zf, wf = z.reshape(count, -1), wz.reshape(count, -1)
        diag0 += (zf[:, p0] * wf[:, p0]).sum(0).double().cpu()
        diag1 += (zf[:, p1] * wf[:, p1]).sum(0).double().cpu()
        cross += (0.5 * (zf[:, p0] * wf[:, p1] + zf[:, p1] * wf[:, p0])).sum(0).double().cpu()
        accumulated += count
    diag0 /= probes; diag1 /= probes; cross /= probes
    # Project the noisy estimate onto the PSD cone of 2x2 blocks, retaining
    # its estimated diagonal and clipping only the covariance term.
    diag0.clamp_(min=0); diag1.clamp_(min=0)
    bound = torch.sqrt(diag0 * diag1)
    cross = torch.maximum(torch.minimum(cross, bound), -bound)
    h = torch.zeros((pairs.shape[0], 2, 2), dtype=torch.float64)
    h[:, 0, 0] = diag0; h[:, 1, 1] = diag1
    h[:, 0, 1] = cross; h[:, 1, 0] = cross
    return h


def _production_map(matrix: torch.Tensor, steps: int, coefficients, eps: float):
    # Local import avoids a module-level dependency cycle and delegates the
    # actual map to the unchanged production reference.
    from .muon_reference import zeropower_newton_schulz
    return zeropower_newton_schulz(matrix, steps=steps, coefficients=coefficients, eps=eps)


def score_pair_candidates(errors: torch.Tensor, hessian_blocks: torch.Tensor,
                          *, mse_scale: float = 1.0, frechet_scale: float = 1.0,
                          lambda_frechet: float = 1.0) -> torch.Tensor:
    """Return normalized local costs for (pair,candidate,2) physical errors."""
    if errors.ndim != 3 or errors.shape[-1] != 2 or hessian_blocks.shape != (errors.shape[0], 2, 2):
        raise ValueError("expected errors (P,C,2) and pair Hessians (P,2,2)")
    lam = float(lambda_frechet)
    mse = errors.square().sum(-1) / max(float(mse_scale), 1e-30)
    local = torch.einsum("pci,pij,pcj->pc", errors.double(), hessian_blocks.double(), errors.double())
    local = local / max(float(frechet_scale), 1e-30)
    return (1.0 - lam) * mse.double() + lam * local


def pair_cost_from_hessian(error: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Exact quadratic score for one 2-vector given a 2x2 response Gram."""
    e = error.double().reshape(2)
    return e @ hessian.double() @ e


def objective_change_from_block(error_delta: torch.Tensor, gradient: torch.Tensor,
                                hessian: torch.Tensor) -> torch.Tensor:
    """Change in a quadratic global objective for one block replacement."""
    d = error_delta.double().reshape(2); g = gradient.double().reshape(2)
    return 2.0 * (g @ d) + d @ hessian.double() @ d
