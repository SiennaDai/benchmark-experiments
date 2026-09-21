"""Offline finite-step Newton--Schulz sensitivity utilities.

This module is deliberately outside the optimizer.  It accepts detached FP32
matrices, delegates every matrix transform to ``muon_reference`` and uses the
existing ``persist_state`` implementation for quantization.  The scalar map
helpers describe the same production normalization and polynomial; they are
diagnostic summaries, not an alternate training implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import muon_reference
from .muon_spectral_sensitivity import decompose, quantize
from .muon_update_fidelity import _ratios

PRODUCTION_STEPS = 5
PRODUCTION_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
PRODUCTION_EPS = 1e-7
DEFAULT_K_GRID = (1, 2, 3, 4, 5, 6, 8, 10)


@dataclass(frozen=True)
class NSNormalization:
    frobenius_norm: float
    epsilon: float
    transposed: bool


def transform(matrix: torch.Tensor, *, steps: int = PRODUCTION_STEPS,
              coefficients=PRODUCTION_COEFFICIENTS,
              eps: float = PRODUCTION_EPS) -> torch.Tensor:
    """Call the exact production implementation on a detached copy."""
    return muon_reference.zeropower_newton_schulz(
        matrix.detach().clone(), int(steps), tuple(coefficients), float(eps)
    )


@torch.no_grad()
def transform_sweep(matrix: torch.Tensor, steps_grid=DEFAULT_K_GRID,
                    *, coefficients=PRODUCTION_COEFFICIENTS,
                    eps: float = PRODUCTION_EPS) -> dict[int, torch.Tensor]:
    """Compute several finite-step outputs in one Newton--Schulz trajectory.

    This is the same normalization, orientation and polynomial step as the
    production function.  The production implementation is still called by
    :func:`transform` for the baseline and is used in tests to validate this
    batched diagnostic path.  Sharing one trajectory avoids restarting the
    expensive matrix multiplications for every K in the offline sweep.
    """
    matrix = matrix.detach().clone()
    if matrix.ndim != 2:
        raise ValueError("Muon only orthogonalizes 2D matrices")
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.float().T if transposed else matrix.float()
    x = x / (x.norm() + float(eps))
    requested = {int(k) for k in steps_grid}
    out: dict[int, torch.Tensor] = {}
    if 0 in requested:
        out[0] = x.T if transposed else x.clone()
    a, b, c = (float(v) for v in coefficients)
    for step in range(1, max(requested, default=0) + 1):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
        if step in requested:
            out[step] = (x.T if transposed else x.clone())
    return out


def exact_polar(matrix: torch.Tensor, *, threshold: float = 1e-6) -> torch.Tensor:
    """Return the reduced-SVD polar factor with an explicit rank convention.

    ``torch.linalg.svd`` supplies the reduced factors, so this is valid for
    both tall and wide matrices.  The threshold is used only to remove
    numerically inactive modes; the default retains all normal FP32 modes.
    For a rank-deficient matrix, the reduced SVD convention is the canonical
    partial polar factor used by this offline study.
    """
    x = matrix.detach().float()
    if x.ndim != 2:
        raise ValueError("exact polar requires a 2D matrix")
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    if s.numel() and s[0].item() > 0 and threshold > 0:
        active = s / s[0] >= threshold
        # Preserve rectangular output shape while applying the reduced polar
        # convention.  In normal full-rank matrices this is simply U @ Vh.
        if not bool(active.all()):
            u, vh = u[:, active], vh[active, :]
    return u @ vh


def metrics(reference: torch.Tensor, observed: torch.Tensor) -> dict:
    """Return the standard update metrics used by existing fidelity reports."""
    return _ratios(reference, observed, "update")


def scalar_normalization(matrix: torch.Tensor, eps: float = PRODUCTION_EPS) -> NSNormalization:
    x = matrix.detach().float()
    return NSNormalization(float(x.norm()), float(eps), bool(x.shape[0] > x.shape[1]))


def scalar_map(values: torch.Tensor, *, matrix_norm: float,
               steps: int, coefficients=PRODUCTION_COEFFICIENTS,
               eps: float = PRODUCTION_EPS) -> torch.Tensor:
    """Apply the production scalar singular-value transfer map.

    Production first divides the matrix by ``||M||_F + eps`` and then applies
    ``g(x)=a*x+b*x^3+c*x^5`` repeatedly to each singular value.  This helper
    intentionally does not call the matrix transform because it is used to
    inspect mode-wise sensitivities.
    """
    a, b, c = (float(v) for v in coefficients)
    x = values.detach().float() / (float(matrix_norm) + float(eps))
    for _ in range(int(steps)):
        x = a * x + b * x.square() * x + c * x.square().square() * x
    return x


def scalar_map_and_derivative(values: torch.Tensor, *, matrix_norm: float,
                              steps: int, coefficients=PRODUCTION_COEFFICIENTS,
                              eps: float = PRODUCTION_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``f_K(sigma)`` and its analytic derivative w.r.t. sigma."""
    a, b, c = (float(v) for v in coefficients)
    scale = float(matrix_norm) + float(eps)
    x = values.detach().float() / scale
    derivative = torch.full_like(x, 1.0 / scale)
    for _ in range(int(steps)):
        x2 = x.square()
        derivative = derivative * (a + 3.0 * b * x2 + 5.0 * c * x2.square())
        x = a * x + b * x2 * x + c * x2.square() * x
    return x, derivative


def singular_value_transfer_rows(matrix: torch.Tensor, steps: int,
                                 *, coefficients=PRODUCTION_COEFFICIENTS,
                                 eps: float = PRODUCTION_EPS) -> list[dict]:
    """Mode-wise transfer values and amplification factors for one matrix."""
    d = decompose(matrix)
    values, derivative = scalar_map_and_derivative(
        d.singular_values, matrix_norm=float(d.matrix.norm()), steps=steps,
        coefficients=coefficients, eps=eps,
    )
    raw = d.singular_values
    direction = values.abs() / raw.abs().clamp_min(torch.finfo(raw.dtype).tiny)
    out = []
    for i in range(raw.numel()):
        out.append({"mode": i, "sigma": float(raw[i]), "normalized_sigma": float(raw[i] / (d.matrix.norm() + eps)),
                    "f_k": float(values[i]), "a_mag": float(derivative[i]), "a_dir": float(direction[i])})
    return out


def controlled_metrics(source: torch.Tensor, observed: torch.Tensor, *, steps: int,
                       coefficients=PRODUCTION_COEFFICIENTS, eps: float = PRODUCTION_EPS) -> dict:
    """Compare controlled matrices through one exact production NS K."""
    ref = transform(source, steps=steps, coefficients=coefficients, eps=eps)
    obs = transform(observed, steps=steps, coefficients=coefficients, eps=eps)
    result = {"steps": int(steps), **metrics(ref, obs)}
    result.update(_ratios(source, observed, "raw"))
    return result


def sv_only_perturbation(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor,
                         indices: torch.Tensor, epsilon: float,
                         matrix_norm: torch.Tensor) -> torch.Tensor:
    """Deterministic equal-energy singular-value perturbation."""
    if indices.numel() == 0 or matrix_norm.item() == 0:
        return (u * s) @ vh
    weights = torch.arange(1, indices.numel() + 1, device=s.device, dtype=s.dtype)
    delta = torch.zeros_like(s)
    delta[indices] = matrix_norm * float(epsilon) * weights / weights.norm()
    return (u * (s + delta)) @ vh


def rotation_perturbation(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor,
                          indices: torch.Tensor, epsilon: float,
                          matrix_norm: torch.Tensor) -> torch.Tensor:
    """Deterministically rotate left singular vectors to a target norm."""
    if indices.numel() < 2 or matrix_norm.item() == 0:
        return (u * s) @ vh
    size = int(indices.numel())
    generator = torch.zeros((size, size), dtype=u.dtype, device=u.device)
    for i in range(size - 1):
        sign = 1.0 if i % 2 == 0 else -1.0
        generator[i, i + 1] = sign
        generator[i + 1, i] = -sign
    generator /= generator.norm().clamp_min(torch.finfo(generator.dtype).tiny)

    def candidate(theta: float) -> torch.Tensor:
        rotation = torch.linalg.matrix_exp(generator * theta)
        rotated = u.clone()
        rotated[:, indices] = u[:, indices] @ rotation
        return (rotated * s) @ vh

    target = matrix_norm * float(epsilon)
    lo, hi = 0.0, math.pi
    for _ in range(10):
        if (candidate(hi) - (u * s) @ vh).norm().item() >= target.item():
            break
        hi *= 2.0
    for _ in range(52):
        mid = (lo + hi) / 2
        if (candidate(mid) - (u * s) @ vh).norm().item() < target.item():
            lo = mid
        else:
            hi = mid
    return candidate((lo + hi) / 2)


def band_indices(s: torch.Tensor, *, threshold: float = 1e-6) -> dict[str, torch.Tensor]:
    """Use the established effective-rank threshold and 10% index bands."""
    if not s.numel() or s[0].item() == 0:
        empty = torch.empty(0, dtype=torch.long, device=s.device)
        return {"head": empty, "middle": empty, "tail": empty, "active": empty}
    active = torch.nonzero(s / s[0] >= threshold, as_tuple=False).flatten()
    n = int(active.numel())
    width = max(1, math.ceil(n * 0.10))
    return {"active": active, "head": active[:width], "middle": active[max(0, n // 2 - width // 2): min(n, n // 2 - width // 2 + width)],
            "tail": active[-width:]}
