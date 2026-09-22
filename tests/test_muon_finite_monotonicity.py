import math

import torch

from optim.muon_finite_monotonicity import (
    cosine64,
    cosine_distortion,
    local_cosine_coefficient,
    polar_factor,
    polar_skew_exact,
    projected_sensitivity,
)
from optim.muon_frechet import frechet_channels
from optim.muon_reference import zeropower_newton_schulz


def test_local_cosine_expansion_matches_exact_polar_finite_difference():
    m = torch.diag(torch.tensor([2.0, 0.7], dtype=torch.float64))
    e = torch.tensor([[0.0, 0.3], [-0.3, 0.0]], dtype=torch.float64)
    o = polar_factor(m)
    v = frechet_channels(m, e, exact_polar=True)["predicted_delta"]
    coef = local_cosine_coefficient(o, v)
    vals = []
    for t in (1e-2, 3e-3):
        d = cosine_distortion(o, polar_factor(m + t * e))
        vals.append(d / (t * t))
    assert abs(vals[-1] - coef) < abs(vals[0] - coef)
    assert math.isclose(vals[-1], coef, rel_tol=2e-3, abs_tol=1e-8)


def test_exact_polar_derivative_is_tangent_to_orthogonal_manifold():
    m = torch.tensor([[2.0, .1], [-.2, .8]], dtype=torch.float64)
    e = torch.tensor([[.3, .7], [-.5, .2]], dtype=torch.float64)
    o = polar_factor(m)
    v = frechet_channels(m, e, exact_polar=True)["predicted_delta"]
    assert abs(float((o * v).sum())) < 1e-10
    s_perp, s_full, _ = projected_sensitivity(o, v)
    assert math.isclose(s_perp, s_full, rel_tol=1e-10, abs_tol=1e-10)


def test_two_mode_skew_formula_matches_numerical_polar_and_is_monotone():
    si, sj, e = 2.3, .6, .4
    previous = -1.0
    for t in (.01, .1, .5, 1.0, 2.0):
        a = torch.tensor([[si, t * e], [-t * e, sj]], dtype=torch.float64)
        q = polar_factor(a)
        pred = polar_skew_exact(si, sj, e, t)
        assert math.isclose(cosine64(torch.eye(2, dtype=torch.float64), q), pred["cosine"], rel_tol=1e-10, abs_tol=1e-10)
        assert math.isclose(math.atan2(float(q[0, 1]), float(q[0, 0])), pred["angle"], rel_tol=1e-10, abs_tol=1e-10)
        assert pred["distortion"] > previous
        previous = pred["distortion"]
        assert math.isclose(math.tan(pred["angle"]), 2 * t * e / (si + sj), rel_tol=1e-10)


def test_exact_polar_diagonal_and_positive_definite_symmetric_controls():
    m = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    base = polar_factor(m)
    for t in (0.0, .2, 1.0):
        diagonal = torch.diag(torch.tensor([.2, -.1], dtype=torch.float64))
        symmetric = torch.tensor([[0.0, .2], [.2, 0.0]], dtype=torch.float64)
        assert cosine_distortion(base, polar_factor(m + t * diagonal)) < 1e-12
        assert cosine_distortion(base, polar_factor(m + t * symmetric)) < 1e-12


def test_production_frechet_derivative_matches_production_jvp():
    torch.manual_seed(411)
    m = torch.randn(7, 4, dtype=torch.float64)
    e = torch.randn_like(m)
    predicted = frechet_channels(m, e)["predicted_delta"]
    _, jvp = torch.func.jvp(lambda x: zeropower_newton_schulz(x), (m,), (e,))
    assert torch.allclose(predicted, jvp.double(), rtol=2e-5, atol=2e-6)
