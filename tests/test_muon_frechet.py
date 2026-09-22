import torch

from optim.muon_frechet import (
    frechet_channels, normalized_spectral_map, polar_channel_matrices,
)
from optim.muon_reference import zeropower_newton_schulz


def test_coordinate_error_symmetric_and_skew_parts_are_exact():
    torch.manual_seed(71)
    m = torch.randn(7, 4, dtype=torch.float64)
    e = torch.randn_like(m)
    result = frechet_channels(m, e)
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    f = u.T @ e @ vh.T
    assert torch.allclose(result["coordinate_error"], f, atol=1e-10, rtol=1e-10)
    s, a = result["symmetric_coordinate_error"], result["skew_coordinate_error"]
    assert torch.allclose(s, s.T, atol=1e-12, rtol=0)
    assert torch.allclose(a, -a.T, atol=1e-12, rtol=0)
    assert torch.allclose(s + a, result["normalized_coordinate_error"], atol=1e-12, rtol=1e-12)


def test_divided_differences_use_continuous_equal_singular_value_limit():
    sigma = torch.tensor([3.0, 3.0, 1.0], dtype=torch.float64)
    m = torch.diag(sigma)
    values, derivative = normalized_spectral_map(sigma, m.norm(), steps=2)
    result = frechet_channels(m, torch.ones_like(m), steps=2)
    assert torch.isfinite(result["symmetric_divided_difference"]).all()
    assert torch.allclose(result["symmetric_divided_difference"][0, 1], derivative[0], atol=1e-10)


def test_exact_polar_channel_coefficients_have_expected_limits():
    sigma = torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64)
    channels = polar_channel_matrices(sigma)
    assert torch.equal(channels["magnitude_derivative"], torch.zeros_like(sigma))
    assert torch.equal(channels["symmetric_divided_difference"], torch.zeros((3, 3), dtype=torch.float64))
    assert torch.allclose(channels["skew_orientation_coefficient"][0, 1], torch.tensor(1/3, dtype=torch.float64))
    assert torch.allclose(channels["rectangular_leakage_coefficient"], 1/sigma)


def test_exact_polar_analytic_channels_vanish_or_match_orientation_formula():
    torch.manual_seed(72)
    m = torch.randn(8, 5, dtype=torch.float64)
    e = torch.randn_like(m)
    result = frechet_channels(m, e, exact_polar=True)
    assert result["components"]["magnitude"].norm() == 0
    assert result["components"]["symmetric"].norm() == 0
    expected = 2/(result["sigma"][:, None] + result["sigma"][None, :])
    expected.fill_diagonal_(0)
    assert torch.allclose(result["skew_factor"], expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result["leakage_coefficients"], 1/result["sigma"], atol=1e-12)


def test_analytic_derivative_matches_production_jvp_for_tall_square_and_wide():
    for shape, seed in (((9, 5), 73), ((5, 9), 74), ((6, 6), 75)):
        torch.manual_seed(seed)
        m = torch.randn(*shape)
        e = torch.randn_like(m) * .01
        pred = frechet_channels(m, e)["predicted_delta"].float()
        _, jvp = torch.func.jvp(
            lambda x: zeropower_newton_schulz(x, steps=5), (m,), (e,)
        )
        cosine = torch.nn.functional.cosine_similarity(pred.flatten(), jvp.flatten(), dim=0)
        rel = torch.linalg.vector_norm(pred-jvp)/torch.linalg.vector_norm(jvp)
        assert cosine > 0.99999
        assert rel < 2e-5


def test_analytic_derivative_matches_finite_difference_as_epsilon_shrinks():
    torch.manual_seed(76)
    m = torch.randn(7, 4)
    e = torch.randn_like(m) * .01
    pred = frechet_channels(m, e)["predicted_delta"].float()
    base = zeropower_newton_schulz(m, steps=5)
    # FP32 roundoff dominates below this range; at eps=0.01 the local error is
    # clearly smaller than at eps=0.2 while remaining above cancellation noise.
    errors = []
    for epsilon in (0.2, 0.05, 0.01):
        fd = (zeropower_newton_schulz(m + epsilon*e, steps=5)-base)/epsilon
        errors.append(float(torch.linalg.vector_norm(fd-pred)/torch.linalg.vector_norm(pred)))
    assert errors[-1] < errors[0]


def test_frozen_scale_and_production_normalization_are_distinct_derivatives():
    torch.manual_seed(77)
    m = torch.randn(8, 5)
    e = torch.randn_like(m)
    full = frechet_channels(m, e)["predicted_delta"]
    frozen = frechet_channels(m, e, differentiate_normalization=False)["predicted_delta"]
    assert torch.linalg.vector_norm(full-frozen) > 1e-6
