import torch

from optim import muon_reference
from optim.muon_gram_perturbation import (
    active_band_indices, gram_perturbation_metrics, spectral_gap_proxy,
    subspace_angle_metrics,
)
from optim.muon_spectral_sensitivity import quantize
from optim.muon_structural_decomposition import exact_polar


def test_gram_perturbation_matches_linear_plus_quadratic_identity_both_sides():
    torch.manual_seed(17)
    m = torch.randn(9, 5)
    e = torch.randn_like(m) * 0.1
    result = gram_perturbation_metrics(m, m + e)
    assert result["right_gram_identity_residual"] < 2e-5
    assert result["left_gram_identity_residual"] < 2e-5
    assert result["right_gram_quadratic_relative_fro"] >= 0
    assert result["left_gram_quadratic_over_delta_fro"] >= 0


def test_gram_identity_is_the_declared_matrix_difference_and_grams_are_symmetric():
    torch.manual_seed(19)
    m = torch.randn(7, 4)
    e = torch.randn_like(m) * .03
    mhat = m + e
    g, ghat = m.T @ m, mhat.T @ mhat
    delta = ghat - g
    declared = m.T @ e + e.T @ m + e.T @ e
    assert torch.allclose(delta, declared, atol=2e-5, rtol=2e-5)
    assert torch.allclose(g, g.T, atol=1e-6, rtol=0)
    assert torch.allclose(ghat, ghat.T, atol=1e-6, rtol=0)
    assert torch.allclose(torch.linalg.svdvals(m), torch.linalg.svdvals(m).sort(descending=True).values)


def test_gram_matrices_are_symmetric_and_trace_metrics_are_finite():
    torch.manual_seed(4)
    m = torch.randn(6, 4)
    out = gram_perturbation_metrics(m, m + .03 * torch.randn_like(m))
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values())
    assert out["right_gram_relative_fro"] >= 0
    assert out["left_gram_relative_spectral"] >= 0


def test_zero_reconstruction_error_has_zero_gram_perturbation():
    m = torch.randn(7, 3)
    out = gram_perturbation_metrics(m, m.clone())
    for key, value in out.items():
        if "relative" in key or "delta_fro" in key or "identity_residual" in key:
            assert abs(value) < 1e-7


def test_prior_active_head_middle_tail_bands_are_deterministic_and_exhaustive_for_active_modes():
    s = torch.tensor([1., .8, .7, .6, .5, .4, .3, .2, .1, .01, 1e-7])
    a = active_band_indices(s)
    assert a == active_band_indices(s)
    assert a["head"] == [0]
    assert a["tail"] == [9]
    assert a["middle"] == [5]
    assert max(a["tail"]) < 10
    assert spectral_gap_proxy(s, a["tail"]) >= 0


def test_identical_subspaces_have_zero_principal_angle_sines():
    q, _ = torch.linalg.qr(torch.randn(10, 5))
    out = subspace_angle_metrics(q, q.clone(), [0, 1, 2])
    assert out["dimension"] == 3
    assert out["max_sin"] < 5e-4
    assert out["fro_sin"] < 8e-4


def test_principal_angle_sines_are_valid_and_orthogonal_subspaces_are_maximal():
    u = torch.eye(6)[:, :2]
    v = torch.eye(6)[:, 2:4]
    out = subspace_angle_metrics(u, v, [0, 1])
    assert 0 <= out["mean_sin"] <= 1
    assert abs(out["mean_sin"] - 1) < 1e-6


def test_production_int4_quantizer_and_k5_transform_are_reused():
    torch.manual_seed(22)
    m = torch.randn(8, 6)
    q = quantize(m, "int4-dynamic-b2048")
    assert q.shape == m.shape and q.dtype == torch.float32
    update = muon_reference.zeropower_newton_schulz(m.clone(), steps=5)
    update_q = muon_reference.zeropower_newton_schulz(m.clone(), steps=5)
    cosine = torch.nn.functional.cosine_similarity(update.flatten(), update_q.flatten(), dim=0)
    assert cosine.item() > 1 - 1e-6


def test_exact_polar_matches_svd_uvh_reference_for_rectangular_matrix():
    torch.manual_seed(28)
    m = torch.randn(8, 5)
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    expected = u @ vh
    actual = exact_polar(m)
    assert actual.shape == m.shape
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.allclose(torch.linalg.matrix_norm(actual, ord="fro"), torch.tensor(5.).sqrt(), atol=2e-5)


def test_no_training_or_new_quantizer_hooks_in_diagnostic_module():
    import inspect
    from optim import muon_gram_perturbation as module
    src = inspect.getsource(module)
    assert "optimizer.step(" not in src
    assert "persist_state(" not in src
    assert "quantize(" not in src
