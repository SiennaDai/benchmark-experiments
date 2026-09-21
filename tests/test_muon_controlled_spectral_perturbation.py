import torch

from optim.muon_controlled_spectral_perturbation import (
    actual_error_decomposition,
    decompose,
    rotation_only,
    singular_value_only,
    spectral_bands,
)
from optim.muon_reference import zeropower_newton_schulz
from optim.muon_spectral_sensitivity import quantize


def matrix():
    torch.manual_seed(12)
    return torch.randn(8, 6)


def test_head_tail_definition_is_deterministic_and_disjoint():
    d = decompose(matrix()); a = spectral_bands(d.singular_values); b = spectral_bands(d.singular_values)
    assert torch.equal(a.head, b.head) and torch.equal(a.tail, b.tail)
    assert set(a.head.tolist()).isdisjoint(a.tail.tolist())
    assert sorted(a.head.tolist() + a.tail.tolist()) == list(range(6))


def test_equal_energy_rotation_respects_epsilon():
    d = decompose(matrix()); bands = spectral_bands(d.singular_values)
    for indices in (bands.head, bands.tail):
        if indices.numel() < 2:
            continue
        observed, _, invalid = rotation_only(d, indices, 0.02)
        assert not invalid
        assert torch.allclose((observed - d.matrix).norm() / d.matrix.norm(), torch.tensor(0.02), atol=2e-5)


def test_singular_value_only_preserves_uv_subspaces():
    d = decompose(matrix()); bands = spectral_bands(d.singular_values)
    observed, invalid = singular_value_only(d, bands.tail, 0.01)
    assert not invalid
    q = torch.linalg.svd(observed, full_matrices=False)
    # Compare the full projectors, which remain stable even if singular values
    # are close and individual singular vectors are not unique.
    assert torch.allclose(q.U @ q.U.T, d.u @ d.u.T, atol=2e-5)
    assert torch.allclose(q.Vh.T @ q.Vh, d.vh.T @ d.vh, atol=2e-5)


def test_rotation_only_preserves_singular_values():
    d = decompose(matrix()); bands = spectral_bands(d.singular_values)
    observed, _, invalid = rotation_only(d, bands.tail, 0.01)
    assert not invalid
    assert torch.allclose(torch.linalg.svdvals(observed), d.singular_values, atol=2e-5, rtol=2e-5)


def test_tau_zero_and_actual_error_decomposition_are_finite():
    d = decompose(matrix()); bands = spectral_bands(d.singular_values)
    assert torch.allclose(d.matrix, (d.u * d.singular_values) @ d.vh, atol=2e-5)
    q = quantize(d.matrix, "int4-dynamic-b2048")
    result = actual_error_decomposition(d, q, head=bands.head, tail=bands.tail)
    assert result["total_projected_error_energy"] >= 0
    assert result["unresolved_error_energy"] >= 0


def test_production_muon_transform_is_callable_without_reimplementation():
    d = decompose(matrix()); observed = d.matrix * 1.01
    expected = zeropower_newton_schulz(d.matrix)
    assert torch.allclose(expected, zeropower_newton_schulz(d.matrix.clone()))
    assert expected.shape == observed.shape
