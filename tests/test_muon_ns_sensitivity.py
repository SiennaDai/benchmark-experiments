import torch

from optim.muon_ns_sensitivity import (
    DEFAULT_K_GRID,
    band_indices,
    exact_polar,
    scalar_map,
    scalar_map_and_derivative,
    transform, transform_sweep,
)
from optim.muon_reference import zeropower_newton_schulz


def test_production_transform_is_exactly_reused():
    matrix = torch.randn(7, 4)
    assert torch.equal(transform(matrix), zeropower_newton_schulz(matrix))
    assert torch.equal(transform_sweep(matrix)[5], transform(matrix))


def test_scalar_map_matches_diagonal_matrix_transform():
    diagonal = torch.tensor([0.9, 0.4, 0.1, 0.02])
    matrix = torch.diag(diagonal)
    transformed = transform(matrix, steps=3)
    mapped = scalar_map(diagonal, matrix_norm=float(matrix.norm()), steps=3)
    assert torch.allclose(torch.diag(transformed), mapped, atol=2e-6, rtol=2e-6)


def test_scalar_derivative_matches_finite_difference():
    values = torch.tensor([0.2, 0.5, 0.9])
    mapped, derivative = scalar_map_and_derivative(values, matrix_norm=1.2, steps=5)
    delta = 1e-3
    plus = scalar_map(values + delta, matrix_norm=1.2, steps=5)
    minus = scalar_map(values - delta, matrix_norm=1.2, steps=5)
    assert torch.allclose(derivative, (plus - minus) / (2 * delta), atol=3e-3, rtol=3e-3)
    assert torch.isfinite(mapped).all()


def test_exact_polar_reconstruction_and_rectangular_shape():
    matrix = torch.randn(5, 3)
    polar = exact_polar(matrix)
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    assert polar.shape == matrix.shape
    assert torch.allclose(polar, u @ vh, atol=2e-5, rtol=2e-5)


def test_singular_value_only_has_zero_polar_error():
    matrix = torch.randn(6, 4)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    altered = (u * (s + torch.tensor([0.1, 0.02, 0.01, 0.005]))) @ vh
    assert torch.allclose(exact_polar(matrix), exact_polar(altered), atol=2e-5, rtol=2e-5)


def test_bands_are_deterministic_and_disjoint():
    s = torch.tensor([4.0, 2.0, 1.0, .1, .01])
    first, second = band_indices(s), band_indices(s)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert set(first["head"].tolist()).isdisjoint(first["tail"].tolist())
    assert set(first["active"].tolist()) == set(range(5))


def test_k_grid_contains_production_k():
    assert 5 in DEFAULT_K_GRID
