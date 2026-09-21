import torch

from optim.muon_spectral_sensitivity import quantize
from optim.muon_tail_correction import (
    BUDGETS, correction, decompose, mode_indices, row_for_budget,
    validate_projection,
)
from optim.muon_reference import zeropower_newton_schulz
from optim.state_simulation import persist_state


def matrix():
    torch.manual_seed(7)
    return torch.randn(12, 8)


def test_baseline_reuses_production_dynamic_quantizer():
    value = matrix()
    expected = persist_state(value.clone(), "int4_dynamic_momentum", "muon_momentum", quantization_granularity="blockwise", quantization_block_size=2048)
    assert torch.equal(quantize(value, "int4-dynamic-b2048"), expected)


def test_rank_zero_reproduces_baseline_and_transform_is_production():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    assert torch.equal(q + correction(d, q, mode_indices(d, 0, "tail"), "diagonal"), q)
    assert torch.equal(zeropower_newton_schulz(value), zeropower_newton_schulz(value.clone()))


def test_tail_modes_are_nested_and_diagonal_projection_is_restricted():
    d = decompose(matrix()); q = quantize(d.matrix, "int4-dynamic-b2048")
    previous = set()
    for budget in BUDGETS:
        indices = mode_indices(d, budget, "tail")
        assert previous.issubset(set(indices.tolist())); previous = set(indices.tolist())
        value = correction(d, q, indices, "diagonal")
        assert validate_projection(d, value, indices, "diagonal")


def test_full_tail_projection_and_matched_controls():
    d = decompose(matrix()); q = quantize(d.matrix, "int4-dynamic-b2048")
    tail = mode_indices(d, 3, "tail"); head = mode_indices(d, 3, "head"); random = mode_indices(d, 3, "random")
    assert validate_projection(d, correction(d, q, tail, "full"), tail, "full")
    assert tail.numel() == head.numel() == random.numel()
    assert torch.equal(random, mode_indices(d, 3, "random"))


def test_row_reports_cost_and_finite_metrics():
    d = decompose(matrix()); q = quantize(d.matrix, "int4-dynamic-b2048")
    row = row_for_budget(d, q, band="tail", kind="diagonal", budget=2)
    assert row["correction_rank"] == 2
    assert row["number_of_scalar_coefficients"] == 2
    assert row["explicit_storage_scalars"] > 2
    assert row["update_cosine"] is not None


def test_full_rank_projection_approaches_residual_in_reduced_square_case():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    indices = mode_indices(d, d.effective_rank, "tail")
    full = correction(d, q, indices, "full")
    residual = d.matrix - q
    projected = d.u.T @ residual @ d.vh.T
    expected = d.u @ projected @ d.vh
    assert torch.allclose(full, expected, atol=1e-5, rtol=1e-5)
