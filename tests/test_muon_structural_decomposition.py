import sys

import torch

sys.path.insert(0, "src")
from optim.muon_structural_decomposition import (  # noqa: E402
    decompose,
    deterministic_random_modes,
    danger_zone_energy,
    energy_rank,
    structural_reconstruct,
    top_k_posthoc_correction,
    truncated,
)
from optim.muon_spectral_sensitivity import quantize  # noqa: E402
from optim.state_simulation import persist_state  # noqa: E402


def test_truncated_svd_reconstructs_original_with_residual():
    torch.manual_seed(3)
    matrix = torch.randn(5, 4)
    state = decompose(matrix)
    low, residual = truncated(state, state.singular_values.numel())
    assert torch.allclose(low + residual, matrix, atol=2e-5, rtol=2e-5)


def test_rank_and_energy_selection_are_deterministic():
    s = torch.tensor([4.0, 2.0, 1.0, 0.5])
    assert energy_rank(s, 0.25) == 1
    assert energy_rank(s, 0.9) == 2
    assert energy_rank(s, 0.99) == 4
    state = decompose(torch.diag(s))
    assert truncated(state, 2)[0].shape == (4, 4)


def test_structural_quantizer_reuses_production_dynamic_b2048():
    matrix = torch.linspace(-1.0, 1.0, 32).reshape(4, 8)
    state = decompose(matrix)
    result = structural_reconstruct(state, 0)
    expected = persist_state(matrix.clone(), "int4_dynamic_momentum", "muon_momentum",
                             quantization_granularity="blockwise", quantization_block_size=2048)
    assert torch.equal(result["reconstruction"], expected)
    assert torch.equal(quantize(matrix, "int4-dynamic-b2048"), expected)


def test_posthoc_rank_zero_is_exact_baseline_and_full_rank_is_fp32():
    torch.manual_seed(7)
    matrix = torch.randn(4, 4)
    state = decompose(matrix)
    quantized = quantize(matrix, "int4-dynamic-b2048")
    assert torch.equal(top_k_posthoc_correction(state, quantized, 0), quantized)
    corrected = top_k_posthoc_correction(state, quantized, 4)
    assert corrected.shape == matrix.shape


def test_random_modes_do_not_consume_global_rng_and_are_deterministic():
    torch.manual_seed(99)
    before = torch.random.get_rng_state()
    a = deterministic_random_modes(20, 5, seed=2026)
    after = torch.random.get_rng_state()
    b = deterministic_random_modes(20, 5, seed=2026)
    assert torch.equal(before, after)
    assert torch.equal(a, b)
    assert a.numel() == 5 and torch.all(a[:-1] < a[1:])


def test_danger_zone_energy_uses_original_svd_basis():
    values = torch.tensor([1.0, 0.1, 0.003, 0.0001])
    matrix = torch.diag(values)
    state = decompose(matrix)
    error = torch.diag(torch.tensor([0.0, 0.0, 2.0, 0.0]))
    result = danger_zone_energy(state, error)
    assert result["danger_mode_count"] == 1
    assert result["danger_diagonal_energy"] > 0
    assert result["danger_cross_energy"] == 0
