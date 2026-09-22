"""Focused contracts for offline paired-vector INT3 helpers."""

import sys

import pytest
import torch

sys.path.insert(0, "src")

from optim.muon_vector_int3 import (  # noqa: E402
    codebook_occupancy,
    fit_covariance_transform,
    fit_kmeans,
    normalize_vector_blocks,
    pair_values,
    pairing_indices,
    polar_codebook,
    quantize_vectors,
    quantize_vectors_per_dimension,
    unpair_values,
    vector_scales,
    vector_storage_bits,
)


PAIRERS = ("contiguous", "row", "column", "checkerboard")


@pytest.mark.parametrize("scheme", PAIRERS)
@pytest.mark.parametrize("shape", [(3, 5), (5, 4), (4, 5), (1, 7)])
def test_pairers_are_deterministic_nonoverlapping_and_exactly_invertible(scheme, shape):
    generator = torch.Generator().manual_seed(2026)
    matrix = torch.randn(shape, generator=generator)
    paired, singles, pair_idx, single_idx = pair_values(matrix, scheme)
    paired_again, singles_again, pair_idx_again, single_idx_again = pair_values(matrix, scheme)

    assert torch.equal(pair_idx, pair_idx_again)
    assert torch.equal(single_idx, single_idx_again)
    assert torch.equal(paired, paired_again)
    assert torch.equal(singles, singles_again)
    assert paired.shape == (pair_idx.shape[0], 2)
    assert singles.numel() == single_idx.numel()
    assert paired.shape[0] * 2 + singles.numel() == matrix.numel()

    covered = torch.cat((pair_idx.reshape(-1), single_idx))
    assert torch.equal(torch.sort(covered).values, torch.arange(matrix.numel()))
    restored = unpair_values(paired, singles, shape, pair_idx, single_idx)
    assert torch.equal(restored, matrix)


@pytest.mark.parametrize("shape", [(3, 5), (5, 4), (4, 5)])
def test_pairing_indices_return_identical_deterministic_partitions(shape):
    for scheme in PAIRERS:
        first = pairing_indices(shape, scheme)
        second = pairing_indices(shape, scheme)
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])


def test_polar_codebook_has_exact_unique_zero_and_never_exceeds_budget():
    for budget in (2, 8, 32, 64):
        cb = polar_codebook(n_angles=16, n_radii=8, max_codewords=budget)
        assert cb.ndim == 2 and cb.shape[1] == 2
        assert 1 <= cb.shape[0] <= budget
        assert torch.isfinite(cb).all()
        assert (cb.square().sum(dim=1) == 0).sum().item() == 1
        assert torch.unique(cb, dim=0).shape[0] == cb.shape[0]
        assert torch.equal(cb, polar_codebook(n_angles=16, n_radii=8, max_codewords=budget))


def test_seeded_kmeans_is_reproducible_and_reserves_an_exact_zero():
    generator = torch.Generator().manual_seed(9)
    samples = torch.randn((200, 2), generator=generator)
    cb_a, meta_a = fit_kmeans(samples, 16, seed=71, iterations=20)
    cb_b, meta_b = fit_kmeans(samples, 16, seed=71, iterations=20)
    assert torch.equal(cb_a, cb_b)
    assert meta_a == meta_b
    assert 1 <= cb_a.shape[0] <= 16
    assert (cb_a.square().sum(dim=1) == 0).sum().item() == 1
    assert torch.isfinite(cb_a).all()
    assert meta_a["seed"] == 71


def test_vector_scales_and_quantization_preserve_expected_shapes_and_finite_values():
    generator = torch.Generator().manual_seed(17)
    pairs = torch.randn((11, 2), generator=generator)
    pairs[0] = 0
    codebook = polar_codebook(n_angles=4, n_radii=4, max_codewords=16)

    scales = vector_scales(pairs, method="p98", block_size=8)
    expected_blocks = (pairs.shape[0] + (8 // 2) - 1) // (8 // 2)
    assert scales.shape == (expected_blocks,)
    assert torch.isfinite(scales).all()
    assert torch.all(scales > 0)
    reconstruction, returned_scales, indices = quantize_vectors(
        pairs, codebook, scales=scales, block_size=8
    )
    assert reconstruction.shape == pairs.shape
    assert returned_scales.shape == scales.shape
    assert indices.shape == (pairs.shape[0],)
    assert torch.equal(returned_scales, scales)
    assert torch.isfinite(reconstruction).all()
    assert torch.all((indices >= 0) & (indices < codebook.shape[0]))
    assert torch.equal(reconstruction[0], torch.zeros(2))


def test_vector_scales_use_block_local_statistics_and_zero_blocks_stay_zero():
    pairs = torch.tensor([[1.0, -2.0], [0.5, -0.25], [10.0, -20.0], [5.0, -10.0],
                          [0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
    scales = vector_scales(pairs, method="absmax", block_size=4)
    assert scales.shape == (4,)
    assert scales[0].item() == 2.0
    assert scales[1].item() == 20.0
    assert scales[2].item() == 0.0
    assert scales[3].item() == 4.0

    codebook = polar_codebook(n_angles=4, n_radii=4, max_codewords=16)
    reconstruction, selected_scales, indices = quantize_vectors(
        pairs, codebook, scales=scales, block_size=4
    )
    assert torch.equal(reconstruction[4:6], torch.zeros((2, 2)))
    assert selected_scales[2].item() == 0.0
    assert torch.equal(indices[4:6], torch.zeros(2, dtype=torch.int64))


def test_calibration_normalization_matches_evaluation_clipping_domain():
    pairs = torch.tensor([[0.1, 0.0], [0.2, -0.1], [100.0, -50.0], [0.3, 0.2]])
    normalized = normalize_vector_blocks(pairs, "p98", block_size=8)
    assert normalized.shape == pairs.shape
    assert torch.isfinite(normalized).all()
    assert torch.all(normalized.abs() <= 1.0)


def test_per_dimension_scales_are_two_per_block_and_deterministic():
    pairs = torch.tensor([[1.0, 10.0], [-2.0, -20.0], [0.5, 5.0], [0.0, 0.0]])
    codebook = polar_codebook(n_angles=4, n_radii=4, max_codewords=16)
    first = quantize_vectors_per_dimension(pairs, codebook, method="absmax", block_size=4)
    second = quantize_vectors_per_dimension(pairs, codebook, method="absmax", block_size=4)
    q, scales, indices = first
    assert q.shape == pairs.shape
    assert scales.shape == (2, 2)
    assert torch.equal(scales, second[1]) and torch.equal(q, second[0])
    assert torch.equal(scales[0], torch.tensor([2.0, 20.0]))
    assert torch.equal(q[-1], torch.zeros(2))
    assert torch.all((indices >= 0) & (indices < codebook.shape[0]))


def test_vector_storage_accounting_matches_explicit_payload_and_metadata_formula():
    result = vector_storage_bits(
        (4, 5), codewords=8, lowrank_rank=2, block_size=4,
        codebook_bits=32, scale_bits=32, factor_bits=16,
    )
    # 10 paired vectors -> 3-bit indices; 5 b=4 scalar blocks -> 160 scale bits.
    assert result["pair_index_bits"] == 30
    assert result["scale_bits"] == 5 * 32
    # BF16 factors: U(4x2) + V(5x2) + sigma(2).
    assert result["factor_bits"] == 16 * (4 * 2 + 5 * 2 + 2)
    assert result["shared_codebook_bits"] == 8 * 2 * 32
    assert result["metadata_bits"] == 96 + 24 + 32 + 5 * 32
    assert result["total_bits_unamortized"] == sum(
        result[key] for key in ("pair_index_bits", "factor_bits", "shared_codebook_bits", "metadata_bits")
    )
    assert result["total_bits_excluding_shared_codebook"] == sum(
        result[key] for key in ("pair_index_bits", "factor_bits", "metadata_bits")
    )

    rotated = vector_storage_bits((4, 5), codewords=8, lowrank_rank=2,
                                  block_size=4, rotation_bits=128,
                                  scales_per_block=2)
    assert rotated["scale_bits"] == 2 * result["scale_bits"]
    assert rotated["rotation_bits"] == 128
    assert rotated["total_bits_unamortized"] == result["total_bits_unamortized"] + result["scale_bits"] + 128

    spectral = vector_storage_bits((4, 5), codewords=8, lowrank_rank=2,
                                   block_size=4, include_spectral_basis=True)
    assert spectral["spectral_basis_bits"] == 32 * (4 * 4 + 5 * 4)

    odd = vector_storage_bits((3, 3), codewords=8, lowrank_rank=0, block_size=4)
    assert odd["pair_index_bits"] == 4 * 3 + 3  # singleton gets scalar INT3 payload


def test_codebook_occupancy_counts_used_and_dead_entries():
    occupancy = codebook_occupancy(torch.tensor([0, 1, 1, 3, 3, 3]), codewords=5)
    assert occupancy["counts"] == [1, 2, 0, 3, 0]
    assert occupancy["occupied"] == 3
    assert occupancy["dead"] == 2


def test_covariance_whitening_transform_and_inverse_roundtrip():
    generator = torch.Generator().manual_seed(31)
    source = torch.randn((1024, 2), generator=generator)
    mixing = torch.tensor([[3.0, 0.7], [-0.4, 0.35]])
    samples = source @ mixing.T

    transform, inverse = fit_covariance_transform(samples, whiten=True, eps=1e-8)
    whitened = samples @ transform.T
    restored = whitened @ inverse.T
    assert transform.shape == inverse.shape == (2, 2)
    assert torch.isfinite(transform).all() and torch.isfinite(inverse).all()
    assert torch.allclose(restored, samples, atol=2e-5, rtol=2e-5)
    covariance = whitened.T @ whitened / whitened.shape[0]
    assert torch.allclose(covariance, torch.eye(2), atol=3e-4, rtol=3e-4)
