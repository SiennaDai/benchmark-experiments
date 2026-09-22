"""Contracts for deterministic vector-INT3 robustness analysis utilities."""
from __future__ import annotations

import sys
from argparse import Namespace

import pytest
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from optim.muon_vector_int3_robustness import (  # noqa: E402
    align_codebook,
    codebook_occupancy_stats,
    evenly_spaced_indices,
    evenly_spaced_sample,
    pack_indices,
    packed_storage_report,
    unpack_indices,
)
from analyze_muon_vector_int3_robustness import codebook_specs, config_key, evaluation_plan  # noqa: E402


def test_evenly_spaced_indices_are_deterministic_ordered_and_cover_endpoints():
    assert evenly_spaced_indices(20, 5) == [0, 5, 10, 14, 19]
    assert evenly_spaced_indices(20, 5) == evenly_spaced_indices(20, 5)
    assert evenly_spaced_indices(20, 1) == [9]
    assert evenly_spaced_indices(4, 8) == [0, 1, 2, 3]
    assert evenly_spaced_indices(0, 3) == []
    assert evenly_spaced_sample(list("abcdefghij"), 4) == ["a", "d", "g", "j"]
    with pytest.raises(ValueError):
        evenly_spaced_indices(-1, 2)


def test_codebook_alignment_is_invariant_to_candidate_permutation():
    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
    permutation = torch.tensor([2, 0, 3, 1])
    candidate = reference[permutation]
    aligned, candidate_indices = align_codebook(reference, candidate)
    assert torch.equal(aligned, reference)
    assert torch.equal(candidate[candidate_indices], aligned)
    assert torch.equal(torch.sort(candidate_indices).values, torch.arange(4))


def test_codebook_alignment_uses_global_minimum_cost_not_greedy_nearest():
    reference = torch.tensor([[0.0], [2.0]])
    candidate = torch.tensor([[1.1], [-1.0]])
    aligned, assignment = align_codebook(reference, candidate)
    total = (reference - aligned).square().sum()
    swapped = (reference - candidate[assignment.flip(0)]).square().sum()
    assert total <= swapped


def test_occupancy_reports_dead_words_entropy_and_top_shares():
    summary = codebook_occupancy_stats(torch.tensor([0, 1, 1, 3, 3, 3]), 5)
    assert summary["used_codewords"] == 3
    assert summary["dead_codewords"] == 2
    assert summary["counts"] == [1, 2, 0, 3, 0]
    assert summary["top1_occupancy_fraction"] == 0.5
    assert summary["top5_occupancy_fraction"] == 1.0
    assert summary["occupancy_entropy_bits"] > 0
    assert codebook_occupancy_stats(torch.empty(0, dtype=torch.long), 8)["occupancy_entropy_bits"] == 0


@pytest.mark.parametrize("bits", [5, 6, 7])
@pytest.mark.parametrize("count", [0, 1, 2, 7, 8, 9, 101])
def test_fixed_bit_packing_round_trip_for_arbitrary_counts(bits, count):
    generator = torch.Generator().manual_seed(bits * 1000 + count)
    values = torch.randint(0, 1 << bits, (count,), generator=generator)
    packed = pack_indices(values, bits)
    decoded = unpack_indices(packed, count, bits)
    assert torch.equal(decoded, values)
    assert packed.numel() == (count * bits + 7) // 8
    report = packed_storage_report(count, bits)
    assert report["theoretical_bits"] == count * bits
    assert report["packed_bytes"] == packed.numel()
    assert report["padding_bits"] == packed.numel() * 8 - count * bits


def test_bit_packing_accepts_bytes_and_rejects_invalid_inputs():
    original = [0, 31, 7, 16, 29]
    packed = pack_indices(original, 5)
    assert unpack_indices(packed.numpy().tobytes(), len(original), 5).tolist() == original
    with pytest.raises(ValueError):
        pack_indices([32], 5)
    with pytest.raises(ValueError):
        unpack_indices(packed, 4, 5)


def test_evaluation_plan_keeps_nondefault_calibration_sweeps_held_out():
    specs = codebook_specs(Namespace(calibration_max_vectors=1200))
    for evaluation_seed in (0, 1):
        plan = evaluation_plan(evaluation_seed, specs)
        for rank in (4, 8):
            keys = set(plan[rank])
            for train_seed in (0, 1):
                # The full 32/64/128 frontier is evaluated in-split and held-out.
                for words in (32, 64, 128):
                    assert config_key(train_seed, rank, words, 8, 1200) in keys
            # Calibration-size and vector-count sweeps are only evaluated on
            # the opposite seed, so held-out samples cannot leak into fitting.
            for count in (1, 2, 4, 16):
                assert config_key(1-evaluation_seed, rank, 64, count, 1200) in keys
                assert config_key(evaluation_seed, rank, 64, count, 1200) not in keys
            for vectors in (300, 600, 2400):
                assert config_key(1-evaluation_seed, rank, 64, 8, vectors) in keys
                assert config_key(evaluation_seed, rank, 64, 8, vectors) not in keys
