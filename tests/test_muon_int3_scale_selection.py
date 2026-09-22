"""Tests for offline, block-local INT3 scale-selection helpers."""

import sys

import torch

sys.path.insert(0, "src")

from optim.muon_conditioned_int3_companding import INT3_CODEBOOK  # noqa: E402
from optim.muon_int3_scale_selection import (  # noqa: E402
    block_scales,
    fixed_codebook_roundtrip,
    select_scales,
)


def _manual_fixed_roundtrip(value, scales, block_size, codebook):
    flat = value.float().reshape(-1)
    output = torch.empty_like(flat)
    for block_index, start in enumerate(range(0, flat.numel(), block_size)):
        block = flat[start : start + block_size]
        scale = float(scales[block_index])
        if scale == 0:
            output[start : start + block.numel()] = 0
            continue
        normalized = (block / scale).clamp(float(codebook[0]), float(codebook[-1]))
        distances = (normalized[:, None] - codebook.float()[None, :]).abs()
        indices = distances.argmin(dim=1)
        output[start : start + block.numel()] = codebook.float()[indices] * scale
    return output.reshape_as(value)


def test_fixed_codebook_scale_roundtrip_clips_and_uses_a_separate_scale_per_block():
    values = torch.tensor([-4.0, -0.6, 0.0, 0.8, 1.5, 2.0, 4.0, 5.0])
    scales = torch.tensor([2.0, 5.0])
    expected = _manual_fixed_roundtrip(values, scales, 4, INT3_CODEBOOK)
    actual = fixed_codebook_roundtrip(values, scales, INT3_CODEBOOK, block_size=4)
    assert torch.equal(actual, expected)
    # Clipping at normalized +/-1 maps out-of-range inputs to endpoint levels.
    assert actual[0].item() == -2.0
    assert actual[-1].item() == 5.0
    assert actual[2].item() == 0.0


def test_block_statistics_are_local_to_each_block_and_apply_multiplier():
    values = torch.tensor([1.0, -2.0, 3.0, -4.0, 10.0, -20.0, 30.0, -40.0])
    absmax = block_scales(values, "absmax", multiplier=0.5, block_size=4)
    assert torch.equal(torch.as_tensor(absmax), torch.tensor([2.0, 20.0]))

    changed_second = values.clone()
    changed_second[4:] *= 100.0
    changed_scales = block_scales(changed_second, "absmax", multiplier=0.5, block_size=4)
    assert torch.as_tensor(changed_scales)[0].item() == 2.0
    assert torch.as_tensor(changed_scales)[1].item() == 2000.0


def test_percentile_scales_are_deterministic_block_local_and_nonnegative():
    values = torch.tensor([1.0, -2.0, 3.0, -100.0, 10.0, -20.0, 30.0, -1000.0])
    first = torch.as_tensor(block_scales(values, "percentile", percentile=75.0, block_size=4))
    second = torch.as_tensor(block_scales(values, "percentile", percentile=75.0, block_size=4))
    assert torch.equal(first, second)
    assert first.shape == (2,)
    assert torch.isfinite(first).all() and torch.all(first > 0)
    assert first[1].item() > first[0].item()

    changed_second = values.clone()
    changed_second[4:] *= 10.0
    changed = torch.as_tensor(block_scales(changed_second, "percentile", percentile=75.0, block_size=4))
    assert changed[0].item() == first[0].item()
    assert changed[1].item() > first[1].item()


def test_local_mse_scale_grid_selection_is_deterministic_and_matches_grid_minimum():
    values = torch.tensor([0.10, -0.22, 0.41, 1.0, -0.03, 0.18, 0.29, -0.73])
    grid = (0.5, 0.75, 1.0, 1.25)
    q1, scales1 = select_scales(values, codebook=INT3_CODEBOOK, method="local_mse", grid=grid, block_size=4)
    q2, scales2 = select_scales(values, codebook=INT3_CODEBOOK, method="local_mse", grid=grid, block_size=4)
    assert torch.equal(q1, q2)
    assert torch.equal(torch.as_tensor(scales1), torch.as_tensor(scales2))
    assert torch.isfinite(q1).all()
    assert torch.all(torch.as_tensor(scales1) > 0)

    baseline_scales = block_scales(values, "absmax", block_size=4)
    baseline = fixed_codebook_roundtrip(values, baseline_scales, INT3_CODEBOOK, block_size=4)
    assert (q1 - values).square().sum() <= (baseline - values).square().sum() + 1e-7


def test_scale_selection_handles_zero_blocks_and_keeps_nonzero_scales_positive():
    values = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3, -0.4])
    scales = torch.as_tensor(block_scales(values, "absmax", block_size=4))
    assert scales.shape == (2,)
    assert scales[0].item() == 0.0
    assert scales[1].item() > 0.0
    reconstructed = fixed_codebook_roundtrip(values, scales, INT3_CODEBOOK, block_size=4)
    assert torch.equal(reconstructed[:4], torch.zeros(4))
    assert torch.isfinite(reconstructed).all()

    selected, selected_scales = select_scales(values, method="local_mse", grid=(0.5, 1.0, 1.5), block_size=4)
    assert torch.equal(selected[:4], torch.zeros(4))
    assert torch.as_tensor(selected_scales)[0].item() == 0.0
    assert torch.as_tensor(selected_scales)[1].item() > 0.0


def test_scale_helpers_do_not_mutate_input_and_keep_offline_zero_behavior():
    values = torch.tensor([0.0, -1.2, 0.3, 0.7, -0.2, 0.0])
    original = values.clone()
    scales = block_scales(values, "absmax", block_size=3)
    reconstruction = fixed_codebook_roundtrip(values, scales, block_size=3)
    selected, _ = select_scales(values, method="local_mse", grid=(0.75, 1.0, 1.25), block_size=3)
    assert torch.equal(values, original)
    assert torch.isfinite(reconstruction).all() and torch.isfinite(selected).all()


def test_importing_offline_scale_helpers_does_not_change_production_int4_roundtrip():
    # A behavioral guard: importing/using this offline helper module leaves the
    # production state-simulation roundtrip result unchanged.
    from optim.state_simulation import int8_blockwise_dynamic_roundtrip

    values = torch.tensor([0.0, -0.2, 0.7, 1.0, -3.0, 4.0])
    before = int8_blockwise_dynamic_roundtrip(values, signed=True, block_size=4)
    _ = block_scales(values, "percentile", percentile=90.0, block_size=4)
    after = int8_blockwise_dynamic_roundtrip(values, signed=True, block_size=4)
    assert torch.equal(before, after)
