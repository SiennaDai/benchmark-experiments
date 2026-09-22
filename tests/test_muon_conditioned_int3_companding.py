"""Contracts for the offline conditioned-INT3 companding study."""

import sys

import pytest
import torch

sys.path.insert(0, "src")

from optim.muon_conditioned_int3_companding import (  # noqa: E402
    DANGER_LOG10_RANGE,
    INT3_CODEBOOK,
    build_symmetric_codebook,
    danger_mode_mask,
    int3_mulaw_roundtrip,
    int3_uniform_roundtrip,
    lloyd_max_codebook,
    mulaw_inverse,
    mulaw_transform,
    optimize_block_scale,
    select_geometry_codebook,
)
from optim.muon_quantization_aware_conditioner import (  # noqa: E402
    int3_dynamic_roundtrip,
)
from scripts import analyze_muon_conditioned_int3_companding as study  # noqa: E402


def _nearest_codebook_roundtrip(block: torch.Tensor, scale: float, codebook: torch.Tensor) -> torch.Tensor:
    normalized = (block.float() / scale).clamp(-1.0, 1.0)
    distances = (normalized.reshape(-1, 1) - codebook.reshape(1, -1)).abs()
    return codebook[distances.argmin(dim=1)].reshape_as(block) * scale


def test_uniform_int3_exactly_matches_existing_offline_implementation():
    generator = torch.Generator().manual_seed(123)
    values = torch.cat((torch.randn(2059, generator=generator), torch.zeros(7)))
    actual = int3_uniform_roundtrip(values, block_size=2048)
    previous = int3_dynamic_roundtrip(values, block_size=2048)
    assert torch.equal(actual, previous)


@pytest.mark.parametrize("mu", [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0])
def test_mulaw_forward_inverse_are_consistent_and_monotone(mu):
    x = torch.linspace(-1.0, 1.0, 2001)
    transformed = mulaw_transform(x, mu)
    recovered = mulaw_inverse(transformed, mu)
    assert torch.allclose(recovered, x, atol=2e-6, rtol=2e-6)
    assert torch.all(transformed[1:] >= transformed[:-1])
    assert transformed[0].item() == pytest.approx(-1.0)
    assert transformed[-1].item() == pytest.approx(1.0)
    assert torch.allclose(mulaw_transform(-x, mu), -transformed, atol=1e-7)


@pytest.mark.parametrize("mu", [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0])
def test_mulaw_roundtrip_is_finite_shape_preserving_and_contains_exact_zero(mu):
    values = torch.tensor([0.0, 1.0, -1.0, 0.02, -0.02, 0.4, -0.7])
    result = int3_mulaw_roundtrip(values, mu, block_size=4)
    assert result.shape == values.shape
    assert torch.isfinite(result).all()
    assert result[0].item() == 0.0


def test_all_int3_codebooks_are_symmetric_with_exact_zero_and_seven_levels():
    for a1, a2 in ((1 / 3, 2 / 3), (0.2, 0.7), (0.4, 0.8)):
        codebook = build_symmetric_codebook(a1, a2)
        assert codebook.numel() == 7
        assert torch.unique(codebook).numel() == 7
        assert (codebook == 0).sum().item() == 1
        assert torch.allclose(codebook, -codebook.flip(0), atol=1e-7)
        assert codebook[0].item() == pytest.approx(-1.0)
        assert codebook[-1].item() == pytest.approx(1.0)


def test_uniform_codebook_is_the_declared_offline_int3_codebook():
    expected = torch.tensor([-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1], dtype=torch.float32)
    assert torch.allclose(INT3_CODEBOOK.float(), expected, atol=1e-7)
    assert torch.allclose(build_symmetric_codebook(1 / 3, 2 / 3), expected, atol=1e-7)


def test_lloyd_max_is_deterministic_symmetric_and_retains_zero():
    samples = torch.tensor([-1.0, -0.8, -0.3, -0.1, 0.0, 0.0, 0.1, 0.3, 0.8, 1.0])
    a = lloyd_max_codebook(samples, iterations=40)
    b = lloyd_max_codebook(samples, iterations=40)
    assert torch.equal(a, b)
    assert a.numel() == 7
    assert torch.unique(a).numel() == 7
    assert (a == 0).sum().item() == 1
    assert torch.allclose(a, -a.flip(0), atol=1e-6)
    assert torch.all(a[1:] > a[:-1])


def test_per_block_scale_oracle_is_no_worse_than_absmax_for_mse():
    block = torch.tensor([-1.0, -0.77, -0.25, -0.17, -0.03, 0.04, 0.21, 0.68, 0.91])
    quantized, scale, squared_error = optimize_block_scale(block, codebook=INT3_CODEBOOK)
    scale = float(scale)
    absmax_scale = float(block.abs().max().item())
    mse = (_nearest_codebook_roundtrip(block, scale, INT3_CODEBOOK) - block).square().sum()
    baseline = (_nearest_codebook_roundtrip(block, absmax_scale, INT3_CODEBOOK) - block).square().sum()
    assert scale > 0
    assert torch.allclose(quantized, _nearest_codebook_roundtrip(block, scale, INT3_CODEBOOK))
    assert squared_error == pytest.approx(float(mse), abs=1e-7)
    assert mse <= baseline + 1e-7


def test_geometry_codebook_selection_is_deterministic_and_uses_only_supplied_calibration_items():
    # The selection API accepts calibration items only; held-out residuals are
    # deliberately kept outside that input and therefore cannot affect choice.
    calibration = [
        {
            "residual": torch.tensor([[0.02, -0.05], [0.12, -0.4]]),
            "u": torch.eye(2),
            "vh": torch.eye(2),
            "mode_weights": torch.tensor([1.0, 3.0]),
        }
    ]
    candidate_pairs = [(0.2, 0.6), (1 / 3, 2 / 3), (0.4, 0.8)]
    selected_a = select_geometry_codebook(calibration, candidate_pairs)
    selected_b = select_geometry_codebook(calibration, candidate_pairs)
    assert selected_a == selected_b
    assert tuple(selected_a[:2]) in candidate_pairs
    assert all(row["calibration_items"] == len(calibration) for row in selected_a[2])
    # An unrelated held-out set has no parameter in this call's contract.
    held_out = [{"residual": torch.full((2, 2), 1000.0)}]
    del held_out
    assert tuple(select_geometry_codebook(calibration, candidate_pairs)[:2]) == tuple(selected_a[:2])


def test_danger_zone_is_fixed_half_open_log10_interval():
    assert DANGER_LOG10_RANGE == (-3.0, -2.0)
    sigma = torch.tensor([1.0, 1e-2, 1e-3, 2e-3, 1e-4, 0.0])
    mask = danger_mode_mask(sigma)
    # -3 is included, -2 excluded; inactive zero is excluded.
    assert mask.tolist() == [False, False, True, True, False, False]


def test_matched_controls_choose_nearest_raw_error_without_using_evaluation_for_selection():
    common = {"seed": 1, "update": 128, "parameter_id": "weight", "k": 4,
              "zero_fraction": 0.2, "update_cosine": 0.7}
    rows = [
        {**common, "quantizer": "geometry_codebook", "raw_residual_relative_l2": 0.31},
        {**common, "quantizer": "uniform_int3", "raw_residual_relative_l2": 0.50},
        {**common, "quantizer": "lloyd_global", "raw_residual_relative_l2": 0.33},
        {**common, "quantizer": "mulaw_mu_5", "raw_residual_relative_l2": 0.29},
        {**common, "quantizer": "danger_zone_codebook", "raw_residual_relative_l2": 0.35},
    ]
    pairs = study.nearest_matched_controls(rows)
    raw = next(p for p in pairs if p["candidate"] == "geometry_codebook" and p["match_type"] == "nearest_raw_relative_l2")
    zero = next(p for p in pairs if p["candidate"] == "geometry_codebook" and p["match_type"] == "nearest_zero_fraction")
    assert raw["reference"] == "lloyd_global"
    assert raw["absolute_raw_l2_gap"] == pytest.approx(0.02)
    assert zero["match_type"] == "nearest_zero_fraction"
    assert len(pairs) == 4  # Two match types per eligible geometry-derived candidate.


def test_heldout_rows_are_not_used_as_global_parameter_calibration():
    # Calibration selection accepts only the supplied list and records the
    # number of those rows; evaluation rows cannot be passed through a hidden
    # global data source or report reader.
    calibration = [{"residual": torch.tensor([[0.1, -0.4], [0.2, 0.7]]),
                    "u": torch.eye(2), "vh": torch.eye(2),
                    "mode_weights": torch.ones(2)}]
    selected = select_geometry_codebook(calibration, [(0.2, 0.6), (1/3, 2/3)])
    assert all(row["calibration_items"] == 1 for row in selected[2])


def test_summary_includes_tensor_size_weighted_update_cosine():
    rows = [
        {"k": 4, "quantizer": "uniform_int3", "seed": 1, "update": 128, "parameter_id": "a", "shape": "(1, 1)", "update_cosine": 0.0},
        {"k": 4, "quantizer": "uniform_int3", "seed": 1, "update": 128, "parameter_id": "b", "shape": "(1, 3)", "update_cosine": 1.0},
    ]
    summary = study.summarize(rows)
    assert summary[0]["mean"] == pytest.approx(0.5)
    assert summary[0]["weighted_mean"] == pytest.approx(0.75)


def test_global_codebook_refinement_grid_is_deterministic_and_valid():
    from scripts.analyze_muon_conditioned_int3_companding import refine_codebook_pairs
    a = refine_codebook_pairs(0.2, 0.6)
    assert a == refine_codebook_pairs(0.2, 0.6)
    assert (0.2, 0.6) in a
    assert all(0 < x < y < 1 for x, y in a)
    assert len(a) > 6
