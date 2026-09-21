import sys

import torch

sys.path.insert(0, "src")
from optim.muon_spectral_gap_theory import (  # noqa: E402
    active_indices,
    band_indices,
    band_separation,
    controlled_gap_spectrum,
    deterministic_tail_error,
    gap_proxies,
    mode_gaps,
    safe_correlation,
)
from optim.muon_spectral_sensitivity import decompose, quantize, subspace_metrics  # noqa: E402
from optim.muon_ns_sensitivity import exact_polar  # noqa: E402
from optim.state_simulation import persist_state  # noqa: E402


def test_mode_gaps_and_active_threshold_are_explicit():
    s = torch.tensor([10.0, 9.0, 5.0, 1.0])
    gaps = mode_gaps(s)
    assert torch.equal(gaps["gap"], torch.tensor([1.0, 1.0, 4.0, 4.0]))
    assert torch.equal(active_indices(s, threshold=0.1), torch.tensor([0, 1, 2, 3]))
    assert torch.equal(active_indices(torch.tensor([1.0, 1e-8]), threshold=1e-6), torch.tensor([0]))


def test_band_separation_is_boundary_gap_for_tail():
    s = torch.tensor([10.0, 9.0, 5.0, 1.0])
    bands = band_indices(s)
    assert bands["tail"].numel() == 1
    assert band_separation(s, bands["tail"]) == 4.0


def test_projector_distance_and_gap_proxy_are_deterministic():
    left = torch.eye(4)[:, :2]
    right = torch.eye(4)[:, 2:]
    assert subspace_metrics(left, left)["projection_distance"] < 1e-7
    # Frobenius distance of two orthogonal rank-2 projectors is sqrt(4)=2.
    assert abs(subspace_metrics(left, right)["projection_distance"] - 2.0) < 1e-6
    error = torch.eye(4) * 0.1
    proxies = gap_proxies(error, torch.tensor([4.0, 3.0, 2.0, 1.0]), band_indices(torch.tensor([4.0, 3.0, 2.0, 1.0])))
    assert proxies["tail_sensitivity_2"] is not None
    assert proxies == gap_proxies(error, torch.tensor([4.0, 3.0, 2.0, 1.0]), band_indices(torch.tensor([4.0, 3.0, 2.0, 1.0])))


def test_controlled_gap_changes_only_boundary_spectrum_and_is_deterministic():
    matrix = torch.randn(8, 5)
    d = decompose(matrix)
    bands = band_indices(d.singular_values)
    changed, actual = controlled_gap_spectrum(d.singular_values, bands["tail"], 0.5)
    assert actual is not None and actual < float((d.singular_values[bands["tail"][0]-1] - d.singular_values[bands["tail"][0]]).abs())
    assert torch.equal(changed, controlled_gap_spectrum(d.singular_values, bands["tail"], 0.5)[0])
    # U/V are unchanged by reconstruction from the same singular vectors.
    rebuilt = (d.u * changed) @ d.vh
    assert subspace_metrics(d.u[:, bands["tail"]], decompose(rebuilt).u[:, bands["tail"]])["projection_distance"] < 1e-5


def test_matched_tail_perturbation_norm_and_gap_proxy():
    matrix = torch.randn(8, 5)
    d = decompose(matrix); bands = band_indices(d.singular_values)
    e1 = deterministic_tail_error(d.u, d.vh, bands["tail"], 0.01 * matrix.norm())
    e2 = deterministic_tail_error(d.u, d.vh, bands["tail"], 0.01 * matrix.norm())
    assert torch.allclose(e1.norm(), e2.norm(), atol=1e-7, rtol=1e-7)
    assert torch.equal(e1, e2)


def test_exact_polar_preserves_u_v_when_only_singular_values_change():
    matrix = torch.randn(7, 4)
    d = decompose(matrix)
    altered = (d.u * (d.singular_values + torch.linspace(0.01, 0.1, d.singular_values.numel()))) @ d.vh
    assert torch.allclose(exact_polar(matrix), exact_polar(altered), atol=2e-5, rtol=2e-5)


def test_production_quantizer_path_is_unchanged():
    matrix = torch.linspace(-1, 1, 4099).reshape(1, -1)
    expected = persist_state(matrix.clone(), "int4_dynamic_momentum", "muon_momentum", quantization_granularity="blockwise", quantization_block_size=2048)
    assert torch.equal(quantize(matrix, "int4-dynamic-b2048"), expected)


def test_correlation_handles_ties_and_sample_count():
    rows = [{"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 3.0}, {"x": 2.0, "y": 4.0}]
    result = safe_correlation(rows, "x", "y")
    assert result["sample_count"] == 3
    assert result["spearman"] is not None
