import sys

import torch

sys.path.insert(0, "src")
from optim import muon_spectral_sensitivity as spectral
from optim.state_simulation import persist_state


def test_svd_reconstruction_and_metrics_are_finite():
    matrix = torch.randn(8, 5)
    decomposition = spectral.decompose(matrix)
    reconstructed = decomposition.u @ torch.diag(decomposition.singular_values) @ decomposition.vh
    assert torch.allclose(reconstructed, matrix, atol=1e-5, rtol=1e-5)
    metrics = spectral.spectral_metrics(decomposition.singular_values)
    assert metrics["effective_rank"] == 5
    assert metrics["stable_rank"] > 0


def test_spectral_basis_projection_reconstructs_error():
    matrix = torch.randn(6, 4)
    decomposition = spectral.decompose(matrix)
    error_basis = torch.randn(4, 4) * .01
    error = decomposition.u @ error_basis @ decomposition.vh
    quantized = matrix + error
    projected = decomposition.u.T @ error @ decomposition.vh.T
    assert torch.allclose(decomposition.u @ projected @ decomposition.vh, error, atol=1e-5, rtol=1e-5)
    metrics = spectral.spectral_error_decomposition(decomposition, quantized)
    assert metrics["total_error_energy"] > 0


def test_subspace_metrics_identity_and_orthogonal_cases():
    basis = torch.eye(5)[:, :2]
    identical = spectral.subspace_metrics(basis, basis)
    assert identical["principal_angle_max"] < 1e-6
    assert identical["projection_distance"] < 1e-6
    rotated = torch.eye(5)[:, 2:4]
    orthogonal = spectral.subspace_metrics(basis, rotated)
    assert abs(orthogonal["principal_angle_max"] - torch.pi / 2) < 1e-5


def test_conditioning_intervention_preserves_orientation_and_tau_zero():
    matrix = torch.randn(7, 4)
    decomposition = spectral.decompose(matrix)
    zero = spectral.conditioning_intervention(decomposition, 0.0, quantizer="int4-dynamic-b2048")
    assert torch.allclose(zero["conditioned_matrix"], matrix, atol=1e-5, rtol=1e-5)
    tau = spectral.conditioning_intervention(decomposition, 1e-2, quantizer="int4-dynamic-b2048")
    expected = decomposition.u @ torch.diag(torch.maximum(decomposition.singular_values, decomposition.singular_values.max() * 1e-2)) @ decomposition.vh
    assert torch.allclose(tau["conditioned_matrix"], expected, atol=1e-5, rtol=1e-5)


def test_production_quantizer_and_transform_are_reused(monkeypatch):
    matrix = torch.randn(4, 4)
    calls = []
    production_persist = spectral.persist_state
    production_transform = spectral.muon_reference.zeropower_newton_schulz

    def traced_persist(*args, **kwargs):
        calls.append("quantizer")
        return production_persist(*args, **kwargs)

    def traced_transform(*args, **kwargs):
        calls.append("transform")
        return production_transform(*args, **kwargs)

    monkeypatch.setattr(spectral, "persist_state", traced_persist)
    monkeypatch.setattr(spectral.muon_reference, "zeropower_newton_schulz", traced_transform)
    q = spectral.quantize(matrix, "int4-dynamic-b2048")
    spectral.update_metrics(matrix, q)
    assert "quantizer" in calls and "transform" in calls


def test_quantizer_matches_existing_production_path():
    matrix = torch.linspace(-1, 1, 4099).reshape(1, -1)
    expected = persist_state(matrix.clone(), "int4_dynamic_momentum", "muon_momentum",
                             quantization_granularity="blockwise", quantization_block_size=2048)
    assert torch.equal(spectral.quantize(matrix, "int4-dynamic-b2048"), expected)


def test_effective_rank_and_condition_metrics_have_documented_threshold():
    singular_values = torch.tensor([10.0, 1.0, 1e-5, 1e-7])
    metrics = spectral.spectral_metrics(singular_values)
    assert metrics["effective_rank"] == 3
    assert metrics["effective_condition_number"] == 1e6
    assert metrics["naive_condition_number"] == 1e8
    assert metrics["stable_rank"] > 1


def test_conditioning_intervention_preserves_singular_subspaces():
    matrix = torch.randn(8, 5)
    decomposition = spectral.decompose(matrix)
    result = spectral.conditioning_intervention(decomposition, 1e-2, quantizer="int8-linear-b2048")
    conditioned = spectral.decompose(result["conditioned_matrix"])
    bands = spectral.band_slices(decomposition.singular_values.numel())
    for sl in bands.values():
        assert spectral.subspace_metrics(decomposition.u[:, sl], conditioned.u[:, sl])["projection_distance"] < 1e-4
        assert spectral.subspace_metrics(decomposition.vh.T[:, sl], conditioned.vh.T[:, sl])["projection_distance"] < 1e-4


def test_tau_zero_reconstruction_and_analysis_are_deterministic():
    matrix = torch.randn(7, 4)
    decomposition = spectral.decompose(matrix)
    first = spectral.conditioning_intervention(decomposition, 0.0, quantizer="int4-dynamic-b2048")
    second = spectral.conditioning_intervention(decomposition, 0.0, quantizer="int4-dynamic-b2048")
    assert torch.allclose(first["conditioned_matrix"], matrix, atol=1e-5, rtol=1e-5)
    assert torch.equal(first["quantized"], second["quantized"])
    assert first["update_cosine"] == second["update_cosine"]


def test_all_existing_quantizers_use_the_same_block_size_path():
    matrix = torch.randn(3, 4097)
    for name, simulation in spectral.QUANTIZER_TO_SIMULATION.items():
        expected = persist_state(matrix.clone(), simulation, "muon_momentum",
                                 quantization_granularity="blockwise", quantization_block_size=2048)
        assert torch.equal(spectral.quantize(matrix, name), expected)
