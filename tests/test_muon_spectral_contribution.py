import torch

from optim.muon_spectral_contribution import (
    ACCOUNTING_BANDS, BANDS, EPSILON, band_component, block_energy, controlled_perturbation,
    decompose, metrics, quantize, restoration,
)
from optim.muon_reference import zeropower_newton_schulz
from optim.state_simulation import persist_state


def matrix():
    torch.manual_seed(23)
    return torch.randn(20, 20)


def test_production_dynamic_quantizer_is_unchanged():
    value = matrix()
    expected = persist_state(value.clone(), "int4_dynamic_momentum", "muon_momentum",
                             quantization_granularity="blockwise", quantization_block_size=2048)
    assert torch.equal(quantize(value, "int4-dynamic-b2048"), expected)


def test_nine_band_components_reconstruct_projected_residual_without_double_counting():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    ehat = d.u.T @ (q - value) @ d.vh.T
    total = torch.zeros_like(value)
    energy = 0.0
    for row_band in ACCOUNTING_BANDS:
        for col_band in ACCOUNTING_BANDS:
            ri, ci = d.bands[row_band], d.bands[col_band]
            block = torch.zeros_like(ehat)
            block[ri[:, None], ci[None, :]] = ehat[ri[:, None], ci[None, :]]
            total = total + d.u @ block @ d.vh
            energy += float(block.square().sum())
    assert torch.allclose(total, d.u @ ehat @ d.vh, atol=1e-5, rtol=1e-5)
    assert abs(energy - float(ehat.square().sum())) < 1e-5


def test_band_energy_sums_and_unresolved_is_explicit():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    row = block_energy(d, q)
    assert abs(sum(row[f"{r}_{c}_energy"] for r in ACCOUNTING_BANDS for c in ACCOUNTING_BANDS)
               - row["projected_residual_energy"]) < 1e-4
    assert row["projected_residual_energy"] - row["total_residual_energy"] < 1e-4


def test_restoring_all_components_recovers_projected_fp32_and_zero_restores_baseline():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    assert torch.equal(restoration(d, q, ()), q)
    corrected = restoration(d, q, ACCOUNTING_BANDS)
    projected = d.u @ (d.u.T @ value @ d.vh.T) @ d.vh
    assert torch.allclose(corrected, projected, atol=1e-5, rtol=1e-5)


def test_band_components_are_disjoint_row_projections():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    components = [band_component(d, q, band) for band in ACCOUNTING_BANDS]
    residual_projection = d.u @ (d.u.T @ (q - value) @ d.vh.T) @ d.vh
    assert torch.allclose(sum(components), residual_projection, atol=1e-5, rtol=1e-5)


def test_matched_controlled_magnitude_norms_and_determinism():
    value = matrix(); d = decompose(value)
    first = controlled_perturbation(d, "tail", "magnitude", EPSILON)[0]
    second = controlled_perturbation(d, "tail", "magnitude", EPSILON)[0]
    assert torch.equal(first, second)
    assert torch.allclose((first - value).norm() / value.norm(), torch.tensor(EPSILON), atol=1e-6, rtol=1e-5)


def test_orientation_controlled_norm_and_singular_values():
    value = matrix(); d = decompose(value)
    rotated, invalid = controlled_perturbation(d, "tail", "orientation", EPSILON)
    assert not invalid
    assert torch.allclose((rotated - value).norm() / value.norm(), torch.tensor(EPSILON), atol=1e-6, rtol=1e-5)
    assert torch.allclose(torch.linalg.svdvals(rotated), d.singular_values, atol=2e-5, rtol=2e-5)


def test_controlled_sensitivity_reuses_production_transform():
    value = matrix(); d = decompose(value); kwargs = {"steps": 5, "coefficients": (3.4445, -4.7750, 2.0315), "eps": 1e-7}
    candidate = controlled_perturbation(d, "head", "magnitude", EPSILON)[0]
    reference = zeropower_newton_schulz(value.clone(), **kwargs)
    observed = zeropower_newton_schulz(candidate.clone(), **kwargs)
    result = metrics(value, candidate, reference_update=reference, transform_kwargs=kwargs)
    expected_cosine = float((reference * observed).sum() / (reference.norm() * observed.norm()))
    assert abs(result["update_cosine"] - expected_cosine) < 1e-6


def test_singleton_band_orientation_is_explicitly_invalid():
    value = torch.diag(torch.tensor([4.0, 1.0, 0.5]))
    d = decompose(value)
    candidate, invalid = controlled_perturbation(d, "head", "orientation", EPSILON)
    # A three-mode matrix's 10% band is one mode; no artificial second mode is used.
    assert invalid
    assert torch.equal(candidate, value)
