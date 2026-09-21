import torch

from optim.muon_reference import zeropower_newton_schulz
from optim.muon_spectral_block_contribution import (
    GROUPS, PRIMARY_BLOCKS, block_component, block_energies, decompose, exact_polar,
    metric_pair, paired_restore_metrics, quantize, restore_blocks, spectral_coordinates,
)
from optim.state_simulation import persist_state


def matrix():
    torch.manual_seed(123)
    return torch.randn(20, 16)


def test_production_quantizer_is_reused_unchanged():
    value = matrix()
    expected = persist_state(value.clone(), "int4_dynamic_momentum", "muon_momentum",
                             quantization_granularity="blockwise", quantization_block_size=2048)
    assert torch.equal(quantize(value, "int4-dynamic-b2048"), expected)


def test_primary_blocks_are_orthogonal_and_energy_accounting_is_exact():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    coords = spectral_coordinates(d, q)
    total = torch.zeros_like(value)
    energy = 0.0
    names = {'H': 'head', 'M': 'middle', 'T': 'tail'}
    for block in PRIMARY_BLOCKS:
        component = block_component(d, q, names[block[0]], names[block[1]])
        total = total + component
        rows, cols = d.bands[{'H': 'head', 'M': 'middle', 'T': 'tail'}[block[0]]], d.bands[{'H': 'head', 'M': 'middle', 'T': 'tail'}[block[1]]]
        energy += float(coords[rows][:, cols].square().sum())
    assert torch.allclose(total, sum((block_component(d, q, names[b[0]], names[b[1]]) for b in PRIMARY_BLOCKS), torch.zeros_like(value)), atol=1e-5)
    assert energy >= 0
    accounting = block_energies(d, q)
    assert sum(accounting.blocks.values()) <= accounting.total + 1e-4


def test_all_accounting_blocks_reconstruct_reduced_projection():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    total = torch.zeros_like(value)
    for row in ("head", "middle", "tail", "other"):
        for col in ("head", "middle", "tail", "other"):
            total += block_component(d, q, row, col)
    projection = d.u @ spectral_coordinates(d, q) @ d.vh
    assert torch.allclose(total, projection, atol=1e-5, rtol=1e-5)


def test_restore_zero_and_all_accounting_blocks():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    assert torch.equal(restore_blocks(d, q, ()), q)
    restored = restore_blocks(d, q, tuple(f"{r}_{c}" for r in ("head", "middle", "tail", "other") for c in ("head", "middle", "tail", "other")))
    projected_error = d.u @ spectral_coordinates(d, q) @ d.vh
    assert torch.allclose(restored, q - projected_error, atol=1e-5, rtol=1e-5)
    assert torch.allclose(restore_blocks(d, q, tuple(f"{r}_{c}" for r in ("head", "middle", "tail", "other") for c in ("head", "middle", "tail", "other")) + ("unresolved",)), value, atol=1e-5, rtol=1e-5)


def test_grouped_restoration_equals_constituent_sum():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    grouped = restore_blocks(d, q, GROUPS["H_to_M"])
    separate = restore_blocks(d, q, ("HM", "MH"))
    assert torch.equal(grouped, separate)


def test_exact_polar_and_production_transform_are_read_only_and_deterministic():
    value = matrix()
    first, second = exact_polar(value), exact_polar(value.clone())
    assert torch.equal(first, second)
    assert first.shape == value.shape
    kwargs = {"steps": 5, "coefficients": (3.4445, -4.7750, 2.0315), "eps": 1e-7}
    assert torch.equal(zeropower_newton_schulz(value), zeropower_newton_schulz(value.clone(), **kwargs))


def test_restore_metrics_report_baseline_and_gain():
    value = matrix(); d = decompose(value); q = quantize(value, "int4-dynamic-b2048")
    kwargs = {"steps": 5, "coefficients": (3.4445, -4.7750, 2.0315), "eps": 1e-7}
    candidate = restore_blocks(d, q, ("MM",))
    result = paired_restore_metrics(value, q, candidate, transform_kwargs=kwargs)
    assert "baseline_update_cosine" in result
    assert result["update_cosine_gain"] is not None
    assert metric_pair(value, value)["cosine"] == 1.0
