import sys

import torch

sys.path.insert(0, "src")
from optim.muon_danger_zone_crossmode_oracle import (  # noqa: E402
    DANGER_LOG10_END,
    DANGER_LOG10_START,
    component_from_mask,
    coordinate_masks,
    danger_mode_mask,
    restore_from_coordinates,
    select_energy_budget,
    select_top_coefficients,
)
from optim.muon_ns_sensitivity import exact_polar  # noqa: E402
from optim.muon_spectral_sensitivity import quantize  # noqa: E402
from optim.state_simulation import persist_state  # noqa: E402


def test_stored_danger_zone_is_fixed_and_half_open():
    s = torch.tensor([1.0, 1e-2, 1e-2 * (1 - 1e-5), 1e-3, 1e-4])
    mask = danger_mode_mask(s)
    assert DANGER_LOG10_START == -3.0 and DANGER_LOG10_END == -2.0
    assert mask.tolist() == [False, False, True, True, False]


def test_diagonal_and_cross_supports_are_disjoint_and_complete():
    s = torch.tensor([1.0, 1e-2, 3e-3, 1e-4])
    masks = coordinate_masks(s)
    danger_union = masks.diagonal | masks.cross
    assert not bool((masks.diagonal & masks.cross).any())
    assert torch.equal(masks.diagonal, torch.diag(masks.danger))
    expected = (masks.danger[:, None] | masks.danger[None, :]) & ~torch.eye(4, dtype=torch.bool)
    assert torch.equal(masks.cross, expected)
    assert torch.equal(danger_union, masks.diagonal | masks.cross)
    assert not bool(torch.diag(masks.cross).any())


def test_energy_budget_hits_target_without_new_support():
    x = torch.tensor([[3.0, -2.0], [1.0, 0.5]])
    selected, count, clipped = select_energy_budget(x, 2.0)
    assert not clipped and count >= 1
    assert torch.isclose(selected.norm(), torch.tensor(2.0), atol=1e-6)
    assert torch.all((selected == 0) | (selected == x) | (selected.abs() < x.abs()))


def test_top_coefficients_is_deterministic_and_bounded():
    x = torch.tensor([[1.0, -4.0], [3.0, 2.0]])
    a = select_top_coefficients(x, 2)
    b = select_top_coefficients(x, 2)
    assert torch.equal(a, b)
    assert int((a != 0).sum()) == 2
    assert torch.equal(a, torch.tensor([[0.0, -4.0], [3.0, 0.0]]))


def test_full_actual_residual_restoration_reconstructs_fp32():
    torch.manual_seed(11)
    m = torch.randn(4, 4)
    q = m + 0.1 * torch.randn(4, 4)
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    ehat = u.T @ (q - m) @ vh.T
    restored = restore_from_coordinates(q, u, vh, ehat)
    assert torch.allclose(restored, m, atol=2e-5, rtol=2e-5)


def test_component_projection_is_only_allowed_support():
    s = torch.tensor([1.0, 1e-2, 3e-3, 1e-4])
    masks = coordinate_masks(s)
    e = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    component = component_from_mask(e, masks.cross)
    assert torch.equal(component[~masks.cross], torch.zeros_like(component[~masks.cross]))
    assert torch.equal(component[masks.cross], e[masks.cross])


def test_baseline_uses_existing_dynamic_b2048_quantizer():
    value = torch.linspace(-1.0, 1.0, 32).reshape(4, 8)
    expected = persist_state(value.clone(), "int4_dynamic_momentum", "muon_momentum",
                             quantization_granularity="blockwise", quantization_block_size=2048)
    assert torch.equal(quantize(value, "int4-dynamic-b2048"), expected)


def test_exact_polar_is_orthogonal_factor_for_square_matrix():
    value = torch.tensor([[2.0, 1.0], [0.5, 3.0]])
    polar = exact_polar(value)
    assert torch.allclose(polar.T @ polar, torch.eye(2), atol=2e-5, rtol=2e-5)
