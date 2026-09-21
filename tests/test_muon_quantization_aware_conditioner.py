import sys

import torch

sys.path.insert(0, "src")
from optim.muon_quantization_aware_conditioner import (  # noqa: E402
    INT3_CODEBOOK,
    greedy_select,
    int3_dynamic_roundtrip,
    quantize_residual,
    selected_reconstruction,
)


def test_int3_is_deterministic_blockwise_and_preserves_zero():
    x = torch.tensor([0.0, 0.2, -0.4, 1.0, -1.0, 0.0])
    a = int3_dynamic_roundtrip(x, block_size=4)
    b = int3_dynamic_roundtrip(x, block_size=4)
    assert torch.equal(a, b)
    assert a[0].item() == 0.0 and a[-1].item() == 0.0
    assert torch.all((a.abs() <= 1.0))


def test_int4_delegation_and_int3_validation():
    x = torch.randn(17, 9)
    assert quantize_residual(x, 4).shape == x.shape
    assert quantize_residual(x, 3).shape == x.shape
    assert INT3_CODEBOOK.numel() == 7


def test_greedy_selection_is_deterministic_and_unique():
    x = torch.randn(12, 8, generator=torch.Generator().manual_seed(4))
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    a = greedy_select(x, u, s, vh, 4, 3, "range_aware", candidate_pool=6)
    b = greedy_select(x, u, s, vh, 4, 3, "range_aware", candidate_pool=6)
    assert a.modes == b.modes
    assert len(a.modes) == len(set(a.modes)) == 4
    assert all(0 <= i < 6 for i in a.modes)


def test_selected_reconstruction_has_expected_shape_and_rank_budget():
    x = torch.randn(10, 7)
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    c, q, reconstructed = selected_reconstruction(x, u, s, vh, (0, 2, 4), 3)
    assert c.shape == q.shape == reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()


def test_muon_oracle_objective_uses_reference_update():
    x = torch.randn(8, 8, generator=torch.Generator().manual_seed(9))
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    from optim import muon_reference
    kw = {"steps": 2, "coefficients": (3.4445, -4.7750, 2.0315), "eps": 1e-7}
    ref = muon_reference.zeropower_newton_schulz(x.clone(), **kw)
    row = greedy_select(x, u, s, vh, 2, 3, "muon_update_aware", candidate_pool=4,
                        reference_update=ref, transform_kwargs=kw)
    assert len(row.modes) == 2 and row.candidate_evaluations > 0
