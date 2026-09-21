import sys

import torch

sys.path.insert(0, "src")
from optim.muon_approx_lowrank import approximate_svd, reconstruct  # noqa: E402


def test_deterministic_initializations_and_shapes():
    x = torch.randn(9, 4, generator=torch.Generator().manual_seed(7))
    a = approximate_svd(x, 2, iterations=2, oversampling=4, initialization="canonical")
    b = approximate_svd(x, 2, iterations=2, oversampling=4, initialization="canonical")
    assert a.rank == 2 and a.working_rank == 4
    assert torch.equal(a.u, b.u) and torch.equal(a.singular_values, b.singular_values)
    assert reconstruct(a).shape == x.shape


def test_randomized_seed_is_reproducible_and_changes_with_seed():
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(3))
    a = approximate_svd(x, 3, iterations=1, oversampling=0, initialization="randomized", seed=2026)
    b = approximate_svd(x, 3, iterations=1, oversampling=0, initialization="randomized", seed=2026)
    c = approximate_svd(x, 3, iterations=1, oversampling=0, initialization="randomized", seed=2027)
    assert torch.equal(a.u, b.u)
    assert not torch.equal(a.u, c.u)


def test_q_zero_and_rank_never_exceed_request():
    x = torch.randn(4, 10)
    a = approximate_svd(x, 8, iterations=0, oversampling=8, initialization="canonical")
    assert a.rank == 4 and a.working_rank == 4
    assert a.m_multiplies == 0 and a.mt_multiplies == 1
    assert a.u.shape[1] == 4 and a.vh.shape[0] == 4


def test_approximate_path_uses_only_projected_svd(monkeypatch):
    calls = []
    original = torch.linalg.svd

    def wrapped(value, *args, **kwargs):
        calls.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", wrapped)
    x = torch.randn(20, 7)
    approximate_svd(x, 2, iterations=2, oversampling=4)
    assert calls == []
