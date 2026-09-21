import sys
from unittest.mock import patch

import torch

sys.path.insert(0, "src")
from optim.muon_approx_lowrank import (  # noqa: E402
    extract, principal_angle_max, reconstruct, subspace_projection_distance,
)


def test_rank_and_reconstruction_without_full_svd():
    x = torch.randn(20, 8, generator=torch.Generator().manual_seed(7))
    with patch.object(torch.linalg, "svd", side_effect=AssertionError("full SVD")):
        f = extract(x, 3, q=2, oversampling=4, initialization="canonical")
    assert f.u.shape == (20, 3)
    assert f.singular_values.shape == (3,)
    assert f.vh.shape == (3, 8)
    assert f.matmul_mx_count == 3
    assert f.matmul_mtx_count == 2
    assert reconstruct(f).shape == x.shape


def test_randomized_initialization_is_reproducible_and_q_zero_is_valid():
    x = torch.randn(12, 7, generator=torch.Generator().manual_seed(3))
    a = extract(x, 4, q=0, oversampling=4, initialization="randomized", seed=2026)
    b = extract(x, 4, q=0, oversampling=4, initialization="randomized", seed=2026)
    assert torch.equal(a.u, b.u)
    assert torch.equal(a.singular_values, b.singular_values)
    assert torch.equal(a.vh, b.vh)
    assert a.matmul_mx_count == 1 and a.matmul_mtx_count == 0


def test_projection_and_angle_metrics():
    x = torch.eye(5)[:, :2]
    y = x.clone()
    assert subspace_projection_distance(x, y) == 0.0
    assert principal_angle_max(x, y) == 0.0
    z = torch.eye(5)[:, 2:4]
    assert subspace_projection_distance(x, z) > 0
    assert principal_angle_max(x, z) is not None


def test_rank_zero_and_oversampling_clip():
    x = torch.randn(4, 3)
    f = extract(x, 0, q=4, oversampling=8)
    assert reconstruct(f).shape == x.shape
    assert f.rank == 0
    f2 = extract(x, 3, q=1, oversampling=20)
    assert f2.u.shape[1] == 3

