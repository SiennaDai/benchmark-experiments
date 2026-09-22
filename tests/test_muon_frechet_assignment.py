import torch

from optim.muon_frechet import frechet_channels
from optim.muon_frechet_assignment import (
    analytic_pair_hessian,
    hutchinson_pair_blocks,
    objective_change_from_block,
    pair_cost_from_hessian,
    score_pair_candidates,
)


def test_exact_pair_hessian_quadratic_matches_direct_analytic_derivative():
    torch.manual_seed(501)
    m = torch.randn(6, 4, dtype=torch.float64)
    coords = (1, 0, 1, 1)
    h = analytic_pair_hessian(m, coords)
    e = torch.zeros_like(m)
    e[1, 0], e[1, 1] = .3, -.7
    direct = frechet_channels(m, e)["predicted_delta"]
    pair = torch.tensor([.3, -.7], dtype=torch.float64)
    assert torch.allclose(pair_cost_from_hessian(pair, h), direct.square().sum(), rtol=1e-10, atol=1e-10)


def test_global_frechet_objective_contains_cross_pair_terms():
    torch.manual_seed(502)
    m = torch.randn(7, 5, dtype=torch.float64)
    e1 = torch.zeros_like(m); e2 = torch.zeros_like(m)
    e1[0, 0] = .7; e2[0, 1] = -.4
    j1 = frechet_channels(m, e1)["predicted_delta"]
    j2 = frechet_channels(m, e2)["predicted_delta"]
    total = frechet_channels(m, e1 + e2)["predicted_delta"]
    assert abs(float(total.square().sum() - j1.square().sum() - j2.square().sum())) > 1e-7


def test_hutchinson_pair_blocks_are_deterministic_and_psd_projected():
    torch.manual_seed(503)
    m = torch.randn(5, 3)
    pairs = torch.tensor([[0, 1], [4, 5], [7, 8]])
    h1 = hutchinson_pair_blocks(m, pairs, probes=16, seed=23)
    h2 = hutchinson_pair_blocks(m, pairs, probes=16, seed=23)
    assert torch.equal(h1, h2)
    assert torch.allclose(h1, h1.transpose(1, 2))
    assert torch.linalg.eigvalsh(h1).min() >= -1e-10


def test_candidate_local_cost_uses_all_64_fixed_codewords_without_index_growth():
    torch.manual_seed(504)
    errors = torch.randn(11, 64, 2)
    h = torch.eye(2, dtype=torch.float64).expand(11, 2, 2).clone()
    costs = score_pair_candidates(errors, h)
    assert costs.shape == (11, 64)
    assert torch.equal(costs.argmin(1), errors.square().sum(-1).argmin(1))
    assert 6 == (64).bit_length() - 1


def test_coordinate_change_quadratic_formula_includes_global_linear_term():
    h = torch.tensor([[2., .5], [.5, 1.]], dtype=torch.float64)
    d = torch.tensor([.2, -.4], dtype=torch.float64)
    g = torch.tensor([1.5, -.3], dtype=torch.float64)
    expected = 2 * g.dot(d) + d @ h @ d
    assert torch.allclose(objective_change_from_block(d, g, h), expected)
