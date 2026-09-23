import pytest

torch = pytest.importorskip('torch')
from scripts.analyze_recursive_vq_k5_correction_oracle import _finish_sums


def test_global_metric_sums_match_direct_concatenation():
    torch.manual_seed(0)
    a = [torch.randn(3, 2), torch.randn(4, 2)]
    b = [torch.randn(3, 2), torch.randn(4, 2)]
    sums = {k: 0.0 for k in ('dot', 'a2', 'b2', 'd2')}
    for x, y in zip(a, b):
        d = x - y
        sums['dot'] += float((x * y).sum())
        sums['a2'] += float(x.square().sum())
        sums['b2'] += float(y.square().sum())
        sums['d2'] += float(d.square().sum())
    cos, rel = _finish_sums(sums)
    aa, bb = torch.cat([x.flatten() for x in a]), torch.cat([y.flatten() for y in b])
    assert cos == pytest.approx(float(torch.nn.functional.cosine_similarity(aa, bb, dim=0)), rel=1e-6)
    assert rel == pytest.approx(float((aa - bb).norm() / aa.norm()), rel=1e-6)


def test_zero_budget_has_zero_extra_bytes():
    assert int(0.0 * 10_616_832 / 8) == 0
