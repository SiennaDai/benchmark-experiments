import pytest

torch = pytest.importorskip("torch")
from scripts.analyze_recursive_vq_correction_oracle import _corr_from_svd, _metrics


def test_low_rank_reconstruction_and_zero_rank():
    torch.manual_seed(0)
    ref = torch.randn(7, 5)
    vq = torch.randn(7, 5)
    u, s, vh = torch.linalg.svd(ref - vq, full_matrices=False)
    assert torch.equal(_corr_from_svd(vq, u, s, vh, 0), vq)
    c = _corr_from_svd(vq, u, s, vh, 2)
    assert c.shape == ref.shape
    assert torch.isfinite(c).all()
    assert _metrics(c, ref)[1] < _metrics(vq, ref)[1]


def test_bf16_factor_storage_formula():
    m, n, r = 1152, 384, 8
    assert 2 * r * (m + n) == 24576
