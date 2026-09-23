import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_vq_preconditioning_oracle import (  # noqa: E402
    inv_mulaw,
    inv_signed_power,
    mulaw,
    row_column_scales,
    signed_power,
)


def test_mulaw_roundtrip_is_deterministic():
    x = torch.linspace(-2, 2, 257)
    y = mulaw(x, 2.0, 50.0)
    assert torch.allclose(inv_mulaw(y, 2.0, 50.0), x.clamp(-2, 2), atol=2e-6, rtol=0)


def test_power_roundtrip_is_deterministic():
    x = torch.linspace(-2, 2, 257)
    y = signed_power(x, 2.0, 0.67)
    assert torch.allclose(inv_signed_power(y, 2.0, 0.67), x.clamp(-2, 2), atol=2e-6, rtol=0)


def test_row_column_scales_are_finite_and_positive():
    r = torch.tensor([[0.0, 2.0], [3.0, 4.0]])
    for mode in ("rms", "l2"):
        dr, dc = row_column_scales(r, mode)
        assert torch.isfinite(dr).all() and torch.isfinite(dc).all()
        assert (dr > 0).all() and (dc > 0).all()


def test_zero_matrix_scaling_is_safe():
    dr, dc = row_column_scales(torch.zeros(3, 4), "rms")
    assert torch.isfinite(dr).all() and torch.isfinite(dc).all()
