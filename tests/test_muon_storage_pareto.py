import sys

import torch

sys.path.insert(0, "src")
from optim.muon_storage_pareto import (  # noqa: E402
    FACTOR_VARIANTS,
    block_count,
    cast_factors,
    direct_storage_bits,
    metadata_bits,
    pareto_mask,
    storage_bits,
)


def test_storage_formula_and_global_components():
    row = storage_bits((10, 20), 2, "fp32")
    assert row["idealized_bits"] == 4 * 10 * 20 + 32 * (10 * 2 + 2 + 20 * 2)
    assert row["fp32_bits"] == 32 * 10 * 20
    assert row["metadata_inclusive_bits"] >= row["idealized_bits"]
    assert block_count(2049) == 2


def test_factor_precisions_and_casts_are_explicit():
    assert FACTOR_VARIANTS["bf16"] == (16, 16, 16)
    assert FACTOR_VARIANTS["fp16_uv_fp32_sigma"] == (16, 32, 16)
    u = torch.tensor([[1.2345]], dtype=torch.float32)
    s = torch.tensor([0.9876], dtype=torch.float32)
    vh = torch.tensor([[0.3333]], dtype=torch.float32)
    u16, s16, v16 = cast_factors(u, s, vh, "bf16")
    assert u16.dtype == s16.dtype == v16.dtype == torch.float32
    assert not torch.equal(u16, u)
    _, sm, _ = cast_factors(u, s, vh, "bf16_uv_fp32_sigma")
    assert torch.equal(sm, s)


def test_metadata_inclusive_storage_never_lower_and_direct_baseline():
    structural = storage_bits((5, 7), 1, "fp16", include_metadata=True)
    direct = direct_storage_bits((5, 7), include_metadata=True)
    assert structural["metadata_inclusive_bits"] >= structural["idealized_bits"]
    assert direct["int4_metadata_inclusive_bits"] >= direct["int4_idealized_bits"]


def test_pareto_filter_and_deterministic_storage():
    points = [
        {"storage_ratio_vs_fp32": 0.2, "update_cosine": 0.7},
        {"storage_ratio_vs_fp32": 0.4, "update_cosine": 0.8},
        {"storage_ratio_vs_fp32": 0.3, "update_cosine": 0.6},
    ]
    assert pareto_mask(points) == [True, True, False]
    low_error = [
        {"storage_ratio_vs_fp32": 0.2, "update_relative_l2": 0.7},
        {"storage_ratio_vs_fp32": 0.4, "update_relative_l2": 0.3},
        {"storage_ratio_vs_fp32": 0.3, "update_relative_l2": 0.8},
    ]
    assert pareto_mask(low_error, y="update_relative_l2", higher_is_better=False) == [True, True, False]
    assert storage_bits((10, 20), 2, "bf16") == storage_bits((10, 20), 2, "bf16")
