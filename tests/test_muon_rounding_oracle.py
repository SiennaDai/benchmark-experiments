import sys

import torch

sys.path.insert(0, "src")
import optim.muon_rounding_oracle as oracle
from optim.state_simulation import create_bitsandbytes_dynamic_map, persist_state


def test_nearest_oracle_exactly_matches_production_dynamic_roundtrip():
    value = torch.linspace(-1, 1, 4099).reshape(1, -1).float()
    parts = oracle._candidate_parts(value)
    expected = persist_state(value.clone(), "int4_dynamic_momentum", "muon_momentum",
                             quantization_granularity="blockwise", quantization_block_size=2048)
    assert torch.equal(oracle.production_nearest(value), expected)
    assert torch.equal(parts.nearest * parts.scale, expected)


def test_candidates_are_existing_codebook_levels_and_fixed_scale_blocks():
    value = torch.randn(2, 4097)
    parts = oracle._candidate_parts(value)
    codebook = create_bitsandbytes_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4)
    assert all(bool(torch.isin(x, codebook).all()) for x in (parts.lower, parts.upper, parts.nearest))
    for start in range(0, value.numel(), 2048):
        stop = min(start + 2048, value.numel())
        expected = value.reshape(-1)[start:stop].abs().amax()
        assert torch.equal(parts.scale.reshape(-1)[start], torch.where(expected == 0, torch.ones_like(expected), expected))


def test_oracle_search_is_deterministic_and_does_not_change_input():
    value = torch.randn(32, 64)
    before = value.clone()
    item = {"tensor": value, "parameter_id": "p", "name": "p"}
    first = oracle.analyze_oracle_tensor(item, update_max_candidates=8, update_max_groups=2)
    second = oracle.analyze_oracle_tensor(item, update_max_candidates=8, update_max_groups=2)
    assert torch.equal(value, before)
    assert [(r["rounding_mode"], r["raw_momentum_cosine"], r["muon_update_cosine"]) for r in first[0]] == [(r["rounding_mode"], r["raw_momentum_cosine"], r["muon_update_cosine"]) for r in second[0]]


def test_update_oracle_acceptance_never_worsens_exact_objective():
    value = torch.randn(16, 32)
    rows, stats = oracle.analyze_oracle_tensor({"tensor": value, "parameter_id": "p", "name": "p"}, update_max_candidates=16, update_max_groups=4)
    nearest = next(r for r in rows if r["rounding_mode"] == "nearest")
    update = next(r for r in rows if r["rounding_mode"] == "muon_update_direction_oracle")
    assert update["muon_update_cosine"] >= nearest["muon_update_cosine"] - 1e-12
    assert stats[1]["accepted_flips"] <= stats[1]["considered_count"]


def test_raw_oracle_acceptance_never_worsens_raw_cosine():
    value = torch.randn(16, 32)
    rows, _ = oracle.analyze_oracle_tensor({"tensor": value, "parameter_id": "p", "name": "p"}, update_max_candidates=4, update_max_groups=1)
    nearest = next(r for r in rows if r["rounding_mode"] == "nearest")
    raw = next(r for r in rows if r["rounding_mode"] == "raw_direction_oracle")
    assert raw["raw_momentum_cosine"] >= nearest["raw_momentum_cosine"] - 1e-12


def test_production_transform_is_called_for_update_oracle(monkeypatch):
    calls = []
    production = oracle.muon_reference.zeropower_newton_schulz

    def traced(*args, **kwargs):
        calls.append(1)
        return production(*args, **kwargs)

    monkeypatch.setattr(oracle.muon_reference, "zeropower_newton_schulz", traced)
    oracle.analyze_oracle_tensor({"tensor": torch.randn(8, 16), "parameter_id": "p", "name": "p"}, update_max_candidates=2, update_max_groups=1)
    assert calls
