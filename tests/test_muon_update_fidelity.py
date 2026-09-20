import copy

import pytest
import torch

from optim.muon_reference import ReferenceMuon, zeropower_newton_schulz
from optim.muon_update_fidelity import (QUANTIZERS, MuonUpdateFidelityObserver,
                                        analyze_momentum, analyze_tensors, load_snapshot,
                                        save_snapshot)
from optim.state_simulation import persist_state


def test_update_analysis_calls_exact_production_muon_transform(monkeypatch):
    import optim.muon_update_fidelity as fidelity
    matrix = torch.tensor([[.2, -.7, .5], [.3, .1, -.4]])
    production = fidelity.muon_reference.zeropower_newton_schulz
    calls = []
    def traced(*args, **kwargs):
        calls.append(args[0].clone())
        return production(*args, **kwargs)
    monkeypatch.setattr(fidelity.muon_reference, "zeropower_newton_schulz", traced)
    row, _ = fidelity.analyze_momentum(matrix, quantizer="int8-linear-b2048")
    expected = production(matrix, 5, (3.4445, -4.7750, 2.0315), 1e-7)
    assert len(calls) == 2
    assert torch.equal(calls[0], matrix) and row["update_metric_status"] == "ok"
    assert torch.allclose(expected, production(calls[0], 5, (3.4445, -4.7750, 2.0315), 1e-7))


def test_observer_does_not_perturb_optimizer_state_parameters_or_rng():
    initial = torch.tensor([[1., 2., -3.], [.5, -.2, 4.]])
    grad = torch.tensor([[.2, -.1, .3], [.4, -.5, .6]])
    def run(enabled):
        p = torch.nn.Parameter(initial.clone()); p.grad = grad.clone()
        opt = ReferenceMuon([{"params": [p], "optimizer_group": "muon", "weight_decay": .1,
                              "state_quantization_granularity": "blockwise"}], state_simulation="int4_dynamic_momentum")
        if enabled:
            observer = MuonUpdateFidelityObserver({id(p): "p"})
            observer.begin_update(1); opt.set_diagnostic_observer(observer.observe)
        opt.step()
        if enabled: observer.finish_update(1)
        return p.detach().clone(), copy.deepcopy(opt.state_dict()), torch.rand(4)
    torch.manual_seed(123); enabled = run(True)
    torch.manual_seed(123); disabled = run(False)
    assert torch.equal(enabled[0], disabled[0])
    assert enabled[1].keys() == disabled[1].keys()
    assert torch.equal(enabled[1]["state"][0]["muon_momentum"], disabled[1]["state"][0]["muon_momentum"])
    assert torch.equal(enabled[2], disabled[2])


def test_offline_quantizers_match_training_persistence_paths():
    value = torch.linspace(-2, 2, 2051).reshape(1, -1)
    for identity, (simulation, _, _, _) in QUANTIZERS.items():
        row, _ = analyze_momentum(value, quantizer=identity)
        expected = persist_state(value, simulation, "muon_momentum", quantization_granularity="blockwise", quantization_block_size=2048)
        actual_error = (expected - value).square().sum().sqrt() / value.square().sum().sqrt()
        assert row["raw_momentum_relative_l2"] == float(actual_error)


def test_aggregate_uses_global_norms_not_mean_of_tensor_metrics():
    tensors = [{"name": "small", "tensor": torch.tensor([[1., .1]])},
               {"name": "large", "tensor": torch.tensor([[100., 0.]])}]
    rows = analyze_tensors(tensors, quantizers=["int4-linear-b2048"])
    aggregate = rows[-1]
    per = [x for x in rows if x["record_type"] == "tensor"]
    assert aggregate["raw_momentum_relative_l2"] != sum(x["raw_momentum_relative_l2"] for x in per) / 2
    # Directly recompute with the same existing persistence path.
    q = [persist_state(x["tensor"], "int4_linear_momentum", "muon_momentum", quantization_granularity="blockwise", quantization_block_size=2048) for x in tensors]
    ref = torch.cat([x["tensor"].reshape(-1) for x in tensors]); obs = torch.cat([x.reshape(-1) for x in q])
    assert aggregate["raw_momentum_relative_l2"] == pytest.approx(float((obs-ref).norm()/ref.norm()))
    assert aggregate["raw_momentum_cosine"] == pytest.approx(float(torch.dot(obs, ref) / (obs.norm() * ref.norm())))


def test_all_required_quantizers_and_snapshot_roundtrip(tmp_path):
    tensor = torch.randn(3, 683)
    for identity in QUANTIZERS:
        row, _ = analyze_momentum(tensor, quantizer=identity)
        assert row["raw_momentum_relative_l2"] >= 0 and row["update_relative_l2"] >= 0
    path = tmp_path / "snapshot.pt"
    save_snapshot(path, metadata={"update": 128, "source_commit": "abc", "recipe_fingerprint": "r", "data_fingerprint": "d", "seeds": {}},
                  tensors=[{"parameter_id": "x", "name": "x", "tensor": tensor}])
    loaded = load_snapshot(path)
    assert loaded["metadata"]["update"] == 128
    assert torch.equal(loaded["tensors"][0]["tensor"], tensor) and loaded["tensors"][0]["tensor"].dtype == torch.float32


def test_non_matrix_is_explicitly_excluded_from_update_metrics():
    row, _ = analyze_momentum(torch.ones(7), quantizer="int8-linear-b2048")
    assert row["update_metric_status"] == "excluded_not_2d_muon_matrix"
