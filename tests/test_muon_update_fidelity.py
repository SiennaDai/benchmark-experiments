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
    features = torch.tensor([[.2, -.1, .3], [.4, -.5, .6]])
    def run(enabled):
        p = torch.nn.Parameter(initial.clone())
        loss = (p * features).square().mean(); loss.backward()
        gradient = p.grad.detach().clone()
        opt = ReferenceMuon([{"params": [p], "optimizer_group": "muon", "weight_decay": .1,
                              "state_quantization_granularity": "blockwise"}], state_simulation="int4_dynamic_momentum")
        if enabled:
            observer = MuonUpdateFidelityObserver({id(p): "p"})
            observer.begin_update(1); opt.set_diagnostic_observer(observer.observe)
        opt.step()
        if enabled: observer.finish_update(1)
        return float(loss.detach()), gradient, p.detach().clone(), copy.deepcopy(opt.state_dict()), torch.rand(4)
    torch.manual_seed(123); enabled = run(True)
    torch.manual_seed(123); disabled = run(False)
    assert enabled[0] == disabled[0] and torch.equal(enabled[1], disabled[1])
    assert torch.equal(enabled[2], disabled[2])
    assert enabled[3].keys() == disabled[3].keys()
    assert torch.equal(enabled[3]["state"][0]["muon_momentum"], disabled[3]["state"][0]["muon_momentum"])
    assert torch.equal(enabled[4], disabled[4])


def test_snapshot_only_observer_does_not_perturb_training_trajectory():
    initial = torch.tensor([[1., 2., -3.], [.5, -.2, 4.]])
    features = torch.tensor([[.2, -.1, .3], [.4, -.5, .6]])
    def run(snapshot_only):
        p = torch.nn.Parameter(initial.clone())
        loss = (p * features).square().mean(); loss.backward()
        gradient = p.grad.detach().clone()
        opt = ReferenceMuon([{"params": [p], "optimizer_group": "muon", "weight_decay": .1,
                              "state_quantization_granularity": "blockwise"}], state_simulation="int4_dynamic_momentum")
        observer = None
        if snapshot_only:
            observer = MuonUpdateFidelityObserver({id(p): "p"}, snapshot_updates=[1],
                                                   online_fidelity_enabled=False)
            observer.begin_update(1); opt.set_diagnostic_observer(observer.observe)
        opt.step()
        snapshot = observer.finish_update(1)[1] if observer is not None else []
        return p.detach().clone(), copy.deepcopy(opt.state_dict()), torch.rand(4), gradient, snapshot
    torch.manual_seed(123); enabled = run(True)
    torch.manual_seed(123); disabled = run(False)
    assert torch.equal(enabled[0], disabled[0])
    assert torch.equal(enabled[1]["state"][0]["muon_momentum"], disabled[1]["state"][0]["muon_momentum"])
    assert torch.equal(enabled[2], disabled[2])
    assert torch.equal(enabled[4][0]["tensor"], enabled[3])


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


def test_snapshot_only_collects_only_selected_updates_and_does_no_analysis(monkeypatch):
    import optim.muon_update_fidelity as fidelity

    copies, analyses, transforms = [], [], []
    original_copy = fidelity._copy_momentum_to_cpu
    monkeypatch.setattr(fidelity, "_copy_momentum_to_cpu",
                        lambda value: (copies.append(value), original_copy(value))[1])
    monkeypatch.setattr(fidelity, "analyze_tensors",
                        lambda *args, **kwargs: analyses.append(True))
    production = fidelity.muon_reference.zeropower_newton_schulz
    monkeypatch.setattr(fidelity.muon_reference, "zeropower_newton_schulz",
                        lambda *args, **kwargs: (transforms.append(True), production(*args, **kwargs))[1])
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    observer = MuonUpdateFidelityObserver({id(parameter): "p"},
                                          snapshot_updates=[7],
                                          online_fidelity_enabled=False)
    for update in range(1, 11):
        observer.begin_update(update)
        observer.observe(parameter=parameter, momentum_pre=torch.full((2, 2), update),
                         momentum_post=None, updated=None, group=None)
        rows, snapshots = observer.finish_update(update)
        assert rows == []
        assert bool(snapshots) is (update == 7)
    assert len(copies) == 1
    assert len(analyses) == 0
    assert len(transforms) == 0


def test_online_fidelity_collects_and_analyzes_every_update(monkeypatch):
    import optim.muon_update_fidelity as fidelity

    copies, updates = [], []
    original_copy = fidelity._copy_momentum_to_cpu
    monkeypatch.setattr(fidelity, "_copy_momentum_to_cpu",
                        lambda value: (copies.append(value), original_copy(value))[1])
    def traced(items, **kwargs):
        updates.append(len(items))
        return []
    monkeypatch.setattr(fidelity, "analyze_tensors", traced)
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    observer = MuonUpdateFidelityObserver({id(parameter): "p"},
                                          snapshot_updates=[2],
                                          online_fidelity_enabled=True)
    for update in range(1, 4):
        observer.begin_update(update)
        observer.observe(parameter=parameter, momentum_pre=torch.ones(2, 2),
                         momentum_post=None, updated=None, group=None)
        observer.finish_update(update)
    assert len(copies) == 3
    assert updates == [1, 1, 1]


def test_snapshot_only_selected_snapshot_matches_online_snapshot():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    momentum = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    def collect(online):
        observer = MuonUpdateFidelityObserver({id(parameter): "p"},
                                              snapshot_updates=[7],
                                              online_fidelity_enabled=online)
        observer.begin_update(7)
        observer.observe(parameter=parameter, momentum_pre=momentum,
                         momentum_post=None, updated=None, group=None)
        return observer.finish_update(7)
    online_rows, online_snapshot = collect(True)
    offline_rows, offline_snapshot = collect(False)
    assert online_rows
    assert offline_rows == []
    assert torch.equal(online_snapshot[0]["tensor"], offline_snapshot[0]["tensor"])
    assert online_snapshot[0]["parameter_id"] == offline_snapshot[0]["parameter_id"] == "p"
    assert analyze_tensors(online_snapshot) == analyze_tensors(offline_snapshot)
