import io
import csv
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from optim.muon_recursive import RecursiveMuon, StructuralVQCodec  # noqa: E402
from optim.muon_reference import ReferenceMuon  # noqa: E402
from scripts.analyze_recursive_muon_error_feedback import analyze  # noqa: E402


def _codec():
    torch.manual_seed(991)
    codebook = torch.randn(64, 2) * .4
    codebook[0] = 0
    return StructuralVQCodec(codebook, rank=2, block_size=16)


def _recursive(parameter, alpha=None):
    kwargs = {} if alpha is None else {"recursive_error_feedback_alpha": alpha}
    return RecursiveMuon([{"params": [parameter], "optimizer_group": "muon", "weight_decay": 0.0}],
                         {id(parameter): _codec()}, lr=.003, muon_momentum=.8,
                         muon_nesterov=True, muon_ns_steps=2, **kwargs)


def _step(opt, p, grad):
    p.grad = grad.clone()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_alpha_zero_default_is_bitwise_same_as_explicit_zero_and_no_error_state():
    torch.manual_seed(10)
    a = torch.nn.Parameter(torch.randn(8, 8)); b = torch.nn.Parameter(a.detach().clone())
    default = _recursive(a); explicit = _recursive(b, 0.0)
    for i in range(4):
        g = torch.randn_like(a) * (i + 1)
        _step(default, a, g); _step(explicit, b, g)
        assert torch.equal(a, b)
        assert torch.equal(default.state[a]["compressed_momentum"].indices,
                           explicit.state[b]["compressed_momentum"].indices)
        assert "momentum_error" not in default.state[a]
        assert "momentum_error" not in explicit.state[b]


def test_first_update_is_identical_for_alpha_zero_half_and_one():
    torch.manual_seed(11)
    initial = torch.randn(8, 8)
    params = [torch.nn.Parameter(initial.clone()) for _ in range(3)]
    opts = [_recursive(params[0]), _recursive(params[1], .5), _recursive(params[2], 1.0)]
    g = torch.randn(8, 8)
    for p, opt in zip(params, opts):
        _step(opt, p, g)
    assert torch.equal(params[0], params[1]) and torch.equal(params[0], params[2])
    assert torch.equal(opts[1].state[params[1]]["momentum_error"],
                       opts[2].state[params[2]]["momentum_error"])


def test_recurrence_injects_alpha_times_mu_times_previous_error():
    torch.manual_seed(12)
    p = torch.nn.Parameter(torch.randn(8, 8)); opt = _recursive(p, .5); seen = []
    opt.set_mechanism_observer(lambda **kwargs: seen.append(kwargs))
    mu, alpha = .8, .5
    _step(opt, p, torch.randn_like(p))
    prev_decoded = opt.codecs[id(p)].decode(opt.state[p]["compressed_momentum"])
    prev_error = opt.state[p]["momentum_error"].clone()
    grad = torch.randn_like(p)
    _step(opt, p, grad)
    got = seen[-1]["momentum_candidate"]
    expected = mu * prev_decoded + grad + alpha * mu * prev_error
    wrong = mu * prev_decoded + grad + alpha * prev_error
    assert torch.allclose(got, expected, atol=2e-6, rtol=1e-6)
    assert not torch.allclose(got, wrong, atol=1e-4, rtol=0)


def test_error_buffer_is_exact_candidate_minus_persisted_decode_and_alpha_one_restores_reference():
    torch.manual_seed(13)
    p_ref = torch.nn.Parameter(torch.randn(8, 8)); p_ef = torch.nn.Parameter(p_ref.detach().clone())
    ref = ReferenceMuon([{"params": [p_ref], "optimizer_group": "muon", "weight_decay": 0.0}],
                        lr=.003, muon_momentum=.8, muon_nesterov=True, muon_ns_steps=2)
    ef = _recursive(p_ef, 1.0); candidates = []
    ef.set_mechanism_observer(lambda **kwargs: candidates.append(kwargs["momentum_candidate"].clone()))
    for i in range(5):
        g = torch.randn_like(p_ref)
        _step(ref, p_ref, g); _step(ef, p_ef, g)
        state = ef.state[p_ef]
        decoded = ef.codecs[id(p_ef)].decode(state["compressed_momentum"])
        assert torch.allclose(state["momentum_error"], candidates[-1] - decoded, atol=1e-7, rtol=0)
        assert "momentum" not in state and "momentum_candidate" not in state
        assert torch.allclose(p_ef, p_ref, atol=3e-6, rtol=1e-6)


def test_error_buffer_checkpoint_resume_and_storage_accounting():
    torch.manual_seed(14)
    p1 = torch.nn.Parameter(torch.randn(8, 8)); p2 = torch.nn.Parameter(p1.detach().clone())
    continuous = _recursive(p1, 1.0); split = _recursive(p2, 1.0)
    gradients = [torch.randn(8, 8) for _ in range(4)]
    for g in gradients[:2]:
        _step(continuous, p1, g); _step(split, p2, g)
    stream = io.BytesIO(); torch.save(split.state_dict(), stream); stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=False)
    p3 = torch.nn.Parameter(p2.detach().clone()); resumed = _recursive(p3, 1.0)
    resumed.load_state_dict(saved)
    for g in gradients[2:]:
        _step(continuous, p1, g); _step(resumed, p3, g)
    assert torch.allclose(p1, p3, atol=1e-7, rtol=0)
    c_state = continuous.state[p1]; r_state = resumed.state[p3]
    assert torch.equal(c_state["compressed_momentum"].indices, r_state["compressed_momentum"].indices)
    assert torch.equal(c_state["momentum_error"], r_state["momentum_error"])
    assert r_state["momentum_error"].device == p3.device
    assert r_state["momentum_error"].dtype == torch.float32
    summary = resumed.recursive_state_summary()
    row = summary["tensors"][0]
    assert row["has_fp32_error_buffer"] and not row["has_fp32_momentum"]
    assert row["error_buffer_bits"] == p3.numel() * 32
    assert summary["error_buffer_bits"] == p3.numel() * 32
    assert summary["persistent_bits"] == (summary["compressed_state_bits"] +
                                             summary["error_buffer_bits"] + summary["codebook_bits"])


def test_diagnostics_do_not_create_persistent_state_or_change_alpha_zero_step():
    torch.manual_seed(15)
    a = torch.nn.Parameter(torch.randn(8, 8)); b = torch.nn.Parameter(a.detach().clone())
    baseline = _recursive(a, 0.0); observed = _recursive(b, 0.0)
    observed.set_mechanism_observer(lambda **kwargs: None)
    g = torch.randn_like(a)
    _step(baseline, a, g); _step(observed, b, g)
    assert torch.equal(a, b)
    assert set(observed.state[b]) == {"compressed_momentum"}
    assert "momentum_error" not in observed.state_dict()["state"][0]


def test_alpha_must_be_finite_and_nonnegative():
    p = torch.nn.Parameter(torch.zeros(8, 8))
    for alpha in (-.1, math.inf, math.nan, True):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            _recursive(p, alpha)


def test_error_feedback_mechanism_observer_records_reconstruction_and_injection(tmp_path):
    from optim.muon_mechanism import MuonMechanismObserver
    p = torch.nn.Parameter(torch.randn(8, 8)); opt = _recursive(p, 1.0)
    observer = MuonMechanismObserver(run_dir=tmp_path, optimizer_name="recursive_muon",
        parameter_names={id(p): "transformer.h.0.test.weight"}, raw_updates=[1, 2],
        scalar_updates=[1, 2], metadata={"muon_momentum": .8, "muon_ns_steps": 2,
        "muon_ns_coefficients": [3.4445, -4.775, 2.0315], "muon_eps": 1e-7,
        "error_feedback_alpha": 1.0})
    opt.set_mechanism_observer(observer.observe)
    for update in (1, 2):
        opt.set_mechanism_update(update); observer.begin_update(update)
        _step(opt, p, torch.randn_like(p))
        observer.finish_update(update=update, processed_target_tokens=update*8192,
                               train_nll=1.0, learning_rate=.001)
    rows = [json.loads(x) for x in (tmp_path / "mechanism/metrics.jsonl").read_text().splitlines()]
    assert rows[0]["error_feedback_alpha"] == 1.0
    assert rows[0]["injected_correction_norm_ratio"] == 0.0
    assert rows[1]["injected_correction_norm_ratio"] > 0
    assert rows[1]["previous_candidate_reconstruction_relative_l2"] < 1e-6
    blob = torch.load(tmp_path / "mechanism/update_000002.pt", map_location="cpu", weights_only=False)
    assert "momentum_error" not in blob["tensors"][0]  # derive it without duplicating the large tensor
    item = blob["tensors"][0]
    recovered_error = item["momentum_candidate"] - item["momentum_persisted_decoded"]
    assert torch.equal(recovered_error, opt.state[p]["momentum_error"].cpu())


def test_four_trajectory_analysis_smoke_and_global_pooling(tmp_path):
    torch.manual_seed(81)
    runs = {key: tmp_path / key for key in ("fp32", "alpha0", "alpha05", "alpha1")}
    mu = .8
    alphas = {"alpha0": 0.0, "alpha05": .5, "alpha1": 1.0}
    shared_config = {"model": {"n_layer": 1}, "data": {"manifest": "frozen"},
        "train": {"target_tokens": 4096*8192}, "schedule": {"total_updates": 4096},
        "precision": {"compute": "fp32"}, "eval": {"every_updates": 128}}
    for key, run in runs.items():
        (run / "mechanism").mkdir(parents=True)
        optimizer_cfg = {"name": "reference_muon" if key == "fp32" else "recursive_muon",
            "muon_momentum": mu, "muon_nesterov": True, "muon_ns_steps": 2,
            "muon_ns_coefficients": [3.4445, -4.775, 2.0315], "muon_eps": 1e-7}
        if key != "fp32":
            optimizer_cfg.update({"recursive_codebook_key": "s0_k8_w64_t8_v1200",
                "recursive_codebook_path": "reports/codebook.pt", "recursive_rank": 8,
                "recursive_block_size": 2048, "recursive_factor_dtype": "bf16",
                "recursive_structure_mode": "exact_svd_oracle", "recursive_representation": "vq_int3",
                "recursive_error_feedback_alpha": alphas[key]})
        config = shared_config | {"optimizer": optimizer_cfg}
        (run / "resolved_config.json").write_text(json.dumps(config))
        summary = {"seed": 1, "data_seed": 1337, "algorithm_seed": 2026,
            "protocol_id": "test", "data_fingerprint": "hash", "schedule_total_updates": 4096,
            "total_updates": 4096, "target_tokens": 4096*8192, "sequence_length": 256,
            "compute_precision": "fp32", "completed_updates": 512, "status": "paused_staged",
            "optimizer_name": "reference_muon" if key == "fp32" else "recursive_muon",
            "git_commit": "test", "optimizer_state_bytes": 1000,
            "optimizer_state": {"compressed_state_bits": 100, "error_buffer_bits": 64,
                                "codebook_bits": 512, "unique_storage_bytes": 1000, "tensors": []}}
        (run / "summary.json").write_text(json.dumps(summary))
        train = [{"event_type": "train", "completed_updates": i,
                  "processed_target_tokens": i*8192, "tokens_this_update": 8192, "lr": .001}
                 for i in range(1, 513)]
        ev = [{"event_type": "eval", "completed_updates": i, "nll": 5.5 + i/10000}
              for i in (0, 128, 256, 384, 512)]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in train+ev))
        prev = torch.zeros(4, 4); prev_candidate = torch.zeros_like(prev)
        for update in (1, 8, 32, 128, 512):
            torch.manual_seed(7000 + update)
            grad = torch.randn(4, 4) * .1
            if key == "fp32":
                entering = prev_candidate
                candidate = mu * entering + grad
                persisted = candidate
            else:
                entering = prev
                e_prev = prev_candidate - prev
                alpha = alphas[key]
                candidate = mu * entering + grad + alpha*mu*e_prev
                persisted = candidate - .01*torch.ones_like(candidate)
            item = {"name": "transformer.h.0.test.weight", "shape": [4, 4],
                "gradient": grad, "momentum_prev_decoded": entering,
                "momentum_prev_candidate": prev_candidate,
                "momentum_candidate": candidate, "momentum_persisted_decoded": persisted}
            metadata = {"seed": 1, "data_seed": 1337, "algorithm_seed": 2026,
                "protocol_id": "test", "data_fingerprint": "hash", "schedule_total_updates": 4096,
                "tokens_per_update": 8192, "sequence_length": 256,
                "muon_momentum": mu, "muon_nesterov": True, "muon_ns_steps": 2,
                "muon_ns_coefficients": [3.4445, -4.775, 2.0315], "muon_eps": 1e-7,
                "optimizer_name": summary["optimizer_name"], "update": update,
                "processed_target_tokens": update*8192,
                "error_feedback_alpha": alphas.get(key, 0.0)}
            torch.save({"format": "recursive_muon_mechanism_snapshot", "version": 1,
                        "metadata": metadata, "tensors": [item]},
                       run / "mechanism" / f"update_{update:06d}.pt")
            prev_candidate = candidate
            prev = persisted
    out = tmp_path / "report"
    summary = analyze(runs, out)
    assert summary["protocol_integrity"]["status"] == "compatible"
    rows = list(csv.DictReader((out / "landmark_metrics.csv").open()))
    assert len(rows) == 15
    row = next(x for x in rows if x["update"] == "512" and x["method"] == "alpha1")
    assert float(row["k5_cosine"]) == pytest.approx(1.0, abs=2e-5)
    assert (out / "comparison.md").exists()
