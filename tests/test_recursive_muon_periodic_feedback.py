import copy
import io
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from config.recipe import RecipeError, load_recipe  # noqa: E402
from optim.muon_recursive import (RecursiveMuon, StructuralVQCodec,
                                  periodic_correction_due,
                                  periodic_error_accumulator_next)  # noqa: E402
from optim.muon_reference import ReferenceMuon  # noqa: E402
from reporting import scientific_differences  # noqa: E402
from benchmark_suite import load_suite  # noqa: E402
from analyze_recursive_muon_periodic_feedback import analyze, _run_config  # noqa: E402


def _codec():
    torch.manual_seed(891)
    codebook = torch.randn(64, 2) * .35
    codebook[0] = 0
    return StructuralVQCodec(codebook, rank=2, block_size=16)


def _opt(parameter, *, mode="none", interval=None, alpha=0.0):
    return RecursiveMuon(
        [{"params": [parameter], "optimizer_group": "muon", "weight_decay": 0.0}],
        {id(parameter): _codec()}, lr=.003, muon_momentum=.8,
        muon_nesterov=True, muon_ns_steps=2,
        recursive_error_feedback_alpha=alpha,
        recursive_error_feedback_mode=mode,
        recursive_error_feedback_interval=interval)


def _step(opt, parameter, update, gradient):
    opt.set_mechanism_update(update)
    parameter.grad = gradient.clone()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_unpaid_residual_decay_is_exact_and_correction_pays_old_accumulator():
    mu = .8
    e1, e2, e3, e4 = [torch.tensor([float(i)]) for i in (1, 2, 3, 4)]
    zero = torch.zeros_like(e1)
    a2 = periodic_error_accumulator_next(zero, e1, mu, correction_applied=False)
    assert a2.item() == pytest.approx(mu)
    a3 = periodic_error_accumulator_next(a2, e2, mu, correction_applied=False)
    assert a3.item() == pytest.approx(mu**2 + mu*2)
    a4 = periodic_error_accumulator_next(a3, e3, mu, correction_applied=False)
    assert a4.item() == pytest.approx(mu**3 + mu**2*2 + mu*3)
    a5 = periodic_error_accumulator_next(a4, e4, mu, correction_applied=True)
    assert a5.item() == pytest.approx(mu*4)
    assert periodic_correction_due(4, 4) and not periodic_correction_due(3, 4)
    assert not periodic_correction_due(1, 4)


def test_periodic_k1_matches_every_step_alpha1_and_first_update_matches():
    torch.manual_seed(91)
    initial = torch.randn(8, 8)
    a = torch.nn.Parameter(initial.clone()); b = torch.nn.Parameter(initial.clone())
    periodic = _opt(a, mode="periodic", interval=1)
    every_step = _opt(b, mode="fractional", alpha=1.0)
    for update in range(1, 6):
        grad = torch.randn_like(a)
        _step(periodic, a, update, grad); _step(every_step, b, update, grad)
        if update == 1:
            assert torch.equal(a, b)
        else:
            assert torch.allclose(a, b, atol=3e-6, rtol=1e-6)
        assert torch.equal(periodic.state[a]["compressed_momentum"].indices,
                           every_step.state[b]["compressed_momentum"].indices)
        assert torch.isfinite(periodic.state[a]["momentum_error_accumulator"]).all()


def test_no_correction_before_interval_matches_alpha_zero_trajectory():
    torch.manual_seed(92)
    initial = torch.randn(8, 8)
    a = torch.nn.Parameter(initial.clone()); b = torch.nn.Parameter(initial.clone())
    periodic = _opt(a, mode="periodic", interval=8)
    baseline = _opt(b, mode="none")
    for update in range(1, 8):
        grad = torch.randn_like(a)
        _step(periodic, a, update, grad); _step(baseline, b, update, grad)
        assert torch.equal(a, b)
        assert torch.equal(periodic.state[a]["compressed_momentum"].indices,
                           baseline.state[b]["compressed_momentum"].indices)


def test_periodic_checkpoint_resume_and_fp32_storage_accounting():
    torch.manual_seed(93)
    p0 = torch.nn.Parameter(torch.randn(8, 8)); p1 = torch.nn.Parameter(p0.detach().clone())
    continuous = _opt(p0, mode="periodic", interval=4)
    split = _opt(p1, mode="periodic", interval=4)
    grads = [torch.randn(8, 8) for _ in range(6)]
    for update, grad in enumerate(grads[:3], 1):
        _step(continuous, p0, update, grad); _step(split, p1, update, grad)
    stream = io.BytesIO(); torch.save(split.state_dict(), stream); stream.seek(0)
    loaded = torch.load(stream, map_location="cpu", weights_only=False)
    p2 = torch.nn.Parameter(p1.detach().clone()); resumed = _opt(p2, mode="periodic", interval=4)
    resumed.load_state_dict(loaded)
    for update, grad in enumerate(grads[3:], 4):
        _step(continuous, p0, update, grad); _step(resumed, p2, update, grad)
    assert torch.equal(p0, p2)
    assert torch.equal(continuous.state[p0]["momentum_error_accumulator"],
                       resumed.state[p2]["momentum_error_accumulator"])
    assert resumed.state[p2]["momentum_error_accumulator"].device == p2.device
    assert resumed.state[p2]["momentum_error_accumulator"].dtype == torch.float32
    summary = resumed.recursive_state_summary()
    row = summary["tensors"][0]
    assert row["periodic_accumulator_bits"] == p2.numel() * 32
    assert row["has_fp32_periodic_accumulator"]
    assert not row["has_fp32_momentum"]
    assert summary["periodic_accumulator_bits"] == p2.numel() * 32
    assert summary["persistent_bits"] == (summary["compressed_state_bits"] +
        summary["periodic_accumulator_bits"] + summary["codebook_bits"])


def test_observer_records_correction_event_and_accumulator_norms(tmp_path):
    from optim.muon_mechanism import MuonMechanismObserver
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = _opt(p, mode="periodic", interval=2)
    observer = MuonMechanismObserver(run_dir=tmp_path, optimizer_name="recursive_muon",
        parameter_names={id(p): "transformer.h.0.test.weight"}, raw_updates=[1],
        scalar_updates=[1], metadata={"muon_momentum": .8, "muon_ns_steps": 2,
            "muon_ns_coefficients": [3.4445, -4.775, 2.0315], "muon_eps": 1e-7,
            "error_feedback_mode": "periodic", "error_feedback_interval": 2})
    opt.set_mechanism_observer(observer.observe)
    for update in (1, 2, 3, 4):
        observer.begin_update(update)
        _step(opt, p, update, torch.randn_like(p))
        observer.finish_update(update=update, processed_target_tokens=update*8192,
                               train_nll=1.0, learning_rate=.001)
    rows = [json.loads(line) for line in (tmp_path / "mechanism/metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == [1, 2, 4]
    assert not rows[0]["correction_applied"]
    assert rows[1]["correction_applied"] and rows[2]["correction_applied"]
    assert rows[1]["periodic_injected_correction_norm_ratio"] > 0


def test_periodic_recipe_protocol_and_suite_vary_only_interval():
    paths = [ROOT / f"recipes/mechanism_recursive_vq_int3_ef_periodic_k{k}_4096_s1.json"
             for k in (4, 16, 64)]
    configs = [load_recipe(path) for path in paths]
    assert [cfg["optimizer"]["recursive_error_feedback_interval"] for cfg in configs] == [4, 16, 64]
    assert all(cfg["derived"]["total_updates"] == cfg["derived"]["schedule_total_updates"] == 4096
               for cfg in configs)
    assert all(cfg["experiment"]["seed"] == 1 and cfg["experiment"]["data_seed"] == 1337 and
               cfg["experiment"]["algorithm_seed"] == 2026 for cfg in configs)
    suite = load_suite(ROOT / "benchmarks/recursive_muon_periodic_feedback_512_s1_v1.json")
    assert [entry["run_id"].split("k")[-1].split("_")[0] for entry in suite["runs"]] == ["4", "16", "64"]
    assert suite["trajectory"]["landmark_updates"] == [1, 8, 32, 128, 512]
    assert scientific_differences(configs, [entry["run_id"] for entry in suite["runs"]], suite["vary"]) == []


def test_invalid_periodic_recipe_requires_interval_and_zero_alpha(tmp_path):
    base = json.loads((ROOT / "recipes/mechanism_recursive_vq_int3_ef_periodic_k4_4096_s1.json").read_text())
    bad = copy.deepcopy(base); del bad["optimizer"]["recursive_error_feedback_interval"]
    path = tmp_path / "missing.json"; path.write_text(json.dumps(bad))
    with pytest.raises(RecipeError, match="recursive_error_feedback_interval"):
        load_recipe(path)
    bad = copy.deepcopy(base); bad["optimizer"]["recursive_error_feedback_alpha"] = .5
    path = tmp_path / "alpha.json"; path.write_text(json.dumps(bad))
    with pytest.raises(RecipeError, match="alpha=0"):
        load_recipe(path)


def test_periodic_analysis_smoke_pools_k5_and_writes_required_outputs(tmp_path):
    methods = ("fp32", "alpha0", "k4", "k16", "k64", "alpha1")
    runs = {method: tmp_path / method for method in methods}
    shared_config = {"model": {"n_layer": 1}, "data": {"manifest": "frozen"},
        "train": {"target_tokens": 4096*8192}, "schedule": {"total_updates": 4096},
        "precision": {"compute": "fp32"}, "eval": {"every_updates": 128}}
    mu = .8
    for method, run in runs.items():
        (run / "mechanism").mkdir(parents=True)
        optimizer = {"name": "reference_muon" if method == "fp32" else "recursive_muon",
            "lr": .001, "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1,
            "fused": False, "foreach": False, "state_simulation": "none",
            "muon_momentum": mu, "muon_nesterov": True, "muon_ns_steps": 2,
            "muon_ns_coefficients": [3.4445, -4.775, 2.0315], "muon_eps": 1e-7}
        if method != "fp32":
            optimizer.update({"recursive_rank": 8, "recursive_block_size": 2048,
                "recursive_factor_dtype": "bf16", "recursive_structure_mode": "exact_svd_oracle",
                "recursive_representation": "vq_int3", "recursive_codebook_path": "reports/cb.pt",
                "recursive_codebook_key": "s0_k8_w64_t8_v1200"})
        if method == "alpha1": optimizer["recursive_error_feedback_alpha"] = 1.0
        if method in {"k4", "k16", "k64"}:
            optimizer["recursive_error_feedback_mode"] = "periodic"
            optimizer["recursive_error_feedback_interval"] = int(method[1:])
        cfg = shared_config | {"optimizer": optimizer}
        (run / "resolved_config.json").write_text(json.dumps(cfg))
        summary = {"seed": 1, "data_seed": 1337, "algorithm_seed": 2026,
            "protocol_id": "slimpajama-hash-split-v1",
            "data_fingerprint": "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18",
            "schedule_total_updates": 4096, "total_updates": 4096,
            "target_tokens": 33554432, "sequence_length": 256, "compute_precision": "fp32",
            "completed_updates": 512, "status": "paused_staged", "recipe_fingerprint": method,
            "git_commit": "test", "optimizer_state_bytes": 1000,
            "optimizer_state": {"unique_storage_bytes": 1000, "persistent_bits": 8000,
                "compressed_state_bits": 7000, "codebook_bits": 512,
                "error_buffer_bits": 0, "periodic_accumulator_bits": 64,
                "muon_scalar_count": 256, "muon_effective_bits_per_value": 2,
                "tensors": []}}
        (run / "summary.json").write_text(json.dumps(summary))
        events = [{"event_type": "train", "completed_updates": u,
            "processed_target_tokens": u*8192, "tokens_this_update": 8192, "lr": .001,
            "train_nll": 6.0-u/10000} for u in range(1, 513)]
        events += [{"event_type": "eval", "completed_updates": u, "nll": 5.5+u/10000}
                   for u in (0, 128, 256, 384, 512)]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in events))
        mechanism_rows = []
        for u in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            interval = optimizer.get("recursive_error_feedback_interval")
            mechanism_rows.append({"update": u, "correction_applied": bool(interval and u % interval == 0),
                "periodic_accumulator_norm_ratio": .1, "periodic_injected_correction_norm_ratio": .05,
                "periodic_accumulator_next_norm_ratio": .03, "error_buffer_norm_ratio": .02,
                "train_nll": 5.0})
        (run / "mechanism/metrics.jsonl").write_text("\n".join(json.dumps(x) for x in mechanism_rows))
        for update in (1, 8, 32, 128, 512):
            torch.manual_seed(500+update)
            gradient = torch.randn(4, 4)
            candidate = gradient*.8 if method != "fp32" else gradient
            if method == "fp32": candidate = gradient
            payload = {"format": "recursive_muon_mechanism_snapshot", "version": 1,
                "metadata": {"update": update, "seed": 1, "schedule_total_updates": 4096,
                    "muon_momentum": mu, "muon_ns_steps": 2,
                    "muon_ns_coefficients": [3.4445, -4.775, 2.0315], "muon_eps": 1e-7},
                "tensors": [{"name": "transformer.h.0.test.weight", "shape": [4,4],
                    "gradient": gradient, "momentum_candidate": candidate}]}
            torch.save(payload, run / "mechanism" / f"update_{update:06d}.pt")
    out = tmp_path / "periodic-report"
    result = analyze(runs, out)
    assert result["protocol_integrity"]["status"] == "compatible"
    assert result["correction_event_count"] == 8 + 6 + 4
    for filename in ("comparison.md", "summary.json", "landmark_metrics.csv",
                     "correction_event_metrics.csv", "storage_breakdown.csv", "provenance.json"):
        assert (out / filename).is_file()
    event_header = (out / "correction_event_metrics.csv").read_text().splitlines()[0]
    assert "local_no_injection_vs_injected_k5_cosine" in event_header


def test_analysis_recovers_omitted_baseline_config_only_on_recipe_fingerprint_match(tmp_path):
    recipe = load_recipe(ROOT / "recipes/mechanism_fp32_muon_4096_s1.json")
    summary = {"recipe_name": recipe["experiment"]["name"],
               "recipe_fingerprint": recipe["fingerprint"]}
    config = _run_config(tmp_path, summary)
    assert config["fingerprint"] == summary["recipe_fingerprint"]
    summary["recipe_fingerprint"] = "wrong-fingerprint"
    with pytest.raises(ValueError, match="fingerprint"):
        _run_config(tmp_path, summary)
