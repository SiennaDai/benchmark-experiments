import sys
import subprocess
import json
import csv
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.recipe import load_recipe  # noqa: E402
from optim.muon_mechanism import MuonMechanismObserver  # noqa: E402
from optim.muon_reference import ReferenceMuon  # noqa: E402
from optim.muon_recursive import RecursiveMuon, StructuralVQCodec  # noqa: E402
from scripts.analyze_recursive_muon_mechanism import (  # noqa: E402
    _direction_from_previous, _k5, _pooled_pairs, validate_pair_metadata,
)


def _metadata():
    return {"seed": 1, "data_seed": 1337, "algorithm_seed": 2026,
            "recipe_fingerprint": "x", "data_fingerprint": "y", "schedule_total_updates": 4096,
            "protocol_id": "test-protocol", "tokens_per_update": 8192, "sequence_length": 256,
            "muon_momentum": .95, "muon_ns_steps": 5,
            "muon_nesterov": True, "muon_ns_coefficients": [3.4445, -4.775, 2.0315], "muon_eps": 1e-7}


def test_mechanism_observer_writes_exact_fp32_raw_fixture(tmp_path):
    observer = MuonMechanismObserver(run_dir=tmp_path, optimizer_name="recursive_muon",
                                     parameter_names={1: "transformer.h.0.test.weight"},
                                     raw_updates=[1], scalar_updates=[1], metadata=_metadata())
    observer.begin_update(1)
    g = torch.randn(4, 4); prev = torch.randn(4, 4); candidate = prev * .95 + g; persisted = candidate.round()
    observer.observe(parameter=torch.nn.Parameter(torch.zeros(4, 4)), gradient=g,
                     momentum_prev=prev, momentum_candidate=candidate,
                     momentum_persisted_decoded=persisted, direction=g + .95 * candidate,
                     updated=torch.zeros(4, 4), group={}, update=1)
    observer.finish_update(update=1, processed_target_tokens=8192, train_nll=1.0, learning_rate=.001)
    blob = torch.load(tmp_path / "mechanism/update_000001.pt", map_location="cpu", weights_only=False)
    item = blob["tensors"][0]
    assert item["gradient"].dtype == torch.float32
    assert torch.equal(item["momentum_candidate"], candidate)
    assert (tmp_path / "mechanism/metrics.jsonl").exists()


def test_mechanism_observer_is_not_optimizer_state_and_does_not_change_step():
    torch.manual_seed(4); grad = torch.randn(4, 4)
    p0 = torch.nn.Parameter(torch.randn(4, 4)); p1 = torch.nn.Parameter(p0.detach().clone())
    o0 = ReferenceMuon([{"params": [p0], "optimizer_group": "muon", "weight_decay": 0.0}], lr=.001)
    o1 = ReferenceMuon([{"params": [p1], "optimizer_group": "muon", "weight_decay": 0.0}], lr=.001)
    seen = []
    o1.set_mechanism_observer(lambda **kwargs: seen.append(kwargs))
    p0.grad = grad.clone(); p1.grad = grad.clone(); o0.step(); o1.step()
    assert torch.equal(p0, p1)
    assert "_mechanism_observer" not in o1.state_dict()
    assert "mechanism" not in o1.state[p1]
    assert len(seen) == 1


def test_recursive_mechanism_callback_observes_persisted_decode_without_state_entry():
    torch.manual_seed(7)
    cb = torch.randn(64, 2) * .2; cb[0] = 0
    p = torch.nn.Parameter(torch.randn(16, 16))
    codec = StructuralVQCodec(cb, rank=8, block_size=2048)
    opt = RecursiveMuon([{"params": [p], "optimizer_group": "muon", "weight_decay": 0.0}], {id(p): codec}, lr=.001)
    seen = []
    opt.set_mechanism_observer(lambda **kwargs: seen.append(kwargs)); opt.set_mechanism_update(1)
    p.grad = torch.randn_like(p); opt.step()
    assert len(seen) == 1
    assert "compressed_momentum" in opt.state[p] and "momentum" not in opt.state[p]
    assert "_mechanism_observer" not in opt.state_dict()
    assert torch.equal(seen[0]["momentum_candidate"], seen[0]["momentum_persisted_decoded"]) is False


def test_mechanism_recipes_keep_formal_horizon():
    for name in ("mechanism_fp32_muon_4096_s1.json", "mechanism_recursive_vq_int3_4096_s1.json"):
        cfg = load_recipe(Path("recipes") / name)
        assert cfg["derived"]["total_updates"] == 4096
        assert cfg["derived"]["schedule_total_updates"] == 4096
        assert cfg["logging"]["muon_mechanism_snapshot_updates"] == [1, 8, 32, 128, 512]


def test_pair_metadata_validation_detects_mismatch():
    a = _metadata(); b = dict(a, recipe_fingerprint="different"); b["data_seed"] = 9
    assert validate_pair_metadata(a, b)
    assert not validate_pair_metadata(a, dict(a))


def test_nesterov_direction_uses_previous_state_not_current_candidate():
    mu = .95
    previous = torch.tensor([[2.0, -1.0]])
    gradient = torch.tensor([[.5, 3.0]])
    candidate = mu * previous + gradient
    exact = _direction_from_previous(previous, gradient, mu)
    expected = gradient + mu * candidate
    assert torch.allclose(exact, expected, atol=1e-6, rtol=0)
    # Passing the already gradient-containing candidate as if it were M_(t-1)
    # would add the current gradient a second time and must be observably wrong.
    wrong = _direction_from_previous(candidate, gradient, mu)
    assert not torch.allclose(wrong, expected, atol=1e-4, rtol=0)


def test_counterfactual_metrics_pool_by_global_norms_not_mean_cosine():
    # Tensor 1 is tiny and orthogonal; tensor 2 dominates the global metric.
    tiny = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    large = (torch.tensor([100.0, 0.0]), torch.tensor([100.0, .1]))
    pooled = _pooled_pairs([tiny, large])
    concatenated = _pooled_pairs([(torch.cat((tiny[0], large[0])),
                                   torch.cat((tiny[1], large[1])))])
    arithmetic_cosine = sum(float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                            for a, b in (tiny, large)) / 2
    assert pooled["cosine"] == pytest.approx(concatenated["cosine"], abs=1e-9)
    assert pooled["relative_l2"] == pytest.approx(concatenated["relative_l2"], abs=1e-9)
    assert arithmetic_cosine < .51


def test_corrected_paired_analysis_rebuilds_all_directions_from_previous_states(tmp_path):
    fp = tmp_path / "fp"; vq = tmp_path / "vq"
    (fp / "mechanism").mkdir(parents=True); (vq / "mechanism").mkdir(parents=True)
    mu = .95
    snapshots = {}
    for run, optimizer_name, is_vq in ((fp, "reference_muon", False), (vq, "recursive_muon", True)):
        for update in (1, 8, 32, 128, 512):
            meta = _metadata() | {"update": update, "processed_target_tokens": update * 8192,
                                  "optimizer_name": optimizer_name, "tensor_count": 1}
            if update == 1:
                prev = torch.zeros(2, 2); prev_candidate = torch.zeros(2, 2)
            else:
                prev = torch.full((2, 2), .1 if not is_vq else .14)
                prev_candidate = prev + (.01 if is_vq else 0.0)
            grad = torch.tensor([[.2, .1], [-.3, .4]]) + (torch.tensor([[.05, -.25], [.1, -.05]]) if is_vq else 0)
            candidate = mu * prev + grad
            persisted = candidate + (torch.tensor([[.005, -.003], [.002, .004]]) if is_vq else 0)
            item = {"name": "transformer.h.0.test.weight", "shape": [2, 2],
                    "gradient": grad, "momentum_prev_decoded": prev,
                    "momentum_prev_candidate": prev_candidate, "momentum_candidate": candidate,
                    "momentum_persisted_decoded": persisted}
            snapshots[(update, is_vq)] = item
            torch.save({"format": "recursive_muon_mechanism_snapshot", "version": 1,
                        "metadata": meta, "tensors": [item]},
                       run / "mechanism" / f"update_{update:06d}.pt")
        summary = {"seed": 1, "data_seed": 1337, "algorithm_seed": 2026,
                   "protocol_id": "test-protocol", "data_fingerprint": "y",
                       "schedule_total_updates": 4096, "target_tokens": 4096 * 8192,
                       "total_updates": 4096, "compute_precision": "fp32",
                       "sequence_length": 256, "completed_updates": 512,
                       "optimizer_name": optimizer_name, "status": "paused_staged", "git_commit": "test"}
        (run / "summary.json").write_text(json.dumps(summary))
        events = [{"event_type": "train", "completed_updates": i,
                   "processed_target_tokens": i * 8192, "tokens_this_update": 8192,
                   "lr": .001} for i in range(1, 513)]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in events))
    out = tmp_path / "corrected_report"
    subprocess.run([sys.executable, "scripts/analyze_recursive_muon_mechanism.py",
                    "--fp32-run", str(fp), "--vq-run", str(vq), "--out", str(out)], check=True)
    with (out / "landmark_metrics.csv").open() as f:
        pooled_by_update = {int(x["update"]): x for x in csv.DictReader(f)}
    pooled = pooled_by_update[8]
    item_fp, item_vq = snapshots[(8, False)], snapshots[(8, True)]
    g_ref, g_vq = item_fp["gradient"], item_vq["gradient"]
    prev_ref, prev_vq = item_fp["momentum_prev_decoded"], item_vq["momentum_prev_decoded"]
    prev_unquant_vq = item_vq["momentum_prev_candidate"]
    d_ref = _direction_from_previous(prev_ref, g_ref, mu)
    d_vq = _direction_from_previous(prev_vq, g_vq, mu)
    d_state = _direction_from_previous(prev_vq, g_ref, mu)
    d_gradient = _direction_from_previous(prev_ref, g_vq, mu)
    d_local = _direction_from_previous(prev_unquant_vq, g_vq, mu)
    assert torch.allclose(d_local - d_vq, mu**2 * (prev_unquant_vq - prev_vq), atol=1e-7, rtol=0)
    map_meta = _metadata()
    o_ref = _k5(d_ref, map_meta); o_vq = _k5(d_vq, map_meta)
    o_state = _k5(d_state, map_meta); o_gradient = _k5(d_gradient, map_meta); o_local = _k5(d_local, map_meta)
    assert float(pooled["nesterov_direction_cosine"]) == pytest.approx(
        _pooled_pairs([(d_ref, d_vq)])["cosine"], abs=1e-7)
    assert float(pooled["k5_update_cosine"]) == pytest.approx(
        _pooled_pairs([(o_ref, o_vq)])["cosine"], abs=1e-7)
    assert float(pooled["state_only_k5_cosine"]) == pytest.approx(
        _pooled_pairs([(o_ref, o_state)])["cosine"], abs=1e-7)
    assert float(pooled["gradient_only_k5_cosine"]) == pytest.approx(
        _pooled_pairs([(o_ref, o_gradient)])["cosine"], abs=1e-7)
    assert float(pooled["local_k5_cosine"]) == pytest.approx(
        _pooled_pairs([(o_vq, o_local)])["cosine"], abs=1e-7)
    assert "global tensor-norm pooling" in (out / "summary.json").read_text()
    assert (out / "comparison.md").exists()
    update1 = pooled_by_update[1]
    assert float(update1["local_k5_cosine"]) == pytest.approx(1.0, abs=1e-7)
    assert float(update1["local_k5_relative_l2"]) == pytest.approx(0.0, abs=1e-7)
