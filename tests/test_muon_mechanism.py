import sys
import subprocess
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.recipe import load_recipe  # noqa: E402
from optim.muon_mechanism import MuonMechanismObserver  # noqa: E402
from optim.muon_reference import ReferenceMuon  # noqa: E402
from optim.muon_recursive import RecursiveMuon, StructuralVQCodec  # noqa: E402
from scripts.analyze_recursive_muon_mechanism import validate_pair_metadata  # noqa: E402


def _metadata():
    return {"seed": 1, "data_seed": 1337, "algorithm_seed": 2026,
            "recipe_fingerprint": "x", "data_fingerprint": "y", "schedule_total_updates": 4096,
            "muon_momentum": .95, "muon_ns_steps": 5,
            "muon_ns_coefficients": [3.4445, -4.775, 2.0315], "muon_eps": 1e-7}


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


def test_tiny_paired_analysis_fixture(tmp_path):
    fp = tmp_path / "fp"; vq = tmp_path / "vq"
    (fp / "mechanism").mkdir(parents=True); (vq / "mechanism").mkdir(parents=True)
    meta = _metadata() | {"update": 1, "processed_target_tokens": 8192,
                          "optimizer_name": "reference_muon", "tensor_count": 1}
    vmeta = dict(meta, optimizer_name="recursive_muon")
    g = torch.randn(4, 4); m = torch.randn(4, 4); cand = .95*m + g; persisted = cand + .01
    item_fp = {"name": "transformer.h.0.test.weight", "shape": [4, 4], "gradient": g,
               "momentum_prev_decoded": m, "momentum_prev_candidate": m,
               "momentum_candidate": cand, "momentum_persisted_decoded": cand}
    item_vq = {**item_fp, "gradient": g + .01, "momentum_candidate": cand + .02,
               "momentum_persisted_decoded": persisted}
    for d, payload in ((fp, {"format": "recursive_muon_mechanism_snapshot", "version": 1, "metadata": meta, "tensors": [item_fp]}),
                       (vq, {"format": "recursive_muon_mechanism_snapshot", "version": 1, "metadata": vmeta, "tensors": [item_vq]})):
        torch.save(payload, d / "mechanism/update_000001.pt"); (d / "metrics.jsonl").write_text("")
    out = tmp_path / "report"
    subprocess.run([sys.executable, "scripts/analyze_recursive_muon_mechanism.py", "--fp32-run", str(fp), "--vq-run", str(vq), "--out", str(out)], check=True)
    assert (out / "landmark_metrics.csv").exists()
