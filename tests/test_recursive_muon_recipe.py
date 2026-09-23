import json
from pathlib import Path

import torch
import pytest

from config.recipe import load_recipe
from train_platform import learning_rate
from scripts.check_recursive_gate import gate_snapshot
from optim.muon_recursive import StructuralVQCodec, build_recursive_codecs


ROOT = Path(__file__).resolve().parents[1]


def test_recursive_prototype_recipes_are_1024_update_paired_protocols():
    paths = [
        ROOT / "recipes/recursive_muon_fp32_reference_1024_s0.json",
        ROOT / "recipes/recursive_muon_structural_int4_1024_s0.json",
        ROOT / "recipes/recursive_muon_structural_vq_int3_1024_s0.json",
    ]
    configs = [load_recipe(p) for p in paths]
    assert {c["derived"]["tokens_per_update"] for c in configs} == {8192}
    assert {c["derived"]["total_updates"] for c in configs} == {1024}
    assert {c["experiment"]["seed"] for c in configs} == {0}
    assert {c["experiment"]["data_seed"] for c in configs} == {1337}
    assert {c["experiment"]["algorithm_seed"] for c in configs} == {2026}


def test_staged_recipes_keep_the_4096_scheduler_horizon():
    paths = [ROOT / "recipes/recursive_muon_structural_int4_4096_s0.json", ROOT / "recipes/recursive_muon_structural_vq_int3_4096_s0.json"]
    configs = [load_recipe(p) for p in paths]
    assert {c["derived"]["total_updates"] for c in configs} == {4096}
    assert {c["derived"]["schedule_total_updates"] for c in configs} == {4096}
    a = learning_rate(128, 4096, .001, 20, .1, "cosine")
    b = learning_rate(128, 4096, .001, 20, .1, "cosine")
    assert a == b


def test_gate_report_requires_landmark_checkpoint_and_uses_event_values(tmp_path):
    for name, nll in (("fp", 1.0), ("candidate", 1.2)):
        run = tmp_path / name; (run / "checkpoints").mkdir(parents=True)
        (run / "metrics.jsonl").write_text(json.dumps({"event_type": "train", "completed_updates": 128, "train_nll": nll}) + "\n" + json.dumps({"event_type": "eval", "completed_updates": 128, "nll": nll + .1}) + "\n")
        torch.save({}, run / "checkpoints/update_000128.pt")
    report = gate_snapshot(tmp_path / "fp", tmp_path / "candidate", 128)
    assert report["checkpoint_present"] is True
    assert report["validation_nll_delta"] == pytest.approx(.2)


def test_vq_recipe_loads_frozen_opposite_seed_codebook():
    cfg = load_recipe(ROOT / "recipes/recursive_muon_structural_vq_int3_1024_s0.json")
    blob = torch.load(ROOT / cfg["optimizer"]["recursive_codebook_path"], map_location="cpu", weights_only=False)
    cb = blob["codebooks"][cfg["optimizer"]["recursive_codebook_key"]]
    assert tuple(cb.shape) == (64, 2)
    assert torch.any(cb.square().sum(dim=1) == 0)
    assert cfg["optimizer"]["recursive_codebook_key"].startswith("s1_")


def test_recursive_codec_has_no_fp32_momentum_shadow():
    model = torch.nn.Linear(16, 16, bias=False)
    cb = torch.zeros((64, 2)); cb[1:, 0] = torch.linspace(-1, 1, 63)
    with pytest.raises(ValueError, match="no eligible"):
        build_recursive_codecs(model, {"recursive_rank": 8, "recursive_block_size": 2048,
                                        "recursive_structure_mode": "exact_svd_oracle",
                                        "recursive_representation": "vq_int3",
                                        "recursive_codebook_path": "reports/muon_vector_int3_robustness/calibration_codebooks.pt",
                                        "recursive_codebook_key": "s1_k8_w64_t8_v1200"}, ROOT)
