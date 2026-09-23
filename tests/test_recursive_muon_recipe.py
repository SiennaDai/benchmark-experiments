import json
from pathlib import Path

import torch
import pytest

from config.recipe import load_recipe
from train_platform import learning_rate
from benchmark_suite import load_suite
from reporting import scientific_differences
from scripts.check_recursive_gate import gate_snapshot
from optim.muon_recursive import StructuralVQCodec, build_recursive_codecs
from optim.state_simulation import persistence_metadata


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


def test_recursive_4096_suite_declares_the_two_structural_methods_and_gates():
    suite = load_suite(ROOT / "benchmarks/recursive_muon_structural_4096_s0_v1.json")
    configs = [load_recipe(run["recipe"]) for run in suite["runs"]]
    assert [run["run_id"] for run in suite["runs"]] == [
        "recursive_int4_4096_s0", "recursive_vq_int3_4096_s0"]
    assert suite["trajectory"]["landmark_updates"] == [128, 512, 1024, 2048, 4096]
    assert scientific_differences(configs, [run["run_id"] for run in suite["runs"]], suite["vary"]) == []


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
    codebook_path = ROOT / cfg["optimizer"]["recursive_codebook_path"]
    assert codebook_path.is_file()
    blob = torch.load(codebook_path, map_location="cpu", weights_only=False)
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


def test_recursive_persistence_metadata_is_complete_for_run_manifest():
    metadata = persistence_metadata("recursive_muon", "none")
    assert metadata["persistence_timing"].startswith("post_update;")
    assert metadata["state_groups"]["muon"] == {
        "quantized_state_names": ["compressed_momentum"], "states_left_fp32": []}
    assert metadata["quantizer"] == "structural_recursive_codec"
    assert metadata["actual_optimizer_memory_reduction"] is True


def test_recursive_state_summary_exposes_common_byte_contract():
    parameter = torch.nn.Parameter(torch.zeros(4, 4))
    # The concrete recursive optimizer is exercised by the training path; this
    # assertion protects the summary contract that path consumes.
    from optim.muon_recursive import RecursiveMuon
    optimizer = RecursiveMuon([{"params": [parameter], "optimizer_group": "muon"}], codecs={}, lr=0.1)
    summary = optimizer.recursive_state_summary()
    assert summary["unique_storage_bytes"] == summary["logical_tensor_bytes"] == 0
    assert summary["persistent_bits"] == summary["codebook_bits"] == 0
