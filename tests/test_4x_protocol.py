from pathlib import Path

from config.recipe import load_recipe
from benchmark_suite import load_suite
from reporting import GENERATED_FIELDS, IDENTITY_FIELDS, flatten, scientific_differences


ROOT = Path(__file__).resolve().parents[1]


def test_4x_recipes_are_paired_and_have_exact_horizon():
    control = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_4x_s0.json")
    treatment = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_bf16_state_4x_s0.json")
    assert control["derived"]["total_updates"] == treatment["derived"]["total_updates"] == 4096
    assert control["derived"]["tokens_per_update"] == treatment["derived"]["tokens_per_update"] == 8192
    assert control["data"]["allow_repeated_epochs"] is False
    assert control["data"]["manifest"] == "data/slimpajama_4x/manifest.json"
    assert scientific_differences([control, treatment], ["control", "treatment"], ["optimizer.state_simulation"]) == []
    ignored = IDENTITY_FIELDS | GENERATED_FIELDS
    differing = {key for key in set(flatten(control)) | set(flatten(treatment))
                 if key not in ignored and flatten(control).get(key) != flatten(treatment).get(key)}
    assert differing == {"optimizer.state_simulation"}


def test_4x_suite_declares_only_the_paired_runs_and_landmarks():
    suite = load_suite(ROOT / "benchmarks/mini_fp32_state_precision_4x_s0_v1.json")
    assert [run["run_id"] for run in suite["runs"]] == ["reference_adamw_4x_s0", "reference_adamw_bf16_state_4x_s0"]
    assert suite["vary"] == ["optimizer.state_simulation"]
    assert suite["trajectory"]["landmark_updates"] == [1024, 2048, 3072, 4096]


SNAPSHOT_LANDMARKS = [128, 512, 1024, 2048, 4096]
SNAPSHOT_DATA_FINGERPRINT = "30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18"


def _snapshot_recipe(seed):
    return load_recipe(ROOT / f"recipes/mini_fp32_reference_muon_fidelity_snapshots_4x_s{seed}.json")


def test_muon_fidelity_snapshot_recipes_have_fixed_fp32_protocol():
    s0, s1 = _snapshot_recipe(0), _snapshot_recipe(1)
    for cfg, seed in ((s0, 0), (s1, 1)):
        assert cfg["experiment"]["seed"] == seed
        assert cfg["experiment"]["data_seed"] == 1337
        assert cfg["experiment"]["algorithm_seed"] == 2026
        assert cfg["data"]["manifest"] == "data/slimpajama_4x/manifest.json"
        assert cfg["data"]["allow_repeated_epochs"] is False
        assert cfg["train"]["target_tokens"] == 33554432
        assert cfg["derived"]["tokens_per_update"] == 8192
        assert cfg["derived"]["total_updates"] == 4096
        assert cfg["derived"]["schedule_total_updates"] == 4096
        assert cfg["schedule"]["warmup_updates"] == 20
        assert cfg["precision"]["compute"] == cfg["precision"]["parameter_dtype"] == cfg["precision"]["gradient_dtype"] == "fp32"
        assert cfg["optimizer"]["name"] == "reference_muon"
        assert cfg["optimizer"]["state_simulation"] == "none"
        assert "state_diagnostics" not in cfg["logging"]
        assert cfg["logging"]["muon_update_fidelity"] is False
        assert cfg["logging"]["muon_momentum_snapshot_updates"] == SNAPSHOT_LANDMARKS

    # The immutable dataset identity is a mount-time manifest property.  The
    # recipe audit still requires both trajectories to reference the same
    # frozen manifest; a mounted manifest can additionally be checked against
    # the protocol fingerprint.
    assert s0["data"]["manifest"] == s1["data"]["manifest"]
    manifest = ROOT / s0["data"]["manifest"]
    if manifest.exists():
        from data.frozen_tokens import load_manifest
        assert load_manifest(manifest)["fingerprint"] == SNAPSHOT_DATA_FINGERPRINT


def test_muon_fidelity_snapshot_recipes_differ_from_canonical_only_as_declared():
    baseline = load_recipe(ROOT / "recipes/mini_fp32_reference_muon_4x_s0.json")
    snapshot = _snapshot_recipe(0)
    differing = {key for key in set(flatten(baseline)) | set(flatten(snapshot))
                 if flatten(baseline).get(key) != flatten(snapshot).get(key)}
    assert differing == {
        "experiment.name",
        "logging.muon_update_fidelity",
        "logging.muon_momentum_snapshot_updates",
        "fingerprint",
        "derived.recipe_path",
    }


def test_muon_fidelity_snapshot_trajectories_differ_only_by_seed_and_identity():
    s0, s1 = _snapshot_recipe(0), _snapshot_recipe(1)
    differing = {key for key in set(flatten(s0)) | set(flatten(s1))
                 if flatten(s0).get(key) != flatten(s1).get(key)}
    assert differing == {"experiment.name", "experiment.seed", "fingerprint", "derived.recipe_path"}
    assert scientific_differences([s0, s1], ["s0", "s1"], ["experiment.seed"]) == []


def test_muon_fidelity_snapshot_suite_declares_exact_two_trajectories():
    suite = load_suite(ROOT / "benchmarks/mini_muon_fidelity_snapshots_4x_v1.json")
    assert [run["run_id"] for run in suite["runs"]] == [
        "muon_fidelity_snapshots_4x_s0", "muon_fidelity_snapshots_4x_s1"
    ]
    assert suite["vary"] == ["experiment.seed"]
    assert suite["trajectory"] == {
        "group_by": ["experiment.data_seed", "experiment.algorithm_seed"],
        "treatment_field": "experiment.seed",
        "control_value": 0,
        "treatment_values": [1],
        "landmark_updates": SNAPSHOT_LANDMARKS,
    }
