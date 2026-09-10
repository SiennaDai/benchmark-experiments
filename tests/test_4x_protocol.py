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
