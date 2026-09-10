import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark_suite import load_suite
from config.recipe import load_recipe
from reporting import GENERATED_FIELDS, IDENTITY_FIELDS, flatten, scientific_differences
from train_platform import learning_rate


ROOT = Path(__file__).resolve().parents[1]


def test_schedule_total_updates_is_optional_and_defaults_to_training_horizon():
    existing = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_s0.json")
    assert existing["derived"]["total_updates"] == 1024
    assert existing["derived"]["schedule_total_updates"] == 1024
    assert "total_updates" not in existing["schedule"]


def test_schedule_controlled_recipes_are_paired_and_stop_at_target_horizon():
    control = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_schedule4096_train1024_s0.json")
    treatment = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_bf16_state_schedule4096_train1024_s0.json")
    assert control["derived"]["total_updates"] == treatment["derived"]["total_updates"] == 1024
    assert control["derived"]["schedule_total_updates"] == treatment["derived"]["schedule_total_updates"] == 4096
    assert control["derived"]["tokens_per_update"] == 8192
    assert control["data"]["allow_repeated_epochs"] is False
    assert control["data"]["manifest"] == "data/slimpajama_4x/manifest.json"
    assert scientific_differences([control, treatment], ["control", "treatment"], ["optimizer.state_simulation"]) == []
    ignored = IDENTITY_FIELDS | GENERATED_FIELDS
    assert {key for key in set(flatten(control)) | set(flatten(treatment)) if key not in ignored and flatten(control).get(key) != flatten(treatment).get(key)} == {"optimizer.state_simulation"}


def test_schedule_horizon_drives_lr_at_update_1024():
    old = learning_rate(1024, 1024, 0.001, 20, 0.1, "cosine")
    controlled = learning_rate(1024, 4096, 0.001, 20, 0.1, "cosine")
    assert old == pytest.approx(0.0001)
    assert controlled == pytest.approx(0.0008718554895857875)
    assert controlled > old


def test_schedule_controlled_suite_declares_only_the_pair():
    suite = load_suite(ROOT / "benchmarks/mini_fp32_state_precision_schedule4096_train1024_s0_v1.json")
    assert [run["run_id"] for run in suite["runs"]] == ["reference_adamw_schedule4096_train1024_s0", "reference_adamw_bf16_state_schedule4096_train1024_s0"]
    assert suite["vary"] == ["optimizer.state_simulation"]
    assert suite["runtime"]["device"] == "cuda:0"


def test_dry_run_reports_training_termination_separately_from_schedule_horizon():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/main.py"), "--recipe",
         str(ROOT / "recipes/mini_fp32_reference_adamw_schedule4096_train1024_s0.json"),
         "--data-root", str(ROOT), "--to-device", "cpu", "--dry-run"],
        check=True, capture_output=True, text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["total_updates"] == 1024
    assert plan["schedule_total_updates"] == 4096
    assert plan["tokens_per_update"] == 8192
