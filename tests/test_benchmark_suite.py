import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from benchmark_suite import SuiteError, artifact_status, load_suite, preflight_suite


ROOT = Path(__file__).resolve().parents[1]


def suite_file(tmp_path, runs, vary=None, wall=1):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"name": "generic_suite", "runs": runs, "vary": vary or ["optimizer.name"],
                                "runtime": {"device": "cpu", "max_wall_seconds": wall}, "report": {"output": str(tmp_path / "report")}}))
    return path


def recipe(tmp_path, cfg, name):
    path = tmp_path / name
    path.write_text(json.dumps(cfg))
    return path


def dry_run(root, recipe_path, data_root, device):
    return {"manifest": str(data_root / "manifest.json"), "fingerprint": "recipe", "resolved_device": device}


def test_valid_suite_and_order(tmp_path, diagnostic_config, monkeypatch):
    a = recipe(tmp_path, diagnostic_config, "a.json")
    bcfg = copy.deepcopy(diagnostic_config); bcfg["optimizer"]["name"] = "reference_adamw"
    b = recipe(tmp_path, bcfg, "b.json")
    suite = load_suite(suite_file(tmp_path, [{"run_id": "z", "recipe": a.name}, {"run_id": "a", "recipe": b.name}]))
    monkeypatch.setattr("benchmark_suite._run_dry_run", dry_run)
    monkeypatch.setattr("data.frozen_tokens.load_manifest", lambda _: {"fingerprint": "data"})
    result = preflight_suite(suite, root=ROOT, data_root=tmp_path)
    assert [x["run_id"] for x in suite["runs"]] == ["z", "a"]
    assert result["data_fingerprint"] == "data"


@pytest.mark.parametrize("value", ["{}", '{"name":"x","runs":[],"vary":[],"runtime":{},"report":{}}'])
def test_malformed_suite_rejected(tmp_path, value):
    p = tmp_path / "bad.json"; p.write_text(value)
    with pytest.raises(SuiteError):
        load_suite(p)


def test_scientific_mismatch_and_identity_handling(tmp_path, diagnostic_config, monkeypatch):
    a = recipe(tmp_path, diagnostic_config, "a.json")
    different = copy.deepcopy(diagnostic_config); different["model"]["vocab_size"] += 1
    # Keep the recipe loader's integer/model invariants valid.
    b = recipe(tmp_path, different, "b.json")
    suite = load_suite(suite_file(tmp_path, [{"run_id": "a", "recipe": a.name}, {"run_id": "b", "recipe": b.name}]))
    monkeypatch.setattr("benchmark_suite._run_dry_run", dry_run)
    monkeypatch.setattr("data.frozen_tokens.load_manifest", lambda _: {"fingerprint": "data"})
    with pytest.raises(SuiteError, match="model.vocab_size"):
        preflight_suite(suite, root=ROOT, data_root=tmp_path)
    different = copy.deepcopy(diagnostic_config); different["experiment"]["name"] = "label_only"
    b = recipe(tmp_path, different, "b.json")
    suite = load_suite(suite_file(tmp_path, [{"run_id": "a", "recipe": a.name}, {"run_id": "b", "recipe": b.name}]))
    assert preflight_suite(suite, root=ROOT, data_root=tmp_path)["differences"] == []


def test_allowed_vary_and_data_mismatch(tmp_path, diagnostic_config, monkeypatch):
    a = recipe(tmp_path, diagnostic_config, "a.json")
    changed = copy.deepcopy(diagnostic_config); changed["optimizer"]["name"] = "reference_adamw"
    b = recipe(tmp_path, changed, "b.json")
    suite = load_suite(suite_file(tmp_path, [{"run_id": "a", "recipe": a.name}, {"run_id": "b", "recipe": b.name}]))
    monkeypatch.setattr("benchmark_suite._run_dry_run", dry_run)
    monkeypatch.setattr("data.frozen_tokens.load_manifest", lambda p: {"fingerprint": "one" if str(p).endswith("a.json.manifest") else "two"})
    # dry-run supplies a common path, so replace it with distinct paths to model mounts.
    monkeypatch.setattr("benchmark_suite._run_dry_run", lambda root, recipe_path, data_root, device: {"manifest": recipe_path + ".manifest"})
    with pytest.raises(SuiteError, match="manifest"):
        preflight_suite(suite, root=ROOT, data_root=tmp_path)


def write_artifact(path, cfg, status="completed", checkpoint=True):
    path.mkdir(parents=True)
    (path / "resolved_config.json").write_text(json.dumps(cfg))
    (path / "summary.json").write_text(json.dumps({"status": status}))
    (path / "data_manifest.json").write_text(json.dumps({"fingerprint": "data"}))
    (path / "source.json").write_text(json.dumps({"upstream_commit": "commit"}))
    if checkpoint:
        (path / "checkpoints").mkdir()
        torch.save({"scientific_fingerprint": cfg["fingerprint"], "data_fingerprint": "data"}, path / "checkpoints/latest.pt")


def test_artifact_states_and_resume_guards(tmp_path, diagnostic_config):
    from config.recipe import load_recipe
    source = recipe(tmp_path, diagnostic_config, "source.json")
    cfg = load_recipe(source)
    run = tmp_path / "r"
    assert artifact_status(run, cfg, "data", "commit")[0] == "new"
    write_artifact(run, cfg, "completed")
    assert artifact_status(run, cfg, "data", "commit")[0] == "completed"
    write_artifact(tmp_path / "paused", cfg, "paused_budget")
    assert artifact_status(tmp_path / "paused", cfg, "data", "commit")[0] == "paused"
    write_artifact(tmp_path / "failed", cfg, "failed")
    assert artifact_status(tmp_path / "failed", cfg, "data", "commit")[0] == "failed"
    write_artifact(tmp_path / "missing", cfg, "paused_budget", checkpoint=False)
    assert artifact_status(tmp_path / "missing", cfg, "data", "commit")[0] == "missing_checkpoint"
    wrong = copy.deepcopy(cfg); wrong["fingerprint"] = "wrong"
    write_artifact(tmp_path / "wrong", wrong, "paused_budget")
    assert artifact_status(tmp_path / "wrong", cfg, "data", "commit")[0] == "fingerprint_mismatch"


def test_runtime_controls_are_not_recipe_fingerprint(tmp_path, diagnostic_config):
    from config.recipe import load_recipe
    path = recipe(tmp_path, diagnostic_config, "r.json")
    assert load_recipe(path)["fingerprint"] == load_recipe(path)["fingerprint"]
    suite = load_suite(suite_file(tmp_path, [{"run_id": "r", "recipe": path.name}], wall=99))
    assert suite["runtime"]["max_wall_seconds"] == 99


def test_replication_metadata_is_optional_and_validated(tmp_path, diagnostic_config):
    a = recipe(tmp_path, diagnostic_config, "a.json")
    path = suite_file(tmp_path, [{"run_id": "a", "recipe": a.name}])
    value = json.loads(path.read_text())
    value["replication"] = {"group_by": ["experiment.seed"], "treatment_field": "optimizer.state_simulation",
                             "control_value": "none", "treatment_values": ["bf16_roundtrip"]}
    path.write_text(json.dumps(value))
    assert load_suite(path)["replication"]["group_by"] == ["experiment.seed"]


def test_report_only_uses_existing_artifacts(tmp_path, diagnostic_config, monkeypatch):
    from config.recipe import load_recipe
    source = recipe(tmp_path, diagnostic_config, "report-recipe.json")
    cfg = load_recipe(source)
    suite_path = suite_file(tmp_path, [{"run_id": "r", "recipe": source.name}])
    write_artifact(tmp_path / "runs" / "r", cfg)
    spec = importlib.util.spec_from_file_location("run_benchmark_for_test", ROOT / "scripts" / "run_benchmark.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "preflight_suite", lambda suite, **_: {"configs": [cfg], "plans": [], "data_fingerprint": None, "differences": [], "checked_at": "now"})
    seen = []
    monkeypatch.setattr(runner, "report", lambda suite, output, runs: seen.extend(runs))
    assert runner.main(["--suite", str(suite_path), "--data-root", str(tmp_path), "--output-root", str(tmp_path / "runs"), "--report-only"]) == 0
    assert seen == [tmp_path / "runs" / "r"]


def test_report_only_can_reuse_external_completed_artifact(tmp_path, diagnostic_config, monkeypatch):
    from config.recipe import load_recipe
    source = recipe(tmp_path, diagnostic_config, "reuse-recipe.json")
    cfg = load_recipe(source)
    suite_path = suite_file(tmp_path, [{"run_id": "r", "recipe": source.name}])
    external = tmp_path / "completed_elsewhere"; write_artifact(external, cfg)
    spec = importlib.util.spec_from_file_location("run_benchmark_reuse_test", ROOT / "scripts" / "run_benchmark.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "preflight_suite", lambda suite, **_: {"configs": [cfg], "plans": [], "data_fingerprint": None, "differences": [], "checked_at": "now"})
    seen = []; monkeypatch.setattr(runner, "report", lambda suite, output, runs: seen.extend(runs))
    assert runner.main(["--suite", str(suite_path), "--data-root", str(tmp_path), "--output-root", str(tmp_path / "new_runs"), "--reuse-run", f"r={external}", "--report-only"]) == 0
    assert seen == [external]
