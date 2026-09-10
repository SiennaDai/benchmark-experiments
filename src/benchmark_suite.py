"""Small, explicit orchestration primitives for sequential benchmark suites."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from config.recipe import RecipeError, load_recipe
from reporting import scientific_differences


class SuiteError(ValueError):
    pass


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SuiteError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_keys(value, required, path):
    if not isinstance(value, dict) or set(value) != set(required):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise SuiteError(f"{path} fields must be exactly {sorted(required)}; got {actual}")


def load_suite(path: str | Path) -> dict:
    """Load a non-templated suite and resolve only its file references."""
    path = Path(path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteError(f"invalid suite JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) - {"name", "runs", "vary", "runtime", "report", "replication", "trajectory", "optimizer_lowp"}:
        raise SuiteError("suite contains unknown fields")
    core = {key: item for key, item in value.items() if key not in {"replication", "trajectory", "optimizer_lowp"}}
    _require_keys(core, {"name", "runs", "vary", "runtime", "report"}, "suite")
    if "replication" in value:
        replication = value["replication"]
        _require_keys(replication, {"group_by", "treatment_field", "control_value", "treatment_values"}, "suite.replication")
        if (not isinstance(replication["group_by"], list) or not replication["group_by"] or
                not all(isinstance(x, str) and x for x in replication["group_by"])):
            raise SuiteError("suite.replication.group_by must be a non-empty array of field paths")
        if not isinstance(replication["treatment_field"], str) or not replication["treatment_field"]:
            raise SuiteError("suite.replication.treatment_field must be a field path")
        if not isinstance(replication["treatment_values"], list) or not replication["treatment_values"]:
            raise SuiteError("suite.replication.treatment_values must be a non-empty array")
    if "trajectory" in value:
        trajectory = value["trajectory"]
        _require_keys(trajectory, {"group_by", "treatment_field", "control_value", "treatment_values", "landmark_updates"}, "suite.trajectory")
        if (not isinstance(trajectory["group_by"], list) or not trajectory["group_by"] or
                not all(isinstance(x, str) and x for x in trajectory["group_by"])):
            raise SuiteError("suite.trajectory.group_by must be a non-empty array of field paths")
        if not isinstance(trajectory["treatment_field"], str) or not trajectory["treatment_field"]:
            raise SuiteError("suite.trajectory.treatment_field must be a field path")
        if not isinstance(trajectory["treatment_values"], list) or not trajectory["treatment_values"]:
            raise SuiteError("suite.trajectory.treatment_values must be a non-empty array")
        if (not isinstance(trajectory["landmark_updates"], list) or not trajectory["landmark_updates"] or
                not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in trajectory["landmark_updates"]) or
                trajectory["landmark_updates"] != sorted(set(trajectory["landmark_updates"]))):
            raise SuiteError("suite.trajectory.landmark_updates must be sorted unique positive integers")
    if "optimizer_lowp" in value:
        _require_keys(value["optimizer_lowp"], {"optimizer_field", "state_field", "fp32_value", "lowp_value", "interaction_order"}, "suite.optimizer_lowp")
        if not all(isinstance(value["optimizer_lowp"][k], str) and value["optimizer_lowp"][k] for k in ("optimizer_field", "state_field", "fp32_value", "lowp_value")) or not isinstance(value["optimizer_lowp"]["interaction_order"], list) or len(value["optimizer_lowp"]["interaction_order"]) != 2:
            raise SuiteError("suite.optimizer_lowp is malformed")
    if not isinstance(value["name"], str) or not value["name"]:
        raise SuiteError("suite.name must be a non-empty string")
    if not isinstance(value["runs"], list) or not value["runs"]:
        raise SuiteError("suite.runs must be a non-empty array")
    if not isinstance(value["vary"], list) or not all(isinstance(x, str) and x for x in value["vary"]):
        raise SuiteError("suite.vary must be an array of non-empty field paths")
    if len(set(value["vary"])) != len(value["vary"]):
        raise SuiteError("suite.vary contains duplicate fields")
    _require_keys(value["runtime"], {"device", "max_wall_seconds"}, "suite.runtime")
    if not isinstance(value["runtime"]["device"], str) or not value["runtime"]["device"]:
        raise SuiteError("suite.runtime.device must be a non-empty string")
    if value["runtime"]["max_wall_seconds"] is not None and (not isinstance(value["runtime"]["max_wall_seconds"], (int, float)) or isinstance(value["runtime"]["max_wall_seconds"], bool) or value["runtime"]["max_wall_seconds"] <= 0):
        raise SuiteError("suite.runtime.max_wall_seconds must be a positive number or null")
    _require_keys(value["report"], {"output"}, "suite.report")
    if not isinstance(value["report"]["output"], str) or not value["report"]["output"]:
        raise SuiteError("suite.report.output must be a non-empty string")
    seen = set(); runs = []
    for index, run in enumerate(value["runs"]):
        _require_keys(run, {"run_id", "recipe"}, f"suite.runs[{index}]")
        if not isinstance(run["run_id"], str) or not RUN_ID.fullmatch(run["run_id"]):
            raise SuiteError(f"suite.runs[{index}].run_id must match {RUN_ID.pattern}")
        if run["run_id"] in seen:
            raise SuiteError(f"duplicate run_id: {run['run_id']}")
        if not isinstance(run["recipe"], str) or not run["recipe"]:
            raise SuiteError(f"suite.runs[{index}].recipe must be a non-empty string")
        seen.add(run["run_id"])
        recipe = Path(run["recipe"]).expanduser()
        runs.append({"run_id": run["run_id"], "recipe": str((path.parent / recipe).resolve()) if not recipe.is_absolute() else str(recipe.resolve())})
    return {**value, "runs": runs, "_path": str(path)}


def _run_dry_run(root: Path, recipe: str, data_root: Path, device: str) -> dict:
    command = [sys.executable, str(root / "src" / "main.py"), "--recipe", recipe, "--data-root", str(data_root), "--to-device", device, "--dry-run"]
    result = subprocess.run(command, text=True, capture_output=True)
    try:
        plan = json.loads(result.stdout)
    except json.JSONDecodeError:
        plan = {"stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        raise SuiteError(f"preflight failed for {recipe}: {plan.get('unsupported_reason') or plan.get('capacity_error') or result.stderr.strip() or result.stdout.strip()}")
    return plan


def preflight_suite(suite: dict, *, root: Path, data_root: Path, include_runtime=True) -> dict:
    """Validate every recipe before any training starts; preserve suite order."""
    configs, plans = [], []
    for entry in suite["runs"]:
        try:
            cfg = load_recipe(entry["recipe"])
        except RecipeError as exc:
            raise SuiteError(f"invalid recipe for {entry['run_id']}: {exc}") from exc
        if not isinstance(cfg.get("fingerprint"), str) or not re.fullmatch(r"[0-9a-f]{64}", cfg["fingerprint"]):
            raise SuiteError(f"invalid recipe fingerprint for {entry['run_id']}")
        configs.append(cfg)
        if include_runtime:
            plans.append(_run_dry_run(root, entry["recipe"], data_root, suite["runtime"]["device"]))
    try:
        differences = scientific_differences(configs, [entry["run_id"] for entry in suite["runs"]], suite["vary"])
    except ValueError as exc:
        raise SuiteError(str(exc)) from exc
    if differences:
        fields = ", ".join(sorted({item["field"] for item in differences}))
        raise SuiteError(f"scientific compatibility failed; non-varied fields differ: {fields}")
    # The main dry-run verifies capacity; load manifests separately only to compare
    # their immutable fingerprints, not their mount paths.
    from data.frozen_tokens import load_manifest
    fingerprints = {load_manifest(plan["manifest"])["fingerprint"] for plan in plans} if plans else set()
    if len(fingerprints) > 1:
        raise SuiteError("data manifest contract failed; recipe manifests have different fingerprints")
    return {"configs": configs, "plans": plans, "data_fingerprint": next(iter(fingerprints), None), "differences": differences,
            "checked_at": now()}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def artifact_status(run_dir: Path, cfg: dict, data_fingerprint: str | None, source_commit: str | None) -> tuple[str, str | None]:
    """Return an actionable status without changing an artifact."""
    if not run_dir.exists():
        return "new", None
    summary_path, resolved_path = run_dir / "summary.json", run_dir / "resolved_config.json"
    if not summary_path.exists() or not resolved_path.exists():
        return "running_artifact", "run directory lacks completed lifecycle artifacts; inspect it before resuming"
    try:
        summary, resolved = json.loads(summary_path.read_text()), json.loads(resolved_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid_artifact", str(exc)
    if resolved.get("fingerprint") != cfg["fingerprint"]:
        return "fingerprint_mismatch", "resolved recipe fingerprint differs from suite recipe"
    manifest = json.loads((run_dir / "data_manifest.json").read_text()) if (run_dir / "data_manifest.json").exists() else {}
    if data_fingerprint and manifest.get("fingerprint") != data_fingerprint:
        return "data_mismatch", "artifact data manifest fingerprint differs from suite preflight"
    source = json.loads((run_dir / "source.json").read_text()) if (run_dir / "source.json").exists() else {}
    if source_commit and source.get("upstream_commit") != source_commit:
        return "source_mismatch", "artifact source commit differs from current checkout"
    status = summary.get("status")
    if status == "completed":
        return "completed", None
    if status == "failed":
        return "failed", summary.get("reason") or "training failed"
    if status in {"paused_budget", "interrupted"}:
        checkpoint = run_dir / "checkpoints" / "latest.pt"
        if not checkpoint.exists():
            return "missing_checkpoint", "paused artifact has no checkpoints/latest.pt"
        try:
            checkpoint_value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc:
            return "invalid_checkpoint", str(exc)
        if checkpoint_value.get("scientific_fingerprint") != cfg["fingerprint"]:
            return "fingerprint_mismatch", "checkpoint recipe fingerprint differs from suite recipe"
        if data_fingerprint and checkpoint_value.get("data_fingerprint") != data_fingerprint:
            return "data_mismatch", "checkpoint data fingerprint differs from suite preflight"
        return "paused", None
    return "invalid_artifact", f"unrecognized artifact status: {status!r}"
