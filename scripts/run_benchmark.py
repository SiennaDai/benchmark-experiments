#!/usr/bin/env python3
"""Run an explicit benchmark suite sequentially, with preflight and resume guards."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_suite import SuiteError, artifact_status, load_suite, now, preflight_suite


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_metadata():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def report(suite, report_dir: Path, run_dirs):
    command = [sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--runs", *map(str, run_dirs), "--vary", *suite["vary"], "--output", str(report_dir)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise SuiteError(f"reporting failed: {result.stderr.strip() or result.stdout.strip()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path, help="directory containing run_id artifacts")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        suite = load_suite(args.suite)
        output_root = args.output_root.expanduser().resolve()
        report_dir = Path(suite["report"]["output"]).expanduser()
        report_dir = (ROOT / report_dir).resolve() if not report_dir.is_absolute() else report_dir.resolve()
        metadata = git_metadata()
        # Report-only intentionally avoids device and data preflight: it only reads
        # immutable run artifacts, but retains the same recipe comparability guard.
        preflight = preflight_suite(suite, root=ROOT, data_root=args.data_root.expanduser().resolve(), include_runtime=not args.report_only)
        resolved = {"suite": {k: v for k, v in suite.items() if k != "_path"}, "suite_path": suite["_path"], "source": metadata,
                    "data_fingerprint": preflight["data_fingerprint"], "recipe_fingerprints": {entry["run_id"]: cfg["fingerprint"] for entry, cfg in zip(suite["runs"], preflight["configs"])}, "resolved_at": now()}
        report_dir.mkdir(parents=True, exist_ok=True)
        write_json(report_dir / "suite_resolved.json", resolved)
        write_json(report_dir / "preflight.json", {"status": "PASS", "checked_at": preflight["checked_at"], "recipe_count": len(suite["runs"]),
                                                      "data_fingerprint": preflight["data_fingerprint"], "runtime_checked": not args.report_only})
        print(f"[suite] {suite['name']}")
        print(f"[check] {len(suite['runs'])}/{len(suite['runs'])} recipes valid")
        print("[check] scientific compatibility PASS")
        if not args.report_only:
            print("[check] data/resource PASS")
        run_dirs = [output_root / entry["run_id"] for entry in suite["runs"]]
        statuses = {}
        if args.preflight_only:
            for entry, directory, cfg in zip(suite["runs"], run_dirs, preflight["configs"]):
                status, reason = artifact_status(directory, cfg, preflight["data_fingerprint"], metadata["commit"])
                statuses[entry["run_id"]] = {"status": status, "reason": reason, "run_dir": str(directory)}
            write_json(report_dir / "run_status.json", {"updated_at": now(), "runs": statuses})
            return 0
        if args.report_only:
            for entry, directory, cfg in zip(suite["runs"], run_dirs, preflight["configs"]):
                status, reason = artifact_status(directory, cfg, None, None)
                statuses[entry["run_id"]] = {"status": status, "reason": reason, "run_dir": str(directory)}
                if status not in {"completed", "paused"}:
                    raise SuiteError(f"report-only requires readable run artifacts: {entry['run_id']}: {status}: {reason or ''}")
            write_json(report_dir / "run_status.json", {"updated_at": now(), "runs": statuses})
            report(suite, report_dir, run_dirs)
            print(f"[report] {report_dir}")
            return 0
        # Every status is established before the first training process starts.
        initial = []
        for entry, directory, cfg in zip(suite["runs"], run_dirs, preflight["configs"]):
            status, reason = artifact_status(directory, cfg, preflight["data_fingerprint"], metadata["commit"])
            initial.append((status, reason))
            if status not in {"new", "completed", "paused"}:
                raise SuiteError(f"cannot run {entry['run_id']}: {status}: {reason or ''}")
        for number, (entry, directory, (status, reason)) in enumerate(zip(suite["runs"], run_dirs, initial), 1):
            print(f"[run {number}/{len(suite['runs'])}] {entry['run_id']}")
            if status == "completed":
                print("[status] completed; skip")
            else:
                command = [sys.executable, str(ROOT / "src" / "main.py"), "--recipe", entry["recipe"], "--run-dir", str(directory),
                           "--data-root", str(args.data_root.expanduser().resolve()), "--to-device", suite["runtime"]["device"]]
                if suite["runtime"]["max_wall_seconds"] is not None:
                    command += ["--max-wall-seconds", str(suite["runtime"]["max_wall_seconds"])]
                if status == "paused":
                    command += ["--resume", str(directory / "checkpoints" / "latest.pt")]
                    print("[status] paused; resume")
                else:
                    print("[status] new; start")
                result = subprocess.run(command)
                if result.returncode not in {0}:
                    raise SuiteError(f"run {entry['run_id']} exited with status {result.returncode}")
            final, final_reason = artifact_status(directory, preflight["configs"][number - 1], preflight["data_fingerprint"], metadata["commit"])
            statuses[entry["run_id"]] = {"status": final, "reason": final_reason, "run_dir": str(directory)}
            write_json(report_dir / "run_status.json", {"updated_at": now(), "runs": statuses})
            if final == "failed":
                raise SuiteError(f"run {entry['run_id']} failed: {final_reason or ''}")
        write_json(report_dir / "run_status.json", {"updated_at": now(), "runs": statuses})
        report(suite, report_dir, run_dirs)
        print(f"[report] {report_dir}")
        return 0
    except SuiteError as exc:
        print(f"[suite error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
