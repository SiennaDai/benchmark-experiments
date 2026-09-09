"""Run artifacts, event logs, environment capture and atomic checkpoints."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def environment_snapshot() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": cuda, "torch_cuda": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
    }


def source_snapshot(root: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.strip()
    return {"remote": git("remote", "get-url", "origin"), "upstream_commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")), "diff": git("diff", "--", "src", "scripts", "recipes", "tests", "docs")}


class EventWriter:
    def __init__(self, path: Path, run_id: str, segment_id: int, start_index: int = 0):
        self.path, self.run_id, self.segment_id, self.index = path, run_id, segment_id, start_index

    def write(self, event_type: str, completed_updates: int, processed_target_tokens: int, **values):
        event = {"schema_version": 1, "event_type": event_type, "run_id": self.run_id, "event_index": self.index, "completed_updates": completed_updates, "processed_target_tokens": processed_target_tokens, "segment_id": self.segment_id, **values}
        self.index += 1
        with self.path.open("a") as f:
            f.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")


def rng_state(cuda_used: bool) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if cuda_used else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
