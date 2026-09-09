"""Validated little-endian token files and deterministic fixed-window sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: str | Path, verify: bool = True) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("dtype") != "uint16" or manifest.get("endianness") != "little":
        raise ValueError("unsupported data manifest")
    for name, split in manifest["splits"].items():
        token_path = (path.parent / split["file"]).resolve()
        if verify and sha256_file(token_path) != split["sha256"]:
            raise ValueError(f"hash mismatch for split {name}: {token_path}")
        if token_path.stat().st_size != split["tokens"] * 2:
            raise ValueError(f"size mismatch for split {name}")
    manifest["_path"] = str(path)
    return manifest


class FrozenWindows:
    def __init__(self, manifest: dict, split: str, sequence_length: int):
        info = manifest["splits"][split]
        self.path = (Path(manifest["_path"]).parent / info["file"]).resolve()
        self.data = np.memmap(self.path, dtype="<u2", mode="r")
        self.sequence_length = sequence_length
        self.num_windows = (len(self.data) - 1) // sequence_length
        self.dropped_tail_tokens = (len(self.data) - 1) % sequence_length

    def window(self, window_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= window_id < self.num_windows:
            raise IndexError(window_id)
        start = window_id * self.sequence_length
        xy = np.asarray(self.data[start:start + self.sequence_length + 1], dtype=np.int64)
        return torch.from_numpy(xy[:-1].copy()), torch.from_numpy(xy[1:].copy())


def validate_data_capacity(manifest: dict, *, sequence_length: int, train_split: str, validation_split: str,
                           eval_target_tokens: int, train_target_tokens: int,
                           micro_batch_size: int, accumulation_steps: int,
                           total_updates: int, allow_repeated_epochs: bool,
                           test_split: str | None = None,
                           test_target_tokens: int | None = None) -> dict:
    """Validate all deterministic token/window capacity constraints before a run."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    def capacity(split: str, requested: int, label: str) -> dict:
        if split not in manifest["splits"]:
            raise ValueError(f"{label} split {split!r} is missing from the manifest")
        available_tokens = manifest["splits"][split]["tokens"]
        available_windows = max(0, (available_tokens - 1) // sequence_length)
        requested_windows = requested // sequence_length
        if requested % sequence_length:
            raise ValueError(f"{label} requested eval tokens must be divisible by sequence_length")
        if requested_windows > available_windows:
            raise ValueError(
                f"{label} capacity insufficient: split={split!r}, available_tokens={available_tokens}, "
                f"sequence_length={sequence_length}, available_windows={available_windows}, "
                f"requested_eval_tokens={requested}, requested_windows={requested_windows}; "
                "expand the data split or lower the recipe eval budget"
            )
        return {"available_tokens": available_tokens, "available_windows": available_windows,
                "requested_tokens": requested, "requested_windows": requested_windows}

    train_info = manifest["splits"].get(train_split)
    if train_info is None:
        raise ValueError(f"train split {train_split!r} is missing from the manifest")
    train_windows = max(0, (train_info["tokens"] - 1) // sequence_length)
    needed_windows = total_updates * micro_batch_size * accumulation_steps
    if not allow_repeated_epochs and needed_windows > train_windows:
        raise ValueError(
            f"training capacity insufficient: split={train_split!r}, available_tokens={train_info['tokens']}, "
            f"sequence_length={sequence_length}, available_windows={train_windows}, "
            f"required_windows={needed_windows}; increase train data or enable data.allow_repeated_epochs"
        )
    result = {"train": {"available_tokens": train_info["tokens"], "available_windows": train_windows,
                         "required_windows": needed_windows},
              "validation": capacity(validation_split, eval_target_tokens, "validation")}
    if test_split is not None and test_target_tokens is not None:
        result["test"] = capacity(test_split, test_target_tokens, "test")
    return result

class DeterministicSampler:
    def __init__(self, num_windows: int, seed: int, allow_repeated_epochs: bool):
        self.num_windows, self.allow_repeated_epochs = num_windows, allow_repeated_epochs
        self.rng = np.random.default_rng(seed)
        self.epoch, self.offset = 0, 0
        self.permutation = self.rng.permutation(num_windows)

    def take(self, count: int) -> list[int]:
        result = []
        while len(result) < count:
            remaining = self.num_windows - self.offset
            n = min(count - len(result), remaining)
            result.extend(int(x) for x in self.permutation[self.offset:self.offset+n])
            self.offset += n
            if self.offset == self.num_windows and len(result) < count:
                if not self.allow_repeated_epochs:
                    raise RuntimeError("training requires more windows than available")
                self.epoch += 1
                self.offset = 0
                self.permutation = self.rng.permutation(self.num_windows)
        return result

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "offset": self.offset, "permutation": self.permutation.tolist(), "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.epoch, self.offset = state["epoch"], state["offset"]
        self.permutation = np.asarray(state["permutation"], dtype=np.int64)
        self.rng.bit_generator.state = state["rng_state"]
