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
