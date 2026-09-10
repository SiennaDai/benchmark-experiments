import json
import subprocess
import sys

import numpy as np
import pytest

from data.frozen_tokens import load_manifest, sha256_file, validate_data_capacity


def make_manifest(tmp_path, values):
    splits = {}
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.bin"
        np.asarray(values, dtype="<u2").tofile(path)
        splits[name] = {"file": path.name, "tokens": len(values), "sha256": sha256_file(path)}
    manifest = {"schema_version": 1, "fingerprint": "x", "dtype": "uint16", "endianness": "little",
                "max_token_id": int(max(values)), "splits": splits}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def args_for(manifest, eval_tokens, *, repeated=True):
    return dict(sequence_length=256, train_split="train", validation_split="validation", eval_target_tokens=eval_tokens,
                train_target_tokens=512, micro_batch_size=1, accumulation_steps=1,
                total_updates=1, allow_repeated_epochs=repeated)


def test_validation_capacity_boundary_and_overflow(tmp_path):
    manifest = load_manifest(make_manifest(tmp_path, range(8193)))
    validate_data_capacity(manifest, **args_for(manifest, 8192))
    with pytest.raises(ValueError, match=r"validation.*available_tokens=8193.*available_windows=32.*requested_windows=33"):
        validate_data_capacity(manifest, **args_for(manifest, 8448))


def test_repeated_epochs_only_affect_training(tmp_path):
    manifest = load_manifest(make_manifest(tmp_path, range(8193)))
    with pytest.raises(ValueError, match="validation.*capacity"):
        validate_data_capacity(manifest, **args_for(manifest, 16384, repeated=True))


def test_4x_training_capacity_boundary_without_repeated_epochs():
    manifest = {"splits": {name: {"tokens": 65537} for name in ("validation", "test")}}
    manifest["splits"]["train"] = {"tokens": 33554433}
    result = validate_data_capacity(manifest, sequence_length=256, train_split="train", validation_split="validation",
                                    eval_target_tokens=65536, train_target_tokens=33554432,
                                    micro_batch_size=4, accumulation_steps=8, total_updates=4096,
                                    allow_repeated_epochs=False, test_split="test", test_target_tokens=65536)
    assert result["train"]["required_windows"] == result["train"]["available_windows"] == 131072


def test_gpu_recipe_dry_run_rejects_small_synthetic_validation(tmp_path):
    output = tmp_path / "data" / "diagnostic"
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--kind", "synthetic", "--output", str(output)], check=True)
    recipe = json.loads(open("recipes/diagnostic_gpu_fp32.json").read())
    recipe["eval"]["max_target_tokens"] = 16384
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    result = subprocess.run([sys.executable, "src/main.py", "--recipe", str(recipe_path), "--data-root", str(tmp_path), "--to-device", "cpu", "--dry-run"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "validation" in result.stdout and "available_windows=32" in result.stdout and "requested_windows=64" in result.stdout
