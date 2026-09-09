import json
import subprocess
import sys
from pathlib import Path

import pytest
from config.recipe import RecipeError, load_recipe

ROOT = Path(__file__).resolve().parents[1]

def save(tmp_path, cfg):
    p = tmp_path / "r.json"; p.write_text(json.dumps(cfg)); return p

def test_valid_is_deterministic(tmp_path, diagnostic_config):
    a = load_recipe(save(tmp_path, diagnostic_config)); b = load_recipe(tmp_path / "r.json")
    assert a == b and a["derived"]["tokens_per_update"] == 512 and a["derived"]["total_updates"] == 100

@pytest.mark.parametrize("mutation", [lambda c:c["model"].update(extra=1), lambda c:c["train"].update(target_tokens=51199), lambda c:c["model"].update(ffn_dim=256), lambda c:c["model"].update(n_head=3), lambda c:c["precision"].update(compute="fp16")])
def test_bad_recipe_rejected(tmp_path, diagnostic_config, mutation):
    mutation(diagnostic_config)
    with pytest.raises(RecipeError): load_recipe(save(tmp_path, diagnostic_config))

def test_duplicate_key_rejected(tmp_path):
    p=tmp_path/"r.json"; p.write_text('{"x":1,"x":2}')
    with pytest.raises(RecipeError, match="duplicate"): load_recipe(p)


def test_data_root_is_runtime_only(tmp_path, diagnostic_config):
    relative_manifest = Path("mounted") / "manifest.json"
    diagnostic_config["data"]["manifest"] = str(relative_manifest)
    recipe = save(tmp_path, diagnostic_config)
    data_root = tmp_path / "data-root"
    manifest = data_root / relative_manifest
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema_version": 1, "dtype": "uint16", "endianness": "little", "max_token_id": 0,
        "fingerprint": "x", "splits": {name: {"file": "empty.bin", "tokens": 16385, "sha256": ""}
        for name in ("train", "validation", "test")}}))
    (manifest.parent / "empty.bin").write_bytes(b"\0\0" * 16385)
    import hashlib
    digest = hashlib.sha256(b"\0\0" * 16385).hexdigest()
    value = json.loads(manifest.read_text())
    for split in value["splits"].values(): split["sha256"] = digest
    manifest.write_text(json.dumps(value))

    scientific = load_recipe(recipe)["fingerprint"]
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/main.py"), "--recipe", str(recipe),
         "--data-root", str(data_root), "--to-device", "cpu", "--dry-run"],
        check=True, capture_output=True, text=True,
    )
    plan = json.loads(result.stdout)

    assert plan["manifest"] == str(manifest.resolve())
    assert plan["data_root"] == str(data_root.resolve())
    assert plan["fingerprint"] == scientific
