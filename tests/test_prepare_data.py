import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_prepare_data():
    spec = importlib.util.spec_from_file_location("prepare_data_for_test", ROOT / "scripts" / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_hub(monkeypatch, calls):
    class Api:
        def dataset_info(self, dataset, **kwargs):
            calls.append(("dataset_info", dataset, kwargs))
            return types.SimpleNamespace(sha="resolved-revision")

        def list_repo_tree(self, dataset, **kwargs):
            calls.append(("list_repo_tree", dataset, kwargs))
            return [types.SimpleNamespace(path="a.parquet", size=10)]

    hub = types.SimpleNamespace(HfApi=Api, hf_hub_download=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_explicit_source_revision_is_resolved_and_recorded(tmp_path, monkeypatch):
    module, calls = load_prepare_data(), []
    fake_hub(monkeypatch, calls)
    module.prepare_slimpajama(tmp_path, 10, True, {"train": 1, "validation": 1, "test": 1}, "requested-revision")
    plan = json.loads((tmp_path / "source_plan.json").read_text())
    assert calls == [("dataset_info", "DKYoon/SlimPajama-6B", {"revision": "requested-revision"}),
                     ("list_repo_tree", "DKYoon/SlimPajama-6B", {"repo_type": "dataset", "revision": "resolved-revision", "recursive": True, "expand": True})]
    assert plan["revision"] == "resolved-revision"


def test_default_source_revision_behavior_uses_dataset_head(tmp_path, monkeypatch):
    module, calls = load_prepare_data(), []
    fake_hub(monkeypatch, calls)
    module.prepare_slimpajama(tmp_path, 10, True, {"train": 1, "validation": 1, "test": 1})
    assert calls[0] == ("dataset_info", "DKYoon/SlimPajama-6B", {})
