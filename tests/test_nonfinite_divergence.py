import json
import math

import pytest
import torch

from config.recipe import load_recipe
from experiment_io import write_json
from train_platform import NonFiniteMetricError, evaluate, require_finite_metric, run


class _Windows:
    sequence_length = 1
    num_windows = 1000

    def window(self, _index):
        return torch.tensor([0]), torch.tensor([0])


class _Model(torch.nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.loss = loss

    def forward(self, _x, _y, get_logits=True):
        # Retain a gradient path for the finite smoke case.
        return {"loss": self.weight * 0 + self.loss}


def _cfg():
    cfg = load_recipe("recipes/diagnostic_cpu.json")
    cfg["optimizer"]["name"] = "reference_adamw"
    cfg["derived"]["total_updates"] = 1
    cfg["derived"]["schedule_total_updates"] = 1
    cfg["train"]["target_tokens"] = 512
    cfg["eval"]["every_updates"] = 1
    cfg["checkpoint"]["every_updates"] = 1
    return cfg


def _patch_small_run(monkeypatch, loss, eval_values):
    import train_platform

    monkeypatch.setattr(train_platform, "load_manifest", lambda _path: {"fingerprint": "data", "max_token_id": 0})
    monkeypatch.setattr(train_platform, "FrozenWindows", lambda *_args: _Windows())
    monkeypatch.setattr(train_platform, "validate_data_capacity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_platform, "build_model", lambda _cfg, _device: _Model(loss))
    values = iter(eval_values)

    def fake_evaluate(*_args, **_kwargs):
        value = next(values)
        return {"evaluated_tokens": 1, "nll": require_finite_metric("validation_nll", value),
                "ppl": 1.0, "elapsed_seconds": 0.01}

    monkeypatch.setattr(train_platform, "evaluate", fake_evaluate)


@pytest.mark.parametrize(("value", "kind"), [(math.inf, "+inf"), (math.nan, "nan")])
def test_nonfinite_train_nll_becomes_structured_terminal_artifact(tmp_path, monkeypatch, value, kind):
    _patch_small_run(monkeypatch, value, [1.0])
    summary = run(_cfg(), tmp_path / kind, to_device="cpu")
    assert summary["status"] == "diverged_nonfinite"
    assert summary["completed_updates"] == 0
    divergence = summary["divergence"]
    assert divergence["update"] == 1
    assert divergence["processed_target_tokens"] == 2
    assert divergence["offending_field"] == "train_nll"
    assert divergence["nonfinite_value"] == kind
    assert divergence["diagnostic_checkpoint_saved"] is False
    assert not (tmp_path / kind / "checkpoints" / "latest.pt").exists()
    events = [json.loads(line) for line in (tmp_path / kind / "metrics.jsonl").read_text().splitlines()]
    assert events[-2]["event_type"] == "divergence"
    assert events[-1]["event_type"] == "lifecycle"
    assert events[-1]["phase"] == "diverged_nonfinite"


def test_nonfinite_validation_nll_becomes_structured_terminal_artifact(tmp_path, monkeypatch):
    _patch_small_run(monkeypatch, 1.0, [1.0, math.inf])
    summary = run(_cfg(), tmp_path / "validation", to_device="cpu")
    assert summary["status"] == "diverged_nonfinite"
    assert summary["completed_updates"] == 1
    assert summary["divergence"]["update"] == 1
    assert summary["divergence"]["offending_field"] == "validation_nll"
    assert summary["divergence"]["nonfinite_value"] == "+inf"
    assert summary["divergence"]["last_finite_train_metric"] == {"update": 1, "train_nll": 1.0}
    # The scheduled checkpoint is not written after the non-finite evaluation.
    assert not (tmp_path / "validation" / "checkpoints" / "latest.pt").exists()


def test_finite_run_still_completes_and_writes_strict_json(tmp_path, monkeypatch):
    _patch_small_run(monkeypatch, 1.0, [1.0, 1.0])
    summary = run(_cfg(), tmp_path / "finite", to_device="cpu")
    assert summary["status"] == "completed"
    assert summary["divergence"] is None
    assert (tmp_path / "finite" / "checkpoints" / "latest.pt").exists()
    assert json.loads((tmp_path / "finite" / "summary.json").read_text())["status"] == "completed"


def test_evaluate_detects_nonfinite_validation_loss_before_event():
    with pytest.raises(NonFiniteMetricError, match="validation_nll: -inf"):
        evaluate(_Model(float("-inf")), _Windows(), 1, 1, {}, torch.device("cpu"))


def test_strict_json_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError, match="Out of range float values"):
        write_json(tmp_path / "strict.json", {"metric": math.inf})
