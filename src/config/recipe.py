"""Strict, self-contained experiment recipe loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class RecipeError(ValueError):
    pass


FIELDS = {
    "experiment": {"name", "protocol_id", "seed", "data_seed", "algorithm_seed"},
    "model": {"family", "n_layer", "n_embd", "n_head", "ffn_dim", "multiple_of", "vocab_size", "sequence_length", "dropout", "bias", "tie_embeddings", "init_std", "rmsnorm_eps", "rope_theta"},
    "data": {"manifest", "train_split", "validation_split", "allow_repeated_epochs"},
    "train": {"target_tokens", "micro_batch_size", "accumulation_steps", "grad_clip_norm"},
    "optimizer": {"name", "lr", "betas", "eps", "weight_decay", "fused", "foreach", "state_simulation"},
    "schedule": {"name", "warmup_updates", "final_lr_ratio"},
    "precision": {"compute", "parameter_dtype", "gradient_dtype", "tf32", "attention_backend", "deterministic", "compile"},
    "eval": {"every_updates", "max_target_tokens", "batch_size", "compute", "attention_backend"},
    "logging": {"every_updates", "diagnostics", "wandb"},
    "checkpoint": {"every_updates", "save_initial", "save_final"},
}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RecipeError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _require_type(value: Any, typ: type | tuple[type, ...], path: str) -> None:
    if typ is int and isinstance(value, bool):
        raise RecipeError(f"{path} must be int")
    if not isinstance(value, typ):
        raise RecipeError(f"{path} has wrong type")


def load_recipe(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        cfg = json.loads(path.read_text(), object_pairs_hook=_pairs)
    except json.JSONDecodeError as exc:
        raise RecipeError(f"invalid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise RecipeError("recipe root must be an object")
    if set(cfg) != set(FIELDS):
        raise RecipeError(f"recipe groups mismatch: missing={sorted(set(FIELDS)-set(cfg))}, unknown={sorted(set(cfg)-set(FIELDS))}")
    for group, fields in FIELDS.items():
        if not isinstance(cfg[group], dict):
            raise RecipeError(f"{group} must be an object")
        missing, unknown = fields - set(cfg[group]), set(cfg[group]) - fields
        if missing or unknown:
            raise RecipeError(f"{group} fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")

    m, t, o, s, p, e = (cfg[k] for k in ("model", "train", "optimizer", "schedule", "precision", "eval"))
    for key in ("n_layer", "n_embd", "n_head", "ffn_dim", "multiple_of", "vocab_size", "sequence_length"):
        _require_type(m[key], int, f"model.{key}")
        if m[key] <= 0:
            raise RecipeError(f"model.{key} must be positive")
    for key in ("target_tokens", "micro_batch_size", "accumulation_steps"):
        _require_type(t[key], int, f"train.{key}")
        if t[key] <= 0:
            raise RecipeError(f"train.{key} must be positive")
    if m["family"] != "llama" or m["dropout"] != 0 or m["bias"] is not False or m["tie_embeddings"] is not True:
        raise RecipeError("first version requires llama, dropout=0, bias=false, tie_embeddings=true")
    expected_ffn = m["multiple_of"] * (((8 * m["n_embd"] // 3) + m["multiple_of"] - 1) // m["multiple_of"])
    if m["ffn_dim"] != expected_ffn:
        raise RecipeError(f"model.ffn_dim must equal upstream value {expected_ffn}")
    if m["n_embd"] % m["n_head"] or (m["n_embd"] // m["n_head"]) % 2:
        raise RecipeError("model head_dim must be an even integer")
    tokens_per_update = t["micro_batch_size"] * m["sequence_length"] * t["accumulation_steps"]
    if t["target_tokens"] % tokens_per_update:
        raise RecipeError(f"train.target_tokens must be divisible by {tokens_per_update}")
    if o["name"] not in {"torch_adamw", "reference_adamw", "bnb_adamw32", "bnb_adamw8"}:
        raise RecipeError("unsupported optimizer.name")
    if o["state_simulation"] not in {"none", "bf16_roundtrip"}:
        raise RecipeError("unsupported optimizer.state_simulation")
    if o["state_simulation"] != "none" and o["name"] != "reference_adamw":
        raise RecipeError("state_simulation is only valid for reference_adamw")
    if not isinstance(o["betas"], list) or len(o["betas"]) != 2:
        raise RecipeError("optimizer.betas must be a two-element array")
    if p["compute"] not in {"fp32", "bf16"} or p["parameter_dtype"] != "fp32" or p["gradient_dtype"] != "fp32":
        raise RecipeError("unsupported precision combination")
    if p["attention_backend"] not in {"math", "auto"} or e["attention_backend"] not in {"math", "auto"}:
        raise RecipeError("unsupported attention backend")
    if p["compile"] is not False or p["tf32"] is not False:
        raise RecipeError("compile and tf32 must be false in protocol v1")
    if s["name"] not in {"cosine", "constant"}:
        raise RecipeError("unsupported schedule.name")
    total_updates = t["target_tokens"] // tokens_per_update
    _require_type(s["warmup_updates"], int, "schedule.warmup_updates")
    if not 0 <= s["warmup_updates"] < total_updates:
        raise RecipeError("schedule.warmup_updates must satisfy 0 <= W < total_updates")
    if not 0 <= s["final_lr_ratio"] <= 1:
        raise RecipeError("schedule.final_lr_ratio must be in [0,1]")
    if e["max_target_tokens"] % m["sequence_length"]:
        raise RecipeError("eval.max_target_tokens must be divisible by sequence_length")
    cfg["derived"] = {"tokens_per_update": tokens_per_update, "total_updates": total_updates, "recipe_path": str(path)}
    cfg["fingerprint"] = scientific_fingerprint(cfg)
    return cfg


def scientific_fingerprint(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if k not in {"derived", "fingerprint"}}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump_resolved(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
