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

# These fields are deliberately opt-in so existing strict recipes retain their
# exact serialized scientific configuration and therefore their fingerprints.
OPTIONAL_FIELDS = {"schedule": {"total_updates"}, "optimizer": {"muon_momentum", "muon_nesterov", "muon_ns_steps", "muon_ns_coefficients", "muon_eps", "state_quantization_granularity", "state_quantization_block_size", "recursive_rank", "recursive_block_size", "recursive_factor_dtype", "recursive_structure_mode", "recursive_representation", "recursive_codebook_path", "recursive_codebook_key"}, "logging": {"state_diagnostics", "muon_update_fidelity", "muon_momentum_snapshot_updates"}}


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
        optional = OPTIONAL_FIELDS.get(group, set())
        missing, unknown = fields - set(cfg[group]), set(cfg[group]) - fields - optional
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
    if o["name"] not in {"torch_adamw", "reference_adamw", "reference_muon", "recursive_muon", "bnb_adamw32", "bnb_adamw8"}:
        raise RecipeError("unsupported optimizer.name")
    simulations = {"none", "bf16_roundtrip", "int8_linear_first_moment", "int8_linear_second_moment", "int8_linear_all_moments", "int8_linear_momentum", "int4_linear_momentum", "int8_dynamic_all_moments", "int8_dynamic_second_moment", "int4_dynamic_all_moments", "int4_dynamic_momentum"}
    if o["state_simulation"] not in simulations:
        raise RecipeError("unsupported optimizer.state_simulation")
    if o["state_simulation"] != "none" and o["name"] not in {"reference_adamw", "reference_muon"}:
        raise RecipeError("state_simulation is only valid for reference_adamw or reference_muon")
    if o["name"] == "reference_adamw" and o["state_simulation"] in {"int8_linear_momentum", "int4_linear_momentum", "int4_dynamic_momentum"}:
        raise RecipeError("momentum simulations are only valid for reference_muon")
    if o["name"] == "reference_muon" and o["state_simulation"] in {"int8_linear_first_moment", "int8_linear_second_moment", "int8_linear_all_moments", "int8_dynamic_all_moments", "int8_dynamic_second_moment", "int4_dynamic_all_moments"}:
        raise RecipeError("AdamW INT8 state simulations are only valid for reference_adamw")
    granularity = o.get("state_quantization_granularity", "per_state_tensor")
    if granularity not in {"per_state_tensor", "blockwise"}:
        raise RecipeError("optimizer.state_quantization_granularity is unsupported")
    block_size = o.get("state_quantization_block_size", 2048)
    _require_type(block_size, int, "optimizer.state_quantization_block_size")
    if block_size <= 0:
        raise RecipeError("optimizer.state_quantization_block_size must be positive")
    if granularity == "blockwise" and not (o["state_simulation"].startswith(("int8_linear_", "int4_linear_")) or o["state_simulation"] in {"int8_dynamic_all_moments", "int8_dynamic_second_moment", "int4_dynamic_all_moments", "int4_dynamic_momentum"}):
        raise RecipeError("blockwise state quantization requires an integer state simulation")
    if o["state_simulation"] in {"int8_dynamic_all_moments", "int8_dynamic_second_moment", "int4_dynamic_all_moments", "int4_dynamic_momentum"} and granularity != "blockwise":
        raise RecipeError("dynamic state simulation requires blockwise granularity")
    if not isinstance(o["betas"], list) or len(o["betas"]) != 2:
        raise RecipeError("optimizer.betas must be a two-element array")
    if o["name"] in {"reference_muon", "recursive_muon"}:
        for key, default, typ in (("muon_momentum", .95, (int, float)), ("muon_nesterov", True, bool), ("muon_ns_steps", 5, int), ("muon_eps", 1e-7, (int, float))):
            value = o.get(key, default)
            _require_type(value, typ, f"optimizer.{key}")
        if not 0 <= o.get("muon_momentum", .95) < 1 or o.get("muon_ns_steps", 5) <= 0 or o.get("muon_eps", 1e-7) <= 0:
            raise RecipeError("invalid reference_muon hyperparameter")
        coefficients = o.get("muon_ns_coefficients", [3.4445, -4.7750, 2.0315])
        if not isinstance(coefficients, list) or len(coefficients) != 3 or not all(isinstance(x, (int, float)) for x in coefficients):
            raise RecipeError("optimizer.muon_ns_coefficients must be a three-element numeric array")
    if o["name"] == "recursive_muon":
        if o.get("state_simulation", "none") != "none":
            raise RecipeError("recursive_muon owns its compressed state; state_simulation must be none")
        if o.get("recursive_rank", 8) <= 0:
            raise RecipeError("optimizer.recursive_rank must be positive")
        if o.get("recursive_block_size", 2048) <= 0:
            raise RecipeError("optimizer.recursive_block_size must be positive")
        if o.get("recursive_factor_dtype", "bf16") != "bf16":
            raise RecipeError("recursive prototype currently requires BF16 structural factors")
        if o.get("recursive_structure_mode", "exact_svd_oracle") not in {"exact_svd_oracle"}:
            raise RecipeError("unsupported recursive_structure_mode")
        if o.get("recursive_representation", "vq_int3") not in {"vq_int3", "int4"}:
            raise RecipeError("unsupported recursive_representation")
        if o.get("recursive_representation", "vq_int3") == "vq_int3" and (not isinstance(o.get("recursive_codebook_path"), str) or not isinstance(o.get("recursive_codebook_key"), str)):
            raise RecipeError("recursive VQ requires recursive_codebook_path and recursive_codebook_key")
    if p["compute"] not in {"fp32", "bf16"} or p["parameter_dtype"] != "fp32" or p["gradient_dtype"] != "fp32":
        raise RecipeError("unsupported precision combination")
    if p["attention_backend"] not in {"math", "auto"} or e["attention_backend"] not in {"math", "auto"}:
        raise RecipeError("unsupported attention backend")
    if p["compile"] is not False or p["tf32"] is not False:
        raise RecipeError("compile and tf32 must be false in protocol v1")
    if s["name"] not in {"cosine", "constant"}:
        raise RecipeError("unsupported schedule.name")
    total_updates = t["target_tokens"] // tokens_per_update
    schedule_total_updates = s.get("total_updates", total_updates)
    _require_type(schedule_total_updates, int, "schedule.total_updates")
    if schedule_total_updates <= 0:
        raise RecipeError("schedule.total_updates must be positive")
    _require_type(s["warmup_updates"], int, "schedule.warmup_updates")
    if not 0 <= s["warmup_updates"] < schedule_total_updates:
        raise RecipeError("schedule.warmup_updates must satisfy 0 <= W < schedule.total_updates")
    if not 0 <= s["final_lr_ratio"] <= 1:
        raise RecipeError("schedule.final_lr_ratio must be in [0,1]")
    if "state_diagnostics" in cfg["logging"] and not isinstance(cfg["logging"]["state_diagnostics"], bool):
        raise RecipeError("logging.state_diagnostics must be bool")
    if "muon_update_fidelity" in cfg["logging"] and not isinstance(cfg["logging"]["muon_update_fidelity"], bool):
        raise RecipeError("logging.muon_update_fidelity must be bool")
    snapshots = cfg["logging"].get("muon_momentum_snapshot_updates", [])
    if not isinstance(snapshots, list) or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in snapshots):
        raise RecipeError("logging.muon_momentum_snapshot_updates must be a list of positive update integers")
    if len(set(snapshots)) != len(snapshots):
        raise RecipeError("logging.muon_momentum_snapshot_updates must not contain duplicates")
    if e["max_target_tokens"] % m["sequence_length"]:
        raise RecipeError("eval.max_target_tokens must be divisible by sequence_length")
    cfg["derived"] = {"tokens_per_update": tokens_per_update, "total_updates": total_updates,
                      "schedule_total_updates": schedule_total_updates, "recipe_path": str(path)}
    cfg["fingerprint"] = scientific_fingerprint(cfg)
    return cfg


def scientific_fingerprint(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if k not in {"derived", "fingerprint"}}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump_resolved(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
