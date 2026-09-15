"""Deterministic numerical simulations for persisted optimizer state.

These helpers deliberately return FP32 tensors.  They simulate the numerical
effect of writing a state tensor at lower precision after an optimizer update;
they do not provide compressed optimizer storage.
"""

from __future__ import annotations

import torch


INT8_LINEAR_SIMULATIONS = frozenset({
    "int8_linear_first_moment",
    "int8_linear_second_moment",
    "int8_linear_all_moments",
    "int8_linear_momentum",
})
STATE_SIMULATIONS = frozenset({"none", "bf16_roundtrip", *INT8_LINEAR_SIMULATIONS})


def int8_linear_roundtrip(state: torch.Tensor) -> torch.Tensor:
    """Return the deterministic signed max-abs INT8 linear simulation of state.

    The scale is chosen independently for every input tensor as
    ``max(abs(state)) / 127``.  A zero tensor has no representable scale, so it
    remains zero.  FP32 quantization/dequantization arithmetic and output make
    this a persistence-numerics simulation rather than a memory implementation.
    """
    value = state.float()
    max_abs = value.abs().max()
    if max_abs.item() == 0:
        return value
    scale = max_abs / 127
    quantized = torch.round(value / scale).clamp(-127, 127)
    return (quantized * scale).float()


def persist_state(state: torch.Tensor, simulation: str, state_name: str) -> torch.Tensor:
    """Apply a post-update persistence simulation to one selected state tensor."""
    if simulation == "none":
        return state
    if simulation == "bf16_roundtrip":
        return state.to(torch.bfloat16).float()
    selected = {
        "int8_linear_first_moment": {"exp_avg"},
        "int8_linear_second_moment": {"exp_avg_sq"},
        "int8_linear_all_moments": {"exp_avg", "exp_avg_sq"},
        "int8_linear_momentum": {"muon_momentum"},
    }
    if simulation in selected:
        return int8_linear_roundtrip(state) if state_name in selected[simulation] else state
    raise ValueError(f"unsupported state simulation: {simulation}")


def persistence_metadata(optimizer_name: str, simulation: str) -> dict:
    """An auditable description of exactly which persisted states are simulated."""
    if optimizer_name == "reference_adamw":
        all_states = ["exp_avg", "exp_avg_sq"]
        selected = {
            "none": [], "bf16_roundtrip": all_states,
            "int8_linear_first_moment": ["exp_avg"],
            "int8_linear_second_moment": ["exp_avg_sq"],
            "int8_linear_all_moments": all_states,
        }.get(simulation)
        if selected is None:
            raise ValueError(f"simulation {simulation} is invalid for {optimizer_name}")
        untouched = [state for state in all_states if state not in selected]
        groups = {"reference_adamw": {"quantized_state_names": selected, "states_left_fp32": untouched}}
    elif optimizer_name == "reference_muon":
        if simulation not in {"none", "bf16_roundtrip", "int8_linear_momentum"}:
            raise ValueError(f"simulation {simulation} is invalid for {optimizer_name}")
        selected = ["muon_momentum"] if simulation != "none" else []
        groups = {
            "muon": {"quantized_state_names": selected, "states_left_fp32": [] if selected else ["muon_momentum"]},
            "auxiliary_adamw": {"quantized_state_names": [], "states_left_fp32": ["exp_avg", "exp_avg_sq"]},
        }
    else:
        return {"simulation": simulation, "state_groups": {"optimizer": {"quantized_state_names": [], "states_left_fp32": []}}}
    metadata = {"simulation": simulation, "state_groups": groups,
                "persistence_timing": "post_update; current FP32 state drives current update; persisted state affects next step"}
    if simulation in INT8_LINEAR_SIMULATIONS:
        metadata.update({"quantizer": "int8_linear_roundtrip", "bits": 8, "signed_range": [-127, 127],
                         "scale": "max_abs / 127", "rounding": "nearest", "granularity": "per_state_tensor",
                         "persistent_storage_in_platform": "fp32_dequantized_simulation",
                         "actual_optimizer_memory_reduction": False})
    elif simulation == "bf16_roundtrip":
        metadata.update({"quantizer": "bf16_roundtrip", "persistent_storage_in_platform": "fp32_dequantized_simulation",
                         "actual_optimizer_memory_reduction": False})
    else:
        metadata.update({"quantizer": "none", "persistent_storage_in_platform": "fp32"})
    return metadata
