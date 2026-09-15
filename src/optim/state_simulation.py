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
QUANTIZATION_GRANULARITIES = frozenset({"per_state_tensor", "blockwise"})


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


def int8_blockwise_linear_roundtrip(state: torch.Tensor, block_size: int = 2048) -> torch.Tensor:
    """Return signed max-abs INT8 simulation independently for contiguous blocks.

    Blocks are formed *within one state tensor* only.  Full blocks are batched
    for efficient device execution; a final partial block gets its own scale
    and is never padded into another block.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    value = state.float()
    flat = value.reshape(-1)
    if flat.numel() == 0:
        return value
    result = torch.empty_like(flat)
    full_elements = (flat.numel() // block_size) * block_size
    if full_elements:
        blocks = flat[:full_elements].reshape(-1, block_size)
        scales = blocks.abs().amax(dim=1, keepdim=True) / 127
        # A zero block is unchanged.  Substituting one only avoids 0/0; its
        # quantized values are still exactly zero after multiplication by scale.
        safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        result[:full_elements] = (torch.round(blocks / safe_scales).clamp(-127, 127) * scales).reshape(-1)
    if full_elements < flat.numel():
        tail = flat[full_elements:]
        scale = tail.abs().max() / 127
        if scale.item() == 0:
            result[full_elements:] = tail
        else:
            result[full_elements:] = torch.round(tail / scale).clamp(-127, 127) * scale
    return result.reshape_as(value).float()


def persist_state(state: torch.Tensor, simulation: str, state_name: str, *,
                  quantization_granularity: str = "per_state_tensor",
                  quantization_block_size: int = 2048) -> torch.Tensor:
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
        if state_name not in selected[simulation]:
            return state
        if quantization_granularity == "per_state_tensor":
            return int8_linear_roundtrip(state)
        if quantization_granularity == "blockwise":
            return int8_blockwise_linear_roundtrip(state, quantization_block_size)
        raise ValueError(f"unsupported quantization granularity: {quantization_granularity}")
    raise ValueError(f"unsupported state simulation: {simulation}")


def persistence_metadata(optimizer_name: str, simulation: str, *,
                         quantization_granularity: str = "per_state_tensor",
                         quantization_block_size: int = 2048) -> dict:
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
        if quantization_granularity not in QUANTIZATION_GRANULARITIES:
            raise ValueError(f"unsupported quantization granularity: {quantization_granularity}")
        metadata.update({"quantizer": "int8_linear_roundtrip", "bits": 8, "signed_range": [-127, 127],
                         "scale": "max_abs / 127", "rounding": "nearest", "granularity": "per_state_tensor",
                         "persistent_storage_in_platform": "fp32_dequantized_simulation",
                         "actual_optimizer_memory_reduction": False})
        if quantization_granularity == "blockwise":
            metadata.update({"quantizer": "linear_max_abs", "granularity": "blockwise",
                             "block_size": quantization_block_size,
                             "partial_final_block": "supported",
                             "simulation": "int8_linear_roundtrip",
                             "state_simulation_mode": simulation})
    elif simulation == "bf16_roundtrip":
        metadata.update({"quantizer": "bf16_roundtrip", "persistent_storage_in_platform": "fp32_dequantized_simulation",
                         "actual_optimizer_memory_reduction": False})
    else:
        metadata.update({"quantizer": "none", "persistent_storage_in_platform": "fp32"})
    return metadata
