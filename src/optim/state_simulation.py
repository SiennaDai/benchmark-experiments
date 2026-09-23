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
    "int4_linear_momentum",
})
INT8_DYNAMIC_SIMULATIONS = frozenset({"int8_dynamic_all_moments", "int8_dynamic_second_moment", "int4_dynamic_all_moments", "int4_dynamic_momentum"})
STATE_SIMULATIONS = frozenset({"none", "bf16_roundtrip", *INT8_LINEAR_SIMULATIONS, *INT8_DYNAMIC_SIMULATIONS})
QUANTIZATION_GRANULARITIES = frozenset({"per_state_tensor", "blockwise"})


def create_bitsandbytes_dynamic_map(*, signed: bool, max_exponent_bits: int = 7,
                                    total_bits: int = 8, device=None) -> torch.Tensor:
    """Create the pinned bitsandbytes ``create_dynamic_map`` reference map.

    This is a direct, dependency-free transcription of bitsandbytes
    ``functional.create_dynamic_map`` for the pinned 4-bit and 8-bit
    generalizations. It intentionally builds the map in FP32 and does not use
    bitsandbytes kernels. ``signed=False`` has nonnegative representable
    values; the signed map has both signs plus zero.
    """
    if total_bits not in {4, 8} or max_exponent_bits != total_bits - 1:
        raise ValueError("dynamic maps use pinned bits=4/8 and max_exponent_bits=total_bits-1")
    data: list[float] = []
    # Reference semantics deliberately reserve one non-sign bit in both maps.
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1
                             if signed else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        boundaries = torch.linspace(0.1, 1, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        values = ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        data.extend(values)
        if signed:
            data.extend([-value for value in values])
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        values = ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        data.extend(values)
        if signed:
            data.extend([-value for value in values])
    data.extend([0.0, 1.0])
    if len(data) != 2 ** total_bits:
        raise AssertionError("pinned dynamic map has an unexpected number of values")
    return torch.tensor(sorted(data), dtype=torch.float32, device=device)


def _nearest_codebook_values(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest lookup with deterministic lower-level selection on exact ties."""
    flat = normalized.reshape(-1).clamp(codebook[0], codebook[-1])
    upper = torch.searchsorted(codebook, flat).clamp(max=codebook.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    choose_upper = (flat - codebook[lower]).abs() > (codebook[upper] - flat).abs()
    return torch.where(choose_upper, codebook[upper], codebook[lower]).reshape_as(normalized)


def int8_blockwise_dynamic_roundtrip(state: torch.Tensor, *, signed: bool,
                                     block_size: int = 2048, total_bits: int = 8) -> torch.Tensor:
    """Blockwise absmax dynamic-map persistence simulation in FP32.

    Each call receives exactly one optimizer-state tensor, so blocks cannot
    cross state-tensor boundaries.  A current update uses the unrounded state;
    this result is only stored for the following update.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    value = state.float()
    flat = value.reshape(-1)
    if not flat.numel():
        return value
    codebook = create_bitsandbytes_dynamic_map(signed=signed, max_exponent_bits=total_bits - 1,
                                               total_bits=total_bits, device=value.device)
    result = torch.empty_like(flat)
    full_elements = (flat.numel() // block_size) * block_size
    if full_elements:
        blocks = flat[:full_elements].reshape(-1, block_size)
        scales = blocks.abs().amax(dim=1, keepdim=True)
        safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        result[:full_elements] = (_nearest_codebook_values(blocks / safe_scales, codebook) * scales).reshape(-1)
    if full_elements < flat.numel():
        tail = flat[full_elements:]
        scale = tail.abs().max()
        result[full_elements:] = tail if scale.item() == 0 else _nearest_codebook_values(tail / scale, codebook) * scale
    return result.reshape_as(value).float()


def int8_linear_roundtrip(state: torch.Tensor, bits: int = 8) -> torch.Tensor:
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
    if bits < 2:
        raise ValueError("signed linear quantization requires at least 2 bits")
    qmax = 2 ** (bits - 1) - 1
    scale = max_abs / qmax
    quantized = torch.round(value / scale).clamp(-qmax, qmax)
    return (quantized * scale).float()


def int8_blockwise_linear_roundtrip(state: torch.Tensor, block_size: int = 2048, bits: int = 8) -> torch.Tensor:
    """Return signed max-abs INT8 simulation independently for contiguous blocks.

    Blocks are formed *within one state tensor* only.  Full blocks are batched
    for efficient device execution; a final partial block gets its own scale
    and is never padded into another block.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if bits < 2:
        raise ValueError("signed linear quantization requires at least 2 bits")
    qmax = 2 ** (bits - 1) - 1
    value = state.float()
    flat = value.reshape(-1)
    if flat.numel() == 0:
        return value
    result = torch.empty_like(flat)
    full_elements = (flat.numel() // block_size) * block_size
    if full_elements:
        blocks = flat[:full_elements].reshape(-1, block_size)
        scales = blocks.abs().amax(dim=1, keepdim=True) / qmax
        # A zero block is unchanged.  Substituting one only avoids 0/0; its
        # quantized values are still exactly zero after multiplication by scale.
        safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        result[:full_elements] = (torch.round(blocks / safe_scales).clamp(-qmax, qmax) * scales).reshape(-1)
    if full_elements < flat.numel():
        tail = flat[full_elements:]
        scale = tail.abs().max() / qmax
        if scale.item() == 0:
            result[full_elements:] = tail
        else:
            result[full_elements:] = torch.round(tail / scale).clamp(-qmax, qmax) * scale
    return result.reshape_as(value).float()


def persist_state(state: torch.Tensor, simulation: str, state_name: str, *,
                  quantization_granularity: str = "per_state_tensor",
                  quantization_block_size: int = 2048) -> torch.Tensor:
    """Apply a post-update persistence simulation to one selected state tensor."""
    if simulation == "none":
        return state
    if simulation == "bf16_roundtrip":
        return state.to(torch.bfloat16).float()
    if simulation in INT8_DYNAMIC_SIMULATIONS:
        selected_dynamic = {
            "int8_dynamic_all_moments": {"exp_avg", "exp_avg_sq"},
            "int8_dynamic_second_moment": {"exp_avg_sq"},
            "int4_dynamic_all_moments": {"exp_avg", "exp_avg_sq"},
            "int4_dynamic_momentum": {"muon_momentum"},
        }[simulation]
        if state_name not in selected_dynamic:
            return state
        if quantization_granularity != "blockwise":
            raise ValueError("dynamic INT8 persistence requires blockwise granularity")
        return int8_blockwise_dynamic_roundtrip(state, signed=state_name != "exp_avg_sq",
                                                block_size=quantization_block_size,
                                                total_bits=4 if simulation.startswith("int4_") else 8)
    selected = {
        "int8_linear_first_moment": {"exp_avg"},
        "int8_linear_second_moment": {"exp_avg_sq"},
        "int8_linear_all_moments": {"exp_avg", "exp_avg_sq"},
        "int8_linear_momentum": {"muon_momentum"},
        "int4_linear_momentum": {"muon_momentum"},
    }
    if simulation in selected:
        if state_name not in selected[simulation]:
            return state
        if quantization_granularity == "per_state_tensor":
            return int8_linear_roundtrip(state, bits=4 if simulation.startswith("int4_") else 8)
        if quantization_granularity == "blockwise":
            return int8_blockwise_linear_roundtrip(state, quantization_block_size, bits=4 if simulation.startswith("int4_") else 8)
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
            "int8_dynamic_all_moments": all_states,
            "int8_dynamic_second_moment": ["exp_avg_sq"],
            "int4_dynamic_all_moments": all_states,
        }.get(simulation)
        if selected is None:
            raise ValueError(f"simulation {simulation} is invalid for {optimizer_name}")
        untouched = [state for state in all_states if state not in selected]
        groups = {"reference_adamw": {"quantized_state_names": selected, "states_left_fp32": untouched}}
    elif optimizer_name == "reference_muon":
        if simulation not in {"none", "bf16_roundtrip", "int8_linear_momentum", "int4_linear_momentum", "int4_dynamic_momentum"}:
            raise ValueError(f"simulation {simulation} is invalid for {optimizer_name}")
        selected = ["muon_momentum"] if simulation != "none" else []
        groups = {
            "muon": {"quantized_state_names": selected, "states_left_fp32": [] if selected else ["muon_momentum"]},
            "auxiliary_adamw": {"quantized_state_names": [], "states_left_fp32": ["exp_avg", "exp_avg_sq"]},
        }
    elif optimizer_name == "recursive_muon":
        # The recursive optimizer owns a genuinely compressed Muon state.  It
        # is not one of the numerical round-trip simulations above, but the
        # run manifest still needs the same auditable persistence contract as
        # the reference optimizers.
        groups = {
            "muon": {"quantized_state_names": ["compressed_momentum"], "states_left_fp32": []},
            "auxiliary_adamw": {"quantized_state_names": [], "states_left_fp32": ["exp_avg", "exp_avg_sq"]},
        }
    else:
        return {"simulation": simulation,
                "state_groups": {"optimizer": {"quantized_state_names": [], "states_left_fp32": []}},
                "persistence_timing": "post_update; current FP32 state drives current update; persisted state affects next step"}
    metadata = {"simulation": simulation, "state_groups": groups,
                "persistence_timing": "post_update; current FP32 state drives current update; persisted state affects next step"}
    if simulation in INT8_LINEAR_SIMULATIONS:
        if quantization_granularity not in QUANTIZATION_GRANULARITIES:
            raise ValueError(f"unsupported quantization granularity: {quantization_granularity}")
        bits = 4 if simulation.startswith("int4_") else 8
        qmax = 2 ** (bits - 1) - 1
        metadata.update({"quantizer": f"int{bits}_linear_roundtrip", "bits": bits, "signed_range": [-qmax, qmax],
                         "scale": f"max_abs / {qmax}", "rounding": "nearest", "granularity": "per_state_tensor",
                         "persistent_storage_in_platform": "fp32_dequantized_simulation",
                         "actual_optimizer_memory_reduction": False})
        if quantization_granularity == "blockwise":
            metadata.update({"quantizer": "linear_max_abs", "granularity": "blockwise",
                             "block_size": quantization_block_size,
                             "partial_final_block": "supported",
                             "simulation": f"int{bits}_linear_roundtrip",
                             "state_simulation_mode": simulation})
    elif simulation in INT8_DYNAMIC_SIMULATIONS:
        if quantization_granularity != "blockwise":
            raise ValueError("dynamic INT8 persistence requires blockwise granularity")
        bits = 4 if simulation.startswith("int4_") else 8
        dynamic_map_provenance = {
            "source": "bitsandbytes.functional.create_dynamic_map",
            "implementation": "pinned_pure_torch_transcription",
            "max_exponent_bits": bits - 1, "total_bits": bits,
        }
        if optimizer_name == "reference_muon":
            dynamic_map_provenance["muon_momentum"] = {
                "codebook": "dynamic", "signed": True, "representable_values": 2 ** bits,
            }
        else:
            dynamic_map_provenance.update({
                "exp_avg": {"codebook": "dynamic", "signed": True, "representable_values": 2 ** bits},
                "exp_avg_sq": {"codebook": "dynamic", "signed": False, "representable_values": 2 ** bits},
            })
        metadata.update({
            "simulation": f"int{bits}_dynamic_roundtrip", "state_simulation_mode": simulation,
            "quantizer": "bitsandbytes_create_dynamic_map_reference", "bits": bits,
            "granularity": "blockwise", "block_size": quantization_block_size,
            "partial_final_block": "supported", "rounding": "nearest_codebook_value",
            "block_scale": "absmax", "persistent_storage_in_platform": "fp32_dequantized_simulation",
            "actual_optimizer_memory_reduction": False,
            "dynamic_map_provenance": dynamic_map_provenance,
        })
    elif simulation == "bf16_roundtrip":
        metadata.update({"quantizer": "bf16_roundtrip", "persistent_storage_in_platform": "fp32_dequantized_simulation",
                         "actual_optimizer_memory_reduction": False})
    elif optimizer_name == "recursive_muon":
        metadata.update({"quantizer": "structural_recursive_codec",
                         "persistent_storage_in_platform": "serialized_compressed_state",
                         "actual_optimizer_memory_reduction": True})
    else:
        metadata.update({"quantizer": "none", "persistent_storage_in_platform": "fp32"})
    return metadata
