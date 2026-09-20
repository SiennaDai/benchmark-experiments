"""Offline direction-aware rounding oracles for Muon INT4 dynamic state.

This module is deliberately separate from the optimizer.  It uses the same
blockwise dynamic-map construction and the production Muon transform, but
never writes optimizer state.  The two oracle searches are deterministic
headroom studies, not deployable quantizers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from . import muon_reference
from .muon_update_fidelity import _aggregate, _number, _ratios
from .state_simulation import create_bitsandbytes_dynamic_map, persist_state


QUANTIZER = "int4-dynamic-b2048"
BLOCK_SIZE = 2048


@dataclass
class BlockCandidates:
    scale: torch.Tensor
    normalized: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    nearest: torch.Tensor
    eligible: torch.Tensor
    midpoint_distance: torch.Tensor


def _candidate_parts(value: torch.Tensor, block_size: int = BLOCK_SIZE) -> BlockCandidates:
    """Return exact dynamic-map block scales and neighboring levels.

    The clamp, absmax scale, final partial block, and lower-on-tie nearest
    rule mirror ``int8_blockwise_dynamic_roundtrip``.  ``lower == upper`` is
    enforced at exact codebook levels so those values are correctly fixed.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    value = value.detach().float()
    flat = value.reshape(-1)
    n = flat.numel()
    if not n:
        empty = flat.clone()
        return BlockCandidates(empty, empty, empty, empty, empty, empty.bool(), empty)
    codebook = create_bitsandbytes_dynamic_map(signed=True, max_exponent_bits=3,
                                               total_bits=4, device=flat.device)
    scale = torch.empty_like(flat)
    normalized = torch.empty_like(flat)
    lower = torch.empty_like(flat)
    upper = torch.empty_like(flat)
    nearest = torch.empty_like(flat)
    midpoint_distance = torch.ones_like(flat)
    eligible = torch.zeros_like(flat, dtype=torch.bool)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = flat[start:stop]
        block_scale = block.abs().amax()
        safe = torch.where(block_scale == 0, torch.ones_like(block_scale), block_scale)
        norm = (block / safe).clamp(codebook[0], codebook[-1])
        upper_idx = torch.searchsorted(codebook, norm).clamp(max=codebook.numel() - 1)
        lower_idx = (upper_idx - 1).clamp(min=0)
        up = codebook[upper_idx]
        lo = codebook[lower_idx]
        exact = up == norm
        lo = torch.where(exact, up, lo)
        choose_upper = (norm - lo).abs() > (up - norm).abs()
        near = torch.where(choose_upper, up, lo)
        gap = (up - lo).abs()
        distance = torch.where(gap != 0, (norm - (lo + up) / 2).abs() / gap,
                               torch.ones_like(norm))
        scale[start:stop] = safe
        normalized[start:stop] = norm
        lower[start:stop] = lo
        upper[start:stop] = up
        nearest[start:stop] = near
        midpoint_distance[start:stop] = distance
        eligible[start:stop] = (~exact) & (lo != up) & torch.isfinite(block)
    shape = value.shape
    return BlockCandidates(scale.reshape(shape), normalized.reshape(shape), lower.reshape(shape),
                           upper.reshape(shape), nearest.reshape(shape), eligible.reshape(shape),
                           midpoint_distance.reshape(shape))


def production_nearest(value: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """The unchanged production INT4 dynamic roundtrip."""
    return persist_state(value.detach().clone(), "int4_dynamic_momentum", "muon_momentum",
                         quantization_granularity="blockwise", quantization_block_size=block_size)


def candidate_quantized(parts: BlockCandidates, choice: torch.Tensor) -> torch.Tensor:
    """Dequantize a lower/upper codebook assignment tensor."""
    levels = torch.where(choice, parts.upper, parts.lower)
    return (levels * parts.scale).reshape_as(parts.scale).float()


def _cosine(dot: torch.Tensor, ref_sq: torch.Tensor, obs_sq: torch.Tensor) -> torch.Tensor:
    if ref_sq.numel() != 1:
        raise ValueError("reference norm must be scalar")
    if ref_sq.item() == 0:
        return torch.full_like(dot.double(), float("nan"))
    denominator = ref_sq.double().sqrt() * obs_sq.double().clamp_min(0).sqrt()
    return torch.where(denominator != 0, dot.double() / denominator, torch.full_like(dot.double(), float("nan")))


def _raw_group_search(source: torch.Tensor, parts: BlockCandidates, *,
                      midpoint_margin: float, max_passes: int, group_size: int) -> tuple[torch.Tensor, dict]:
    """Deterministic block-local grouped coordinate descent for raw cosine."""
    source = source.detach().float()
    q = (parts.nearest * parts.scale).clone()
    eligible = parts.eligible & (parts.midpoint_distance <= midpoint_margin)
    flat_x, flat_q, flat_lo, flat_up, flat_ok = [x.reshape(-1) for x in (source, q, parts.lower, parts.upper, eligible)]
    accepted = 0
    considered = int(flat_ok.sum().item())
    for start in range(0, flat_x.numel(), BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, flat_x.numel())
        idx = torch.nonzero(flat_ok[start:stop], as_tuple=False).flatten() + start
        if not idx.numel():
            continue
        current = flat_q[start:stop].clone()
        x = flat_x[start:stop]
        lo, up = flat_lo[start:stop] * parts.scale.reshape(-1)[start:stop], flat_up[start:stop] * parts.scale.reshape(-1)[start:stop]
        # Flip relative to nearest; if nearest picked upper, alternate lower.
        alt = torch.where((current - lo).abs() <= (current - up).abs(), up, lo)
        for _ in range(max_passes):
            local = idx - start
            delta = alt[local] - current[local]
            dot = (x * current).sum().double(); obs_sq = current.square().sum().double(); ref_sq = x.square().sum().double()
            new_dot = dot + x[local].double() * delta.double()
            new_sq = obs_sq + 2 * current[local].double() * delta.double() + delta.double().square()
            scores = _cosine(new_dot, ref_sq, new_sq)
            current_score = _cosine(dot, ref_sq, obs_sq)
            gain = scores - current_score
            positive = gain > 0
            if not bool(positive.any()):
                break
            ranked = torch.argsort(gain.masked_fill(~positive, float("-inf")), descending=True)
            chosen = ranked[:group_size]
            chosen = chosen[torch.isfinite(gain[chosen])]
            if not chosen.numel():
                break
            chosen_local = local[chosen]
            proposal = current.clone(); proposal[chosen_local] = alt[chosen_local]
            proposal_score = _cosine((x * proposal).sum(), ref_sq, proposal.square().sum())
            if not (proposal_score > current_score):
                break
            current = proposal
            # The accepted group is no longer a candidate in subsequent passes.
            keep = torch.ones(idx.numel(), dtype=torch.bool, device=idx.device)
            keep[chosen] = False
            idx = idx[keep]
            if not idx.numel():
                break
            accepted += int(chosen.numel())
        flat_q[start:stop] = current
    return flat_q.reshape_as(source), {"total_count": int(flat_ok.numel()), "eligible_count": int(eligible.sum()), "considered_count": considered,
                                       "accepted_flips": accepted, "exact_evaluations": 0}


def _proxy_rank(source: torch.Tensor, parts: BlockCandidates, *, midpoint_margin: float) -> torch.Tensor:
    q = parts.nearest.reshape(-1); x = source.detach().float().reshape(-1)
    lo, up = parts.lower.reshape(-1), parts.upper.reshape(-1)
    eligible = parts.eligible.reshape(-1) & (parts.midpoint_distance.reshape(-1) <= midpoint_margin)
    alt = torch.where((q - lo).abs() <= (q - up).abs(), up, lo)
    delta = alt - q
    # First-order raw direction proxy.  Tie-breaking is stable by flat index.
    score = (x * delta).double() / (1e-30 + delta.double().abs())
    score = score.masked_fill(~eligible, float("-inf"))
    order = torch.argsort(score, descending=True, stable=True)
    return order[eligible[order]]


@torch.no_grad()
def _update_group_search(source: torch.Tensor, parts: BlockCandidates, *,
                         ns_steps: int, ns_coefficients, ns_eps: float,
                         midpoint_margin: float, max_candidates: int,
                         group_size: int, max_groups: int) -> tuple[torch.Tensor, dict]:
    """Deterministic grouped approximate search with exact Muon acceptance.

    All eligible candidates are counted.  Only the top ``max_candidates`` by
    the documented first-order raw proxy are sent to exact transform
    evaluation; this explicit cap is recorded in search statistics.
    """
    source = source.detach().float()
    parts = parts
    nearest = production_nearest(source)
    q = nearest.clone()
    eligible = parts.eligible & (parts.midpoint_distance <= midpoint_margin)
    all_count = int(eligible.sum().item())
    ranked = _proxy_rank(source, parts, midpoint_margin=midpoint_margin)[:max_candidates]
    lo, up, nearest_flat = parts.lower.reshape(-1), parts.upper.reshape(-1), parts.nearest.reshape(-1)
    scale_flat = parts.scale.reshape(-1)
    alt = torch.where((nearest_flat - lo).abs() <= (nearest_flat - up).abs(), up, lo) * scale_flat
    current_update = muon_reference.zeropower_newton_schulz(source.clone(), ns_steps, ns_coefficients, ns_eps)
    current_q_update = muon_reference.zeropower_newton_schulz(q.clone(), ns_steps, ns_coefficients, ns_eps)
    current_cos = _cosine((current_update * current_q_update).sum(), current_update.square().sum(), current_q_update.square().sum())
    accepted = 0; exact_evals = 0
    for group_start in range(0, ranked.numel(), group_size):
        if exact_evals >= max_groups:
            break
        group = ranked[group_start:group_start + group_size]
        proposal_flat = q.reshape(-1).clone()
        proposal_flat[group] = alt[group]
        proposal = proposal_flat.reshape_as(q)
        proposal_update = muon_reference.zeropower_newton_schulz(proposal.clone(), ns_steps, ns_coefficients, ns_eps)
        proposal_cos = _cosine((current_update * proposal_update).sum(), current_update.square().sum(), proposal_update.square().sum())
        exact_evals += 1
        if bool(proposal_cos > current_cos):
            q = proposal
            current_cos = proposal_cos
            accepted += int(group.numel())
    return q, {"total_count": int(eligible.numel()), "eligible_count": all_count, "considered_count": int(ranked.numel()),
               "accepted_flips": accepted, "exact_evaluations": exact_evals,
               "screening_cap": max_candidates}


@torch.no_grad()
def analyze_oracle_tensor(item: dict, *, ns_steps: int = 5,
                          ns_coefficients=(3.4445, -4.7750, 2.0315), ns_eps: float = 1e-7,
                          midpoint_margin: float = 0.25, raw_max_passes: int = 2,
                          raw_group_size: int = 32, update_max_candidates: int = 32,
                          update_group_size: int = 8, update_max_groups: int = 4) -> tuple[list[dict], list[dict]]:
    """Return nearest/raw-direction/Muon-update rows and search statistics."""
    source = item["tensor"].detach().float()
    parts = _candidate_parts(source)
    nearest = production_nearest(source)
    raw_q, raw_stats = _raw_group_search(source, parts, midpoint_margin=midpoint_margin,
                                          max_passes=raw_max_passes, group_size=raw_group_size)
    if source.ndim == 2:
        update_q, update_stats = _update_group_search(source, parts, ns_steps=ns_steps,
                                                       ns_coefficients=ns_coefficients, ns_eps=ns_eps,
                                                       midpoint_margin=midpoint_margin,
                                                       max_candidates=update_max_candidates,
                                                       group_size=update_group_size, max_groups=update_max_groups)
    else:
        update_q, update_stats = nearest.clone(), {"total_count": int(parts.eligible.numel()), "eligible_count": int((parts.eligible & (parts.midpoint_distance <= midpoint_margin)).sum()),
                                                   "considered_count": 0, "accepted_flips": 0, "exact_evaluations": 0,
                                                   "screening_cap": update_max_candidates}
    rows = []
    stats = []
    for mode, q in (("nearest", nearest), ("raw_direction_oracle", raw_q), ("muon_update_direction_oracle", update_q)):
        row = {"rounding_mode": mode, "shape": list(source.shape), "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")),
               "parameter_name": item.get("name", "<unknown>"), "quantizer": QUANTIZER, "bits": 4,
               "codebook": "dynamic", "block_size": BLOCK_SIZE,
               "raw_zero_fraction": _number((q == 0).float().mean()) if q.numel() else None,
               **_ratios(source, q, "raw_momentum")}
        if source.ndim != 2:
            row.update({"update_relative_l2": None, "update_cosine": None, "update_norm_ratio": None,
                        "muon_update_relative_l2": None, "muon_update_cosine": None, "muon_update_norm_ratio": None,
                        "update_metric_status": "excluded_not_2d_muon_matrix"})
        else:
            ref = muon_reference.zeropower_newton_schulz(source.clone(), ns_steps, ns_coefficients, ns_eps)
            obs = muon_reference.zeropower_newton_schulz(q.clone(), ns_steps, ns_coefficients, ns_eps)
            row.update(_ratios(ref, obs, "update"))
            row.update({f"muon_{key}": row[key] for key in ("update_relative_l2", "update_cosine", "update_norm_ratio")})
        reduction = {"raw_ref_sq": source.square().sum(), "raw_obs_sq": q.square().sum(),
                     "raw_error_sq": (q - source).square().sum(), "raw_dot": (q * source).sum(),
                     "update_eligible": source.ndim == 2}
        if source.ndim == 2:
            ref = muon_reference.zeropower_newton_schulz(source.clone(), ns_steps, ns_coefficients, ns_eps)
            obs = muon_reference.zeropower_newton_schulz(q.clone(), ns_steps, ns_coefficients, ns_eps)
            reduction.update({"update_ref_sq": ref.square().sum(), "update_obs_sq": obs.square().sum(),
                              "update_error_sq": (obs - ref).square().sum(), "update_dot": (obs * ref).sum()})
        row["_reduction"] = reduction
        rows.append(row)
    for mode, data in (("raw_direction_oracle", raw_stats), ("muon_update_direction_oracle", update_stats)):
        stats.append({"rounding_mode": mode, "parameter_id": rows[0]["parameter_id"], "parameter_name": rows[0]["parameter_name"], **data})
    return rows, stats


@torch.no_grad()
def aggregate_rows_from_tensors(items: Iterable[dict], *, mode: str, **kwargs) -> dict:
    """Compute aggregate metrics directly from all tensors for one mode."""
    records = []
    for item in items:
        source = item["tensor"].detach().float(); parts = _candidate_parts(source); nearest = production_nearest(source)
        if mode == "nearest": q = nearest
        elif mode == "raw_direction_oracle": q, _ = _raw_group_search(source, parts, midpoint_margin=kwargs.get("midpoint_margin", .25), max_passes=kwargs.get("raw_max_passes", 2), group_size=kwargs.get("raw_group_size", 32))
        else: q, _ = _update_group_search(source, parts, ns_steps=kwargs.get("ns_steps", 5), ns_coefficients=kwargs.get("ns_coefficients", (3.4445, -4.775, 2.0315)), ns_eps=kwargs.get("ns_eps", 1e-7), midpoint_margin=kwargs.get("midpoint_margin", .25), max_candidates=kwargs.get("update_max_candidates", 32), group_size=kwargs.get("update_group_size", 8), max_groups=kwargs.get("update_max_groups", 4))
        r = {"raw_ref_sq": source.square().sum(), "raw_obs_sq": q.square().sum(), "raw_error_sq": (q-source).square().sum(), "raw_dot": (q*source).sum(), "update_eligible": source.ndim == 2}
        if source.ndim == 2:
            ref = muon_reference.zeropower_newton_schulz(source.clone(), kwargs.get("ns_steps",5), kwargs.get("ns_coefficients",(3.4445,-4.775,2.0315)), kwargs.get("ns_eps",1e-7)); obs = muon_reference.zeropower_newton_schulz(q.clone(), kwargs.get("ns_steps",5), kwargs.get("ns_coefficients",(3.4445,-4.775,2.0315)), kwargs.get("ns_eps",1e-7)); r.update({"update_ref_sq":ref.square().sum(),"update_obs_sq":obs.square().sum(),"update_error_sq":(obs-ref).square().sum(),"update_dot":(obs*ref).sum()})
        records.append(r)
    return _aggregate(records, "raw") | {f"muon_{k}": v for k, v in _aggregate(records, "update").items() if k.startswith("update_") or k == "update_metric_status"}
