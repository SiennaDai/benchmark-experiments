"""Read-only, bounded AdamW state-persistence diagnostics.

The collector deliberately observes detached tensors after the FP32 moment
update and after persistence simulation.  It never writes an optimizer tensor
or calls a random operation.  Distribution percentiles use a bounded,
deterministic evenly-spaced sample from each state tensor; counts, extrema and
L2 error are exact reductions over all elements.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from .state_simulation import create_bitsandbytes_dynamic_map, _nearest_codebook_values


def _kind(value: float):
    if math.isnan(value): return {"nonfinite": "nan"}
    if math.isinf(value): return {"nonfinite": "+inf" if value > 0 else "-inf"}
    return value


def _scalar(value):
    return _kind(float(value.detach().item()) if torch.is_tensor(value) else float(value))


def _percentiles(values: list[torch.Tensor], names=(50, 90, 99, 99.9)) -> dict:
    if not values:
        return {f"p{str(n).replace('.', '')}": None for n in names}
    x = torch.cat(values).float()
    x = x[torch.isfinite(x)]
    if not x.numel(): return {f"p{str(n).replace('.', '')}": {"nonfinite": "no_finite_values"} for n in names}
    return {f"p{str(n).replace('.', '')}": _scalar(torch.quantile(x, n / 100)) for n in names}


def _evenly_spaced_indices(n: int, limit: int, device: torch.device | str) -> torch.Tensor:
    """Return bounded, endpoint-preserving integer sample indices.

    Do not use ``linspace(...).long()`` here.  CUDA's default floating-point
    linspace loses integer precision for sufficiently large flattened tensors,
    which can turn a nominal endpoint into an out-of-bounds index.  This
    construction is integer-only and deterministic.  For two or more samples
    it includes both endpoints; a one-element sample intentionally selects
    the first element, matching the bounded sampler contract.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if limit < 1:
        raise ValueError("limit must be positive")
    count = min(n, limit)
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.int64)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.int64)
    return torch.arange(count, device=device, dtype=torch.int64) * (n - 1) // (count - 1)


def _sample(value: torch.Tensor, limit=1024) -> torch.Tensor:
    flat = value.detach().reshape(-1)
    if flat.numel() <= limit:
        return flat
    return flat[_evenly_spaced_indices(flat.numel(), limit, flat.device)]


class AdamWStateDiagnostics:
    """Per-update aggregator installed as an optional ReferenceAdamW observer."""
    schema_version = 1

    def __init__(self, parameter_names: dict[int, str], *, tensor_landmarks=(),
                 quantization_granularity="per_state_tensor", quantization_block_size=2048,
                 state_simulation="none"):
        self.parameter_names = parameter_names
        self.tensor_landmarks = set(tensor_landmarks)
        if quantization_granularity not in {"per_state_tensor", "blockwise"}:
            raise ValueError("unsupported quantization granularity")
        if not isinstance(quantization_block_size, int) or quantization_block_size <= 0:
            raise ValueError("quantization block size must be positive")
        self.quantization_granularity = quantization_granularity
        self.quantization_block_size = quantization_block_size
        self.state_simulation = state_simulation
        self._items = []

    def _scales(self, value: torch.Tensor) -> list[torch.Tensor]:
        """Return the exact scale units used by persistence, without mutation."""
        flat = value.detach().float().reshape(-1)
        if self.quantization_granularity == "per_state_tensor":
            return [flat.abs().max() / 127] if flat.numel() else []
        divisor = 1 if self.state_simulation == "int8_dynamic_all_moments" else 127
        return [block.abs().max() / divisor for block in flat.split(self.quantization_block_size) if block.numel()]

    @torch.no_grad()
    def _dynamic_occupancy(self, values, *, signed: bool):
        """Bounded block-normalized dynamic-codebook occupancy observation."""
        if self.state_simulation != "int8_dynamic_all_moments":
            return {}
        samples = []
        for value in values:
            flat = value.detach().float().reshape(-1)
            full = (flat.numel() // self.quantization_block_size) * self.quantization_block_size
            if full:
                blocks = flat[:full].reshape(-1, self.quantization_block_size)
                # At most 16 deterministic positions per block keeps diagnostic
                # overhead bounded while retaining each block's own absmax.
                count = min(16, self.quantization_block_size)
                positions = _evenly_spaced_indices(self.quantization_block_size, count, flat.device)
                scales = blocks.abs().amax(dim=1, keepdim=True)
                safe = torch.where(scales == 0, torch.ones_like(scales), scales)
                samples.append((blocks[:, positions] / safe).reshape(-1))
            if full < flat.numel():
                tail = flat[full:]
                scale = tail.abs().max()
                safe = scale if scale.item() != 0 else torch.ones_like(scale)
                samples.append(_sample(tail / safe, limit=16))
        if not samples:
            return {"codebook_occupancy_sample_elements": 0, "codebook_levels_used": 0,
                    "codebook_levels_used_fraction": 0.0, "fraction_mapped_to_zero_code": 0.0,
                    "fraction_mapped_to_smallest_positive_code": 0.0}
        normalized = torch.cat(samples)
        codebook = create_bitsandbytes_dynamic_map(signed=signed, device=normalized.device)
        mapped = _nearest_codebook_values(normalized, codebook)
        zero = codebook[torch.searchsorted(codebook, torch.tensor(0., device=codebook.device))]
        positive = codebook[codebook > 0][0]
        levels = torch.unique(mapped)
        return {"codebook_occupancy_sample_elements": int(mapped.numel()),
                "codebook_levels_used": int(levels.numel()),
                "codebook_levels_used_fraction": float(levels.numel() / codebook.numel()),
                "fraction_mapped_to_zero_code": float((mapped == zero).float().mean().item()),
                "fraction_mapped_to_smallest_positive_code": float((mapped == positive).float().mean().item())}

    @torch.no_grad()
    def observe(self, *, parameter, exp_avg_pre, exp_avg_post, exp_avg_sq_pre,
                exp_avg_sq_post, updated, group, step):
        # All quantities are detached/read-only.  `updated` is the already
        # computed FP32 current-step result, before it is copied into parameter.
        self._items.append({"name": self.parameter_names.get(id(parameter), "<unknown>"),
                            "p_sq": float(parameter.detach().float().square().sum().item()),
                            "update_sq": float((updated.detach() - parameter.detach()).float().square().sum().item()),
                            "m_pre": exp_avg_pre.detach(), "m_post": exp_avg_post.detach(),
                            "v_pre": exp_avg_sq_pre.detach(), "v_post": exp_avg_sq_post.detach(),
                            "beta2": group["betas"][1], "eps": group["eps"], "step": step})

    def _state_stats(self, pre_list, post_list, *, include_denominator=False, beta2=None, eps=None, step=None):
        n_tensors, n_elements = len(pre_list), sum(x.numel() for x in pre_list)
        if not n_tensors:
            return {"number_of_state_tensors": 0, "number_of_state_elements": 0}
        zero_pre = zero_post = newly = nonfinite_pre = nonfinite_post = 0
        sum_pre = sum_post = sq_pre = sq_error = 0.0; max_error = 0.0
        mins=[]; min_positive=[]; maxs=[]; post_mins=[]; post_positive=[]; post_maxs=[]; scales=[]; samples=[]; amp_samples=[]
        amp_counts = {2: 0, 10: 0, 100: 0}; amp_total = 0; denom_pre_min = float("inf"); denom_post_min = float("inf"); inv_pre_max = 0.; inv_post_max = 0.
        tensor_rows=[]
        for pre, post in zip(pre_list, post_list):
            finite_pre, finite_post = torch.isfinite(pre), torch.isfinite(post)
            nonfinite_pre += int((~finite_pre).sum().item()); nonfinite_post += int((~finite_post).sum().item())
            zero_pre += int((pre == 0).sum().item()); zero_post += int((post == 0).sum().item()); newly += int(((pre != 0) & (post == 0)).sum().item())
            good = finite_pre & finite_post
            if good.any():
                a, b = pre[good].float(), post[good].float(); err = b-a
                sum_pre += float(a.sum().item()); sum_post += float(b.sum().item()); sq_pre += float(a.square().sum().item()); sq_error += float(err.square().sum().item()); max_error=max(max_error, float(err.abs().max().item()))
                mins.append(a.min()); maxs.append(a.max()); post_mins.append(b.min()); post_maxs.append(b.max())
                if (a > 0).any(): min_positive.append(a[a > 0].min())
                if (b > 0).any(): post_positive.append(b[b > 0].min())
                scales.extend(self._scales(pre))
                if include_denominator:
                    correction = 1 - beta2 ** step
                    denom_pre = (a / correction).clamp_min(0).sqrt().add(eps)
                    denom_post = (b / correction).clamp_min(0).sqrt().add(eps)
                    amp = denom_pre / denom_post
                    amp = amp[torch.isfinite(amp)]
                    if amp.numel():
                        amp_counts[2] += int((amp > 2).sum().item()); amp_counts[10] += int((amp > 10).sum().item()); amp_counts[100] += int((amp > 100).sum().item()); amp_total += amp.numel(); amp_samples.append(_sample(amp))
                        denom_pre_min=min(denom_pre_min, float(denom_pre.min().item())); denom_post_min=min(denom_post_min, float(denom_post.min().item())); inv_pre_max=max(inv_pre_max, float((1/denom_pre).max().item())); inv_post_max=max(inv_post_max, float((1/denom_post).max().item()))
        def mn(xs): return _scalar(torch.stack(xs).min()) if xs else None
        def mx(xs): return _scalar(torch.stack(xs).max()) if xs else None
        scale_percentiles = _percentiles([x.reshape(-1) for x in scales], names=(50,90,99))
        output = {"number_of_state_tensors": n_tensors, "number_of_state_elements": n_elements,
                  "pre_quant_zero_fraction": zero_pre / n_elements, "post_quant_zero_fraction": zero_post / n_elements, "newly_zero_fraction": newly / n_elements,
                  "pre_quant_nonfinite_count": nonfinite_pre, "post_quant_nonfinite_count": nonfinite_post,
                  "pre_quant_min": mn(mins), "pre_quant_min_positive": mn(min_positive), "pre_quant_max": mx(maxs), "pre_quant_mean": _kind(sum_pre / n_elements),
                  "post_quant_min": mn(post_mins), "post_quant_min_positive": mn(post_positive), "post_quant_max": mx(post_maxs), "post_quant_mean": _kind(sum_post / n_elements),
                  "global_relative_l2_quantization_error": _kind(math.sqrt(sq_error) / max(math.sqrt(sq_pre), 1e-30)), "global_max_abs_quantization_error": _kind(max_error),
                  "scale_min": mn(scales), "scale_median": scale_percentiles["p50"], "scale_p90": scale_percentiles["p90"], "scale_p99": scale_percentiles["p99"], "scale_max": mx(scales)}
        if self.state_simulation == "int8_dynamic_all_moments":
            output["dynamic_codebook_occupancy"] = self._dynamic_occupancy(pre_list, signed=not include_denominator)
        if include_denominator:
            amp_percentiles = _percentiles(amp_samples, names=(50,90,99,99.9))
            output.update({"denom_pre_min": _kind(denom_pre_min), "denom_post_min": _kind(denom_post_min), "inverse_denom_pre_max": _kind(inv_pre_max), "inverse_denom_post_max": _kind(inv_post_max),
                           "amplification_mean": _scalar(torch.cat(amp_samples).mean()) if amp_samples else None,
                           "amplification_median": amp_percentiles["p50"], "amplification_p90": amp_percentiles["p90"], "amplification_p99": amp_percentiles["p99"], "amplification_p999": amp_percentiles["p999"],
                           "amplification_max": _scalar(torch.cat(amp_samples).max()) if amp_samples else None,
                           **{f"fraction_amplification_gt_{threshold}": (count / amp_total if amp_total else None) for threshold,count in amp_counts.items()}})
        return output

    @torch.no_grad()
    def finish_update(self, *, update, processed_target_tokens, train_nll, pre_clip_grad_norm, learning_rate):
        items, self._items = self._items, []
        pre_m, post_m = [x["m_pre"] for x in items], [x["m_post"] for x in items]
        pre_v, post_v = [x["v_pre"] for x in items], [x["v_post"] for x in items]
        p_sq=sum(x["p_sq"] for x in items); u_sq=sum(x["update_sq"] for x in items)
        beta2, eps, step = (items[0]["beta2"], items[0]["eps"], items[0]["step"]) if items else (None,None,None)
        result = {"schema_version": self.schema_version, "update": update, "processed_target_tokens": processed_target_tokens,
                "timing": "after_fp32_moment_update_and_current_parameter_update; post_persistence_is_next_step_state",
                "second_moment": self._state_stats(pre_v, post_v, include_denominator=True, beta2=beta2, eps=eps, step=step),
                "first_moment": self._state_stats(pre_m, post_m),
                "actual_update": {"global_parameter_norm_before_update": _kind(math.sqrt(p_sq)), "global_parameter_update_l2_norm": _kind(math.sqrt(u_sq)), "relative_parameter_update_norm": _kind(math.sqrt(u_sq) / max(math.sqrt(p_sq), 1e-30)), "pre_clip_grad_norm": _scalar(pre_clip_grad_norm), "train_nll": _scalar(train_nll), "learning_rate": _scalar(learning_rate)}}
        if update in self.tensor_landmarks:
            rows = []
            for item in items:
                row = self._state_stats([item["v_pre"]], [item["v_post"]], include_denominator=True,
                                        beta2=item["beta2"], eps=item["eps"], step=item["step"])
                row["parameter_name"] = item["name"]
                rows.append(row)
            result["state_tensor_diagnostics"] = rows
        return result
