"""Read-only fidelity measurements for FP32 Muon momentum snapshots.

This module intentionally delegates both operations with scientific semantics:
``persist_state`` is the training persistence quantizer and
``muon_reference.zeropower_newton_schulz`` is the production Muon transform.
It never writes a tensor supplied by the optimizer.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch

from . import muon_reference
from .state_simulation import persist_state

QUANTIZERS = {
    "int8-linear-b2048": ("int8_linear_momentum", 8, "linear", 2048),
    "int4-linear-b2048": ("int4_linear_momentum", 4, "linear", 2048),
    "int4-dynamic-b2048": ("int4_dynamic_momentum", 4, "dynamic", 2048),
}


def _number(x: torch.Tensor) -> float | dict:
    value = float(x.detach().item())
    if math.isfinite(value):
        return value
    return {"nonfinite": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}


def _ratios(reference: torch.Tensor, observed: torch.Tensor, prefix: str) -> dict:
    """Metric reductions with explicit unavailable/nonfinite outcomes."""
    ref, obs = reference.detach().float(), observed.detach().float()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(obs).all())
    if not finite:
        return {f"{prefix}_relative_l2": {"nonfinite": "input"},
                f"{prefix}_cosine": {"nonfinite": "input"},
                f"{prefix}_norm_ratio": {"nonfinite": "input"},
                f"{prefix}_metric_status": "nonfinite_input"}
    ref_sq, obs_sq = ref.square().sum(), obs.square().sum()
    ref_norm, obs_norm = ref_sq.sqrt(), obs_sq.sqrt()
    if ref_norm.item() == 0:
        return {f"{prefix}_relative_l2": None, f"{prefix}_cosine": None,
                f"{prefix}_norm_ratio": None, f"{prefix}_metric_status": "zero_reference_norm"}
    status = "ok" if obs_norm.item() != 0 else "zero_observed_norm"
    return {f"{prefix}_relative_l2": _number((obs - ref).square().sum().sqrt() / ref_norm),
            f"{prefix}_cosine": _number((ref * obs).sum() / (ref_norm * obs_norm)) if obs_norm.item() else None,
            f"{prefix}_norm_ratio": _number(obs_norm / ref_norm), f"{prefix}_metric_status": status}


@torch.no_grad()
def analyze_momentum(momentum: torch.Tensor, *, quantizer: str,
                     ns_steps: int = 5, ns_coefficients=(3.4445, -4.7750, 2.0315),
                     ns_eps: float = 1e-7) -> tuple[dict, dict]:
    """Analyze one detached momentum tensor using production quantization/transform.

    Returns a row and unrounded reductions used for global aggregation.
    Non-2D tensors are deliberately excluded from update metrics: ReferenceMuon
    only invokes the matrix transform for its explicitly selected 2D group.
    """
    if quantizer not in QUANTIZERS:
        raise ValueError(f"unsupported Muon quantizer: {quantizer}")
    simulation, bits, codebook, block_size = QUANTIZERS[quantizer]
    source = momentum.detach()
    # persist_state returns a new FP32 tensor for selected simulations; clone the
    # source first so future implementation changes cannot make observation alias.
    quantized = persist_state(source.clone(), simulation, "muon_momentum",
                              quantization_granularity="blockwise", quantization_block_size=block_size)
    row = {"quantizer": quantizer, "bits": bits, "codebook": codebook, "codebook_family": codebook, "block_size": block_size,
           "shape": list(source.shape), "raw_zero_fraction": _number((quantized == 0).float().mean()) if source.numel() else None,
           **_ratios(source, quantized, "raw_momentum")}
    reductions = {"raw_ref_sq": source.float().square().sum(), "raw_obs_sq": quantized.float().square().sum(),
                  "raw_error_sq": (quantized.float() - source.float()).square().sum(),
                  "raw_dot": (quantized.float() * source.float()).sum(), "update_eligible": source.ndim == 2}
    if source.ndim != 2:
        row.update({"update_relative_l2": None, "update_cosine": None, "update_norm_ratio": None,
                    "update_metric_status": "excluded_not_2d_muon_matrix"})
        row.update({"muon_update_relative_l2": None, "muon_update_cosine": None, "muon_update_norm_ratio": None})
        return row, reductions
    # Exact production transform; inputs are detached diagnostic copies.
    update_ref = muon_reference.zeropower_newton_schulz(source.clone(), ns_steps, ns_coefficients, ns_eps)
    update_quant = muon_reference.zeropower_newton_schulz(quantized.clone(), ns_steps, ns_coefficients, ns_eps)
    row.update(_ratios(update_ref, update_quant, "update"))
    # The long names make paired raw-vs-update comparisons unambiguous while
    # retaining the concise fields requested by the JSONL schema.
    row.update({f"muon_{key}": row[key] for key in ("update_relative_l2", "update_cosine", "update_norm_ratio")})
    reductions.update({"update_ref_sq": update_ref.float().square().sum(), "update_obs_sq": update_quant.float().square().sum(),
                       "update_error_sq": (update_quant.float() - update_ref.float()).square().sum(),
                       "update_dot": (update_quant.float() * update_ref.float()).sum()})
    return row, reductions


def _aggregate(reductions: list[dict], kind: str) -> dict:
    selected = [r for r in reductions if kind == "raw" or r["update_eligible"]]
    if not selected:
        return {f"{kind}_relative_l2": None, f"{kind}_cosine": None, f"{kind}_norm_ratio": None,
                f"{kind}_metric_status": "no_eligible_tensors"}
    values = {name: torch.stack([r[f"{kind}_{name}"].detach().double() for r in selected]).sum()
              for name in ("ref_sq", "obs_sq", "error_sq", "dot")}
    if not all(torch.isfinite(x) for x in values.values()):
        return {f"{kind}_relative_l2": {"nonfinite": "input_or_transform"}, f"{kind}_cosine": {"nonfinite": "input_or_transform"},
                f"{kind}_norm_ratio": {"nonfinite": "input_or_transform"}, f"{kind}_metric_status": "nonfinite"}
    ref_norm, obs_norm = values["ref_sq"].sqrt(), values["obs_sq"].sqrt()
    if ref_norm.item() == 0:
        return {f"{kind}_relative_l2": None, f"{kind}_cosine": None, f"{kind}_norm_ratio": None,
                f"{kind}_metric_status": "zero_reference_norm"}
    return {f"{kind}_relative_l2": float(values["error_sq"].sqrt() / ref_norm),
            f"{kind}_cosine": float(values["dot"] / (ref_norm * obs_norm)) if obs_norm.item() else None,
            f"{kind}_norm_ratio": float(obs_norm / ref_norm),
            f"{kind}_metric_status": "ok" if obs_norm.item() else "zero_observed_norm"}


@torch.no_grad()
def analyze_tensors(tensors: Iterable[dict], *, quantizers=QUANTIZERS, ns_steps=5,
                    ns_coefficients=(3.4445, -4.7750, 2.0315), ns_eps=1e-7) -> list[dict]:
    """Return tensor rows plus one global-norm aggregate row per quantizer."""
    tensors = list(tensors); records = []
    for identity in quantizers:
        reductions = []
        for item in tensors:
            row, reduction = analyze_momentum(item["tensor"], quantizer=identity, ns_steps=ns_steps,
                                              ns_coefficients=ns_coefficients, ns_eps=ns_eps)
            row.update({"record_type": "tensor", "parameter_id": item.get("parameter_id", item.get("name", "<unknown>")), "state_identifier": "muon_momentum",
                        "parameter_name": item.get("name", "<unknown>")})
            records.append(row); reductions.append(reduction)
        aggregate = {"record_type": "aggregate", "parameter_id": None, "parameter_name": None, "state_identifier": "muon_momentum",
                        "quantizer": identity, "bits": QUANTIZERS[identity][1], "codebook": QUANTIZERS[identity][2], "codebook_family": QUANTIZERS[identity][2],
                        "block_size": QUANTIZERS[identity][3], "tensor_count": len(tensors),
                        "update_eligible_tensor_count": sum(r["update_eligible"] for r in reductions),
                        **_aggregate(reductions, "raw"), **_aggregate(reductions, "update")}
        aggregate.update({f"muon_{key}": aggregate[key] for key in ("update_relative_l2", "update_cosine", "update_norm_ratio")})
        aggregate.update({f"raw_momentum_{key}": aggregate[f"raw_{key}"] for key in ("relative_l2", "cosine", "norm_ratio")})
        records.append(aggregate)
    return records


def save_snapshot(path: str | Path, *, metadata: dict, tensors: Iterable[dict]) -> None:
    """Persist only CPU FP32 momentum copies and provenance; never a checkpoint."""
    values = [{"parameter_id": item["parameter_id"], "state_identifier": "muon_momentum", "name": item.get("name", "<unknown>"),
               "shape": list(item["tensor"].shape), "tensor": item["tensor"].detach().to("cpu", torch.float32).clone()}
              for item in tensors]
    torch.save({"format": "muon_momentum_snapshot", "version": 1, "metadata": metadata, "tensors": values}, path)


def load_snapshot(path: str | Path) -> dict:
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if snapshot.get("format") != "muon_momentum_snapshot" or snapshot.get("version") != 1:
        raise ValueError("not a supported Muon momentum snapshot")
    for item in snapshot["tensors"]:
        if item["tensor"].dtype != torch.float32 or list(item["tensor"].shape) != item["shape"]:
            raise ValueError("snapshot tensor is not preserved FP32 momentum")
    return snapshot


def _copy_momentum_to_cpu(momentum: torch.Tensor) -> torch.Tensor:
    """Make the detached FP32 CPU copy used by the snapshot artifact."""
    return momentum.detach().to("cpu", torch.float32).clone()


class MuonUpdateFidelityObserver:
    """Read-only Muon observer with explicit online and snapshot-only modes.

    In online mode, every update is collected and analyzed.  In snapshot-only
    mode, collection is enabled only at the selected snapshot landmarks; the
    callback is otherwise a no-op.  Keeping this decision at the update
    boundary is important because ``observe`` runs once per Muon parameter
    inside the optimizer step.
    """
    def __init__(self, parameter_names: dict[int, str], *, snapshot_updates=(),
                 quantizers=QUANTIZERS, online_fidelity_enabled: bool = True):
        self.parameter_names = parameter_names
        self.snapshot_updates = set(snapshot_updates)
        self.quantizers = tuple(quantizers)
        self.online_fidelity_enabled = bool(online_fidelity_enabled)
        self.current_update = None
        self._collect_current_update = False
        self.items: list[dict] = []

    def begin_update(self, update: int) -> None:
        self.current_update = update
        self._collect_current_update = self.online_fidelity_enabled or update in self.snapshot_updates
        # A new update must never inherit retained tensors from an incomplete
        # optimizer step.  This is also useful for callers that reuse an
        # observer after an interrupted step.
        self.items.clear()

    @torch.no_grad()
    def observe(self, *, parameter, momentum_pre, momentum_post, updated, group) -> None:
        if not self._collect_current_update:
            return
        # CPU clone gives snapshot/replay an independent tensor identity.  It is
        # also the only retained object; neither momentum_post nor updated is read.
        self.items.append({"parameter_id": self.parameter_names.get(id(parameter), f"parameter:{id(parameter)}"),
                           "name": self.parameter_names.get(id(parameter), "<unknown>"),
                           "tensor": _copy_momentum_to_cpu(momentum_pre)})

    @torch.no_grad()
    def finish_update(self, update: int) -> tuple[list[dict], list[dict]]:
        if self.current_update != update:
            raise RuntimeError("Muon observer update boundary was not set")
        items, self.items = self.items, []
        if not self._collect_current_update:
            return [], []
        # Snapshot-only mode deliberately performs no quantization or Muon
        # transform during training.  Those diagnostics are produced by the
        # existing offline replay path after the run.
        rows = analyze_tensors(items, quantizers=self.quantizers) if self.online_fidelity_enabled else []
        for row in rows:
            row["update"] = update
        return rows, items if update in self.snapshot_updates else []
