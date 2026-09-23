"""Optional read-only mechanism observer for paired recursive Muon runs.

The observer is deliberately outside optimizer state.  It records only at
declared landmarks and writes raw FP32 CPU tensors; disabling it leaves the
optimizer recurrence untouched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .muon_reference import zeropower_newton_schulz


def _cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", torch.float32).clone()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float().reshape(-1) @ b.float().reshape(-1)) /
                 (a.float().norm() * b.float().norm()).clamp_min(1e-30))


class MuonMechanismObserver:
    """Collect persistence/local diagnostics without entering optimizer state."""

    schema_version = 1

    def __init__(self, *, run_dir: str | Path, optimizer_name: str,
                 parameter_names: dict[int, str], raw_updates, scalar_updates,
                 metadata: dict):
        self.run_dir = Path(run_dir)
        self.optimizer_name = optimizer_name
        self.parameter_names = parameter_names
        self.raw_updates = set(int(x) for x in raw_updates)
        self.scalar_updates = set(int(x) for x in scalar_updates) | self.raw_updates
        self.metadata = dict(metadata)
        self.mechanism_dir = self.run_dir / "mechanism"
        self.mechanism_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.mechanism_dir / "metrics.jsonl"
        self._items = []
        self._previous_candidates: dict[int, torch.Tensor] = {}
        self._update = None

    def begin_update(self, update: int) -> None:
        self._update = int(update)
        self._items = []

    @property
    def collecting(self) -> bool:
        return self._update in self.scalar_updates

    @torch.no_grad()
    def observe(self, *, parameter, gradient, momentum_prev, momentum_candidate,
                momentum_persisted_decoded, direction, updated, group, **kwargs):
        if not self.collecting:
            # Still retain the previous unquantized candidate for the next
            # update's local counterfactual; this is observer-only transient
            # state and is never placed in a checkpoint.
            self._previous_candidates[id(parameter)] = momentum_candidate.detach().float().clone()
            return
        pid = id(parameter)
        previous_candidate = self._previous_candidates.get(pid)
        if previous_candidate is None:
            previous_candidate = torch.zeros_like(momentum_candidate, dtype=torch.float32)
        pre = momentum_prev.detach().float()
        candidate = momentum_candidate.detach().float()
        persisted = momentum_persisted_decoded.detach().float()
        grad = gradient.detach().float()
        q = persisted - candidate
        self._items.append({
            "parameter_id": self.parameter_names.get(pid, f"parameter:{pid}"),
            "name": self.parameter_names.get(pid, "<unknown>"),
            "shape": list(candidate.shape),
            "gradient": _cpu(grad) if self._update in self.raw_updates else None,
            "momentum_prev_decoded": _cpu(pre) if self._update in self.raw_updates else None,
            "momentum_prev_candidate": _cpu(previous_candidate) if self._update in self.raw_updates else None,
            "momentum_candidate": _cpu(candidate) if self._update in self.raw_updates else None,
            "momentum_persisted_decoded": _cpu(persisted) if self._update in self.raw_updates else None,
            "q_sq": float(q.square().sum().item()),
            "candidate_sq": float(candidate.square().sum().item()),
            "persisted_sq": float(persisted.square().sum().item()),
            "persisted_dot": float((persisted * candidate).sum().item()),
        })
        self._previous_candidates[pid] = candidate.clone()

    @torch.no_grad()
    def finish_update(self, *, update: int, processed_target_tokens: int,
                      train_nll: float, learning_rate: float) -> None:
        if self._update != int(update):
            raise RuntimeError("mechanism observer update boundary mismatch")
        items, self._items = self._items, []
        if not items:
            return
        q2 = sum(x["q_sq"] for x in items); c2 = sum(x["candidate_sq"] for x in items)
        p2 = sum(x["persisted_sq"] for x in items); pdot = sum(x["persisted_dot"] for x in items)
        row = {
            "schema_version": self.schema_version, "update": int(update),
            "processed_target_tokens": int(processed_target_tokens),
            "optimizer_name": self.optimizer_name,
            "train_nll": float(train_nll), "learning_rate": float(learning_rate),
            "tensor_count": len(items),
            "persistence_quantization_relative_l2": math.sqrt(q2 / max(c2, 1e-30)),
            "persistence_quantization_cosine": pdot / max(math.sqrt(p2 * c2), 1e-30) if p2 and c2 else 1.0,
            "persistence_norm_ratio": math.sqrt(p2 / max(c2, 1e-30)),
            "raw_snapshot": self._update in self.raw_updates,
        }
        # At raw landmarks calculate the true one-step local K5 effect while
        # holding the current gradient fixed.  This is not a trajectory.
        if self._update in self.raw_updates:
            local_dot = local_a2 = local_b2 = local_d2 = 0.0
            for x in items:
                g = x["gradient"]; prev = x["momentum_prev_decoded"]; prev_c = x["momentum_prev_candidate"]
                actual = x["momentum_candidate"]; mu = float(self.metadata["muon_momentum"])
                d_actual = g + mu * actual
                d_no_prev = g + mu * (mu * prev_c + g)
                oa = zeropower_newton_schulz(d_actual, self.metadata["muon_ns_steps"], self.metadata["muon_ns_coefficients"], self.metadata["muon_eps"])
                ob = zeropower_newton_schulz(d_no_prev, self.metadata["muon_ns_steps"], self.metadata["muon_ns_coefficients"], self.metadata["muon_eps"])
                local_dot += float((oa * ob).sum()); local_a2 += float(oa.square().sum()); local_b2 += float(ob.square().sum()); local_d2 += float((oa-ob).square().sum())
            row.update({"local_one_step_k5_cosine": local_dot / max(math.sqrt(local_a2 * local_b2), 1e-30),
                        "local_one_step_k5_relative_l2": math.sqrt(local_d2 / max(local_a2, 1e-30))})
            raw_fields = ["gradient", "momentum_prev_decoded", "momentum_prev_candidate", "momentum_candidate"]
            if self.optimizer_name == "recursive_muon":
                raw_fields.append("momentum_persisted_decoded")
            payload = {"format": "recursive_muon_mechanism_snapshot", "version": 1,
                       "metadata": {**self.metadata, "update": int(update), "processed_target_tokens": int(processed_target_tokens),
                                    "optimizer_name": self.optimizer_name, "tensor_count": len(items),
                                    "raw_fields": raw_fields, "persisted_equals_candidate": self.optimizer_name == "reference_muon"},
                       "tensors": [{k: v for k, v in x.items() if k in {"parameter_id", "name", "shape", *raw_fields}} for x in items]}
            torch.save(payload, self.mechanism_dir / f"update_{int(update):06d}.pt")
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
