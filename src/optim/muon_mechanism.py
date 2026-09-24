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
        interval = self.metadata.get("error_feedback_interval")
        periodic_event = (self.metadata.get("error_feedback_mode") == "periodic" and
                          isinstance(interval, int) and interval > 0 and
                          self._update is not None and self._update % interval == 0)
        return self._update in self.scalar_updates or periodic_event

    @torch.no_grad()
    def observe(self, *, parameter, gradient, momentum_prev, momentum_candidate,
                momentum_persisted_decoded, direction, updated, group,
                momentum_error_prev=None, momentum_error=None,
                error_feedback_alpha=0.0, error_feedback_mode="none",
                error_feedback_interval=None, correction_applied=False,
                error_accumulator_prev=None, injected_correction=None,
                error_accumulator_next=None, **kwargs):
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
        error_prev = (torch.zeros_like(candidate) if momentum_error_prev is None
                      else momentum_error_prev.detach().float())
        current_error = (torch.zeros_like(candidate) if momentum_error is None
                         else momentum_error.detach().float())
        mu = float(group.get("muon_momentum", self.metadata.get("muon_momentum", .95)))
        injected = error_prev * (float(error_feedback_alpha) * mu)
        accumulator_prev = (torch.zeros_like(candidate) if error_accumulator_prev is None
                            else error_accumulator_prev.detach().float())
        injected_value = (injected if injected_correction is None
                          else injected_correction.detach().float())
        accumulator_next = (torch.zeros_like(candidate) if error_accumulator_next is None
                            else error_accumulator_next.detach().float())
        reconstructed_prev = pre + error_prev
        prior_candidate = (self._previous_candidates.get(pid) if pid in self._previous_candidates
                           else torch.zeros_like(candidate))
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
            "error_prev_sq": float(error_prev.square().sum().item()),
            "error_current_sq": float(current_error.square().sum().item()),
            "accumulator_prev_sq": float(accumulator_prev.square().sum().item()),
            "accumulator_next_sq": float(accumulator_next.square().sum().item()),
            "correction_applied": bool(correction_applied),
            "error_feedback_interval": error_feedback_interval,
            "error_feedback_mode": error_feedback_mode,
            "injected_actual_sq": float(injected_value.square().sum().item()),
            "mu_error_prev_sq": float((mu * error_prev).square().sum().item()),
            "injected_sq": float(injected.square().sum().item()),
            "injected_grad_dot": float((injected * grad).sum().item()),
            "gradient_sq": float(grad.square().sum().item()),
            "reconstructed_prev_sq": float(reconstructed_prev.square().sum().item()),
            "prior_candidate_sq": float(prior_candidate.square().sum().item()),
            "reconstructed_prior_dot": float((reconstructed_prev * prior_candidate).sum().item()),
            "reconstructed_prior_diff_sq": float((reconstructed_prev - prior_candidate).square().sum().item()),
            "reconstructed_prior_max_abs": float((reconstructed_prev - prior_candidate).abs().max().item()) if candidate.numel() else 0.0,
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
        e_prev2 = sum(x["error_prev_sq"] for x in items)
        e2 = sum(x["error_current_sq"] for x in items)
        accumulator_prev2 = sum(x["accumulator_prev_sq"] for x in items)
        accumulator_next2 = sum(x["accumulator_next_sq"] for x in items)
        injected_actual2 = sum(x["injected_actual_sq"] for x in items)
        mu_e2 = sum(x["mu_error_prev_sq"] for x in items)
        injected2 = sum(x["injected_sq"] for x in items)
        injected_dot = sum(x["injected_grad_dot"] for x in items)
        gradient2 = sum(x["gradient_sq"] for x in items)
        rec2 = sum(x["reconstructed_prev_sq"] for x in items)
        prior2 = sum(x["prior_candidate_sq"] for x in items)
        recdot = sum(x["reconstructed_prior_dot"] for x in items)
        recdiff2 = sum(x["reconstructed_prior_diff_sq"] for x in items)
        row = {
            "schema_version": self.schema_version, "update": int(update),
            "processed_target_tokens": int(processed_target_tokens),
            "optimizer_name": self.optimizer_name,
            "train_nll": float(train_nll), "learning_rate": float(learning_rate),
            "tensor_count": len(items),
            "persistence_quantization_relative_l2": math.sqrt(q2 / max(c2, 1e-30)),
            "persistence_quantization_cosine": pdot / max(math.sqrt(p2 * c2), 1e-30) if p2 and c2 else 1.0,
            "persistence_norm_ratio": math.sqrt(p2 / max(c2, 1e-30)),
            "error_feedback_alpha": float(self.metadata.get("error_feedback_alpha", 0.0)),
            "error_feedback_mode": self.metadata.get("error_feedback_mode", "none"),
            "error_feedback_interval": self.metadata.get("error_feedback_interval"),
            "correction_applied": any(x["correction_applied"] for x in items),
            "periodic_accumulator_norm_ratio": math.sqrt(accumulator_prev2 / max(c2, 1e-30)),
            "periodic_accumulator_next_norm_ratio": math.sqrt(accumulator_next2 / max(c2, 1e-30)),
            "periodic_injected_correction_norm_ratio": math.sqrt(injected_actual2 / max(c2, 1e-30)),
            "error_buffer_norm_ratio": math.sqrt(e2 / max(c2, 1e-30)),
            "previous_error_buffer_norm_ratio": math.sqrt(e_prev2 / max(c2, 1e-30)),
            "mu_error_prev_norm_ratio": math.sqrt(mu_e2 / max(c2, 1e-30)),
            "injected_correction_norm_ratio": math.sqrt(injected2 / max(c2, 1e-30)),
            "injected_correction_gradient_cosine": injected_dot / max(math.sqrt(injected2 * gradient2), 1e-30) if injected2 else 1.0,
            "previous_candidate_reconstruction_cosine": recdot / max(math.sqrt(rec2 * prior2), 1e-30) if rec2 and prior2 else 1.0,
            "previous_candidate_reconstruction_relative_l2": math.sqrt(recdiff2 / max(prior2, 1e-30)),
            "previous_candidate_reconstruction_max_abs": max(x["reconstructed_prior_max_abs"] for x in items),
            "raw_snapshot": self._update in self.raw_updates,
        }
        # At raw landmarks calculate the true one-step local K5 effect while
        # holding the current gradient fixed.  This is not a trajectory.
        if self._update in self.raw_updates:
            local_dot = local_a2 = local_b2 = local_d2 = 0.0
            for x in items:
                g = x["gradient"]; prev = x["momentum_prev_decoded"]
                actual = x["momentum_candidate"]; mu = float(self.metadata["muon_momentum"])
                d_actual = g + mu * actual
                m_no_prev_error = mu * prev + g
                d_no_prev = g + mu * m_no_prev_error
                oa = zeropower_newton_schulz(d_actual, self.metadata["muon_ns_steps"], self.metadata["muon_ns_coefficients"], self.metadata["muon_eps"])
                ob = zeropower_newton_schulz(d_no_prev, self.metadata["muon_ns_steps"], self.metadata["muon_ns_coefficients"], self.metadata["muon_eps"])
                local_dot += float((oa * ob).sum()); local_a2 += float(oa.square().sum()); local_b2 += float(ob.square().sum()); local_d2 += float((oa-ob).square().sum())
            row.update({"local_one_step_k5_cosine": local_dot / max(math.sqrt(local_a2 * local_b2), 1e-30),
                        "local_one_step_k5_relative_l2": math.sqrt(local_d2 / max(local_a2, 1e-30))})
            raw_fields = ["gradient", "momentum_prev_decoded", "momentum_prev_candidate", "momentum_candidate"]
            if self.optimizer_name == "recursive_muon":
                # The error buffer is exactly candidate - decoded persistence;
                # the raw snapshot already stores the two operands, avoiding
                # another full FP32 copy per tensor and landmark.
                raw_fields.append("momentum_persisted_decoded")
            payload = {"format": "recursive_muon_mechanism_snapshot", "version": 1,
                       "metadata": {**self.metadata, "update": int(update), "processed_target_tokens": int(processed_target_tokens),
                                    "optimizer_name": self.optimizer_name, "tensor_count": len(items),
                                    "raw_fields": raw_fields, "persisted_equals_candidate": self.optimizer_name == "reference_muon"},
                       "tensors": [{k: v for k, v in x.items() if k in {"parameter_id", "name", "shape", *raw_fields}} for x in items]}
            torch.save(payload, self.mechanism_dir / f"update_{int(update):06d}.pt")
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
