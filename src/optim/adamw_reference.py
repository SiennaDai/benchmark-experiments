"""Readable FP32 AdamW with an optional persisted-state BF16 simulation."""

import torch

from .state_simulation import STATE_SIMULATIONS, persist_state


class ReferenceAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, state_simulation="none",
                 state_quantization_granularity="per_state_tensor", state_quantization_block_size=2048):
        if state_simulation not in STATE_SIMULATIONS:
            raise ValueError("unsupported state simulation")
        if state_simulation == "int8_linear_momentum":
            raise ValueError("int8_linear_momentum is only valid for ReferenceMuon")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, state_simulation=state_simulation,
                                     state_quantization_granularity=state_quantization_granularity,
                                     state_quantization_block_size=state_quantization_block_size))
        self._diagnostic_observer = None

    def set_diagnostic_observer(self, observer):
        """Install a read-only observer, or ``None`` to disable diagnostics.

        The observer is intentionally outside the state dict and may only read
        detached tensors supplied by ``step``.  This keeps diagnostics opt-in
        and preserves ordinary optimizer/checkpoint semantics.
        """
        self._diagnostic_observer = observer

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("ReferenceAdamW only supports dense gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                g = p.grad.float()
                m = state["exp_avg"].mul(b1).add(g, alpha=1-b1)
                v = state["exp_avg_sq"].mul(b2).addcmul(g, g, value=1-b2)
                t = state["step"]
                m_hat = m / (1-b1**t)
                v_hat = v / (1-b2**t)
                updated = p.float().mul(1-group["lr"]*group["weight_decay"]).addcdiv(m_hat, v_hat.sqrt().add(group["eps"]), value=-group["lr"])
                # Current update uses m/v above.  Simulate lower-precision
                # persistence only after it, for the next optimizer step.
                # Old checkpoints predate these optional group keys.  Their
                # historical behavior is the per-state-tensor default.
                persistence = dict(quantization_granularity=group.get("state_quantization_granularity", "per_state_tensor"),
                                   quantization_block_size=group.get("state_quantization_block_size", 2048))
                persisted_m = persist_state(m, group["state_simulation"], "exp_avg", **persistence)
                persisted_v = persist_state(v, group["state_simulation"], "exp_avg_sq", **persistence)
                if self._diagnostic_observer is not None:
                    self._diagnostic_observer(parameter=p, exp_avg_pre=m, exp_avg_post=persisted_m,
                        exp_avg_sq_pre=v, exp_avg_sq_post=persisted_v, updated=updated, group=group, step=t)
                p.copy_(updated.to(p.dtype))
                state["exp_avg"] = persisted_m
                state["exp_avg_sq"] = persisted_v
        return loss
