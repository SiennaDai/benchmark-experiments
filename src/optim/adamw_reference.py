"""Readable FP32 AdamW with an optional persisted-state BF16 simulation."""

import torch


class ReferenceAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, state_simulation="none"):
        if state_simulation not in {"none", "bf16_roundtrip"}:
            raise ValueError("unsupported state simulation")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, state_simulation=state_simulation))

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
                p.copy_(updated.to(p.dtype))
                if group["state_simulation"] == "bf16_roundtrip":
                    state["exp_avg"] = m.to(torch.bfloat16).float()
                    state["exp_avg_sq"] = v.to(torch.bfloat16).float()
                else:
                    state["exp_avg"] = m
                    state["exp_avg_sq"] = v
        return loss
