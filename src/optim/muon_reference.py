"""Auditable FP32 reference Muon with an FP32 auxiliary AdamW group.

The Muon rule follows Keller Jordan's reference `muon.py` and PyTorch's
``torch.optim.Muon`` defaults, except that this numerical benchmark deliberately
performs orthogonalization in FP32 rather than using a BF16 performance kernel.
"""
import torch


def zeropower_newton_schulz(matrix, steps=5, coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7):
    """Approximate polar factor using X <- aX + (b A + c A²) X, A=X Xᵀ."""
    if matrix.ndim != 2:
        raise ValueError("Muon only orthogonalizes 2D matrices")
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.float().T if transposed else matrix.float()
    x = x / (x.norm() + eps)
    a, b, c = coefficients
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


class ReferenceMuon(torch.optim.Optimizer):
    """Mixed optimizer: Muon hidden matrices; auxiliary parameters use AdamW.

    Momentum: B_t=mu B_(t-1)+g_t.  The Nesterov direction is g_t+mu B_t.
    Decoupled weight decay is applied before the orthogonalized update.  BF16
    simulation only roundtrips ``muon_momentum`` after that update; auxiliary
    AdamW moments always remain FP32.
    """
    def __init__(self, params, lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=.1,
                 state_simulation="none", muon_momentum=.95, muon_nesterov=True,
                 muon_ns_steps=5, muon_ns_coefficients=(3.4445, -4.7750, 2.0315), muon_eps=1e-7):
        if state_simulation not in {"none", "bf16_roundtrip"}: raise ValueError("unsupported state simulation")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        state_simulation=state_simulation, muon_momentum=muon_momentum,
                        muon_nesterov=muon_nesterov, muon_ns_steps=muon_ns_steps,
                        muon_ns_coefficients=tuple(muon_ns_coefficients), muon_eps=muon_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            is_muon = group.get("optimizer_group") == "muon"
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.float()
                state = self.state[p]
                if is_muon:
                    momentum = state.get("muon_momentum", torch.zeros_like(p, dtype=torch.float32))
                    momentum = momentum.mul(group["muon_momentum"]).add(grad)
                    direction = grad.add(momentum, alpha=group["muon_momentum"]) if group["muon_nesterov"] else momentum
                    update = zeropower_newton_schulz(direction, group["muon_ns_steps"], group["muon_ns_coefficients"], group["muon_eps"])
                    p.copy_(p.float().mul(1 - group["lr"] * group["weight_decay"]).add(update, alpha=-group["lr"]).to(p.dtype))
                    state["muon_momentum"] = momentum.to(torch.bfloat16).float() if group["state_simulation"] == "bf16_roundtrip" else momentum
                else:
                    step = state.get("step", 0) + 1; state["step"] = step
                    m = state.get("exp_avg", torch.zeros_like(p, dtype=torch.float32)).mul(group["betas"][0]).add(grad, alpha=1-group["betas"][0])
                    v = state.get("exp_avg_sq", torch.zeros_like(p, dtype=torch.float32)).mul(group["betas"][1]).addcmul(grad, grad, value=1-group["betas"][1])
                    p.copy_(p.float().mul(1-group["lr"]*group["weight_decay"]).addcdiv(m/(1-group["betas"][0]**step), (v/(1-group["betas"][1]**step)).sqrt().add(group["eps"]), value=-group["lr"]).to(p.dtype))
                    state["exp_avg"], state["exp_avg_sq"] = m, v
        return loss
