import torch

from optim.muon_reference import ReferenceMuon, zeropower_newton_schulz
from train_platform import parameter_groups


def test_muon_one_step_matches_independent_fp32_calculation():
    p = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]])); p.grad = torch.tensor([[.2, -.1], [.3, .4]])
    before, grad = p.detach().clone(), p.grad.clone()
    opt = ReferenceMuon([{"params": [p], "optimizer_group": "muon", "weight_decay": .1}], lr=.01, muon_momentum=.95, muon_nesterov=True)
    expected = before * (1 - .01 * .1) - .01 * zeropower_newton_schulz(grad + .95 * grad)
    opt.step()
    assert torch.allclose(p, expected)


def test_muon_roundtrip_only_changes_persisted_muon_momentum():
    m = torch.nn.Parameter(torch.ones(2, 2)); a = torch.nn.Parameter(torch.ones(2))
    m.grad = torch.tensor([[.1234567, .2222222], [.3333333, .4444444]]); a.grad = torch.tensor([.2, -.1])
    opt = ReferenceMuon([{"params": [m], "optimizer_group": "muon", "weight_decay": 0.}, {"params": [a], "optimizer_group": "auxiliary_adamw", "weight_decay": 0.}], state_simulation="bf16_roundtrip")
    opt.step()
    assert torch.equal(opt.state[m]["muon_momentum"], m.grad.to(torch.bfloat16).float())
    assert opt.state[a]["exp_avg"].dtype == torch.float32 and opt.state[a]["exp_avg_sq"].dtype == torch.float32


def test_muon_grouping_hidden_only_and_tied_tensor_is_unique(diagnostic_config):
    from train_platform import build_model
    model = build_model(diagnostic_config, torch.device("cpu"))
    groups, records = parameter_groups(model, .1, "reference_muon")
    assert all(r["group"] == "muon" for r in records if r["name"].startswith("transformer.h.") and len(r["shape"]) == 2)
    assert all(r["group"] == "auxiliary_adamw" for r in records if not r["name"].startswith("transformer.h."))
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
