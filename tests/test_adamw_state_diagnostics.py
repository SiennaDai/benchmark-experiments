import copy, json
import torch

from config.recipe import load_recipe
from benchmark_suite import load_suite
from optim.adamw_reference import ReferenceAdamW
from optim.adamw_state_diagnostics import AdamWStateDiagnostics


def _step(opt, p, grad):
    p.grad = grad.clone(); opt.step(); opt.zero_grad(set_to_none=True)


def test_state_diagnostics_are_read_only_and_nonperturbing():
    torch.manual_seed(3); a=torch.nn.Parameter(torch.randn(5)); b=torch.nn.Parameter(a.detach().clone())
    plain=ReferenceAdamW([a],lr=.01,betas=(.9,.95)); observed=ReferenceAdamW([b],lr=.01,betas=(.9,.95),state_simulation="int8_linear_second_moment")
    collector=AdamWStateDiagnostics({id(b): "p"}); observed.set_diagnostic_observer(collector.observe)
    for update in range(1,4):
        grad=torch.randn(5); _step(plain,a,grad); _step(observed,b,grad)
        result=collector.finish_update(update=update,processed_target_tokens=update,train_nll=1.,pre_clip_grad_norm=1.,learning_rate=.01)
        assert result["second_moment"]["number_of_state_elements"] == 5
    # Compare an observed optimizer to an identical unobserved one: observer only
    # uses detached reductions and never changes numerical optimizer behavior.
    c=torch.nn.Parameter(torch.randn(5)); d=torch.nn.Parameter(c.detach().clone()); x=ReferenceAdamW([c],lr=.01); y=ReferenceAdamW([d],lr=.01); y.set_diagnostic_observer(AdamWStateDiagnostics({id(d):"d"}).observe)
    for _ in range(3):
        g=torch.randn(5); _step(x,c,g); _step(y,d,g)
    assert torch.equal(c,d)
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(x.state[c][key], y.state[d][key])
    before = torch.get_rng_state().clone()
    collector.finish_update(update=99,processed_target_tokens=99,train_nll=1.,pre_clip_grad_norm=1.,learning_rate=.01)
    assert torch.equal(before, torch.get_rng_state())


def test_quantized_v_diagnostics_and_fp32_identity():
    p=torch.nn.Parameter(torch.tensor([1.,-2.,3.])); opt=ReferenceAdamW([p],lr=.1,betas=(0.,.5),state_simulation="int8_linear_second_moment")
    c=AdamWStateDiagnostics({id(p):"p"});opt.set_diagnostic_observer(c.observe); _step(opt,p,torch.tensor([.1,.7,2.]))
    r=c.finish_update(update=1,processed_target_tokens=1,train_nll=1.,pre_clip_grad_norm=1.,learning_rate=.1)["second_moment"]
    assert r["global_relative_l2_quantization_error"] > 0 and r["post_quant_zero_fraction"] >= r["pre_quant_zero_fraction"]
    p=torch.nn.Parameter(torch.tensor([1.,-2.]));opt=ReferenceAdamW([p],lr=.1);c=AdamWStateDiagnostics({id(p):"p"});opt.set_diagnostic_observer(c.observe);_step(opt,p,torch.tensor([.3,.5]))
    assert c.finish_update(update=1,processed_target_tokens=1,train_nll=1.,pre_clip_grad_norm=1.,learning_rate=.1)["second_moment"]["global_relative_l2_quantization_error"] == 0.


def test_denominator_and_zero_edge_cases_are_json_safe():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));o=ReferenceAdamW([p],lr=.1);c=AdamWStateDiagnostics({id(p):"p"});o.set_diagnostic_observer(c.observe);_step(o,p,torch.zeros(2))
    r=c.finish_update(update=1,processed_target_tokens=1,train_nll=1.,pre_clip_grad_norm=1.,learning_rate=.1)
    assert r["second_moment"]["pre_quant_min_positive"] is None
    json.dumps(r, allow_nan=False)


def test_diagnostic_recipes_have_short_stop_and_long_schedule():
    configs=[load_recipe(f"recipes/diagnostic_adamw_{kind}_state_4xprefix_s0.json") for kind in ("fp32","int8_m","int8_v")]
    assert all(c["derived"]["total_updates"] == 150 and c["derived"]["schedule_total_updates"] == 4096 for c in configs)
    base=copy.deepcopy(configs[0]); base["experiment"]["name"] = configs[1]["experiment"]["name"]
    assert {k for k in base["optimizer"] if base["optimizer"][k] != configs[1]["optimizer"][k]} == {"state_simulation"}
    suite = load_suite("benchmarks/diagnostic_adamw_int8_mechanism_4xprefix_s0_v1.json")
    assert len(suite["runs"]) == 3 and suite["vary"] == ["optimizer.state_simulation"]
