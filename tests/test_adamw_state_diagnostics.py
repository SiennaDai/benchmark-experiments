import copy, json
import pytest
import torch

from config.recipe import load_recipe
from benchmark_suite import load_suite
from optim.adamw_reference import ReferenceAdamW
from optim.adamw_state_diagnostics import AdamWStateDiagnostics, MuonStateDiagnostics, _evenly_spaced_indices, _sample
from optim.muon_reference import ReferenceMuon


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


def test_blockwise_diagnostics_report_per_block_scales_without_mutation():
    p = torch.nn.Parameter(torch.tensor([1., -2., 3., -4., .5]))
    opt = ReferenceAdamW([p], lr=.1, betas=(0., .5), state_simulation="int8_linear_second_moment",
                         state_quantization_granularity="blockwise", state_quantization_block_size=2)
    collector = AdamWStateDiagnostics({id(p): "p"}, quantization_granularity="blockwise", quantization_block_size=2)
    opt.set_diagnostic_observer(collector.observe)
    _step(opt, p, torch.tensor([.1, .7, 2., .3, .5]))
    before = {key: value.clone() for key, value in opt.state[p].items() if torch.is_tensor(value)}
    result = collector.finish_update(update=1, processed_target_tokens=1, train_nll=1., pre_clip_grad_norm=1., learning_rate=.1)
    stats = result["second_moment"]
    assert stats["scale_min"] is not None and stats["scale_max"] is not None
    for key, value in before.items():
        assert torch.equal(value, opt.state[p][key])


def test_dynamic_diagnostics_report_block_absmax_scales_and_occupancy_without_mutation():
    p = torch.nn.Parameter(torch.tensor([1., -2., 3., -4., .5]))
    opt = ReferenceAdamW([p], lr=.1, betas=(0., .5), state_simulation="int8_dynamic_all_moments",
                         state_quantization_granularity="blockwise", state_quantization_block_size=2)
    collector = AdamWStateDiagnostics({id(p): "p"}, quantization_granularity="blockwise",
                                      quantization_block_size=2, state_simulation="int8_dynamic_all_moments")
    opt.set_diagnostic_observer(collector.observe)
    _step(opt, p, torch.tensor([.1, .7, 2., .3, .5]))
    before = {key: value.clone() for key, value in opt.state[p].items() if torch.is_tensor(value)}
    result = collector.finish_update(update=1, processed_target_tokens=1, train_nll=1., pre_clip_grad_norm=1., learning_rate=.1)
    occupancy = result["second_moment"]["dynamic_codebook_occupancy"]
    assert result["second_moment"]["scale_min"] is not None
    assert occupancy["codebook_occupancy_sample_elements"] > 0
    assert 0 <= occupancy["fraction_mapped_to_zero_code"] <= 1
    for key, value in before.items():
        assert torch.equal(value, opt.state[p][key])


def test_dynamic_v_only_diagnostics_report_unsigned_second_moment_occupancy_without_mutation():
    p = torch.nn.Parameter(torch.tensor([1., -2., 3., -4., .5]))
    opt = ReferenceAdamW([p], lr=.1, betas=(0., .5), state_simulation="int8_dynamic_second_moment",
                         state_quantization_granularity="blockwise", state_quantization_block_size=2)
    collector = AdamWStateDiagnostics({id(p): "p"}, quantization_granularity="blockwise",
                                      quantization_block_size=2, state_simulation="int8_dynamic_second_moment")
    opt.set_diagnostic_observer(collector.observe)
    _step(opt, p, torch.tensor([.1, .7, 2., .3, .5]))
    before = {key: value.clone() for key, value in opt.state[p].items() if torch.is_tensor(value)}
    result = collector.finish_update(update=1, processed_target_tokens=1, train_nll=1., pre_clip_grad_norm=1., learning_rate=.1)
    occupancy = result["second_moment"]["dynamic_codebook_occupancy"]
    assert result["second_moment"]["scale_min"] is not None
    assert occupancy["codebook_occupancy_sample_elements"] > 0
    assert 0 <= occupancy["fraction_mapped_to_zero_code"] <= 1
    for key, value in before.items():
        assert torch.equal(value, opt.state[p][key])


def test_muon_linear_and_dynamic_diagnostics_share_read_only_momentum_schema():
    for simulation, bits in (("int8_linear_momentum", 8), ("int4_dynamic_momentum", 4)):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = ReferenceMuon([{"params": [parameter], "optimizer_group": "muon", "weight_decay": 0.}],
                                  lr=.1, state_simulation=simulation,
                                  state_quantization_granularity="blockwise", state_quantization_block_size=2)
        collector = MuonStateDiagnostics(quantization_block_size=2, quantization_bits=bits,
                                         state_simulation=simulation)
        optimizer.set_diagnostic_observer(collector.observe)
        _step(optimizer, parameter, torch.tensor([[.3, -.5], [.1, .2]]))
        before = optimizer.state[parameter]["muon_momentum"].clone()
        result = collector.finish_update(update=1, processed_target_tokens=1, train_nll=1.,
                                         pre_clip_grad_norm=1., learning_rate=.1)
        state = result["muon_momentum"]
        assert state["post_quant_momentum_l2_norm"] >= 0
        assert 0 <= state["cosine_similarity_pre_post"] <= 1
        assert state["block_scale_min"] is not None
        assert state["codebook_occupancy"]["representable_levels"] == (255 if bits == 8 else 16)
        assert 0 <= state["codebook_occupancy"]["fraction_mapped_to_zero_code"] <= 1
        assert torch.equal(before, optimizer.state[parameter]["muon_momentum"])


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


@pytest.mark.parametrize("n,limit", [
    (0, 1024), (7, 1024), (1024, 1024), (1025, 1024),
    (2**24 + 123, 1024), (2**24 + 123, 1),
])
def test_evenly_spaced_indices_are_integer_bounded_and_endpoint_preserving(n, limit):
    # Exercise n > float32's exact-integer range without allocating a tensor
    # of that size; the helper is the complete index-construction path.
    idx = _evenly_spaced_indices(n, limit, "cpu")
    assert idx.dtype == torch.int64
    assert idx.numel() == min(n, limit)
    if n:
        assert idx.min().item() >= 0
        assert idx.max().item() < n
        assert idx[0].item() == 0
        if limit > 1:
            assert idx[-1].item() == n - 1
        assert torch.all(idx[1:] >= idx[:-1])
    assert torch.equal(idx, _evenly_spaced_indices(n, limit, "cpu"))


def test_sample_preserves_small_and_boundary_tensor_endpoints():
    for n, limit in ((7, 1024), (1024, 1024), (1025, 1024), (9, 1)):
        value = torch.arange(n)
        sampled = _sample(value, limit)
        assert sampled[0].item() == 0
        if limit > 1:
            assert sampled[-1].item() == n - 1
        assert sampled.numel() == min(n, limit)


def test_evenly_spaced_indices_reject_invalid_limit():
    with pytest.raises(ValueError, match="positive"):
        _evenly_spaced_indices(4, 0, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_evenly_spaced_indices_match_cuda_without_float_indexing():
    n, limit = 2**24 + 123, 1024
    cpu = _evenly_spaced_indices(n, limit, "cpu")
    cuda = _evenly_spaced_indices(n, limit, "cuda").cpu()
    assert torch.equal(cpu, cuda)
