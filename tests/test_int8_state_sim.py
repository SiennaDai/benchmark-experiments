from pathlib import Path

import torch

from benchmark_suite import load_suite
from config.recipe import load_recipe
from optim.adamw_reference import ReferenceAdamW
from optim.muon_reference import ReferenceMuon
from optim.state_simulation import int8_blockwise_linear_roundtrip, int8_linear_roundtrip, persistence_metadata
from reporting import GENERATED_FIELDS, IDENTITY_FIELDS, flatten, scientific_differences


ROOT = Path(__file__).resolve().parents[1]


def test_int8_linear_quantizer_is_exact_deterministic_and_fp32():
    value = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    actual = int8_linear_roundtrip(value)
    expected = torch.tensor([-2.0, -2.0 * 64 / 127, 0.0, 2.0 * 64 / 127, 2.0])
    assert actual.dtype == torch.float32
    assert torch.equal(actual, int8_linear_roundtrip(value))
    assert torch.allclose(actual, expected)
    assert actual[0] == -2 and actual[-1] == 2
    scale = value.abs().max().float() / 127
    assert torch.equal(actual[[0, -1]] / scale, torch.tensor([-127.0, 127.0]))


def test_int8_linear_quantizer_handles_zero_and_scales_each_tensor_independently():
    zero = torch.zeros(3, dtype=torch.float64)
    assert torch.equal(int8_linear_roundtrip(zero), torch.zeros(3, dtype=torch.float32))
    small, large = torch.tensor([-.5, .25]), torch.tensor([-8.0, 4.0])
    assert int8_linear_roundtrip(small)[0] == small[0]
    assert int8_linear_roundtrip(large)[0] == large[0]
    # If a global scale had been used, 0.25 in the small tensor would round to zero.
    assert int8_linear_roundtrip(small)[1] != 0


def test_blockwise_int8_partitions_independently_without_crossing_tensor_boundaries():
    # Two full blocks and a partial block each have an independent max-abs scale.
    value = torch.tensor([127., 1., 10., 5., 4.], dtype=torch.float64)
    actual = int8_blockwise_linear_roundtrip(value, block_size=2)
    expected = torch.tensor([127., 1., 10., 64. * 10. / 127., 4.], dtype=torch.float32)
    assert actual.dtype == torch.float32 and actual.shape == value.shape
    assert torch.equal(actual, expected)
    # A second state tensor cannot share the first tensor's scale.
    other = torch.tensor([.5, .25], dtype=torch.float32)
    assert int8_blockwise_linear_roundtrip(other, 2048)[1] != 0


@torch.no_grad()
def test_blockwise_int8_boundaries_zero_blocks_and_nearest_rounding():
    assert torch.equal(int8_blockwise_linear_roundtrip(torch.zeros(2050), 2048), torch.zeros(2050))
    value = torch.tensor([-2., -1., 0., 1., 2.])
    actual = int8_blockwise_linear_roundtrip(value, 5)
    expected = torch.tensor([-2., -2. * 64 / 127, 0., 2. * 64 / 127, 2.])
    assert torch.allclose(actual, expected)
    assert torch.equal(actual, int8_blockwise_linear_roundtrip(value, 5))


def _adam_pair(simulation):
    control = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    treatment = torch.nn.Parameter(control.detach().clone())
    plain = ReferenceAdamW([control], lr=.1, betas=(.5, .5), eps=1e-8)
    simulated = ReferenceAdamW([treatment], lr=.1, betas=(.5, .5), eps=1e-8, state_simulation=simulation)
    gradient = torch.tensor([.37, -.11])
    control.grad, treatment.grad = gradient.clone(), gradient.clone()
    plain.step(); simulated.step()
    return control, treatment, plain, simulated, gradient


def test_adam_int8_selection_and_current_update_timing():
    for simulation, changed in (("int8_linear_first_moment", {"exp_avg"}),
                                ("int8_linear_second_moment", {"exp_avg_sq"}),
                                ("int8_linear_all_moments", {"exp_avg", "exp_avg_sq"})):
        control, treatment, plain, simulated, _ = _adam_pair(simulation)
        # First parameter update received the same unrounded moment values.
        assert torch.equal(control, treatment)
        for name in ("exp_avg", "exp_avg_sq"):
            actual, original = simulated.state[treatment][name], plain.state[control][name]
            assert actual.dtype == torch.float32
            if name in changed:
                assert torch.equal(actual, int8_linear_roundtrip(original))
            else:
                assert torch.equal(actual, original)


def test_adam_next_step_uses_prior_int8_dequantized_state():
    control, treatment, plain, simulated, _ = _adam_pair("int8_linear_first_moment")
    gradient = torch.tensor([.13, .29])
    control.grad, treatment.grad = gradient.clone(), gradient.clone()
    plain.step(); simulated.step()
    assert not torch.equal(control, treatment)


def test_none_and_bf16_behavior_are_preserved():
    _, _, plain, none, _ = _adam_pair("none")
    assert torch.equal(plain.state[next(iter(plain.state))]["exp_avg"], none.state[next(iter(none.state))]["exp_avg"])
    _, treatment, plain, bf16, _ = _adam_pair("bf16_roundtrip")
    original = plain.state[next(iter(plain.state))]["exp_avg"]
    assert torch.equal(bf16.state[treatment]["exp_avg"], original.to(torch.bfloat16).float())


def test_muon_int8_changes_only_momentum_and_leaves_auxiliary_adamw_fp32():
    matrix = torch.nn.Parameter(torch.ones(2, 2)); auxiliary = torch.nn.Parameter(torch.ones(2))
    matrix.grad = torch.tensor([[.37, -.11], [.02, .21]]); auxiliary.grad = torch.tensor([.2, -.1])
    opt = ReferenceMuon([{"params": [matrix], "optimizer_group": "muon", "weight_decay": 0.},
                         {"params": [auxiliary], "optimizer_group": "auxiliary_adamw", "weight_decay": 0.}],
                        state_simulation="int8_linear_momentum")
    opt.step()
    assert torch.equal(opt.state[matrix]["muon_momentum"], int8_linear_roundtrip(matrix.grad))
    assert opt.state[auxiliary]["exp_avg"].dtype == torch.float32
    assert opt.state[auxiliary]["exp_avg_sq"].dtype == torch.float32
    assert torch.allclose(opt.state[auxiliary]["exp_avg"], auxiliary.grad * .1)
    assert torch.allclose(opt.state[auxiliary]["exp_avg_sq"], auxiliary.grad.square() * .001)


def test_blockwise_adam_and_muon_select_only_intended_persistence_states():
    control, treatment, plain, simulated, _ = _adam_pair("int8_linear_all_moments")
    # Rebuild with blockwise persistence to make the group configuration explicit.
    treatment = torch.nn.Parameter(torch.tensor([1., -2., 3.]))
    simulated = ReferenceAdamW([treatment], lr=.1, betas=(.5, .5), state_simulation="int8_linear_all_moments",
                               state_quantization_granularity="blockwise", state_quantization_block_size=2)
    treatment.grad = torch.tensor([.37, -.11, .02]); simulated.step()
    for name in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(simulated.state[treatment][name], int8_blockwise_linear_roundtrip(
            treatment.grad * (.5 if name == "exp_avg" else .5) if name == "exp_avg" else treatment.grad.square() * .5, 2))
    matrix, auxiliary = torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2))
    matrix.grad, auxiliary.grad = torch.tensor([[.37, -.11], [.02, .21]]), torch.tensor([.2, -.1])
    muon = ReferenceMuon([{"params": [matrix], "optimizer_group": "muon", "weight_decay": 0.},
                           {"params": [auxiliary], "optimizer_group": "auxiliary_adamw", "weight_decay": 0.}],
                         state_simulation="int8_linear_momentum", state_quantization_granularity="blockwise",
                         state_quantization_block_size=2)
    muon.step()
    assert torch.equal(muon.state[matrix]["muon_momentum"], int8_blockwise_linear_roundtrip(matrix.grad, 2))
    assert muon.state[auxiliary]["exp_avg"].dtype == torch.float32
    assert muon.state[auxiliary]["exp_avg_sq"].dtype == torch.float32


def test_blockwise_persistence_is_post_update_and_changes_the_next_step_only():
    a, b = torch.nn.Parameter(torch.tensor([1., -2., 3.])), torch.nn.Parameter(torch.tensor([1., -2., 3.]))
    plain = ReferenceAdamW([a], lr=.1, betas=(.5, .5))
    blockwise = ReferenceAdamW([b], lr=.1, betas=(.5, .5), state_simulation="int8_linear_all_moments",
                               state_quantization_granularity="blockwise", state_quantization_block_size=2)
    first = torch.tensor([.37, -.11, .02]); a.grad = first.clone(); b.grad = first.clone()
    plain.step(); blockwise.step()
    assert torch.equal(a, b)  # current update used the same unrounded moments
    second = torch.tensor([.13, .29, -.17]); a.grad = second.clone(); b.grad = second.clone()
    plain.step(); blockwise.step()
    assert not torch.equal(a, b)  # only next-step state sees persisted roundtrip


def test_int8_metadata_makes_selected_and_fp32_states_auditable():
    adam = persistence_metadata("reference_adamw", "int8_linear_first_moment")
    assert adam["bits"] == 8 and adam["granularity"] == "per_state_tensor"
    assert adam["state_groups"]["reference_adamw"] == {"quantized_state_names": ["exp_avg"], "states_left_fp32": ["exp_avg_sq"]}
    muon = persistence_metadata("reference_muon", "int8_linear_momentum")
    assert muon["state_groups"]["muon"]["quantized_state_names"] == ["muon_momentum"]
    assert muon["state_groups"]["auxiliary_adamw"]["quantized_state_names"] == []
    assert muon["state_groups"]["auxiliary_adamw"]["states_left_fp32"] == ["exp_avg", "exp_avg_sq"]
    blockwise = persistence_metadata("reference_adamw", "int8_linear_all_moments",
                                    quantization_granularity="blockwise", quantization_block_size=2048)
    assert blockwise["granularity"] == "blockwise" and blockwise["block_size"] == 2048


def test_int8_recipes_are_exact_treatments_and_suite_is_compatible():
    baseline = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_4x_s0.json")
    adam_recipes = [load_recipe(ROOT / "recipes" / name) for name in (
        "mini_fp32_reference_adamw_int8_m_state_4x_s0.json",
        "mini_fp32_reference_adamw_int8_v_state_4x_s0.json",
        "mini_fp32_reference_adamw_int8_state_4x_s0.json")]
    ignored = IDENTITY_FIELDS | GENERATED_FIELDS
    for treatment in adam_recipes:
        assert scientific_differences([baseline, treatment], ["control", "treatment"], ["optimizer.state_simulation"]) == []
        changes = {key for key in set(flatten(baseline)) | set(flatten(treatment)) if key not in ignored and flatten(baseline).get(key) != flatten(treatment).get(key)}
        assert changes == {"optimizer.state_simulation"}
        assert treatment["derived"]["total_updates"] == 4096
        assert treatment["derived"]["schedule_total_updates"] == 4096
        assert treatment["derived"]["tokens_per_update"] == 8192
    muon_baseline = load_recipe(ROOT / "recipes/mini_fp32_reference_muon_4x_s0.json")
    muon_treatment = load_recipe(ROOT / "recipes/mini_fp32_reference_muon_int8_state_4x_s0.json")
    assert scientific_differences([muon_baseline, muon_treatment], ["control", "treatment"], ["optimizer.state_simulation"]) == []
    muon_changes = {key for key in set(flatten(muon_baseline)) | set(flatten(muon_treatment)) if key not in ignored and flatten(muon_baseline).get(key) != flatten(muon_treatment).get(key)}
    assert muon_changes == {"optimizer.state_simulation"}
    suite = load_suite(ROOT / "benchmarks/mini_optimizer_int8_state_4x_s0_v1.json")
    assert len(suite["runs"]) == 4
    assert suite["vary"] == ["optimizer.name", "optimizer.state_simulation"]
    configs = [load_recipe(run["recipe"]) for run in suite["runs"]]
    assert scientific_differences(configs, [run["run_id"] for run in suite["runs"]], suite["vary"]) == []


def test_blockwise_recipes_and_suite_hold_the_4x_protocol_constant():
    adam = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_int8_blockwise_linear_state_4x_s0.json")
    muon = load_recipe(ROOT / "recipes/mini_fp32_reference_muon_int8_blockwise_linear_state_4x_s0.json")
    for cfg in (adam, muon):
        assert cfg["derived"]["total_updates"] == cfg["derived"]["schedule_total_updates"] == 4096
        assert cfg["derived"]["tokens_per_update"] == 8192
        assert cfg["optimizer"]["state_quantization_granularity"] == "blockwise"
        assert cfg["optimizer"]["state_quantization_block_size"] == 2048
    suite = load_suite(ROOT / "benchmarks/mini_optimizer_int8_blockwise_linear_4x_s0_v1.json")
    configs = [load_recipe(run["recipe"]) for run in suite["runs"]]
    assert len(configs) == 2
    assert scientific_differences(configs, [run["run_id"] for run in suite["runs"]], suite["vary"]) == []
