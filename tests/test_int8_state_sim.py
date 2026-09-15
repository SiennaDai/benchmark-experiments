from pathlib import Path

import torch

from benchmark_suite import load_suite
from config.recipe import load_recipe
from optim.adamw_reference import ReferenceAdamW
from optim.muon_reference import ReferenceMuon
from optim.state_simulation import int8_linear_roundtrip, persistence_metadata
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


def test_int8_metadata_makes_selected_and_fp32_states_auditable():
    adam = persistence_metadata("reference_adamw", "int8_linear_first_moment")
    assert adam["bits"] == 8 and adam["granularity"] == "per_state_tensor"
    assert adam["state_groups"]["reference_adamw"] == {"quantized_state_names": ["exp_avg"], "states_left_fp32": ["exp_avg_sq"]}
    muon = persistence_metadata("reference_muon", "int8_linear_momentum")
    assert muon["state_groups"]["muon"]["quantized_state_names"] == ["muon_momentum"]
    assert muon["state_groups"]["auxiliary_adamw"]["quantized_state_names"] == []
    assert muon["state_groups"]["auxiliary_adamw"]["states_left_fp32"] == ["exp_avg", "exp_avg_sq"]


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
