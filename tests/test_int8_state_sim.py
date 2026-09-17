from pathlib import Path

import torch

from benchmark_suite import load_suite
from config.recipe import load_recipe
from optim.adamw_reference import ReferenceAdamW
from optim.muon_reference import ReferenceMuon
from optim.state_simulation import (create_bitsandbytes_dynamic_map, int8_blockwise_dynamic_roundtrip,
                                    int8_blockwise_linear_roundtrip, int8_linear_roundtrip, persistence_metadata)
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


def test_pinned_bitsandbytes_dynamic_maps_match_reference_construction_values():
    signed = create_bitsandbytes_dynamic_map(signed=True)
    unsigned = create_bitsandbytes_dynamic_map(signed=False)
    assert signed.dtype == unsigned.dtype == torch.float32
    assert signed.numel() == unsigned.numel() == 256
    assert torch.equal(signed, create_bitsandbytes_dynamic_map(signed=True))
    assert (signed == 0).sum() == (unsigned == 0).sum() == 1
    assert torch.all(signed[1:] >= signed[:-1]) and torch.all(unsigned[1:] >= unsigned[:-1])
    # Known FP32 values from bitsandbytes functional.create_dynamic_map
    # (signed=True/False, max_exponent_bits=7, total_bits=8).
    assert torch.allclose(signed[:3], torch.tensor([-.9929687381, -.9789062738, -.96484375]))
    assert torch.allclose(unsigned[:5], torch.tensor([0., 3.2500003e-7, 7.7499999e-7, 2.1249998e-6, 4.3749997e-6]))
    assert signed[-1] == unsigned[-1] == 1


def test_four_bit_codebooks_and_linear_range_are_deterministic_and_leave_int8_unchanged():
    signed = create_bitsandbytes_dynamic_map(signed=True, total_bits=4, max_exponent_bits=3)
    unsigned = create_bitsandbytes_dynamic_map(signed=False, total_bits=4, max_exponent_bits=3)
    assert signed.numel() == unsigned.numel() == 16
    assert torch.allclose(signed, torch.tensor([-.8875, -.6625, -.4375, -.2125, -.0775, -.0325, -.0055, 0., .0055, .0325, .0775, .2125, .4375, .6625, .8875, 1.]))
    assert torch.allclose(unsigned, torch.tensor([0., .00325, .00775, .02125, .04375, .06625, .08875, .15625, .26875, .38125, .49375, .60625, .71875, .83125, .94375, 1.]))
    value = torch.tensor([-2., -1., 0., 1., 2.])
    # torch.round uses deterministic nearest-even ties: +/-1 maps to +/-3 at scale 2/7.
    assert torch.allclose(int8_linear_roundtrip(value, bits=4), torch.tensor([-2., -6/7, 0., 6/7, 2.]))
    assert torch.equal(int8_linear_roundtrip(value), int8_linear_roundtrip(value, bits=8))
    assert torch.equal(int8_blockwise_linear_roundtrip(value, 2), int8_blockwise_linear_roundtrip(value, 2, bits=8))


def test_four_bit_blockwise_dynamic_and_linear_persist_only_selected_states():
    value = torch.tensor([-2., -.2, 0., .2, 2., .01])
    actual = int8_blockwise_dynamic_roundtrip(value, signed=True, block_size=5, total_bits=4)
    assert actual.dtype == torch.float32 and actual.shape == value.shape
    assert torch.equal(actual, int8_blockwise_dynamic_roundtrip(value, signed=True, block_size=5, total_bits=4))
    matrix, auxiliary = torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2))
    matrix.grad, auxiliary.grad = torch.tensor([[.37, -.11], [.02, .21]]), torch.tensor([.2, -.1])
    opt = ReferenceMuon([{"params": [matrix], "optimizer_group": "muon", "weight_decay": 0.}, {"params": [auxiliary], "optimizer_group": "auxiliary_adamw", "weight_decay": 0.}], state_simulation="int4_linear_momentum", state_quantization_granularity="blockwise", state_quantization_block_size=2)
    opt.step()
    assert torch.equal(opt.state[matrix]["muon_momentum"], int8_blockwise_linear_roundtrip(matrix.grad, 2, bits=4))
    assert opt.state[auxiliary]["exp_avg"].dtype == opt.state[auxiliary]["exp_avg_sq"].dtype == torch.float32


def test_muon_int4_dynamic_persists_only_signed_momentum_and_metadata_is_auditable():
    matrix, auxiliary = torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2))
    matrix.grad, auxiliary.grad = torch.tensor([[.37, -.11], [.02, .21]]), torch.tensor([.2, -.1])
    opt = ReferenceMuon([{"params": [matrix], "optimizer_group": "muon", "weight_decay": 0.},
                         {"params": [auxiliary], "optimizer_group": "auxiliary_adamw", "weight_decay": 0.}],
                        state_simulation="int4_dynamic_momentum", state_quantization_granularity="blockwise",
                        state_quantization_block_size=2)
    opt.step()
    assert torch.equal(opt.state[matrix]["muon_momentum"], int8_blockwise_dynamic_roundtrip(
        matrix.grad, signed=True, block_size=2, total_bits=4))
    assert opt.state[auxiliary]["exp_avg"].dtype == opt.state[auxiliary]["exp_avg_sq"].dtype == torch.float32
    metadata = persistence_metadata("reference_muon", "int4_dynamic_momentum", quantization_granularity="blockwise")
    assert metadata["bits"] == 4 and metadata["block_size"] == 2048
    assert metadata["state_groups"]["auxiliary_adamw"]["states_left_fp32"] == ["exp_avg", "exp_avg_sq"]
    assert metadata["dynamic_map_provenance"]["muon_momentum"] == {
        "codebook": "dynamic", "signed": True, "representable_values": 16}


def test_int4_recipes_and_suite_preserve_fixed_4x_protocol():
    adam = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_int4_blockwise_dynamic_state_4x_s0.json")
    muon = load_recipe(ROOT / "recipes/mini_fp32_reference_muon_int4_blockwise_linear_state_4x_s0.json")
    for cfg in (adam, muon):
        assert cfg["derived"]["total_updates"] == cfg["derived"]["schedule_total_updates"] == 4096
        assert cfg["derived"]["tokens_per_update"] == 8192
        assert cfg["optimizer"]["state_quantization_block_size"] == 2048
    assert adam["optimizer"]["state_simulation"] == "int4_dynamic_all_moments"
    assert muon["optimizer"]["state_simulation"] == "int4_linear_momentum"
    metadata = persistence_metadata("reference_muon", "int4_linear_momentum", quantization_granularity="blockwise")
    assert metadata["bits"] == 4 and metadata["signed_range"] == [-7, 7]
    suite = load_suite(ROOT / "benchmarks/mini_optimizer_int4_state_4x_s0_v1.json")
    configs = [load_recipe(run["recipe"]) for run in suite["runs"]]
    assert scientific_differences(configs, [run["run_id"] for run in suite["runs"]], suite["vary"]) == []


def test_muon_int8_replay_and_int4_dynamic_recipes_share_the_fixed_protocol():
    replay = load_recipe(ROOT / "recipes/mini_fp32_reference_muon_int8_blockwise_linear_state_4x_s0_replay.json")
    dynamic = load_recipe(ROOT / "recipes/mini_fp32_reference_muon_int4_blockwise_dynamic_state_4x_s0.json")
    for cfg in (replay, dynamic):
        assert cfg["derived"]["total_updates"] == cfg["derived"]["schedule_total_updates"] == 4096
        assert cfg["derived"]["tokens_per_update"] == 8192
        assert cfg["optimizer"]["state_quantization_block_size"] == 2048
        assert cfg["logging"]["state_diagnostics"] is True
    assert replay["optimizer"]["state_simulation"] == "int8_linear_momentum"
    assert dynamic["optimizer"]["state_simulation"] == "int4_dynamic_momentum"
    suite = load_suite(ROOT / "benchmarks/mini_muon_int8_replay_int4_dynamic_4x_s0_v1.json")
    configs = [load_recipe(run["recipe"]) for run in suite["runs"]]
    assert scientific_differences(configs, [run["run_id"] for run in suite["runs"]], suite["vary"]) == []


def test_muon_int4_b256_recipes_change_only_block_size_from_b2048_and_suite_is_compatible():
    pairs = (
        ("mini_fp32_reference_muon_int4_blockwise_linear_state_4x_s0.json",
         "mini_fp32_reference_muon_int4_blockwise_linear_b256_state_4x_s0.json"),
        ("mini_fp32_reference_muon_int4_blockwise_dynamic_state_4x_s0.json",
         "mini_fp32_reference_muon_int4_blockwise_dynamic_b256_state_4x_s0.json"),
    )
    ignored = IDENTITY_FIELDS | GENERATED_FIELDS
    for b2048_name, b256_name in pairs:
        b2048, b256 = (load_recipe(ROOT / "recipes" / name) for name in (b2048_name, b256_name))
        assert b2048["optimizer"]["state_quantization_block_size"] == 2048
        assert b256["optimizer"]["state_quantization_block_size"] == 256
        assert b256["derived"]["total_updates"] == b256["derived"]["schedule_total_updates"] == 4096
        assert b256["derived"]["tokens_per_update"] == 8192
        assert b256["data"]["allow_repeated_epochs"] is False
        changes = {key for key in set(flatten(b2048)) | set(flatten(b256))
                   if key not in ignored and flatten(b2048).get(key) != flatten(b256).get(key)}
        assert changes == {"optimizer.state_quantization_block_size"}
    suite = load_suite(ROOT / "benchmarks/mini_muon_int4_b256_ablation_4x_s0_v1.json")
    configs = [load_recipe(run["recipe"]) for run in suite["runs"]]
    assert scientific_differences(configs, [run["run_id"] for run in suite["runs"]], suite["vary"]) == []


def test_b256_blockwise_quantizers_keep_partial_blocks_within_one_momentum_tensor():
    value = torch.cat((torch.tensor([7.]), torch.full((255,), .5), torch.tensor([2., 1.])))
    linear = int8_blockwise_linear_roundtrip(value, block_size=256, bits=4)
    dynamic = int8_blockwise_dynamic_roundtrip(value, signed=True, block_size=256, total_bits=4)
    # The two-element final block has its own absmax.  It is not padded into
    # or scaled by the preceding 256-element block.
    assert linear.shape == dynamic.shape == value.shape
    assert linear.dtype == dynamic.dtype == torch.float32
    assert linear[-2] == dynamic[-2] == 2
    assert linear[-1] != 0 and dynamic[-1] != 0
    other_tensor = torch.tensor([2., 1.])
    assert torch.equal(linear[-2:], int8_blockwise_linear_roundtrip(other_tensor, 256, bits=4))
    assert torch.equal(dynamic[-2:], int8_blockwise_dynamic_roundtrip(other_tensor, signed=True, block_size=256, total_bits=4))


def test_blockwise_dynamic_roundtrip_handles_signed_unsigned_zero_and_partial_blocks():
    signed = torch.tensor([-2., -.2, 0., .2, 2., .01], dtype=torch.float64)
    unsigned = torch.tensor([0., .001, .1, 1., .01], dtype=torch.float64)
    for value, is_signed in ((signed, True), (unsigned, False)):
        actual = int8_blockwise_dynamic_roundtrip(value, signed=is_signed, block_size=5)
        assert actual.dtype == torch.float32 and actual.shape == value.shape
        assert torch.equal(actual, int8_blockwise_dynamic_roundtrip(value, signed=is_signed, block_size=5))
        if is_signed:
            assert actual[0] < 0 and actual[0].abs() <= value[0].abs()
        else:
            assert actual[0] == 0
    assert torch.equal(int8_blockwise_dynamic_roundtrip(torch.zeros(2050), signed=False), torch.zeros(2050))


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


def test_dynamic_adam_persists_both_moments_after_current_fp32_update():
    a, b = torch.nn.Parameter(torch.tensor([1., -2., 3.])), torch.nn.Parameter(torch.tensor([1., -2., 3.]))
    plain = ReferenceAdamW([a], lr=.1, betas=(.5, .5))
    dynamic = ReferenceAdamW([b], lr=.1, betas=(.5, .5), state_simulation="int8_dynamic_all_moments",
                             state_quantization_granularity="blockwise", state_quantization_block_size=2)
    first = torch.tensor([.37, -.11, .02]); a.grad = first.clone(); b.grad = first.clone()
    plain.step(); dynamic.step()
    assert torch.equal(a, b)
    assert torch.equal(dynamic.state[b]["exp_avg"], int8_blockwise_dynamic_roundtrip(plain.state[a]["exp_avg"], signed=True, block_size=2))
    assert torch.equal(dynamic.state[b]["exp_avg_sq"], int8_blockwise_dynamic_roundtrip(plain.state[a]["exp_avg_sq"], signed=False, block_size=2))
    second = torch.tensor([.13, .29, -.17]); a.grad = second.clone(); b.grad = second.clone()
    plain.step(); dynamic.step()
    assert not torch.equal(a, b)


def test_dynamic_adam_second_moment_only_preserves_m_and_persists_unsigned_v():
    a, b = torch.nn.Parameter(torch.tensor([1., -2., 3.])), torch.nn.Parameter(torch.tensor([1., -2., 3.]))
    plain = ReferenceAdamW([a], lr=.1, betas=(.5, .5))
    dynamic_v = ReferenceAdamW([b], lr=.1, betas=(.5, .5), state_simulation="int8_dynamic_second_moment",
                               state_quantization_granularity="blockwise", state_quantization_block_size=2)
    first = torch.tensor([.37, -.11, .02]); a.grad = first.clone(); b.grad = first.clone()
    plain.step(); dynamic_v.step()
    assert torch.equal(a, b)  # current update still used unquantized FP32 moments
    assert torch.equal(dynamic_v.state[b]["exp_avg"], plain.state[a]["exp_avg"])
    assert torch.equal(dynamic_v.state[b]["exp_avg_sq"], int8_blockwise_dynamic_roundtrip(
        plain.state[a]["exp_avg_sq"], signed=False, block_size=2))


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


def test_dynamic_recipe_and_metadata_pin_blockwise_signed_m_unsigned_v():
    cfg = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_int8_blockwise_dynamic_state_4x_s0.json")
    assert cfg["derived"]["total_updates"] == cfg["derived"]["schedule_total_updates"] == 4096
    assert cfg["optimizer"]["state_simulation"] == "int8_dynamic_all_moments"
    assert cfg["optimizer"]["state_quantization_granularity"] == "blockwise"
    assert cfg["optimizer"]["state_quantization_block_size"] == 2048
    meta = persistence_metadata("reference_adamw", "int8_dynamic_all_moments", quantization_granularity="blockwise")
    assert meta["simulation"] == "int8_dynamic_roundtrip"
    assert meta["state_groups"]["reference_adamw"]["quantized_state_names"] == ["exp_avg", "exp_avg_sq"]
    assert meta["dynamic_map_provenance"]["exp_avg"]["signed"] is True
    assert meta["dynamic_map_provenance"]["exp_avg_sq"]["signed"] is False
    suite = load_suite(ROOT / "benchmarks/mini_adamw_int8_blockwise_dynamic_4x_s0_v1.json")
    assert len(suite["runs"]) == 1


def test_dynamic_v_only_recipe_and_ablation_suite_differ_only_by_selected_state():
    all_moments = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_int8_blockwise_dynamic_state_4x_s0.json")
    second_moment = load_recipe(ROOT / "recipes/mini_fp32_reference_adamw_int8_blockwise_dynamic_v_state_4x_s0.json")
    assert scientific_differences([all_moments, second_moment], ["all", "v_only"], ["optimizer.state_simulation"]) == []
    ignored = IDENTITY_FIELDS | GENERATED_FIELDS
    changed = {key for key in set(flatten(all_moments)) | set(flatten(second_moment))
               if key not in ignored and flatten(all_moments).get(key) != flatten(second_moment).get(key)}
    assert changed == {"optimizer.state_simulation"}
    assert second_moment["derived"]["total_updates"] == second_moment["derived"]["schedule_total_updates"] == 4096
    assert second_moment["derived"]["tokens_per_update"] == 8192
    meta = persistence_metadata("reference_adamw", "int8_dynamic_second_moment",
                               quantization_granularity="blockwise", quantization_block_size=2048)
    assert meta["state_groups"]["reference_adamw"] == {
        "quantized_state_names": ["exp_avg_sq"], "states_left_fp32": ["exp_avg"]}
    assert meta["dynamic_map_provenance"]["exp_avg_sq"] == {
        "codebook": "dynamic", "signed": False, "representable_values": 256}
    suite = load_suite(ROOT / "benchmarks/mini_adamw_int8_blockwise_dynamic_ablation_4x_s0_v1.json")
    configs = [load_recipe(run["recipe"]) for run in suite["runs"]]
    assert len(configs) == 2
    assert scientific_differences(configs, [run["run_id"] for run in suite["runs"]], suite["vary"]) == []
