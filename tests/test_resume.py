import copy
import random
import numpy as np
import pytest
import torch
from data.frozen_tokens import DeterministicSampler
from experiment_io import restore_rng,rng_state
from optim.adamw_reference import ReferenceAdamW


def test_int8_optimizer_state_survives_checkpoint_save_and_resume(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = ReferenceAdamW([parameter], lr=.01, state_simulation="int8_linear_all_moments")
    parameter.grad = torch.tensor([.37, -.11]); optimizer.step()
    path = tmp_path / "int8-state.pt"
    torch.save({"parameter": parameter.detach(), "optimizer": optimizer.state_dict()}, path)
    restored_parameter = torch.nn.Parameter(torch.zeros_like(parameter))
    restored_optimizer = ReferenceAdamW([restored_parameter], lr=.01, state_simulation="int8_linear_all_moments")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored_parameter.data.copy_(checkpoint["parameter"]); restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_parameter.grad = torch.tensor([.13, .29]); parameter.grad = restored_parameter.grad.clone()
    optimizer.step(); restored_optimizer.step()
    assert torch.equal(parameter, restored_parameter)


def test_dynamic_int8_optimizer_state_survives_checkpoint_save_and_resume(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, .3]))
    optimizer = ReferenceAdamW([parameter], lr=.01, state_simulation="int8_dynamic_all_moments",
                               state_quantization_granularity="blockwise", state_quantization_block_size=2)
    parameter.grad = torch.tensor([.37, -.11, .22]); optimizer.step()
    path = tmp_path / "dynamic-int8-state.pt"
    torch.save({"parameter": parameter.detach(), "optimizer": optimizer.state_dict()}, path)
    restored_parameter = torch.nn.Parameter(torch.zeros_like(parameter))
    restored_optimizer = ReferenceAdamW([restored_parameter], lr=.01, state_simulation="int8_dynamic_all_moments",
                                        state_quantization_granularity="blockwise", state_quantization_block_size=2)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored_parameter.data.copy_(checkpoint["parameter"]); restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_parameter.grad = torch.tensor([.13, .29, -.17]); parameter.grad = restored_parameter.grad.clone()
    optimizer.step(); restored_optimizer.step()
    assert torch.equal(parameter, restored_parameter)


def test_old_optimizer_checkpoint_without_blockwise_group_keys_still_resumes():
    parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
    old = ReferenceAdamW([parameter], lr=.01, state_simulation="int8_linear_all_moments")
    parameter.grad = torch.tensor([.37, -.11]); old.step()
    checkpoint = copy.deepcopy(old.state_dict())
    for group in checkpoint["param_groups"]:
        group.pop("state_quantization_granularity", None)
        group.pop("state_quantization_block_size", None)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = ReferenceAdamW([restored_parameter], lr=.01, state_simulation="int8_linear_all_moments")
    restored.load_state_dict(checkpoint)
    parameter.grad = restored_parameter.grad = torch.tensor([.13, .29])
    old.step(); restored.step()
    assert torch.equal(parameter, restored_parameter)

def test_optimizer_sampler_and_rng_resume_exact():
    def setup():
        random.seed(7);np.random.seed(7);torch.manual_seed(7);p=torch.nn.Parameter(torch.tensor([1.,-2.]));return p,ReferenceAdamW([p],lr=.01),DeterministicSampler(13,5,True)
    a,oa,sa=setup(); trajectory=[]
    for step in range(20):
        ids=sa.take(2);a.grad=torch.tensor([ids[0]/10,ids[1]/10]);oa.step();trajectory.append((ids,a.detach().clone()))
        if step==6: checkpoint={"p":a.detach().clone(),"o":copy.deepcopy(oa.state_dict()),"s":copy.deepcopy(sa.state_dict()),"r":rng_state(False)}
    b,ob,sb=setup();b.data.copy_(checkpoint["p"]);ob.load_state_dict(checkpoint["o"]);sb.load_state_dict(checkpoint["s"]);restore_rng(checkpoint["r"])
    for step in range(7,20):
        ids=sb.take(2);b.grad=torch.tensor([ids[0]/10,ids[1]/10]);ob.step();assert ids==trajectory[step][0] and torch.equal(b,trajectory[step][1])


def test_cpu_checkpoint_rng_save_and_resume_is_exact(tmp_path):
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    checkpoint = {"rng": rng_state(False)}
    path = tmp_path / "checkpoint.pt"; torch.save(checkpoint, path)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restore_rng(loaded["rng"])
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


class _RemappedRngTensor:
    """CUDA-like checkpoint value usable in CPU-only regression tests."""
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.calls = []

    def detach(self):
        return self

    def to(self, *, device, dtype):
        self.calls.append((device, dtype))
        return torch.tensor([1, 2, 3], dtype=dtype, device=device)


def test_restore_rng_normalizes_remapped_cpu_and_cuda_states(monkeypatch):
    cpu_state, cuda_state = _RemappedRngTensor(), _RemappedRngTensor()
    restored = {}
    monkeypatch.setattr(torch, "set_rng_state", lambda value: restored.setdefault("cpu", value))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: restored.setdefault("cuda", values))
    restore_rng({"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": cpu_state, "torch_cuda": [cuda_state]})
    assert cpu_state.calls == [("cpu", torch.uint8)]
    assert cuda_state.calls == [("cpu", torch.uint8)]
    assert restored["cpu"].device.type == "cpu" and restored["cpu"].dtype == torch.uint8
    assert restored["cuda"][0].device.type == "cpu" and restored["cuda"][0].dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to create CUDA-remapped RNG tensors")
def test_restore_rng_accepts_cuda_remapped_rng_tensors():
    original_cpu, original_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        restore_rng({"python": random.getstate(), "numpy": np.random.get_state(),
                     "torch_cpu": original_cpu.to("cuda"), "torch_cuda": [value.to("cuda") for value in original_cuda]})
        assert torch.equal(torch.get_rng_state(), original_cpu)
        assert all(torch.equal(actual, expected) for actual, expected in zip(torch.cuda.get_rng_state_all(), original_cuda))
    finally:
        torch.set_rng_state(original_cpu)
        torch.cuda.set_rng_state_all(original_cuda)
