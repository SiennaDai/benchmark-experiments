import sys

import torch

sys.path.insert(0, "src")

from optim.muon_recursive import (  # noqa: E402
    RecursiveMuon, StructuralINT4Codec, StructuralVQCodec,
    pack_indices, unpack_indices,
)


def codebook():
    g = torch.Generator().manual_seed(2026)
    x = torch.randn((64, 2), generator=g) * .25
    x[0] = 0
    return x


def test_fixed_width_pack_roundtrip_is_exact():
    x = torch.arange(64, dtype=torch.long).repeat(3)
    packed = pack_indices(x, 6)
    assert packed.numel() == (x.numel() * 6 + 7) // 8
    assert torch.equal(unpack_indices(packed, x.numel()), x)


def test_vq_state_has_no_fp32_momentum_and_decodes_shape():
    torch.manual_seed(2); m = torch.randn(16, 16)
    codec = StructuralVQCodec(codebook(), rank=8)
    state = codec.encode(m); decoded = codec.decode(state)
    assert tuple(state.shape) == (16, 16)
    assert state.u.dtype == torch.bfloat16 and state.vh.dtype == torch.bfloat16
    assert state.indices.dtype == torch.uint8
    assert decoded.shape == m.shape and torch.isfinite(decoded).all()
    assert not hasattr(state, "momentum")


def test_int4_state_roundtrip_is_packed_and_finite():
    torch.manual_seed(4); m = torch.randn(16, 16)
    codec = StructuralINT4Codec(rank=8)
    state = codec.encode(m); decoded = codec.decode(state)
    assert state.codes.dtype == torch.uint8 and state.codes.numel() == (m.numel()+1)//2
    assert decoded.shape == m.shape and torch.isfinite(decoded).all()


def test_recursive_optimizer_reads_compressed_state_on_next_step():
    p = torch.nn.Parameter(torch.eye(16))
    codec = StructuralVQCodec(codebook(), rank=8)
    opt = RecursiveMuon([{"params": [p], "optimizer_group": "muon", "weight_decay": 0.0}],
                        {id(p): codec}, lr=.001, muon_momentum=.95)
    p.grad = torch.ones_like(p); opt.step()
    state = opt.state[p]
    assert "compressed_momentum" in state and "muon_momentum" not in state
    first = codec.decode(state["compressed_momentum"])
    p.grad = torch.zeros_like(p); opt.step()
    second = codec.decode(opt.state[p]["compressed_momentum"])
    assert torch.allclose(second, first * .95, atol=2e-2, rtol=0)


def test_recursive_checkpoint_state_roundtrip():
    p = torch.nn.Parameter(torch.eye(16)); codec = StructuralVQCodec(codebook(), rank=8)
    opt = RecursiveMuon([{"params": [p], "optimizer_group": "muon", "weight_decay": 0.0}], {id(p): codec}, lr=.001)
    p.grad = torch.randn_like(p); opt.step()
    blob = {"model": p.detach().clone(), "optimizer": opt.state_dict()}
    q = torch.nn.Parameter(blob["model"].clone()); opt2 = RecursiveMuon([{"params": [q], "optimizer_group": "muon", "weight_decay": 0.0}], {id(q): codec}, lr=.001)
    q.data.copy_(blob["model"]); opt2.load_state_dict(blob["optimizer"])
    assert torch.equal(opt2.state[q]["compressed_momentum"].indices, opt.state[p]["compressed_momentum"].indices)
