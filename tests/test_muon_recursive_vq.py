import sys
import inspect

import torch

sys.path.insert(0, "src")

from optim.muon_recursive import (  # noqa: E402
    RecursiveMuon, StructuralINT4Codec, StructuralVQCodec,
    _block_stat_scales, pack_indices, unpack_indices,
)
from optim.muon_vector_int3 import pair_values, unpair_values, vector_scales  # noqa: E402


def codebook():
    g = torch.Generator().manual_seed(2026)
    x = torch.randn((64, 2), generator=g) * .25
    x[0] = 0
    return x


def legacy_pack(indices):
    out = torch.zeros(((indices.numel() * 6 + 7) // 8,), dtype=torch.uint8)
    for i, value in enumerate(indices.reshape(-1).tolist()):
        bit = i * 6; integer = int(value) << (bit % 8); byte = bit // 8
        out[byte] |= integer & 0xFF
        if min(8 - bit % 8, 6) < 6:
            out[byte + 1] |= (integer >> 8) & 0xFF
    return out


def test_fixed_width_pack_roundtrip_is_exact():
    for count in (1, 2, 3, 4, 5, 7, 8, 31, 32, 1009, 1024, 10_000):
        x = torch.randint(0, 64, (count,), generator=torch.Generator().manual_seed(count))
        packed = pack_indices(x, 6)
        assert packed.numel() == (x.numel() * 6 + 7) // 8
        assert torch.equal(packed, legacy_pack(x))
        assert torch.equal(unpack_indices(packed, x.numel()), x)


def test_recursive_contiguous_pairing_matches_generic_analysis_path():
    for shape in ((1152, 384), (384, 384), (1024, 384), (384, 1024)):
        matrix = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        pairs, singles, pair_index, single_index = pair_values(matrix, "contiguous")
        assert singles.numel() == 0
        assert torch.equal(pairs, matrix.reshape(-1, 2))
        assert torch.equal(unpair_values(pairs, singles, shape, pair_index, single_index), matrix)


def test_recursive_vectorized_p98_matches_block_loop():
    for flat in (torch.randn(4096), torch.zeros(4096), torch.cat((torch.zeros(2047), torch.tensor([100.0]), torch.randn(2048)) )):
        assert torch.allclose(_block_stat_scales(flat, 2048, percentile=.98),
                              vector_scales(flat.reshape(-1, 2), "p98", block_size=2048),
                              atol=1e-6, rtol=0)


def test_recursive_runtime_hot_path_has_no_pair_index_or_python_index_pack_loop():
    source = inspect.getsource(StructuralVQCodec.encode)
    assert "pair_values(" not in source
    assert ".cpu()" not in source
    assert "values.tolist()" not in inspect.getsource(pack_indices)
    assert "for i in range" not in inspect.getsource(unpack_indices)


def test_vq_state_has_no_fp32_momentum_and_decodes_shape():
    torch.manual_seed(2); m = torch.randn(16, 16)
    codec = StructuralVQCodec(codebook(), rank=8)
    state = codec.encode(m); decoded = codec.decode(state)
    assert tuple(state.shape) == (16, 16)
    assert state.u.dtype == torch.bfloat16 and state.vh.dtype == torch.bfloat16
    assert state.indices.dtype == torch.uint8
    assert decoded.shape == m.shape and torch.isfinite(decoded).all()
    assert not hasattr(state, "momentum")


def test_vq_optimized_codec_matches_legacy_assignment_and_decode():
    torch.manual_seed(7)
    matrix = torch.randn(32, 32)
    cb = codebook()
    codec = StructuralVQCodec(cb, rank=8, block_size=2048)
    state = codec.encode(matrix)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    low = (u[:, :8] * s[:8]) @ vh[:8]
    residual = matrix - low
    pairs = residual.reshape(-1, 2)
    scales = vector_scales(pairs, "p98", block_size=2048)
    pair_scales = torch.repeat_interleave(scales, 1024)[:pairs.shape[0]]
    normalized = (pairs / pair_scales.clamp_min(torch.finfo(torch.float32).tiny)[:, None]).clamp(-1, 1)
    indices = torch.cdist(normalized, cb).argmin(dim=1)
    indices = torch.where(pair_scales > 0, indices, torch.zeros_like(indices))
    # The persistent representation intentionally stores structural factors in
    # BF16, so compare against the decoded BF16 low-rank component rather than
    # the pre-serialization FP32 factors.
    persisted_low = (state.u.float() * state.singular_values.float()) @ state.vh.float()
    expected = persisted_low + (cb[indices] * pair_scales[:, None]).reshape(matrix.shape)
    assert torch.allclose(codec.decode(state), expected, atol=2e-5, rtol=2e-5)
    assert state.indices.device == matrix.device


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
    assert state["compressed_momentum"].u.device == p.device
    first = codec.decode(state["compressed_momentum"])
    p.grad = torch.zeros_like(p); opt.step()
    second = codec.decode(opt.state[p]["compressed_momentum"])
    assert torch.allclose(second, first * .95, atol=2e-2, rtol=0)


def test_recursive_checkpoint_state_roundtrip():
    p = torch.nn.Parameter(torch.eye(16)); codec = StructuralVQCodec(codebook(), rank=8)
    opt = RecursiveMuon([{"params": [p], "optimizer_group": "muon", "weight_decay": 0.0}], {id(p): codec}, lr=.001)
    p.grad = torch.randn_like(p); opt.step()
    blob = {"model": p.detach().clone(), "optimizer": opt.state_dict()}
    serialized = blob["optimizer"]["state"]
    encoded_payload = next(iter(serialized.values()))["compressed_momentum"]
    assert encoded_payload["_recursive_state_type"] == "vq"
    q = torch.nn.Parameter(blob["model"].clone()); opt2 = RecursiveMuon([{"params": [q], "optimizer_group": "muon", "weight_decay": 0.0}], {id(q): codec}, lr=.001)
    q.data.copy_(blob["model"]); opt2.load_state_dict(blob["optimizer"])
    assert torch.equal(opt2.state[q]["compressed_momentum"].indices, opt.state[p]["compressed_momentum"].indices)


def test_recursive_checkpoint_resume_matches_continuous_cpu():
    torch.manual_seed(21)
    gradients = [torch.randn(16, 16) for _ in range(4)]
    p_cont = torch.nn.Parameter(torch.eye(16)); c_cont = StructuralVQCodec(codebook(), rank=8)
    o_cont = RecursiveMuon([{"params": [p_cont], "optimizer_group": "muon", "weight_decay": 0.0}],
                           {id(p_cont): c_cont}, lr=.001)
    for grad in gradients:
        p_cont.grad = grad; o_cont.step()

    p_split = torch.nn.Parameter(torch.eye(16)); c_split = StructuralVQCodec(codebook(), rank=8)
    o_split = RecursiveMuon([{"params": [p_split], "optimizer_group": "muon", "weight_decay": 0.0}],
                            {id(p_split): c_split}, lr=.001)
    for grad in gradients[:2]:
        p_split.grad = grad; o_split.step()
    checkpoint = {"model": p_split.detach().clone(), "optimizer": o_split.state_dict()}
    p_resume = torch.nn.Parameter(checkpoint["model"].clone()); c_resume = StructuralVQCodec(codebook(), rank=8)
    o_resume = RecursiveMuon([{"params": [p_resume], "optimizer_group": "muon", "weight_decay": 0.0}],
                             {id(p_resume): c_resume}, lr=.001)
    o_resume.load_state_dict(checkpoint["optimizer"])
    for grad in gradients[2:]:
        p_resume.grad = grad; o_resume.step()
    assert torch.equal(p_cont, p_resume)
    assert torch.equal(o_cont.state[p_cont]["compressed_momentum"].indices,
                       o_resume.state[p_resume]["compressed_momentum"].indices)
