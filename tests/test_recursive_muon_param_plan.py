import sys

import pytest
import torch

sys.path.insert(0, "src")

from config.recipe import RecipeError, _validate_recursive_param_plan  # noqa: E402
from optim.muon_recursive import (  # noqa: E402
    RecursiveMuon, SplitVQCodec, SplitVQState, StructuralVQCodec,
    build_recursive_codecs, pack_indices, unpack_indices,
)


def codebook(words=64):
    g = torch.Generator().manual_seed(2026)
    x = torch.randn((words, 2), generator=g) * .25
    x[0] = 0
    return x


def test_five_bit_pack_round_trips_and_is_compact():
    for n in (0, 1, 7, 8, 9, 63, 64, 1001, 2048):
        values = torch.randint(0, 32, (n,))
        packed = pack_indices(values, 5)
        assert packed.numel() == (n * 5 + 7) // 8
        assert torch.equal(unpack_indices(packed, n, 5), values)


def test_five_bit_packing_is_little_endian_bit_stream():
    values = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    packed = pack_indices(values, 5)
    acc = sum(int(v) << (5 * i) for i, v in enumerate(values.tolist()))
    expected = torch.tensor([(acc >> (8 * j)) & 0xFF for j in range(packed.numel())],
                            dtype=torch.uint8)
    assert torch.equal(packed, expected)


def test_codec_accepts_32_word_codebook_with_five_bit_indices():
    codec = StructuralVQCodec(codebook(32), rank=4, block_size=512)
    assert codec.index_bits == 5
    m = torch.randn(32, 32)
    state = codec.encode(m)
    assert state.indices.numel() == (state.pair_count * 5 + 7) // 8
    assert codec.decode(state).shape == m.shape
    assert StructuralVQCodec(codebook(64), rank=4).index_bits == 6


def test_split_codec_round_trips_and_serializes():
    split = SplitVQCodec([
        (0, 16, StructuralVQCodec(codebook(32), rank=2, block_size=256)),
        (16, 32, StructuralVQCodec(codebook(64), rank=4, block_size=256)),
    ], shape=(32, 32))
    m = torch.randn(32, 32)
    state = split.encode(m)
    assert isinstance(state, SplitVQState)
    decoded = split.decode(state)
    payload = RecursiveMuon._serialize_encoded(state)
    restored = RecursiveMuon._deserialize_encoded(payload)
    assert isinstance(restored, SplitVQState)
    assert torch.equal(split.decode(restored), decoded)
    assert split.storage_bits(state) > 0


def test_column_pairing_round_trips_non_square():
    for shape in ((32, 64), (64, 32)):
        m = torch.randn(*shape)
        col = StructuralVQCodec(codebook(32), rank=4, block_size=32, pair_axis="col")
        row = StructuralVQCodec(codebook(32), rank=4, block_size=32)
        out_col = col.decode(col.encode(m))
        out_row = row.decode(row.encode(m))
        assert out_col.shape == m.shape and torch.isfinite(out_col).all()
        err_col = float((out_col - m).norm())
        err_row = float((out_row - m).norm())
        assert err_col < err_row * 1.25, (err_col, err_row)


def test_build_codecs_honours_pair_axis(tmp_path):
    path = tmp_path / "books.pt"
    torch.save({"codebooks": {"w32": codebook(32)}}, path)
    cfg = {"recursive_codebook_path": str(path), "recursive_codebook_key": "w32",
           "recursive_rank": 2, "recursive_block_size": 384,
           "recursive_param_plan": [
               {"pattern": "transformer.h.*.attn.c_proj.weight", "rank": 2, "pair_axis": "col"},
               {"pattern": "transformer.h.*.mlp.*", "rank": 2},
           ]}
    model = _FakeModel()
    codecs = build_recursive_codecs(model, cfg, tmp_path)
    assert codecs[id(model.c_proj)].pair_axis == "col"
    assert codecs[id(model.w1)].pair_axis == "row"
    state = codecs[id(model.c_proj)].encode(model.c_proj.detach())
    assert codecs[id(model.c_proj)].decode(state).shape == model.c_proj.shape


def test_split_codec_rejects_gaps():
    with pytest.raises(ValueError):
        SplitVQCodec([(0, 8, StructuralVQCodec(codebook(), rank=1)),
                      (9, 16, StructuralVQCodec(codebook(), rank=1))], shape=(16, 8))


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = torch.nn.Parameter(torch.randn(192, 64))
        self.c_proj = torch.nn.Parameter(torch.randn(64, 64))
        self.w1 = torch.nn.Parameter(torch.randn(128, 64))

    def named_parameters(self, prefix=""):
        yield "transformer.h.0.attn.c_attn.weight", self.c_attn
        yield "transformer.h.0.attn.c_proj.weight", self.c_proj
        yield "transformer.h.0.mlp.w1.weight", self.w1


def test_build_codecs_honours_param_plan(tmp_path):
    path = tmp_path / "books.pt"
    torch.save({"codebooks": {"w64": codebook(64), "w32": codebook(32)}}, path)
    cfg = {
        "recursive_codebook_path": str(path), "recursive_codebook_key": "w64",
        "recursive_rank": 8, "recursive_block_size": 2048,
        "recursive_param_plan": [
            {"pattern": "transformer.h.*.attn.c_attn.weight", "splits": [
                {"rows": [0, 64], "rank": 2, "codebook_key": "w32"},
                {"rows": [64, 128], "rank": 2, "codebook_key": "w32"},
                {"rows": [128, 192], "rank": 6}]},
            {"pattern": "transformer.h.*.mlp.*", "rank": 2, "codebook_key": "w32"},
        ],
    }
    model = _FakeModel()
    codecs = build_recursive_codecs(model, cfg, tmp_path)
    split = codecs[id(model.c_attn)]
    assert isinstance(split, SplitVQCodec)
    assert [c.rank for c in split.subcodecs] == [2, 2, 6]
    assert [c.index_bits for c in split.subcodecs] == [5, 5, 6]
    assert codecs[id(model.c_proj)].rank == 8
    assert codecs[id(model.w1)].rank == 2 and codecs[id(model.w1)].index_bits == 5
    state = split.encode(model.c_attn.detach())
    assert split.decode(state).shape == (192, 64)


def test_randomized_svd_codec_matches_exact_and_is_deterministic():
    m = torch.randn(64, 64)
    exact = StructuralVQCodec(codebook(64), rank=8, block_size=512)
    rand = StructuralVQCodec(codebook(64), rank=8, block_size=512,
                             svd_mode="randomized", svd_power_iters=2)
    assert rand.svd_mode == "randomized"
    a = rand.decode(rand.encode(m))
    b = rand.decode(rand.encode(m))
    assert torch.equal(a, b)
    ref = exact.decode(exact.encode(m))
    rel_exact = float((ref - m).norm() / m.norm())
    rel_rand = float((a - m).norm() / m.norm())
    assert rel_rand < rel_exact * 1.05, (rel_exact, rel_rand)


def test_randomized_svd_zero_rank_is_allowed():
    codec = StructuralVQCodec(codebook(32), rank=0, block_size=256, svd_mode="randomized")
    m = torch.randn(16, 16)
    assert codec.decode(codec.encode(m)).shape == m.shape


def test_param_plan_validation_rejects_bad_entries():
    with pytest.raises(RecipeError):
        _validate_recursive_param_plan([])
    with pytest.raises(RecipeError):
        _validate_recursive_param_plan([{"pattern": "x", "rank": 2, "splits": []}])
    with pytest.raises(RecipeError):
        _validate_recursive_param_plan([{"pattern": "x", "splits": [
            {"rows": [0, 8], "rank": 2}, {"rows": [9, 16], "rank": 2}]}])
    with pytest.raises(RecipeError):
        _validate_recursive_param_plan([{"pattern": "x", "rank": 0}])
    _validate_recursive_param_plan([{"pattern": "x", "rank": 2}])
