"""Recursive structural-state Muon prototype.

This module is intentionally separate from :mod:`muon_reference`.  It is an
offline/experimental optimizer whose Muon momentum is retained as a serialized
structural representation between calls to ``step``. An opt-in causal oracle
may additionally retain only the FP32 persistence residual; it never retains
a parallel FP32 momentum shadow. The implementation uses an exact truncated SVD for the
structure extraction (and therefore labels itself an oracle-structure
prototype); this keeps the persistence semantics correct while a practical
warm-start extractor is evaluated separately.
"""
from __future__ import annotations

import math
import copy
import fnmatch
from pathlib import Path
from dataclasses import dataclass

import torch

from .muon_reference import zeropower_newton_schulz
from .state_simulation import create_bitsandbytes_dynamic_map


def _packed_device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


def _percentile_order_statistic(values: torch.Tensor, percentile: float, dim: int) -> torch.Tensor:
    """Selection-based percentile (order statistic instead of a full sort).

    ``torch.quantile`` sorts every block and interpolates between the two
    neighbouring order statistics; ``kthvalue`` selects a single rank in O(n).
    The selected rank matches the un-interpolated quantile position, so the two
    differ only by the linear-interpolation term (<0.1% on 2048-value blocks),
    which does not change the codec's fidelity.
    """
    n = values.shape[dim]
    k = min(n, max(1, int(round(percentile * (n - 1))) + 1))
    return torch.kthvalue(values, k, dim=dim).values


def _block_stat_scales(flat: torch.Tensor, block_size: int, *, percentile: float | None = None) -> torch.Tensor:
    """Vectorized block statistics with one optional short tail block."""
    if flat.ndim != 1 or block_size <= 0:
        raise ValueError("flat input and positive block_size are required")
    full = (flat.numel() // block_size) * block_size
    pieces = []
    if full:
        blocks = flat[:full].reshape(-1, block_size)
        abs_blocks = blocks.abs()
        if percentile is None:
            scales = abs_blocks.amax(dim=1)
        else:
            scales = _percentile_order_statistic(abs_blocks, percentile, dim=1)
        nonzero = abs_blocks.amax(dim=1) != 0
        scales = torch.where(nonzero, scales.clamp_min(torch.finfo(torch.float32).tiny), torch.zeros_like(scales))
        pieces.append(scales)
    if full != flat.numel():
        tail = flat[full:]
        abs_tail = tail.abs()
        scale = (abs_tail.amax() if percentile is None
                 else _percentile_order_statistic(abs_tail, percentile, dim=0))
        scale = torch.where(abs_tail.amax() != 0, scale.clamp_min(torch.finfo(torch.float32).tiny), torch.zeros_like(scale))
        pieces.append(scale.reshape(1))
    return torch.cat(pieces) if pieces else flat.new_empty((0,))


def _block_mean_abs(flat: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-block mean |value|; the MSE-optimal 1-bit reconstruction amplitude."""
    if flat.ndim != 1 or block_size <= 0:
        raise ValueError("flat input and positive block_size are required")
    full = (flat.numel() // block_size) * block_size
    pieces = []
    if full:
        pieces.append(flat[:full].reshape(-1, block_size).abs().mean(dim=1))
    if full != flat.numel():
        pieces.append(flat[full:].abs().mean().reshape(1))
    return torch.cat(pieces) if pieces else flat.new_empty((0,))


def _repeat_block_scales(scales: torch.Tensor, count: int, block_size: int) -> torch.Tensor:
    """Expand one scalar scale per block to one scalar per value."""
    if count == 0:
        return scales.new_empty((0,))
    return torch.repeat_interleave(scales, block_size)[:count]


def periodic_error_accumulator_next(accumulator: torch.Tensor, residual: torch.Tensor,
                                    momentum: float, *, correction_applied: bool) -> torch.Tensor:
    """Advance unpaid persistence residuals to the next update.

    ``accumulator`` is injected and paid before the current candidate when a
    correction event is due.  Otherwise its contribution remains outstanding
    and decays once through the momentum recurrence.
    """
    if correction_applied:
        return residual.mul(momentum)
    return accumulator.mul(momentum).add(residual, alpha=momentum)


def periodic_correction_due(update: int, interval: int) -> bool:
    if update <= 0 or interval <= 0:
        raise ValueError("update and correction interval must be positive")
    return update % interval == 0


def pack_indices(indices: torch.Tensor, bits: int, *, count: int | None = None) -> torch.Tensor:
    """Pack fixed-width unsigned indices into bytes, least-significant-bit first.

    The operation is vectorized and stays on the input device.  This matters
    for recursive CUDA runs: the packed state is copied to CPU only after the
    device-side packing has completed, rather than iterating over every pair in
    Python.
    """
    if bits not in (1, 2, 4, 5, 6, 8, 10, 12):
        raise ValueError("the recursive prototype packs 1/2/4/5/6/8/10/12-bit indices")
    values = indices.detach().to(dtype=torch.int64).reshape(-1)
    if bool((values < 0).any()) or bool((values >= (1 << bits)).any()):
        raise ValueError("index is outside the declared codebook")
    n = int(values.numel() if count is None else count)
    if n != values.numel():
        raise ValueError("count does not match indices")
    if n == 0:
        return torch.empty((0,), dtype=torch.uint8, device=values.device)
    if bits != 6:
        # Generic little-endian bit packing: the group size is the smallest
        # number of indices that fills whole bytes (2 for 4-bit, 8 for 5-bit).
        group = 8 // math.gcd(bits, 8)
        per_group = group * bits // 8
        out = torch.empty((math.ceil(n * bits / 8),), dtype=torch.uint8, device=values.device)
        groups = n // group
        if groups:
            v = values[:groups * group].reshape(groups, group)
            weights = torch.bitwise_left_shift(
                torch.ones((group,), dtype=torch.int64, device=values.device),
                bits * torch.arange(group, dtype=torch.int64, device=values.device))
            acc = (v * weights).sum(dim=1)
            shifts = torch.arange(per_group, dtype=torch.int64, device=values.device) * 8
            packed = torch.bitwise_and(
                torch.bitwise_right_shift(acc.unsqueeze(1), shifts), 0xFF).to(torch.uint8)
            out[:groups * per_group].reshape(groups, per_group).copy_(packed)
        tail = n - groups * group
        if tail:
            base = groups * per_group
            tail_values = values[groups * group:]
            weights = torch.bitwise_left_shift(
                torch.ones((tail,), dtype=torch.int64, device=values.device),
                bits * torch.arange(tail, dtype=torch.int64, device=values.device))
            acc = (tail_values * weights).sum()
            tail_bytes = math.ceil(tail * bits / 8)
            shifts = torch.arange(tail_bytes, dtype=torch.int64, device=values.device) * 8
            out[base:base + tail_bytes] = torch.bitwise_and(
                torch.bitwise_right_shift(acc, shifts), 0xFF).to(torch.uint8)
        return out
    # Four 6-bit indices occupy exactly three bytes.  This is the same
    # least-significant-bit-first layout used by the legacy scalar loop, but
    # uses tensor slices instead of one Python operation per index.
    groups = n // 4
    out = torch.empty((math.ceil(n * bits / 8),), dtype=torch.uint8, device=values.device)
    if groups:
        v = values[:groups * 4].reshape(groups, 4)
        out[:groups * 3].reshape(groups, 3).copy_(torch.stack((
            v[:, 0] | ((v[:, 1] & 0x03) << 6),
            (v[:, 1] >> 2) | ((v[:, 2] & 0x0F) << 4),
            (v[:, 2] >> 4) | (v[:, 3] << 2),
        ), dim=1).to(torch.uint8))
    tail = n - groups * 4
    if tail:
        v = values[groups * 4:]
        base = groups * 3
        out[base] = (v[0] | ((v[1] & 0x03) << 6) if tail >= 2 else v[0]).to(torch.uint8)
        if tail >= 2:
            out[base + 1] = ((v[1] >> 2) | ((v[2] & 0x0F) << 4) if tail >= 3 else (v[1] >> 2)).to(torch.uint8)
        if tail >= 3:
            out[base + 2] = (v[2] >> 4).to(torch.uint8)
    return out


def unpack_indices(packed: torch.Tensor, count: int, bits: int = 6) -> torch.Tensor:
    """Inverse of :func:`pack_indices`; unused trailing bits are ignored."""
    if bits not in (1, 2, 4, 5, 6, 8, 10, 12) or count < 0:
        raise ValueError("the recursive prototype uses nonnegative count and 1/2/4/5/6/8/10/12-bit indices")
    raw = packed.detach().to(dtype=torch.int64).reshape(-1)
    if raw.numel() != math.ceil(count * bits / 8):
        raise ValueError("packed byte count does not match index count")
    if count == 0:
        return torch.empty((0,), dtype=torch.long, device=raw.device)
    if bits != 6:
        group = 8 // math.gcd(bits, 8)
        per_group = group * bits // 8
        mask = (1 << bits) - 1
        out = torch.empty((count,), dtype=torch.long, device=raw.device)
        groups = count // group
        if groups:
            b = raw[:groups * per_group].reshape(groups, per_group)
            shifts = torch.arange(per_group, dtype=torch.int64, device=raw.device) * 8
            acc = torch.bitwise_left_shift(b, shifts).sum(dim=1)
            v = out[:groups * group].reshape(groups, group)
            positions = bits * torch.arange(group, dtype=torch.int64, device=raw.device)
            v[:, :] = torch.bitwise_and(
                torch.bitwise_right_shift(acc.unsqueeze(1), positions), mask)
        tail = count - groups * group
        if tail:
            base = groups * per_group
            tail_bytes = math.ceil(tail * bits / 8)
            tb = raw[base:base + tail_bytes]
            shifts = torch.arange(tail_bytes, dtype=torch.int64, device=raw.device) * 8
            acc = torch.bitwise_left_shift(tb, shifts).sum()
            positions = bits * torch.arange(tail, dtype=torch.int64, device=raw.device)
            out[groups * group:] = torch.bitwise_and(
                torch.bitwise_right_shift(acc, positions), mask)
        return out
    groups = count // 4
    out = torch.empty((count,), dtype=torch.long, device=raw.device)
    if groups:
        b = raw[:groups * 3].reshape(groups, 3)
        v = out[:groups * 4].reshape(groups, 4)
        v[:, 0] = b[:, 0] & 0x3F
        v[:, 1] = (b[:, 0] >> 6) | ((b[:, 1] & 0x0F) << 2)
        v[:, 2] = (b[:, 1] >> 4) | ((b[:, 2] & 0x03) << 4)
        v[:, 3] = (b[:, 2] >> 2) & 0x3F
    tail = count - groups * 4
    if tail:
        base_b = groups * 3; base_v = groups * 4
        out[base_v] = raw[base_b] & 0x3F
        if tail >= 2:
            out[base_v + 1] = (raw[base_b] >> 6) | ((raw[base_b + 1] & 0x0F) << 2)
        if tail >= 3:
            out[base_v + 2] = (raw[base_b + 1] >> 4) | ((raw[base_b + 2] & 0x03) << 4)
    return out


@dataclass
class StructuralVQState:
    """Serialized persistent state for one 2-D Muon momentum tensor.

    ``u``, ``singular_values`` and ``vh`` are BF16, ``scales`` are FP32, and
    ``indices`` are packed fixed-width values (6-bit for 64-word codebooks,
    5-bit for 32-word codebooks).  No FP32 momentum is stored.
    """

    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    scales: torch.Tensor
    indices: torch.Tensor
    pair_count: int
    shape: tuple[int, int]
    codebook_key: str

    def to(self, device: torch.device) -> "StructuralVQState":
        return StructuralVQState(self.u.to(device), self.singular_values.to(device), self.vh.to(device),
                                 self.scales.to(device), self.indices.to(device), self.pair_count,
                                 self.shape, self.codebook_key)

    def state_dict(self) -> dict:
        return {
            "u": self.u.cpu(), "singular_values": self.singular_values.cpu(),
            "vh": self.vh.cpu(), "scales": self.scales.cpu(),
            "indices": self.indices.cpu(), "pair_count": self.pair_count,
            "shape": tuple(self.shape), "codebook_key": self.codebook_key,
        }

    @classmethod
    def from_state_dict(cls, value: dict) -> "StructuralVQState":
        return cls(value["u"], value["singular_values"], value["vh"], value["scales"],
                   value["indices"], int(value["pair_count"]), tuple(value["shape"]),
                   str(value["codebook_key"]))


def encode_compressed_error(error: torch.Tensor, *, block_size: int = 2048,
                            bits: int = 1) -> CompressedErrorState:
    """Compress the persistence residual for error feedback.

    ``bits=1`` keeps one sign per scalar plus a per-block mean-absolute
    amplitude (the MSE-optimal scale for a fixed sign pattern); ``bits=2`` uses
    four levels (+-0.5, +-1.5 in units of that amplitude).
    """
    if bits not in (1, 2):
        raise ValueError("error feedback supports 1 or 2 bits")
    value = error.detach().float()
    if value.ndim != 2:
        raise ValueError("error feedback requires a 2-D tensor")
    flat = value.reshape(-1)
    scales = _block_mean_abs(flat, block_size)
    amp = _repeat_block_scales(scales, flat.numel(), block_size)
    safe = amp.clamp_min(torch.finfo(torch.float32).tiny)
    unit = flat / safe
    if bits == 1:
        codes = (flat < 0).to(torch.long)
    else:
        codes = (flat < 0).to(torch.long) * 2 + (unit.abs() >= 1).to(torch.long)
    return CompressedErrorState(scales.float(), pack_indices(codes, bits),
                                int(flat.numel()), tuple(value.shape), bits)


def decode_compressed_error(state: CompressedErrorState, *, device=None,
                            block_size: int = 2048) -> torch.Tensor:
    device = device or torch.device("cpu")
    state = state.to(device)
    codes = unpack_indices(state.codes, state.count, state.bits).float()
    amp = _repeat_block_scales(state.scales, state.count, block_size)
    if state.bits == 1:
        values = (1.0 - 2.0 * codes) * amp
    else:
        sign = torch.where(codes < 2, 1.0, -1.0)
        mag = torch.where(torch.remainder(codes, 2) > 0, 1.5, 0.5)
        values = sign * mag * amp
    return values.reshape(state.shape)


def compressed_error_bits(state: CompressedErrorState) -> int:
    return int(state.scales.numel() * 32 + state.codes.numel() * 8)


class StructuralVQCodec:
    """Structural low-rank (exact or randomized SVD) + p98 2-D VQ codec."""

    def __init__(self, codebook: torch.Tensor, *, rank: int = 8,
                 block_size: int = 2048, codebook_key: str = "s0_k8_w64_t8_v1200",
                 svd_mode: str = "exact", svd_power_iters: int = 2,
                 svd_oversample: int = 4, svd_seed: int = 20260101,
                 pair_axis: str = "row"):
        codebook = codebook.detach().float().cpu()
        if (codebook.ndim != 2 or codebook.shape[1] not in (2, 4)
                or codebook.shape[0] < 2 or (codebook.shape[0] & (codebook.shape[0] - 1))
                or not torch.isfinite(codebook).all()):
            raise ValueError("recursive prototype requires a finite (2**b x 2) or (2**b x 4) codebook")
        if not bool((codebook.square().sum(dim=1) == 0).any()):
            raise ValueError("codebook must contain an exact zero codeword")
        if rank < 0 or block_size <= 0:
            raise ValueError("rank and block_size must be positive")
        if svd_mode not in {"exact", "randomized"}:
            raise ValueError("svd_mode must be 'exact' or 'randomized'")
        if svd_power_iters < 0 or svd_oversample < 0:
            raise ValueError("svd power iterations and oversampling must be nonnegative")
        if pair_axis not in {"row", "col"}:
            raise ValueError("pair_axis must be 'row' or 'col'")
        self.codebook = codebook
        self.group_size = int(codebook.shape[1])
        self.index_bits = int(round(math.log2(codebook.shape[0])))
        if block_size % self.group_size:
            raise ValueError("block_size must be divisible by the group size")
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.codebook_key = str(codebook_key)
        self.svd_mode = str(svd_mode)
        self.svd_power_iters = int(svd_power_iters)
        self.svd_oversample = int(svd_oversample)
        self.svd_seed = int(svd_seed)
        self.pair_axis = str(pair_axis)
        self._codebook_cache: dict[tuple[str, int | None], torch.Tensor] = {}

    def _top_factors(self, value: torch.Tensor, k: int):
        """Return truncated ``U, S, Vh`` of ``value`` under the configured extractor."""
        if k == 0:
            empty = value.new_empty((value.shape[0], 0))
            return empty, value.new_empty((0,)), value.new_empty((0, value.shape[1]))
        if self.svd_mode == "exact":
            u, s, vh = torch.linalg.svd(value, full_matrices=False)
            return u[:, :k], s[:k], vh[:k]
        # Deterministic randomized range finder: the seed is fixed per codec, so
        # a resumed run reproduces the same projection at every update.
        n = value.shape[1]
        width = min(n, k + self.svd_oversample)
        generator = torch.Generator(device=value.device)
        generator.manual_seed(self.svd_seed)
        omega = torch.randn(n, width, generator=generator, device=value.device,
                            dtype=value.dtype)
        y = value @ omega
        for _ in range(self.svd_power_iters):
            y = value @ (value.transpose(0, 1) @ y)
        q, _ = torch.linalg.qr(y)
        b = q.transpose(0, 1) @ value
        ub, s, vh = torch.linalg.svd(b, full_matrices=False)
        return (q @ ub[:, :k]), s[:k], vh[:k]

    def _codebook_on(self, device: torch.device) -> torch.Tensor:
        key = _packed_device_key(device)
        if key not in self._codebook_cache:
            self._codebook_cache[key] = self.codebook.to(device=device)
        return self._codebook_cache[key]

    @torch.no_grad()
    def encode(self, momentum: torch.Tensor) -> StructuralVQState:
        # Run the expensive SVD, scale estimation, and nearest-codeword search
        # on the same device as the live optimizer state.  Only the serialized
        # representation is copied to CPU after encoding; moving the input to
        # CPU here would make a CUDA recursive run silently CPU-bound.
        value = momentum.detach().float()
        if value.ndim != 2:
            raise ValueError("structural recursive state requires a 2-D matrix")
        k = min(self.rank, int(min(value.shape)))
        uk, sk, vhk = self._top_factors(value, k)
        low = (uk * sk) @ vhk if k else torch.zeros_like(value)
        residual = value - low
        if residual.numel() % self.group_size:
            raise ValueError("recursive grouped VQ requires a size divisible by the group size")
        # All current formal Muon matrices are even-sized.  Avoid the generic
        # offline pairing helper here: it builds Python lists/sets over every
        # scalar and is not part of the recursive representation semantics.
        work = (residual if self.pair_axis == "row"
                else residual.transpose(0, 1).contiguous())
        groups = work.reshape(-1, self.group_size)
        scales = _block_stat_scales(work.reshape(-1), self.block_size, percentile=.98)
        pair_scales = _repeat_block_scales(scales, groups.shape[0],
                                           self.block_size // self.group_size)
        safe_scales = pair_scales.clamp_min(torch.finfo(torch.float32).tiny)
        normalized = (groups / safe_scales[:, None]).clamp(-1, 1)
        codebook = self._codebook_on(value.device)
        indices = torch.empty((groups.shape[0],), dtype=torch.long, device=value.device)
        # A small number of large chunks bounds temporary memory without
        # reintroducing a block-sized Python loop or GPU synchronization.
        for start in range(0, groups.shape[0], 65536):
            end = min(start + 65536, groups.shape[0])
            indices[start:end] = torch.cdist(normalized[start:end], codebook).argmin(dim=1)
        indices = torch.where(pair_scales > 0, indices, torch.zeros_like(indices))
        return StructuralVQState(uk.to(torch.bfloat16), sk.to(torch.bfloat16),
                                 vhk.to(torch.bfloat16), scales.float(),
                                 pack_indices(indices, self.index_bits), int(indices.numel()),
                                 tuple(value.shape), self.codebook_key)

    @torch.no_grad()
    def decode(self, state: StructuralVQState, *, device=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        if state.u.device != device:
            state = state.to(device)
        u = state.u.to(device=device, dtype=torch.float32)
        s = state.singular_values.to(device=device, dtype=torch.float32)
        vh = state.vh.to(device=device, dtype=torch.float32)
        low = (u * s) @ vh if s.numel() else torch.zeros(state.shape, device=device)
        indices = unpack_indices(state.indices, state.pair_count, self.index_bits)
        scales = state.scales.to(device=device, dtype=torch.float32)
        pair_scales = _repeat_block_scales(scales, state.pair_count,
                                           self.block_size // self.group_size)
        pairs = self._codebook_on(device)[indices] * pair_scales[:, None]
        if self.pair_axis == "row":
            return low + pairs.reshape(state.shape)
        return low + pairs.reshape(state.shape[1], state.shape[0]).transpose(0, 1)

    def storage_bits(self, state: StructuralVQState) -> int:
        """Exact serialized payload bits, excluding Python/container overhead."""
        return int(state.u.numel() * 16 + state.singular_values.numel() * 16 +
                   state.vh.numel() * 16 + state.scales.numel() * 32 +
                   state.indices.numel() * 8)


@dataclass
class SplitVQState:
    """Row-partitioned VQ state: one :class:`StructuralVQState` per row block."""

    parts: list
    shape: tuple[int, int]

    def to(self, device: torch.device) -> "SplitVQState":
        return SplitVQState([part.to(device) for part in self.parts], self.shape)

    def state_dict(self) -> dict:
        return {"parts": [part.state_dict() for part in self.parts],
                "shape": tuple(self.shape)}

    @classmethod
    def from_state_dict(cls, value: dict) -> "SplitVQState":
        return cls([StructuralVQState.from_state_dict(part) for part in value["parts"]],
                   tuple(value["shape"]))


@dataclass
class CompressedErrorState:
    """1-2 bit error feedback: per-block mean-absolute scale plus sign/level bits."""

    scales: torch.Tensor
    codes: torch.Tensor
    count: int
    shape: tuple[int, int]
    bits: int = 1

    def to(self, device: torch.device) -> "CompressedErrorState":
        return CompressedErrorState(self.scales.to(device), self.codes.to(device),
                                    self.count, self.shape, self.bits)

    def state_dict(self) -> dict:
        return {"scales": self.scales.cpu(), "codes": self.codes.cpu(),
                "count": self.count, "shape": tuple(self.shape), "bits": self.bits}

    @classmethod
    def from_state_dict(cls, value: dict) -> "CompressedErrorState":
        return cls(value["scales"], value["codes"], int(value["count"]),
                   tuple(value["shape"]), int(value.get("bits", 1)))


class SplitVQCodec:
    """Compose independent row-block codecs into one per-parameter codec.

    The fused QKV projection is a single Muon parameter, but Q, K and V need
    different ranks and index widths.  This wrapper encodes each contiguous row
    band with its own :class:`StructuralVQCodec` and decodes by concatenation.
    """

    def __init__(self, blocks, *, shape: tuple[int, int]):
        if not blocks:
            raise ValueError("split codec requires at least one row block")
        shape = tuple(int(x) for x in shape)
        cursor = 0
        for start, end, _codec in blocks:
            if start != cursor or end <= start:
                raise ValueError("split codec row blocks must be contiguous and ordered")
            cursor = end
        if cursor != shape[0]:
            raise ValueError("split codec row blocks must cover the full matrix")
        self.blocks = list(blocks)
        self.shape = shape

    @property
    def subcodecs(self):
        return [codec for _start, _end, codec in self.blocks]

    @torch.no_grad()
    def encode(self, momentum: torch.Tensor) -> SplitVQState:
        value = momentum.detach().float()
        if tuple(value.shape) != self.shape:
            raise ValueError("split codec shape mismatch")
        return SplitVQState([codec.encode(value[start:end])
                             for start, end, codec in self.blocks], self.shape)

    @torch.no_grad()
    def decode(self, state: SplitVQState, *, device=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        return torch.cat([codec.decode(part, device=device)
                          for (_start, _end, codec), part in zip(self.blocks, state.parts)], dim=0)

    def storage_bits(self, state: SplitVQState) -> int:
        return int(sum(codec.storage_bits(part) for (_start, _end, codec), part
                       in zip(self.blocks, state.parts)))


@dataclass
class StructuralINT4State:
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    scales: torch.Tensor
    codes: torch.Tensor
    count: int
    shape: tuple[int, int]

    def to(self, device: torch.device) -> "StructuralINT4State":
        return StructuralINT4State(self.u.to(device), self.singular_values.to(device), self.vh.to(device),
                                   self.scales.to(device), self.codes.to(device), self.count, self.shape)


class StructuralINT4Codec:
    """Packed structural INT4 residual codec using the production dynamic map."""

    def __init__(self, *, rank: int = 8, block_size: int = 2048):
        self.rank, self.block_size = int(rank), int(block_size)
        self.map = create_bitsandbytes_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4)
        self._map_cache: dict[tuple[str, int | None], torch.Tensor] = {}

    def _map_on(self, device: torch.device) -> torch.Tensor:
        key = _packed_device_key(device)
        if key not in self._map_cache:
            self._map_cache[key] = self.map.to(device=device)
        return self._map_cache[key]

    @torch.no_grad()
    def encode(self, momentum: torch.Tensor) -> StructuralINT4State:
        # Keep the heavy exact-SVD and residual quantization work on the live
        # parameter device.  Only packed persistent state is copied to CPU.
        value = momentum.detach().float()
        if value.ndim != 2:
            raise ValueError("structural recursive state requires a 2-D matrix")
        u, s, vh = torch.linalg.svd(value, full_matrices=False)
        k = min(self.rank, int(s.numel())); uk, sk, vhk = u[:, :k], s[:k], vh[:k]
        low = (uk * sk) @ vhk if k else torch.zeros_like(value)
        flat = (value - low).reshape(-1)
        scales = _block_stat_scales(flat, self.block_size)
        value_scales = _repeat_block_scales(scales, flat.numel(), self.block_size)
        safe_scales = value_scales.clamp_min(torch.finfo(torch.float32).tiny)
        normalized = (flat / safe_scales).reshape(-1, 1)
        codebook = self._map_on(value.device).reshape(-1, 1)
        codes = torch.empty_like(flat, dtype=torch.long)
        for start in range(0, flat.numel(), 65536):
            end = min(start + 65536, flat.numel())
            codes[start:end] = torch.cdist(normalized[start:end], codebook).argmin(dim=1)
        codes = torch.where(value_scales > 0, codes, torch.zeros_like(codes))
        raw = codes.to(torch.int64)
        packed = torch.empty((math.ceil(raw.numel() / 2),), dtype=torch.uint8, device=value.device)
        even = raw[0::2]; odd = raw[1::2]
        pairs = raw.numel() // 2
        if pairs:
            packed[:pairs] = (even[:pairs] | (odd[:pairs] << 4)).to(torch.uint8)
        if raw.numel() % 2:
            packed[-1] = raw[-1].to(torch.uint8)
        return StructuralINT4State(uk.to(torch.bfloat16), sk.to(torch.bfloat16),
                                   vhk.to(torch.bfloat16), scales.float(),
                                   packed, int(raw.numel()), tuple(value.shape))

    @torch.no_grad()
    def decode(self, state: StructuralINT4State, *, device=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        if state.u.device != device:
            state = state.to(device)
        u = state.u.to(device, torch.float32); s = state.singular_values.to(device, torch.float32); vh = state.vh.to(device, torch.float32)
        low = (u * s) @ vh if s.numel() else torch.zeros(state.shape, device=device)
        packed = state.codes.to(device=device, dtype=torch.int64); raw = torch.empty(state.count, dtype=torch.long, device=device)
        pairs = state.count // 2
        if pairs:
            raw[0:2 * pairs:2] = packed[:pairs] & 15
            raw[1:2 * pairs:2] = (packed[:pairs] >> 4) & 15
        if state.count % 2:
            raw[-1] = packed[-1] & 15
        value_scales = _repeat_block_scales(state.scales.to(device=device, dtype=torch.float32), state.count, self.block_size)
        flat = self._map_on(device)[raw] * value_scales
        return low + flat.reshape(state.shape)

    def storage_bits(self, state: StructuralINT4State) -> int:
        return int(state.u.numel()*16 + state.singular_values.numel()*16 + state.vh.numel()*16 + state.scales.numel()*32 + state.codes.numel()*8)


def build_recursive_codecs(model, cfg: dict, root: Path) -> dict[int, object]:
    """Build codecs for the experimental recipe without changing production paths.

    ``recursive_param_plan`` optionally overrides rank / block size / codebook
    per parameter (first matching ``fnmatch`` pattern wins) and can split one
    parameter into contiguous row bands with independent codecs.
    """
    rank = int(cfg.get("recursive_rank", 8)); block = int(cfg.get("recursive_block_size", 2048))
    mode = cfg.get("recursive_structure_mode", "exact_svd_oracle")
    if mode != "exact_svd_oracle":
        raise ValueError("only exact_svd_oracle is currently implemented")
    kind = cfg.get("recursive_representation", "vq_int3")
    plan = list(cfg.get("recursive_param_plan") or [])
    if plan and kind != "vq_int3":
        raise ValueError("recursive_param_plan requires recursive_representation=vq_int3")
    codebooks: dict[str, torch.Tensor] = {}
    blob = None
    if cfg.get("recursive_codebook_path"):
        path = Path(cfg["recursive_codebook_path"])
        if not path.is_absolute(): path = (root / path).resolve()
        blob = torch.load(path, map_location="cpu", weights_only=False)

    def codebook_for(key):
        if key is None:
            return None
        if blob is None:
            raise ValueError("recursive_param_plan needs recursive_codebook_path")
        if key not in codebooks:
            codebooks[key] = blob["codebooks"][key].float()
        return codebooks[key]

    default_key = cfg.get("recursive_codebook_key")
    svd_kwargs = {"svd_mode": cfg.get("recursive_svd_mode", "exact"),
                  "svd_power_iters": int(cfg.get("recursive_svd_power_iters", 2)),
                  "svd_oversample": int(cfg.get("recursive_svd_oversample", 4))}
    codecs = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or not name.startswith("transformer.h."):
            continue
        if kind != "vq_int3":
            codecs[id(parameter)] = StructuralINT4Codec(rank=rank, block_size=block)
            continue
        entry = next((e for e in plan if fnmatch.fnmatchcase(name, str(e["pattern"]))), None)
        if entry is not None and "splits" in entry:
            blocks = []
            for split in entry["splits"]:
                start, end = (int(x) for x in split["rows"])
                key = split.get("codebook_key", default_key)
                blocks.append((start, end, StructuralVQCodec(
                    codebook_for(key), rank=int(split["rank"]),
                    block_size=int(split.get("block_size", block)),
                    codebook_key=key or "",
                    pair_axis=split.get("pair_axis", "row"), **svd_kwargs)))
            codecs[id(parameter)] = SplitVQCodec(blocks, shape=tuple(parameter.shape))
        else:
            local_rank = int(entry["rank"]) if entry is not None and "rank" in entry else rank
            local_block = int(entry.get("block_size", block)) if entry is not None else block
            key = entry.get("codebook_key", default_key) if entry is not None else default_key
            codecs[id(parameter)] = StructuralVQCodec(
                codebook_for(key), rank=local_rank, block_size=local_block,
                codebook_key=key or "",
                pair_axis=(entry.get("pair_axis", "row") if entry is not None else "row"),
                **svd_kwargs)
    if not codecs:
        raise ValueError("recursive_muon found no eligible 2-D Muon parameters")
    return codecs


class RecursiveMuon(torch.optim.Optimizer):
    """Muon whose selected 2-D momentum states recursively persist via codec.

    This is deliberately an experimental optimizer. Non-Muon parameters use
    ordinary FP32 AdamW moments. ``recursive_error_feedback_alpha`` controls an
    optional persistent FP32 residual of the compressed Muon state, not an
    uncompressed momentum copy. Periodic mode stores a decayed unpaid-residual
    accumulator and injects it only at configured update multiples.
    """

    def __init__(self, params, codecs: dict[int, StructuralVQCodec], *, lr=1e-3,
                 betas=(.9, .999), eps=1e-8, weight_decay=.1, muon_momentum=.95,
                 muon_nesterov=True, muon_ns_steps=5,
                 muon_ns_coefficients=(3.4445, -4.7750, 2.0315), muon_eps=1e-7,
                 recursive_error_feedback_alpha=0.0,
                 recursive_error_feedback_mode=None,
                 recursive_error_feedback_interval=None,
                 recursive_error_feedback_bits=0):
        if isinstance(recursive_error_feedback_alpha, bool):
            raise ValueError("recursive_error_feedback_alpha must be finite and nonnegative")
        alpha = float(recursive_error_feedback_alpha)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("recursive_error_feedback_alpha must be finite and nonnegative")
        mode = recursive_error_feedback_mode or ("fractional" if alpha > 0 else "none")
        if mode not in {"none", "fractional", "periodic"}:
            raise ValueError("recursive_error_feedback_mode must be none, fractional, or periodic")
        if mode == "periodic":
            if isinstance(recursive_error_feedback_interval, bool) or not isinstance(recursive_error_feedback_interval, int) or recursive_error_feedback_interval <= 0:
                raise ValueError("periodic feedback requires a positive integer interval")
            if alpha != 0:
                raise ValueError("periodic feedback requires recursive_error_feedback_alpha=0")
        elif recursive_error_feedback_interval is not None:
            raise ValueError("recursive_error_feedback_interval is only valid in periodic mode")
        if mode == "none" and alpha != 0:
            raise ValueError("none feedback mode requires alpha=0")
        if mode == "fractional" and alpha <= 0:
            raise ValueError("fractional feedback mode requires alpha>0")
        if recursive_error_feedback_bits not in (0, 1, 2):
            raise ValueError("recursive_error_feedback_bits must be 0, 1 or 2")
        if recursive_error_feedback_bits and (mode != "fractional" or alpha <= 0):
            raise ValueError("compressed error feedback requires fractional mode with alpha>0")

        self.codecs = codecs
        self.recursive_error_feedback_bits = int(recursive_error_feedback_bits)
        self.recursive_error_feedback_alpha = alpha
        self.recursive_error_feedback_mode = mode
        self.recursive_error_feedback_interval = recursive_error_feedback_interval
        self._mechanism_observer = None
        self._mechanism_update = None
        self._training_update = None
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        muon_momentum=muon_momentum, muon_nesterov=muon_nesterov,
                        muon_ns_steps=muon_ns_steps, muon_ns_coefficients=tuple(muon_ns_coefficients),
                        muon_eps=muon_eps, recursive_error_feedback_alpha=alpha,
                        recursive_error_feedback_mode=mode,
                        recursive_error_feedback_interval=recursive_error_feedback_interval,
                        recursive_error_feedback_bits=int(recursive_error_feedback_bits))
        super().__init__(params, defaults)

    def set_mechanism_observer(self, observer):
        self._mechanism_observer = observer

    def set_mechanism_update(self, update):
        self._mechanism_update = int(update)

    def set_training_update(self, update):
        """Supply the global recipe update index for deterministic schedules."""
        self._training_update = int(update)
        self._mechanism_update = int(update)

    @staticmethod
    def _serialize_encoded(value):
        if isinstance(value, StructuralVQState):
            return {"_recursive_state_type": "vq", **value.state_dict()}
        if isinstance(value, SplitVQState):
            return {"_recursive_state_type": "split_vq",
                    "parts": [RecursiveMuon._serialize_encoded(part) for part in value.parts],
                    "shape": tuple(value.shape)}
        if isinstance(value, CompressedErrorState):
            return {"_recursive_state_type": "compressed_error", **value.state_dict()}
        if isinstance(value, StructuralINT4State):
            return {"_recursive_state_type": "int4", "u": value.u.cpu(),
                    "singular_values": value.singular_values.cpu(), "vh": value.vh.cpu(),
                    "scales": value.scales.cpu(), "codes": value.codes.cpu(),
                    "count": value.count, "shape": tuple(value.shape)}
        return value.detach().cpu() if torch.is_tensor(value) else copy.deepcopy(value)

    @staticmethod
    def _deserialize_encoded(value):
        if not isinstance(value, dict):
            return value
        kind = value.get("_recursive_state_type")
        if kind == "vq":
            payload = {key: item for key, item in value.items() if key != "_recursive_state_type"}
            return StructuralVQState.from_state_dict(payload)
        if kind == "split_vq":
            return SplitVQState([RecursiveMuon._deserialize_encoded(part)
                                 for part in value["parts"]], tuple(value["shape"]))
        if kind == "compressed_error":
            payload = {key: item for key, item in value.items() if key != "_recursive_state_type"}
            return CompressedErrorState.from_state_dict(payload)
        if kind == "int4":
            return StructuralINT4State(value["u"], value["singular_values"], value["vh"],
                                       value["scales"], value["codes"], int(value["count"]),
                                       tuple(value["shape"]))
        return value

    def state_dict(self):
        """Serialize compressed state on CPU without changing live residency."""
        source = super().state_dict()
        result = {"state": {}, "param_groups": copy.deepcopy(source["param_groups"])}
        for parameter_id, state in source["state"].items():
            result["state"][parameter_id] = {
                key: self._serialize_encoded(value) for key, value in state.items()
            }
        return result

    def load_state_dict(self, state_dict):
        incoming = {"state": {}, "param_groups": copy.deepcopy(state_dict["param_groups"])}
        for parameter_id, state in state_dict["state"].items():
            incoming["state"][parameter_id] = {
                key: self._deserialize_encoded(value) for key, value in state.items()
            }
        super().load_state_dict(incoming)
        # Checkpoint values are CPU-portable; resume puts only compressed state
        # back on the active parameter device.  No FP32 momentum is created.
        for parameter, state in self.state.items():
            encoded = state.get("compressed_momentum")
            if isinstance(encoded, (StructuralVQState, StructuralINT4State, SplitVQState)):
                state["compressed_momentum"] = encoded.to(parameter.device)
            error_state = state.get("compressed_error")
            if isinstance(error_state, CompressedErrorState):
                state["compressed_error"] = error_state.to(parameter.device)
            error = state.get("momentum_error")
            if torch.is_tensor(error):
                state["momentum_error"] = error.to(device=parameter.device, dtype=torch.float32)
            accumulator = state.get("momentum_error_accumulator")
            if torch.is_tensor(accumulator):
                state["momentum_error_accumulator"] = accumulator.to(device=parameter.device, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if not any(group.get("optimizer_group") == "muon" for group in self.param_groups):
            raise RuntimeError("recursive_muon requires an optimizer_group='muon' parameter group")
        for group in self.param_groups:
            is_muon = group.get("optimizer_group") == "muon"
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().float()
                state = self.state[p]
                if is_muon:
                    codec = self.codecs.get(id(p))
                    if codec is None:
                        raise KeyError("missing recursive codec for Muon parameter")
                    compressed = state.get("compressed_momentum")
                    previous_decoded = torch.zeros_like(p, dtype=torch.float32) if compressed is None else codec.decode(compressed, device=p.device)
                    mu = group["muon_momentum"]
                    fb_bits = int(group.get("recursive_error_feedback_bits",
                                            self.recursive_error_feedback_bits))
                    if fb_bits:
                        error_state = state.get("compressed_error")
                        previous_error = (decode_compressed_error(error_state, device=p.device)
                                          if isinstance(error_state, CompressedErrorState)
                                          else torch.zeros_like(p, dtype=torch.float32))
                    else:
                        previous_error = state.get("momentum_error")
                        if previous_error is None:
                            previous_error = torch.zeros_like(p, dtype=torch.float32)
                    mode = group.get("recursive_error_feedback_mode", self.recursive_error_feedback_mode)
                    interval = group.get("recursive_error_feedback_interval", self.recursive_error_feedback_interval)
                    update = self._training_update if self._training_update is not None else self._mechanism_update
                    correction_applied = mode == "periodic" and update is not None and periodic_correction_due(update, interval)
                    previous_accumulator = state.get("momentum_error_accumulator")
                    if previous_accumulator is None:
                        previous_accumulator = torch.zeros_like(p, dtype=torch.float32)
                    momentum = previous_decoded.mul(mu).add(grad)
                    alpha = group.get("recursive_error_feedback_alpha", 0.0)
                    if alpha:
                        # Restore a controlled fraction of the previous
                        # persistence residual: mu * (decoded + alpha * error).
                        momentum.add_(previous_error, alpha=alpha * mu)
                    elif mode == "periodic" and correction_applied:
                        momentum.add_(previous_accumulator)
                    direction = grad.add(momentum, alpha=group["muon_momentum"]) if group["muon_nesterov"] else momentum
                    update = zeropower_newton_schulz(direction, group["muon_ns_steps"], group["muon_ns_coefficients"], group["muon_eps"])
                    updated = p.float().mul(1 - group["lr"] * group["weight_decay"]).add(update, alpha=-group["lr"])
                    encoded = codec.encode(momentum)
                    persisted_decoded = codec.decode(encoded, device=p.device) if (alpha or mode == "periodic" or self._mechanism_observer is not None) else None
                    current_error = momentum - persisted_decoded if (alpha or mode == "periodic") else None
                    next_accumulator = (periodic_error_accumulator_next(previous_accumulator, current_error, mu,
                                        correction_applied=correction_applied) if mode == "periodic" else None)
                    if self._mechanism_observer is not None:
                        self._mechanism_observer(parameter=p, gradient=grad, momentum_prev=previous_decoded,
                            momentum_candidate=momentum, momentum_persisted_decoded=persisted_decoded,
                            direction=direction, updated=updated, group=group, update=self._mechanism_update,
                            compressed_state=encoded, momentum_error_prev=previous_error,
                            momentum_error=current_error, error_feedback_alpha=alpha,
                            error_feedback_mode=mode,
                            error_feedback_interval=interval,
                            correction_applied=correction_applied,
                            error_accumulator_prev=previous_accumulator if mode == "periodic" else None,
                            injected_correction=previous_accumulator if correction_applied else torch.zeros_like(momentum),
                            error_accumulator_next=next_accumulator)
                    p.copy_(updated.to(p.dtype))
                    state["compressed_momentum"] = encoded
                    if alpha:
                        if fb_bits:
                            state["compressed_error"] = encode_compressed_error(
                                current_error, bits=fb_bits)
                            state.pop("momentum_error", None)
                        else:
                            state["momentum_error"] = current_error
                    if mode == "periodic":
                        state["momentum_error_accumulator"] = next_accumulator
                else:
                    step = state.get("step", 0) + 1; state["step"] = step
                    beta1, beta2 = group["betas"]
                    m = state.get("exp_avg", torch.zeros_like(p, dtype=torch.float32)).mul(beta1).add(grad, alpha=1-beta1)
                    v = state.get("exp_avg_sq", torch.zeros_like(p, dtype=torch.float32)).mul(beta2).addcmul(grad, grad, value=1-beta2)
                    denom = (v / (1-beta2**step)).sqrt().add(group["eps"])
                    update = (m / (1-beta1**step)) / denom
                    p.copy_(p.float().mul(1-group["lr"]*group["weight_decay"]).add(update, alpha=-group["lr"]).to(p.dtype))
                    state["exp_avg"], state["exp_avg_sq"] = m, v
        return loss

    def recursive_state_summary(self) -> dict:
        rows = []
        for p, state in self.state.items():
            encoded = state.get("compressed_momentum")
            if encoded is not None:
                error = state.get("momentum_error")
                error_state = state.get("compressed_error")
                codec = self.codecs[id(p)]
                if isinstance(error_state, CompressedErrorState):
                    error_bits = compressed_error_bits(error_state)
                else:
                    error_bits = int(error.numel() * error.element_size() * 8) if torch.is_tensor(error) else 0
                rows.append({"parameter_id": id(p), "shape": list(encoded.shape),
                             "storage_bits": codec.storage_bits(encoded),
                             "error_buffer_bits": error_bits,
                             "periodic_accumulator_bits": int(state["momentum_error_accumulator"].numel() * state["momentum_error_accumulator"].element_size() * 8) if torch.is_tensor(state.get("momentum_error_accumulator")) else 0,
                             "has_fp32_momentum": False, "has_fp32_error_buffer": torch.is_tensor(error),
                             "has_fp32_periodic_accumulator": torch.is_tensor(state.get("momentum_error_accumulator"))})
            else:
                # Auxiliary AdamW state is intentionally unchanged and remains
                # FP32; include it in accounting rather than silently dropping
                # it from the optimizer-state summary.
                for key, value in state.items():
                    if torch.is_tensor(value):
                        rows.append({"parameter_id": id(p), "key": key,
                                     "shape": list(value.shape),
                                     "storage_bits": int(value.numel() * value.element_size() * 8),
                                     "has_fp32_momentum": False})
        codebook_ids = set(); codebook_bits = 0
        for codec in self.codecs.values():
            for sub in (codec.subcodecs if isinstance(codec, SplitVQCodec) else [codec]):
                if hasattr(sub, "codebook"):
                    identity = ("vq", getattr(sub, "codebook_key", "default"))
                    if identity not in codebook_ids:
                        codebook_ids.add(identity); codebook_bits += int(sub.codebook.numel() * 32)
        compressed_bits = sum(r["storage_bits"] for r in rows)
        muon_rows = [r for r in rows if "key" not in r]
        compressed_muon_bits = sum(r["storage_bits"] for r in muon_rows)
        muon_scalar_count = sum(math.prod(r["shape"]) for r in muon_rows)
        error_buffer_bits = sum(r.get("error_buffer_bits", 0) for r in rows)
        periodic_accumulator_bits = sum(r.get("periodic_accumulator_bits", 0) for r in rows)
        persistent_bits = compressed_bits + error_buffer_bits + periodic_accumulator_bits + codebook_bits
        muon_total_bits = compressed_muon_bits + error_buffer_bits + periodic_accumulator_bits + codebook_bits
        # Keep the common run-summary contract in addition to the recursive
        # storage-specific fields.  The summary writer records bytes, while
        # this optimizer also exposes the exact serialized bit count.
        persistent_bytes = (persistent_bits + 7) // 8
        return {"logical_tensor_bytes": persistent_bytes,
                "unique_storage_bytes": persistent_bytes,
                "tensors": rows, "persistent_bits": persistent_bits,
                "codebook_bits": codebook_bits,
                "compressed_state_bits": compressed_bits, "error_buffer_bits": error_buffer_bits,
                "periodic_accumulator_bits": periodic_accumulator_bits,
                "compressed_muon_state_bits": compressed_muon_bits,
                "muon_scalar_count": muon_scalar_count,
                "muon_total_state_bits": muon_total_bits,
                "error_buffer_bytes": (error_buffer_bits + 7) // 8,
                "muon_effective_bits_per_value": muon_total_bits / muon_scalar_count if muon_scalar_count else 0.0,
                "error_feedback_alpha": self.recursive_error_feedback_alpha,
                "error_feedback_mode": self.recursive_error_feedback_mode,
                "error_feedback_interval": self.recursive_error_feedback_interval,
                "compressed_muon_tensor_count": sum("key" not in r for r in rows),
                "structure_mode": "exact_svd_oracle",
                "svd_mode": next((c.svd_mode for c in self.codecs.values()
                                  if isinstance(c, StructuralVQCodec)),
                                 next((c.subcodecs[0].svd_mode for c in self.codecs.values()
                                       if isinstance(c, SplitVQCodec)), None))}
