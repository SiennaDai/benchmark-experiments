"""Recursive structural-state Muon prototype.

This module is intentionally separate from :mod:`muon_reference`.  It is an
offline/experimental optimizer whose Muon momentum is *only* retained as a
serialized structural representation between calls to ``step``.  There is no
FP32 momentum shadow.  The implementation uses an exact truncated SVD for the
structure extraction (and therefore labels itself an oracle-structure
prototype); this keeps the persistence semantics correct while a practical
warm-start extractor is evaluated separately.
"""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import torch

from .muon_reference import zeropower_newton_schulz
from .muon_vector_int3 import pair_values, unpair_values, vector_scales
from .state_simulation import create_bitsandbytes_dynamic_map


def pack_indices(indices: torch.Tensor, bits: int, *, count: int | None = None) -> torch.Tensor:
    """Pack fixed-width unsigned indices into bytes, least-significant-bit first."""
    if bits not in (6,):
        raise ValueError("the recursive prototype currently packs the fixed 64-word/6-bit format")
    values = indices.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
    if bool((values < 0).any()) or bool((values >= (1 << bits)).any()):
        raise ValueError("index is outside the declared codebook")
    n = int(values.numel() if count is None else count)
    if n != values.numel():
        raise ValueError("count does not match indices")
    out = torch.zeros((math.ceil(n * bits / 8),), dtype=torch.uint8)
    for i, value in enumerate(values.tolist()):
        bit = i * bits
        integer = int(value) << (bit % 8)
        byte = bit // 8
        width = min(8 - bit % 8, bits)
        out[byte] |= integer & 0xFF
        if width < bits:
            out[byte + 1] |= (integer >> 8) & 0xFF
    return out


def unpack_indices(packed: torch.Tensor, count: int, bits: int = 6) -> torch.Tensor:
    """Inverse of :func:`pack_indices`; unused trailing bits are ignored."""
    if bits != 6 or count < 0:
        raise ValueError("the recursive prototype uses nonnegative count and 6-bit indices")
    raw = packed.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
    if raw.numel() != math.ceil(count * bits / 8):
        raise ValueError("packed byte count does not match index count")
    out = torch.empty((count,), dtype=torch.long)
    for i in range(count):
        bit = i * bits
        integer = int(raw[bit // 8].item()) >> (bit % 8)
        if bit % 8 + bits > 8:
            integer |= int(raw[bit // 8 + 1].item()) << (8 - bit % 8)
        out[i] = integer & ((1 << bits) - 1)
    return out


@dataclass
class StructuralVQState:
    """Serialized persistent state for one 2-D Muon momentum tensor.

    ``u``, ``singular_values`` and ``vh`` are BF16, ``scales`` are FP32, and
    ``indices`` are packed 6-bit values.  No FP32 momentum is stored.
    """

    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    scales: torch.Tensor
    indices: torch.Tensor
    pair_count: int
    shape: tuple[int, int]
    codebook_key: str

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


class StructuralVQCodec:
    """Exact-SVD structural conditioner + fixed 64-word p98 2-D VQ codec."""

    def __init__(self, codebook: torch.Tensor, *, rank: int = 8,
                 block_size: int = 2048, codebook_key: str = "s0_k8_w64_t8_v1200"):
        codebook = codebook.detach().float().cpu()
        if codebook.shape != (64, 2) or not torch.isfinite(codebook).all():
            raise ValueError("recursive prototype requires a finite 64x2 codebook")
        if not bool((codebook.square().sum(dim=1) == 0).any()):
            raise ValueError("codebook must contain an exact zero codeword")
        if rank < 0 or block_size <= 0:
            raise ValueError("rank and block_size must be positive")
        self.codebook = codebook
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.codebook_key = str(codebook_key)

    @torch.no_grad()
    def encode(self, momentum: torch.Tensor) -> StructuralVQState:
        value = momentum.detach().float().cpu()
        if value.ndim != 2:
            raise ValueError("structural recursive state requires a 2-D matrix")
        u, s, vh = torch.linalg.svd(value, full_matrices=False)
        k = min(self.rank, int(s.numel()))
        uk, sk, vhk = u[:, :k], s[:k], vh[:k]
        low = (uk * sk) @ vhk if k else torch.zeros_like(value)
        residual = value - low
        pairs, singles, pair_index, single_index = pair_values(residual, "contiguous")
        if singles.numel():
            raise ValueError("recursive matrices must contain an even number of values")
        scales = vector_scales(pairs, "p98", block_size=self.block_size)
        indices = torch.empty((pairs.shape[0],), dtype=torch.long)
        nvec = self.block_size // 2
        for start in range(0, pairs.shape[0], nvec):
            end = min(start + nvec, pairs.shape[0])
            alpha = scales[start // nvec]
            if alpha.item() == 0:
                indices[start:end] = 0
            else:
                normalized = (pairs[start:end] / alpha).clamp(-1, 1)
                indices[start:end] = torch.cdist(normalized, self.codebook).argmin(dim=1)
        return StructuralVQState(uk.to(torch.bfloat16), sk.to(torch.bfloat16),
                                 vhk.to(torch.bfloat16), scales.float(),
                                 pack_indices(indices, 6), int(indices.numel()),
                                 tuple(value.shape), self.codebook_key)

    @torch.no_grad()
    def decode(self, state: StructuralVQState, *, device=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        u = state.u.to(device=device, dtype=torch.float32)
        s = state.singular_values.to(device=device, dtype=torch.float32)
        vh = state.vh.to(device=device, dtype=torch.float32)
        low = (u * s) @ vh if s.numel() else torch.zeros(state.shape, device=device)
        indices = unpack_indices(state.indices, state.pair_count).to(device)
        scales = state.scales.to(device=device, dtype=torch.float32)
        pairs = torch.empty((state.pair_count, 2), device=device)
        nvec = self.block_size // 2
        for start in range(0, state.pair_count, nvec):
            end = min(start + nvec, state.pair_count)
            alpha = scales[start // nvec]
            pairs[start:end] = 0 if alpha.item() == 0 else self.codebook.to(device)[indices[start:end]] * alpha
        singles = torch.empty((0,), device=device)
        pi, si = pair_values(torch.zeros(state.shape), "contiguous")[2:]
        return low + unpair_values(pairs, singles, state.shape, pi.to(device), si.to(device))

    def storage_bits(self, state: StructuralVQState) -> int:
        """Exact serialized payload bits, excluding Python/container overhead."""
        return int(state.u.numel() * 16 + state.singular_values.numel() * 16 +
                   state.vh.numel() * 16 + state.scales.numel() * 32 +
                   state.indices.numel() * 8)


@dataclass
class StructuralINT4State:
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    scales: torch.Tensor
    codes: torch.Tensor
    count: int
    shape: tuple[int, int]


class StructuralINT4Codec:
    """Packed structural INT4 residual codec using the production dynamic map."""

    def __init__(self, *, rank: int = 8, block_size: int = 2048):
        self.rank, self.block_size = int(rank), int(block_size)
        self.map = create_bitsandbytes_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4)

    @torch.no_grad()
    def encode(self, momentum: torch.Tensor) -> StructuralINT4State:
        value = momentum.detach().float().cpu()
        if value.ndim != 2:
            raise ValueError("structural recursive state requires a 2-D matrix")
        u, s, vh = torch.linalg.svd(value, full_matrices=False)
        k = min(self.rank, int(s.numel())); uk, sk, vhk = u[:, :k], s[:k], vh[:k]
        low = (uk * sk) @ vhk if k else torch.zeros_like(value)
        flat = (value - low).reshape(-1); scales = []; codes = torch.empty_like(flat, dtype=torch.long)
        for start in range(0, flat.numel(), self.block_size):
            end = min(start + self.block_size, flat.numel()); block = flat[start:end]
            alpha = block.abs().amax(); scales.append(alpha)
            if alpha.item() == 0: codes[start:end] = 0
            else:
                codes[start:end] = torch.cdist((block / alpha).reshape(-1, 1), self.map.reshape(-1, 1)).argmin(1)
        raw = codes.to(torch.int64); packed = torch.zeros((math.ceil(raw.numel() / 2),), dtype=torch.uint8)
        packed[:raw.numel() // 2] = (raw[0::2][:packed.numel()] | (raw[1::2][:packed.numel()] << 4)).to(torch.uint8)
        if raw.numel() % 2: packed[-1] = raw[-1].to(torch.uint8)
        return StructuralINT4State(uk.to(torch.bfloat16), sk.to(torch.bfloat16), vhk.to(torch.bfloat16),
                                   torch.stack(scales).float(), packed, int(raw.numel()), tuple(value.shape))

    @torch.no_grad()
    def decode(self, state: StructuralINT4State, *, device=None) -> torch.Tensor:
        device = device or torch.device("cpu"); u = state.u.to(device, torch.float32); s = state.singular_values.to(device, torch.float32); vh = state.vh.to(device, torch.float32)
        low = (u * s) @ vh if s.numel() else torch.zeros(state.shape, device=device)
        packed = state.codes.to(device=device, dtype=torch.int64); raw = torch.empty(state.count, dtype=torch.long, device=device)
        raw[0::2] = packed[: (state.count + 1)//2] & 15; raw[1::2] = (packed[: state.count//2] >> 4) & 15
        flat = torch.empty(state.count, device=device); cmap = self.map.to(device)
        for start in range(0, state.count, self.block_size):
            end = min(start + self.block_size, state.count); flat[start:end] = cmap[raw[start:end]] * state.scales[start // self.block_size].to(device)
        return low + flat.reshape(state.shape)

    def storage_bits(self, state: StructuralINT4State) -> int:
        return int(state.u.numel()*16 + state.singular_values.numel()*16 + state.vh.numel()*16 + state.scales.numel()*32 + state.codes.numel()*8)


def build_recursive_codecs(model, cfg: dict, root: Path) -> dict[int, object]:
    """Build codecs for the experimental recipe without changing production paths."""
    rank = int(cfg.get("recursive_rank", 8)); block = int(cfg.get("recursive_block_size", 2048))
    mode = cfg.get("recursive_structure_mode", "exact_svd_oracle")
    if mode != "exact_svd_oracle":
        raise ValueError("only exact_svd_oracle is currently implemented")
    codecs = {}
    if cfg.get("recursive_codebook_path"):
        path = Path(cfg["recursive_codebook_path"])
        if not path.is_absolute(): path = (root / path).resolve()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        codebook = blob["codebooks"][cfg["recursive_codebook_key"]].float()
    else:
        codebook = None
    kind = cfg.get("recursive_representation", "vq_int3")
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or not name.startswith("transformer.h."):
            continue
        codecs[id(parameter)] = (StructuralVQCodec(codebook, rank=rank, block_size=block,
                                                    codebook_key=cfg.get("recursive_codebook_key", ""))
                                 if kind == "vq_int3" else StructuralINT4Codec(rank=rank, block_size=block))
    if not codecs:
        raise ValueError("recursive_muon found no eligible 2-D Muon parameters")
    return codecs


class RecursiveMuon(torch.optim.Optimizer):
    """Muon whose selected 2-D momentum states recursively persist via codec.

    This is deliberately an experimental optimizer and is not wired into the
    production adapter.  Non-Muon parameters use ordinary FP32 AdamW moments.
    """

    def __init__(self, params, codecs: dict[int, StructuralVQCodec], *, lr=1e-3,
                 betas=(.9, .999), eps=1e-8, weight_decay=.1, muon_momentum=.95,
                 muon_nesterov=True, muon_ns_steps=5,
                 muon_ns_coefficients=(3.4445, -4.7750, 2.0315), muon_eps=1e-7):
        self.codecs = codecs
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        muon_momentum=muon_momentum, muon_nesterov=muon_nesterov,
                        muon_ns_steps=muon_ns_steps, muon_ns_coefficients=tuple(muon_ns_coefficients),
                        muon_eps=muon_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
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
                    momentum = torch.zeros_like(p, dtype=torch.float32) if compressed is None else codec.decode(compressed, device=p.device)
                    momentum = momentum.mul(group["muon_momentum"]).add(grad)
                    direction = grad.add(momentum, alpha=group["muon_momentum"]) if group["muon_nesterov"] else momentum
                    update = zeropower_newton_schulz(direction, group["muon_ns_steps"], group["muon_ns_coefficients"], group["muon_eps"])
                    p.copy_(p.float().mul(1 - group["lr"] * group["weight_decay"]).add(update, alpha=-group["lr"]).to(p.dtype))
                    state["compressed_momentum"] = codec.encode(momentum)
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
                rows.append({"parameter_id": id(p), "shape": list(encoded.shape),
                             "storage_bits": self.codecs[id(p)].storage_bits(encoded),
                             "has_fp32_momentum": False})
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
            if hasattr(codec, "codebook"):
                identity = ("vq", getattr(codec, "codebook_key", "default"))
                if identity not in codebook_ids:
                    codebook_ids.add(identity); codebook_bits += int(codec.codebook.numel() * 32)
        persistent_bits = sum(r["storage_bits"] for r in rows) + codebook_bits
        # Keep the common run-summary contract in addition to the recursive
        # storage-specific fields.  The summary writer records bytes, while
        # this optimizer also exposes the exact serialized bit count.
        persistent_bytes = (persistent_bits + 7) // 8
        return {"logical_tensor_bytes": persistent_bytes,
                "unique_storage_bytes": persistent_bytes,
                "tensors": rows, "persistent_bits": persistent_bits,
                "codebook_bits": codebook_bits,
                "compressed_muon_tensor_count": sum("key" not in r for r in rows),
                "structure_mode": "exact_svd_oracle"}
