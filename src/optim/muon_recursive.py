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
from pathlib import Path
from dataclasses import dataclass

import torch

from .muon_reference import zeropower_newton_schulz
from .state_simulation import create_bitsandbytes_dynamic_map


def _packed_device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


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
            scales = torch.quantile(abs_blocks, percentile, dim=1)
        nonzero = abs_blocks.amax(dim=1) != 0
        scales = torch.where(nonzero, scales.clamp_min(torch.finfo(torch.float32).tiny), torch.zeros_like(scales))
        pieces.append(scales)
    if full != flat.numel():
        tail = flat[full:]
        abs_tail = tail.abs()
        scale = abs_tail.amax() if percentile is None else torch.quantile(abs_tail, percentile)
        scale = torch.where(abs_tail.amax() != 0, scale.clamp_min(torch.finfo(torch.float32).tiny), torch.zeros_like(scale))
        pieces.append(scale.reshape(1))
    return torch.cat(pieces) if pieces else flat.new_empty((0,))


def _repeat_block_scales(scales: torch.Tensor, count: int, block_size: int) -> torch.Tensor:
    """Expand one scalar scale per block to one scalar per value."""
    if count == 0:
        return scales.new_empty((0,))
    return torch.repeat_interleave(scales, block_size)[:count]


def pack_indices(indices: torch.Tensor, bits: int, *, count: int | None = None) -> torch.Tensor:
    """Pack fixed-width unsigned indices into bytes, least-significant-bit first.

    The operation is vectorized and stays on the input device.  This matters
    for recursive CUDA runs: the packed state is copied to CPU only after the
    device-side packing has completed, rather than iterating over every pair in
    Python.
    """
    if bits not in (6,):
        raise ValueError("the recursive prototype currently packs the fixed 64-word/6-bit format")
    values = indices.detach().to(dtype=torch.int64).reshape(-1)
    if bool((values < 0).any()) or bool((values >= (1 << bits)).any()):
        raise ValueError("index is outside the declared codebook")
    n = int(values.numel() if count is None else count)
    if n != values.numel():
        raise ValueError("count does not match indices")
    if n == 0:
        return torch.empty((0,), dtype=torch.uint8, device=values.device)
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
    if bits != 6 or count < 0:
        raise ValueError("the recursive prototype uses nonnegative count and 6-bit indices")
    raw = packed.detach().to(dtype=torch.int64).reshape(-1)
    if raw.numel() != math.ceil(count * bits / 8):
        raise ValueError("packed byte count does not match index count")
    if count == 0:
        return torch.empty((0,), dtype=torch.long, device=raw.device)
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
        self._codebook_cache: dict[tuple[str, int | None], torch.Tensor] = {}

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
        u, s, vh = torch.linalg.svd(value, full_matrices=False)
        k = min(self.rank, int(s.numel()))
        uk, sk, vhk = u[:, :k], s[:k], vh[:k]
        low = (uk * sk) @ vhk if k else torch.zeros_like(value)
        residual = value - low
        if residual.numel() % 2:
            raise ValueError("recursive contiguous VQ requires even-sized matrices")
        # All current formal Muon matrices are even-sized.  Avoid the generic
        # offline pairing helper here: it builds Python lists/sets over every
        # scalar and is not part of the recursive representation semantics.
        pairs = residual.reshape(-1, 2)
        scales = _block_stat_scales(residual.reshape(-1), self.block_size, percentile=.98)
        pair_scales = _repeat_block_scales(scales, pairs.shape[0], self.block_size // 2)
        safe_scales = pair_scales.clamp_min(torch.finfo(torch.float32).tiny)
        normalized = (pairs / safe_scales[:, None]).clamp(-1, 1)
        codebook = self._codebook_on(value.device)
        indices = torch.empty((pairs.shape[0],), dtype=torch.long, device=value.device)
        # A small number of large chunks bounds temporary memory without
        # reintroducing a block-sized Python loop or GPU synchronization.
        for start in range(0, pairs.shape[0], 65536):
            end = min(start + 65536, pairs.shape[0])
            indices[start:end] = torch.cdist(normalized[start:end], codebook).argmin(dim=1)
        indices = torch.where(pair_scales > 0, indices, torch.zeros_like(indices))
        return StructuralVQState(uk.to(torch.bfloat16), sk.to(torch.bfloat16),
                                 vhk.to(torch.bfloat16), scales.float(),
                                 pack_indices(indices, 6), int(indices.numel()),
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
        indices = unpack_indices(state.indices, state.pair_count)
        scales = state.scales.to(device=device, dtype=torch.float32)
        pair_scales = _repeat_block_scales(scales, state.pair_count, self.block_size // 2)
        pairs = self._codebook_on(device)[indices] * pair_scales[:, None]
        return low + pairs.reshape(state.shape)

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

    This is deliberately an experimental optimizer. Non-Muon parameters use
    ordinary FP32 AdamW moments. ``recursive_error_feedback_alpha`` controls an
    optional persistent FP32 residual of the compressed Muon state, not an
    uncompressed momentum copy.
    """

    def __init__(self, params, codecs: dict[int, StructuralVQCodec], *, lr=1e-3,
                 betas=(.9, .999), eps=1e-8, weight_decay=.1, muon_momentum=.95,
                 muon_nesterov=True, muon_ns_steps=5,
                 muon_ns_coefficients=(3.4445, -4.7750, 2.0315), muon_eps=1e-7,
                 recursive_error_feedback_alpha=0.0):
        if isinstance(recursive_error_feedback_alpha, bool):
            raise ValueError("recursive_error_feedback_alpha must be finite and nonnegative")
        alpha = float(recursive_error_feedback_alpha)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("recursive_error_feedback_alpha must be finite and nonnegative")
        self.codecs = codecs
        self.recursive_error_feedback_alpha = alpha
        self._mechanism_observer = None
        self._mechanism_update = None
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        muon_momentum=muon_momentum, muon_nesterov=muon_nesterov,
                        muon_ns_steps=muon_ns_steps, muon_ns_coefficients=tuple(muon_ns_coefficients),
                        muon_eps=muon_eps, recursive_error_feedback_alpha=alpha)
        super().__init__(params, defaults)

    def set_mechanism_observer(self, observer):
        self._mechanism_observer = observer

    def set_mechanism_update(self, update):
        self._mechanism_update = int(update)

    @staticmethod
    def _serialize_encoded(value):
        if isinstance(value, StructuralVQState):
            return {"_recursive_state_type": "vq", **value.state_dict()}
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
            if isinstance(encoded, (StructuralVQState, StructuralINT4State)):
                state["compressed_momentum"] = encoded.to(parameter.device)
            error = state.get("momentum_error")
            if torch.is_tensor(error):
                state["momentum_error"] = error.to(device=parameter.device, dtype=torch.float32)

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
                    previous_error = state.get("momentum_error")
                    if previous_error is None:
                        previous_error = torch.zeros_like(p, dtype=torch.float32)
                    momentum = previous_decoded.mul(mu).add(grad)
                    alpha = group.get("recursive_error_feedback_alpha", 0.0)
                    if alpha:
                        # Restore a controlled fraction of the previous
                        # persistence residual: mu * (decoded + alpha * error).
                        momentum.add_(previous_error, alpha=alpha * mu)
                    direction = grad.add(momentum, alpha=group["muon_momentum"]) if group["muon_nesterov"] else momentum
                    update = zeropower_newton_schulz(direction, group["muon_ns_steps"], group["muon_ns_coefficients"], group["muon_eps"])
                    updated = p.float().mul(1 - group["lr"] * group["weight_decay"]).add(update, alpha=-group["lr"])
                    encoded = codec.encode(momentum)
                    persisted_decoded = codec.decode(encoded, device=p.device) if (alpha or self._mechanism_observer is not None) else None
                    current_error = momentum - persisted_decoded if alpha else None
                    if self._mechanism_observer is not None:
                        self._mechanism_observer(parameter=p, gradient=grad, momentum_prev=previous_decoded,
                            momentum_candidate=momentum, momentum_persisted_decoded=persisted_decoded,
                            direction=direction, updated=updated, group=group, update=self._mechanism_update,
                            compressed_state=encoded, momentum_error_prev=previous_error,
                            momentum_error=current_error, error_feedback_alpha=alpha)
                    p.copy_(updated.to(p.dtype))
                    state["compressed_momentum"] = encoded
                    if alpha:
                        state["momentum_error"] = current_error
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
                rows.append({"parameter_id": id(p), "shape": list(encoded.shape),
                             "storage_bits": self.codecs[id(p)].storage_bits(encoded),
                             "error_buffer_bits": int(error.numel() * error.element_size() * 8) if torch.is_tensor(error) else 0,
                             "has_fp32_momentum": False, "has_fp32_error_buffer": torch.is_tensor(error)})
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
        compressed_bits = sum(r["storage_bits"] for r in rows)
        muon_rows = [r for r in rows if "key" not in r]
        compressed_muon_bits = sum(r["storage_bits"] for r in muon_rows)
        muon_scalar_count = sum(math.prod(r["shape"]) for r in muon_rows)
        error_buffer_bits = sum(r.get("error_buffer_bits", 0) for r in rows)
        persistent_bits = compressed_bits + error_buffer_bits + codebook_bits
        muon_total_bits = compressed_muon_bits + error_buffer_bits + codebook_bits
        # Keep the common run-summary contract in addition to the recursive
        # storage-specific fields.  The summary writer records bytes, while
        # this optimizer also exposes the exact serialized bit count.
        persistent_bytes = (persistent_bits + 7) // 8
        return {"logical_tensor_bytes": persistent_bytes,
                "unique_storage_bytes": persistent_bytes,
                "tensors": rows, "persistent_bits": persistent_bits,
                "codebook_bits": codebook_bits,
                "compressed_state_bits": compressed_bits, "error_buffer_bits": error_buffer_bits,
                "compressed_muon_state_bits": compressed_muon_bits,
                "muon_scalar_count": muon_scalar_count,
                "muon_total_state_bits": muon_total_bits,
                "error_buffer_bytes": (error_buffer_bits + 7) // 8,
                "muon_effective_bits_per_value": muon_total_bits / muon_scalar_count if muon_scalar_count else 0.0,
                "error_feedback_alpha": self.recursive_error_feedback_alpha,
                "compressed_muon_tensor_count": sum("key" not in r for r in rows),
                "structure_mode": "exact_svd_oracle"}
