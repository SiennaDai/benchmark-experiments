"""Small deterministic utilities for offline vector-INT3 robustness checks.

This module is deliberately separate from quantization and optimizer code. It
provides stable tensor sampling, codebook comparison, occupancy summaries, and
fixed-width index packing for analysis artifacts.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Select ``count`` deterministic, evenly spaced indices from a sequence.

    Endpoints are included when at least two items are selected. A single item
    selects the midpoint. If ``count`` is at least ``length``, every index is
    returned exactly once.
    """
    length, count = int(length), int(count)
    if length < 0 or count < 0:
        raise ValueError("length and count must be nonnegative")
    if count == 0 or length == 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [(length - 1) // 2]
    # Round a uniform endpoint-inclusive grid, then assert uniqueness. Since
    # count < length, its spacing is greater than one and rounded points remain
    # distinct.
    return [int(round(x)) for x in np.linspace(0, length - 1, count)]


def evenly_spaced_sample(items: Sequence, count: int) -> list:
    """Return a deterministic evenly spaced subset while preserving order."""
    return [items[i] for i in evenly_spaced_indices(len(items), count)]


def _hungarian_square(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost square assignment (row -> column), O(n^3), deterministic."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("assignment cost must be a square matrix")
    n = cost.shape[0]
    # Potentials implementation of the Hungarian algorithm, using 1-based
    # indexing as in the standard shortest augmenting path formulation.
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if not np.isfinite(delta):
                raise ValueError("assignment cost contains no finite matching")
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def align_codebook(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align candidate codeword order to reference by minimum total L2 cost.

    Returns ``(aligned_candidate, permutation)`` where ``aligned_candidate[i]``
    is the candidate codeword assigned to ``reference[i]`` and ``permutation``
    contains the corresponding original candidate indices. Codebooks must have
    equal, nonempty ``(K,D)`` shapes.
    """
    ref = torch.as_tensor(reference).detach().to(dtype=torch.float64, device="cpu")
    cand = torch.as_tensor(candidate).detach().to(dtype=torch.float64, device="cpu")
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape != cand.shape or ref.shape[0] == 0:
        raise ValueError("codebooks must have identical nonempty (K,D) shapes")
    if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(cand).all()):
        raise ValueError("codebooks must contain only finite values")
    costs = torch.cdist(ref, cand).square().numpy()
    permutation = _hungarian_square(costs)
    permutation_tensor = torch.from_numpy(permutation.copy())
    aligned = torch.as_tensor(candidate).detach().cpu()[permutation_tensor]
    return aligned, permutation_tensor


def codebook_occupancy_stats(indices: torch.Tensor, codewords: int) -> dict[str, float | int | list[int]]:
    """Summarize codeword usage without interpreting it as entropy coding."""
    codewords = int(codewords)
    if codewords <= 0:
        raise ValueError("codewords must be positive")
    idx = torch.as_tensor(indices).detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if idx.numel() and bool(((idx < 0) | (idx >= codewords)).any()):
        raise ValueError("indices must lie in [0, codewords)")
    counts_t = torch.bincount(idx, minlength=codewords)
    counts = counts_t.tolist()
    total = int(idx.numel())
    occupied = int((counts_t > 0).sum())
    if total:
        probabilities = counts_t[counts_t > 0].double() / total
        entropy = float(-(probabilities * probabilities.log2()).sum())
        top = sorted(counts, reverse=True)
        top1 = top[0] / total
        top5 = sum(top[:5]) / total
    else:
        entropy = top1 = top5 = 0.0
    return {
        "total_indices": total,
        "codewords": codewords,
        "used_codewords": occupied,
        "dead_codewords": codewords - occupied,
        "occupancy_entropy_bits": entropy,
        "occupancy_entropy_normalized": entropy / math.log2(codewords) if codewords > 1 else 0.0,
        "top1_occupancy_fraction": top1,
        "top5_occupancy_fraction": top5,
        "counts": counts,
    }


def pack_indices(indices: torch.Tensor | Sequence[int], bits: int) -> torch.Tensor:
    """Pack fixed-width unsigned indices into a little-endian bitstream.

    The first index occupies the least-significant ``bits`` of the stream.
    Output is a one-dimensional CPU ``torch.uint8`` tensor. A final partial
    byte is zero padded.
    """
    bits = int(bits)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1,16]")
    values = np.asarray(torch.as_tensor(indices).detach().cpu().reshape(-1), dtype=np.int64)
    if np.any(values < 0) or np.any(values >= (1 << bits)):
        raise ValueError("index is outside the representable unsigned range")
    n = values.size
    output = np.zeros((n * bits + 7) // 8, dtype=np.uint8)
    if n == 0:
        return torch.from_numpy(output)
    starts = np.arange(n, dtype=np.int64) * bits
    for source_bit in range(bits):
        set_mask = ((values >> source_bit) & 1).astype(np.uint8)
        destination = starts + source_bit
        byte = destination >> 3
        offset = destination & 7
        np.bitwise_or.at(output, byte, set_mask << offset.astype(np.uint8))
    return torch.from_numpy(output)


def unpack_indices(packed: torch.Tensor | bytes | bytearray | np.ndarray,
                   count: int, bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_indices` for a declared item count."""
    bits, count = int(bits), int(count)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1,16]")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if isinstance(packed, (bytes, bytearray)):
        raw = np.frombuffer(packed, dtype=np.uint8)
    else:
        raw = np.asarray(torch.as_tensor(packed).detach().cpu(), dtype=np.uint8).reshape(-1)
    expected_bytes = (count * bits + 7) // 8
    if raw.size != expected_bytes:
        raise ValueError(f"packed stream has {raw.size} bytes; expected {expected_bytes}")
    values = np.zeros(count, dtype=np.int64)
    if count == 0:
        return torch.from_numpy(values)
    starts = np.arange(count, dtype=np.int64) * bits
    for destination_bit in range(bits):
        source = starts + destination_bit
        bit_values = (raw[source >> 3] >> (source & 7)) & 1
        values |= bit_values.astype(np.int64) << destination_bit
    return torch.from_numpy(values)


def packed_storage_report(count: int, bits: int) -> dict[str, int | float]:
    """Compare ideal fixed-width payload bits with byte-aligned packed size."""
    count, bits = int(count), int(bits)
    if count < 0:
        raise ValueError("count must be nonnegative")
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1,16]")
    theoretical_bits = count * bits
    packed_bytes = (theoretical_bits + 7) // 8
    packed_bits = packed_bytes * 8
    return {
        "count": count,
        "bits_per_index": bits,
        "theoretical_bits": theoretical_bits,
        "theoretical_bytes_fractional": theoretical_bits / 8,
        "packed_bytes": packed_bytes,
        "packed_bits": packed_bits,
        "padding_bits": packed_bits - theoretical_bits,
    }
