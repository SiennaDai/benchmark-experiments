"""Offline 2-D vector quantization helpers for conditioned Muon INT3 studies.

This module is analysis-only.  It has no optimizer or persistence hooks.  A
pair is assigned one fixed-length index (``ceil(log2(K))`` bits); unmatched
values in odd-shaped matrices are explicitly returned as scalar leftovers.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from .muon_quantization_aware_conditioner import BLOCK_SIZE


def pairing_indices(shape: tuple[int, int], scheme: str = "contiguous"):
    """Deterministically partition matrix flat indices into pairs + leftovers.

    ``contiguous`` pairs row-major neighbors; ``row`` pairs within rows;
    ``column`` pairs vertically; ``checkerboard`` pairs diagonal neighbors in
    2x2 tiles.  Unpaired indices are represented separately and preserved in
    their original order.
    """
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("pairing requires a nonempty 2-D shape")
    m, n = map(int, shape)
    remaining = set(range(m * n))
    pairs: list[tuple[int, int]] = []
    if scheme == "contiguous":
        pairs = [(i, i + 1) for i in range(0, m * n - 1, 2)]
    elif scheme == "row":
        for r in range(m):
            base = r * n
            pairs.extend((base + c, base + c + 1) for c in range(0, n - 1, 2))
    elif scheme == "column":
        for c in range(n):
            pairs.extend((r * n + c, (r + 1) * n + c) for r in range(0, m - 1, 2))
    elif scheme == "checkerboard":
        for r in range(0, m - 1, 2):
            for c in range(0, n - 1, 2):
                pairs.extend(((r * n + c, (r + 1) * n + c + 1),
                              (r * n + c + 1, (r + 1) * n + c)))
    else:
        raise ValueError(f"unknown pairing scheme: {scheme}")
    for a, b in pairs:
        if a not in remaining or b not in remaining:
            raise RuntimeError("pairing generated overlapping indices")
        remaining.remove(a); remaining.remove(b)
    singles = sorted(remaining)
    # Pair any unmatched entries in deterministic row-major order.  This keeps
    # vector count maximal while preserving explicit singletons only if N odd.
    if len(singles) > 1:
        pairs.extend((singles[i], singles[i + 1]) for i in range(0, len(singles) - 1, 2))
        singles = singles[len(singles) // 2 * 2:]
    return (torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
            torch.tensor(singles, dtype=torch.long))


def pair_values(matrix: torch.Tensor, scheme: str = "contiguous"):
    """Return (Npair,2) values, singleton values, and inverse-map indices."""
    if matrix.ndim != 2:
        raise ValueError("pair_values expects a matrix")
    pairs, singles = pairing_indices(tuple(matrix.shape), scheme)
    flat = matrix.detach().float().reshape(-1)
    return flat[pairs.to(flat.device)], flat[singles.to(flat.device)], pairs, singles


def unpair_values(pairs_value: torch.Tensor, singles_value: torch.Tensor,
                  shape: tuple[int, int], pair_index: torch.Tensor,
                  single_index: torch.Tensor) -> torch.Tensor:
    """Invert :func:`pair_values` exactly in the original matrix layout."""
    n = int(shape[0]) * int(shape[1]); device = pairs_value.device
    if pairs_value.ndim != 2 or pairs_value.shape[1] != 2:
        raise ValueError("pairs_value must have shape (N,2)")
    if pairs_value.shape[0] * 2 + singles_value.numel() != n:
        raise ValueError("pair/single count does not match requested shape")
    out = torch.empty(n, dtype=pairs_value.dtype, device=device)
    pi = pair_index.to(device); si = single_index.to(device)
    out[pi.reshape(-1)] = pairs_value.reshape(-1)
    if si.numel(): out[si] = singles_value.to(device).reshape(-1)
    return out.reshape(shape)


def polar_codebook(n_angles: int = 8, n_radii: int = 8,
                   radial: str = "sqrt", *, max_codewords: int = 64,
                   device=None) -> torch.Tensor:
    """Fixed deterministic polar grid with an exact unique zero codeword.

    The requested angular/radial product includes the zero radius; duplicate
    zero-angle points are deduplicated, so the final codebook never exceeds the
    declared budget.
    """
    if n_angles < 1 or n_radii < 1:
        raise ValueError("polar grid dimensions must be positive")
    if radial == "uniform":
        radii = torch.linspace(0, 1, n_radii)
    elif radial == "sqrt":
        radii = torch.linspace(0, 1, n_radii).sqrt()
    elif radial == "log":
        radii = torch.cat((torch.zeros(1), torch.logspace(-2, 0, n_radii - 1))) if n_radii > 1 else torch.zeros(1)
    else:
        raise ValueError("radial must be uniform, sqrt, or log")
    angles = torch.arange(n_angles, dtype=torch.float32) * (2 * math.pi / n_angles)
    points = torch.stack(torch.meshgrid(radii, angles, indexing="ij"), dim=-1)
    rho, theta = points[..., 0].reshape(-1), points[..., 1].reshape(-1)
    cb = torch.stack((rho * theta.cos(), rho * theta.sin()), dim=-1)
    cb = torch.unique(cb, dim=0, sorted=True)
    zero = torch.zeros((1, 2), dtype=torch.float32)
    cb = torch.cat((zero, cb[(cb.square().sum(dim=1) > 1e-12)]), dim=0)
    # If a grid exceeds the bit budget, retain zero and deterministic farthest
    # points by radius; callers in this study use grids <= 64.
    if cb.shape[0] > max_codewords:
        order = torch.argsort(cb[1:].square().sum(1), descending=True, stable=True)
        cb = torch.cat((cb[:1], cb[1:][order[:max_codewords - 1]]), dim=0)
    return cb.to(device=device) if device is not None else cb


def fit_covariance_transform(samples: torch.Tensor, *, whiten: bool = False,
                             eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one global symmetric PCA rotation or whitening map and inverse."""
    x = samples.detach().float().reshape(-1, 2)
    cov = x.T @ x / max(x.shape[0], 1)
    vals, vecs = torch.linalg.eigh(cov)
    vals = vals.clamp_min(float(eps))
    if whiten:
        w = vecs @ torch.diag(vals.rsqrt()) @ vecs.T
        wi = vecs @ torch.diag(vals.sqrt()) @ vecs.T
    else:
        w = vecs.T
        wi = vecs
    return w.float(), wi.float()


def fit_kmeans(samples: torch.Tensor, k: int, *, seed: int = 2026,
               iterations: int = 30, max_samples: int = 200_000) -> tuple[torch.Tensor, dict]:
    """Deterministic shared 2-D Lloyd codebook, reserving one exact zero.

    Training uses only the passed calibration samples.  A seeded k-means++
    initialization is followed by fixed-count Lloyd updates; zero is removed
    from the learned centers and then prepended as an exact codeword.
    """
    if k < 2:
        raise ValueError("k must be at least two to include zero and a learned center")
    x = samples.detach().float().reshape(-1, 2).cpu()
    if x.shape[0] > max_samples:
        idx = torch.linspace(0, x.shape[0] - 1, max_samples).round().long()
        x = x[idx]
    if x.shape[0] < k - 1:
        raise ValueError("not enough calibration vectors for requested codebook")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    ncenters = k - 1
    # Seeded k-means++ provides spread without evaluation-data dependence.
    centers = [x[torch.randint(x.shape[0], (1,), generator=g).item()]]
    for _ in range(1, ncenters):
        d2 = torch.cdist(x, torch.stack(centers)).square().amin(dim=1)
        total = d2.sum()
        ix = (torch.randint(x.shape[0], (1,), generator=g).item() if total <= 0
              else int(torch.multinomial(d2 / total, 1, generator=g).item()))
        centers.append(x[ix])
    c = torch.stack(centers)
    for _ in range(iterations):
        labels = torch.cdist(x, c).argmin(dim=1)
        updated = c.clone()
        for j in range(ncenters):
            mask = labels == j
            if bool(mask.any()): updated[j] = x[mask].mean(dim=0)
        if torch.allclose(updated, c, rtol=0, atol=1e-7): c = updated; break
        c = updated
    cb = torch.cat((torch.zeros((1, 2)), c), dim=0)
    # Deterministically remove duplicate centers, retaining exact zero.
    unique = [cb[0]]
    for row in cb[1:]:
        if all(float((row - z).square().sum()) > 1e-12 for z in unique): unique.append(row)
    cb = torch.stack(unique)
    return cb, {"seed": seed, "iterations": iterations, "samples": int(x.shape[0]),
                "requested_codewords": k, "actual_codewords": int(cb.shape[0]),
                "dead_codewords": k - int(cb.shape[0])}


def vector_scales(pairs: torch.Tensor, method: str = "p98", *,
                  block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """One shared positive scale per 2048 scalar residual values (1024 pairs)."""
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (N,2)")
    nvec = block_size // 2
    out = []
    for start in range(0, pairs.shape[0], nvec):
        b = pairs[start:start + nvec].float().reshape(-1)
        if not b.numel() or not bool(b.abs().any()): out.append(b.new_zeros(())); continue
        if method == "absmax": alpha = b.abs().amax()
        elif method == "p98": alpha = torch.quantile(b.abs(), .98)
        elif method == "rms2": alpha = b.square().mean().sqrt() * 2.0
        else: raise ValueError(f"unsupported vector scale rule: {method}")
        out.append(alpha.clamp_min(torch.finfo(torch.float32).tiny))
    return torch.stack(out) if out else torch.empty(0, dtype=torch.float32, device=pairs.device)


def normalize_vector_blocks(pairs: torch.Tensor, method: str = "p98", *,
                            per_dimension: bool = False,
                            block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Normalize calibration vectors and apply the evaluation clipping rule.

    Codebooks must be trained on the same bounded input domain presented to
    nearest-neighbor assignment.  Scalar-scale variants group 2048 values;
    per-dimension variants use one statistic per coordinate over each block.
    """
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be (N, 2)")
    out = pairs.detach().float().clone(); nvec = block_size // 2
    for start in range(0, pairs.shape[0], nvec):
        b = pairs[start:start + nvec].detach().float()
        if per_dimension:
            if method == "p98": alpha = torch.quantile(b.abs(), .98, dim=0)
            elif method == "absmax": alpha = b.abs().amax(dim=0)
            elif method == "rms2": alpha = b.square().mean(dim=0).sqrt() * 2
            else: raise ValueError(f"unsupported vector scale rule: {method}")
            alpha = alpha.clamp_min(1e-20)
        else:
            flat = b.reshape(-1)
            if method == "p98": alpha = torch.quantile(flat.abs(), .98)
            elif method == "absmax": alpha = flat.abs().amax()
            elif method == "rms2": alpha = flat.square().mean().sqrt() * 2
            else: raise ValueError(f"unsupported vector scale rule: {method}")
            alpha = alpha.clamp_min(1e-20)
        out[start:start + nvec] = (b / alpha).clamp(-1, 1)
    return out


@torch.no_grad()
def quantize_vectors(pairs: torch.Tensor, codebook: torch.Tensor, *,
                     scales: torch.Tensor | None = None, scale_method: str = "p98",
                     block_size: int = BLOCK_SIZE, transform: torch.Tensor | None = None,
                     inverse_transform: torch.Tensor | None = None,
                     chunk_size: int = 16_384) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize paired values by nearest 2-D codeword after shared scaling.

    Coordinates are clipped independently to [-1,1].  A supplied global 2x2
    transform is applied after scaling and inverted after codeword lookup.
    Ties follow ``argmin``'s first-index convention and are deterministic.
    """
    x = pairs.detach().float()
    if x.ndim != 2 or x.shape[1] != 2: raise ValueError("pairs must be (N,2)")
    cb = codebook.detach().to(device=x.device, dtype=torch.float32)
    if cb.ndim != 2 or cb.shape[1] != 2 or cb.shape[0] < 2 or cb.shape[0] > 128:
        raise ValueError("codebook must be (K,2), 2<=K<=128")
    if not bool(torch.isfinite(cb).all()): raise ValueError("codebook must be finite")
    if not bool((cb.square().sum(dim=1) == 0).any()): raise ValueError("codebook must contain exact zero")
    if scales is None: scales = vector_scales(x, scale_method, block_size=block_size)
    scales = scales.detach().to(device=x.device, dtype=torch.float32).reshape(-1)
    expected = math.ceil(x.shape[0] / (block_size // 2))
    if scales.numel() != expected: raise ValueError("scale count does not match pair blocks")
    recon = torch.empty_like(x); indices = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    w = transform.to(x) if transform is not None else None
    wi = inverse_transform.to(x) if inverse_transform is not None else None
    if (w is None) != (wi is None): raise ValueError("transform and inverse_transform must be supplied together")
    nvec = block_size // 2
    for bi, start in enumerate(range(0, x.shape[0], nvec)):
        end = min(start + nvec, x.shape[0]); alpha = scales[bi]
        if alpha.item() == 0:
            recon[start:end] = 0; indices[start:end] = 0; continue
        z = (x[start:end] / alpha).clamp(-1, 1)
        if w is not None: z = z @ w.T
        for first in range(0, z.shape[0], chunk_size):
            zz = z[first:first + chunk_size]
            ix = torch.cdist(zz, cb).argmin(dim=1)
            q = cb[ix]
            if wi is not None: q = q @ wi.T
            indices[start + first:start + first + len(ix)] = ix
            recon[start + first:start + first + len(ix)] = q * alpha
    return recon, scales, indices


@torch.no_grad()
def quantize_vectors_per_dimension(pairs: torch.Tensor, codebook: torch.Tensor, *,
                                  method: str = "p98", block_size: int = BLOCK_SIZE,
                                  chunk_size: int = 16_384):
    """Secondary variant with two block scales, one per coordinate dimension."""
    x = pairs.detach().float()
    if x.ndim != 2 or x.shape[1] != 2: raise ValueError("pairs must be (N,2)")
    cb = codebook.detach().to(device=x.device, dtype=torch.float32)
    nvec=block_size//2; recon=torch.empty_like(x); indices=torch.empty(x.shape[0],dtype=torch.long,device=x.device); scales=[]
    for start in range(0,x.shape[0],nvec):
        end=min(start+nvec,x.shape[0]); b=x[start:end]
        if not bool(b.abs().any()):
            scales.append(torch.zeros(2,dtype=torch.float32,device=x.device))
            recon[start:end]=0;indices[start:end]=0
            continue
        if method=="p98": alpha=torch.quantile(b.abs(),.98,dim=0)
        elif method=="absmax": alpha=b.abs().amax(dim=0)
        elif method=="rms2": alpha=b.square().mean(dim=0).sqrt()*2
        else: raise ValueError(f"unsupported per-dimension scale method: {method}")
        alpha=alpha.clamp_min(torch.finfo(torch.float32).tiny); scales.append(alpha)
        z=(b/alpha).clamp(-1,1)
        for first in range(0,z.shape[0],chunk_size):
            zz=z[first:first+chunk_size]; ix=torch.cdist(zz,cb).argmin(dim=1)
            indices[start+first:start+first+len(ix)]=ix
            recon[start+first:start+first+len(ix)]=cb[ix]*alpha
    return recon,torch.stack(scales) if scales else torch.empty((0,2),device=x.device),indices


def vector_storage_bits(shape: tuple[int, int], *, codewords: int,
                        lowrank_rank: int, block_size: int = BLOCK_SIZE,
                        codebook_bits: int = 32, scale_bits: int = 32,
                        factor_bits: int = 16, rotation_bits: int = 0,
                        include_spectral_basis: bool = False,
                        scales_per_block: int = 1) -> dict[str, int | float]:
    """Metadata-inclusive shared-codebook structural state estimate.

    Pair indices use fixed-length ceil(log2(K)) bits; one possible singleton
    uses 3 scalar bits.  Shared codebook and transform are charged once per
    report (script amortizes them across every evaluated tensor).
    """
    m, n = map(int, shape); numel = m * n
    if codewords < 2 or codewords > 128: raise ValueError("codewords must be in [2,128]")
    pairs = numel // 2; singles = numel % 2
    index_bits = math.ceil(math.log2(codewords))
    pair_payload = pairs * index_bits + singles * 3
    block_count = math.ceil(numel / block_size)
    scales = block_count * scale_bits * int(scales_per_block)
    factor = factor_bits * (m * lowrank_rank + n * lowrank_rank + lowrank_rank)
    codebook = codewords * 2 * codebook_bits
    metadata = 96 + 24 + 32 + scales + int(rotation_bits)
    basis = 32 * (m * min(m, n) + n * min(m, n)) if include_spectral_basis else 0
    return {"pair_index_bits": pair_payload, "scale_bits": scales,
            "factor_bits": factor, "shared_codebook_bits": codebook,
            "metadata_bits": metadata, "rotation_bits": int(rotation_bits),
            "spectral_basis_bits": basis,
            "total_bits_unamortized": pair_payload + factor + codebook + metadata + basis,
            "total_bits_excluding_shared_codebook": pair_payload + factor + metadata + basis,
            "fp32_bits": 32 * numel,
            "storage_ratio_unamortized": (pair_payload + factor + codebook + metadata + basis) / (32 * numel)}


def codebook_occupancy(indices: torch.Tensor, codewords: int) -> dict:
    counts = torch.bincount(indices.detach().cpu().long(), minlength=codewords)
    return {"occupied": int((counts > 0).sum()), "dead": int((counts == 0).sum()),
            "counts": counts.tolist()}
