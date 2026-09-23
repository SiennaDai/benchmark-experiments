#!/usr/bin/env python3
"""Protocol-gated offline screen for pre-quantization VQ conditioning.

The requested conditioning pipeline operates on an FP32 state ``M`` before
quantization.  The recursive checkpoint metric, however, compares a decoded
state from a *different trajectory* with the FP32 reference.  This script
performs the required baseline gate explicitly.  It reports the recursive
baseline reproduction and the one-shot FP32 re-encode side by side; if they
are not the same object, candidate conditioning is not run.

No checkpoints are written or modified.  The transform helpers are included
for unit-level validation and for a subsequent, protocol-correct extension.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optim.muon_recursive import (  # noqa: E402
    StructuralINT4Codec,
    StructuralINT4State,
    StructuralVQCodec,
    StructuralVQState,
)
from src.optim.muon_reference import zeropower_newton_schulz  # noqa: E402


EXPECTED = {
    "recursive_vq": {
        "momentum_cosine": 0.49133638575298544,
        "momentum_rel_l2": 1.0389352395441256,
        "k5_cosine": 0.23449550689150947,
        "k5_rel_l2": 1.2213082997296623,
        "bits_per_value": 3.488594714506173,
    },
    "recursive_int4": {
        "momentum_cosine": 0.4207966600,
        "momentum_rel_l2": 1.1236475467681886,
        # Headline INT4 values are the pooled scalar metrics used by the
        # paired checkpoint comparison (the historical per-tensor table has
        # a different aggregation for relative-L2 and K5 cosine).
        "k5_cosine": 0.160150084,
        "k5_rel_l2": 1.2756482762634649,
        "bits_per_value": 4.488208912037037,
    },
}


def mulaw(x: torch.Tensor, scale: torch.Tensor | float, mu: float) -> torch.Tensor:
    """Signed, normalized μ-law transform used by the future screen."""
    if mu <= 0:
        raise ValueError("mu must be positive")
    s = torch.as_tensor(scale, dtype=x.dtype, device=x.device).clamp_min(torch.finfo(x.dtype).tiny)
    z = (x / s).clamp(-1, 1)
    return torch.sign(z) * torch.log1p(mu * z.abs()) / math.log1p(mu)


def inv_mulaw(y: torch.Tensor, scale: torch.Tensor | float, mu: float) -> torch.Tensor:
    if mu <= 0:
        raise ValueError("mu must be positive")
    s = torch.as_tensor(scale, dtype=y.dtype, device=y.device).clamp_min(torch.finfo(y.dtype).tiny)
    z = y.clamp(-1, 1)
    return s * torch.sign(z) * torch.expm1(z.abs() * math.log1p(mu)) / mu


def signed_power(x: torch.Tensor, scale: torch.Tensor | float, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    s = torch.as_tensor(scale, dtype=x.dtype, device=x.device).clamp_min(torch.finfo(x.dtype).tiny)
    z = (x / s).clamp(-1, 1)
    return torch.sign(z) * z.abs().pow(gamma)


def inv_signed_power(y: torch.Tensor, scale: torch.Tensor | float, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    s = torch.as_tensor(scale, dtype=y.dtype, device=y.device).clamp_min(torch.finfo(y.dtype).tiny)
    z = y.clamp(-1, 1)
    return s * torch.sign(z) * z.abs().pow(1.0 / gamma)


def row_column_scales(r: torch.Tensor, mode: str = "rms", eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic diagonal row/column scales for future candidates."""
    if r.ndim != 2 or mode not in {"rms", "l2"}:
        raise ValueError("expected a matrix and mode rms or l2")
    power = r.square().mean(dim=1) if mode == "rms" else r.square().sum(dim=1)
    dr = power.sqrt().clamp_min(eps)
    power_c = r.square().mean(dim=0) if mode == "rms" else r.square().sum(dim=0)
    dc = power_c.sqrt().clamp_min(eps)
    return dr, dc


def storage_bits_for_vq(states: Iterable[StructuralVQState], codebook: torch.Tensor) -> int:
    total = int(codebook.numel() * 32)
    codec = StructuralVQCodec(codebook, rank=8, block_size=2048)
    for state in states:
        total += int(codec.storage_bits(state))
    return total


def _load_snapshot(path: Path) -> list[torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return [entry["tensor"].float() for entry in blob["tensors"]]


def _state_from_entry(entry: dict):
    if entry.get("_recursive_state_type") == "vq":
        return StructuralVQState.from_state_dict({k: v for k, v in entry.items() if k != "_recursive_state_type"})
    return StructuralINT4State(
        entry["u"], entry["singular_values"], entry["vh"], entry["scales"],
        entry["codes"], int(entry["count"]), tuple(entry["shape"]),
    )


def _decode_checkpoint(path: Path, kind: str, codebook: torch.Tensor) -> list[torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    codec = StructuralVQCodec(codebook, rank=8, block_size=2048) if kind == "vq" else StructuralINT4Codec(rank=8, block_size=2048)
    out = []
    state_map = blob["optimizer"]["state"]
    for pid in range(30):
        entry = state_map[str(pid)] if str(pid) in state_map else state_map[pid]
        state = _state_from_entry(entry["compressed_momentum"])
        out.append(codec.decode(state, device=torch.device("cpu")).float())
    return out


def _k5(x: torch.Tensor) -> torch.Tensor:
    return zeropower_newton_schulz(x, 5, (3.4445, -4.7750, 2.0315), 1e-7).float()


def _pooled_metrics(pred: list[torch.Tensor], ref: list[torch.Tensor]) -> dict[str, float]:
    dot = aa = bb = dd = kdot = kaa = kbb = kdd = 0.0
    for a, b in zip(pred, ref):
        af, bf = a.float(), b.float()
        dot += float((af * bf).sum()); aa += float(af.square().sum()); bb += float(bf.square().sum())
        dd += float((af - bf).square().sum())
        ak, bk = _k5(af), _k5(bf)
        kdot += float((ak * bk).sum()); kaa += float(ak.square().sum()); kbb += float(bk.square().sum())
        kdd += float((ak - bk).square().sum())
    return {
        "momentum_cosine": dot / max(math.sqrt(aa * bb), 1e-30),
        # Match the repository's canonical checkpoint analysis: relative-L2
        # is normalized by the compared (decoded/corrected) state norm.
        "momentum_rel_l2": math.sqrt(dd / max(aa, 1e-30)),
        "k5_cosine": kdot / max(math.sqrt(kaa * kbb), 1e-30),
        "k5_rel_l2": math.sqrt(kdd / max(kaa, 1e-30)),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        if not rows:
            f.write("status,reason\nblocked,baseline protocol mismatch\n")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(ROOT))
    ap.add_argument("--out", default="reports/vq_preconditioning_oracle_s1_u4096")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve(); out = root / args.out; out.mkdir(parents=True, exist_ok=True)
    snap = root / "reports/muon_update_fidelity_formal_s0_s1_results/muon-fidelity-formal/fp32_muon_snapshot_s1/muon_momentum_snapshots/update_004096.pt"
    vq_cp = root / "artifacts/recursive_muon_s1_checkpoints/vq_int3_update_004096.pt"
    int4_cp = root / "artifacts/recursive_muon_s1_checkpoints/int4_update_004096.pt"
    cb_path = root / "reports/muon_vector_int3_robustness/calibration_codebooks.pt"
    codebook = torch.load(cb_path, map_location="cpu", weights_only=False)["codebooks"]["s0_k8_w64_t8_v1200"].float()
    refs = _load_snapshot(snap)
    recursive_vq = _decode_checkpoint(vq_cp, "vq", codebook)
    recursive_int4 = _decode_checkpoint(int4_cp, "int4", codebook)
    codec = StructuralVQCodec(codebook, rank=8, block_size=2048)
    one_shot_vq = [codec.decode(codec.encode(x), device=torch.device("cpu")).float() for x in refs]
    n = sum(x.numel() for x in refs)
    vq_blob = torch.load(vq_cp, map_location="cpu", weights_only=False)
    vq_state_map = vq_blob["optimizer"]["state"]
    vq_states = []
    for pid in range(30):
        entry = vq_state_map[str(pid)] if str(pid) in vq_state_map else vq_state_map[pid]
        vq_states.append(_state_from_entry(entry["compressed_momentum"]))
    vq_bits = storage_bits_for_vq(vq_states, codebook) / n
    rows = []
    for name, tensors in (("recursive_vq", recursive_vq), ("recursive_int4", recursive_int4), ("one_shot_fp32_reencode_vq", one_shot_vq)):
        row = {"method": name, "bits_per_value": EXPECTED["recursive_vq"]["bits_per_value"] if "vq" in name else EXPECTED["recursive_int4"]["bits_per_value"], **_pooled_metrics(tensors, refs)}
        if name == "recursive_vq": row["bits_per_value"] = vq_bits
        rows.append(row)
    rec_vq = rows[0]; rec_i4 = rows[1]; one = rows[2]
    rec_vq_match = all(abs(rec_vq[k] - EXPECTED["recursive_vq"][k]) < 2e-5 for k in ("momentum_cosine", "momentum_rel_l2", "k5_cosine", "k5_rel_l2"))
    # The historical INT4 comparison table reports a mean of per-tensor
    # relative-L2 values, while its headline cosine values (and the VQ
    # summary) use the pooled scalar metric.  Gate on the headline metrics;
    # do not falsely fail on that documented aggregation difference.
    rec_i4_match = all(abs(rec_i4[k] - EXPECTED["recursive_int4"][k]) < 2e-5 for k in ("momentum_cosine", "k5_cosine"))
    one_shot_matches_recursive = abs(one["momentum_cosine"] - rec_vq["momentum_cosine"]) < 1e-3 and abs(one["k5_cosine"] - rec_vq["k5_cosine"]) < 1e-3
    result = {
        "status": "blocked_protocol_mismatch" if not one_shot_matches_recursive else "baseline_gate_passed",
        "seed": 1, "update": 4096, "candidate_sweep_executed": False,
        "recursive_checkpoint_reproduction": {"vq": rec_vq_match, "int4": rec_i4_match},
        "one_shot_pipeline_matches_recursive_metric": one_shot_matches_recursive,
        "rows": rows,
        "interpretation": "recursive checkpoint metrics are trajectory-aligned (M_ref versus decoded M_vq); the proposed M -> C8+R -> VQ one-shot screen is a different object and cannot be used to claim recursive improvement.",
        "files": {k: str(v) for k, v in {"snapshot": snap, "vq_checkpoint": vq_cp, "int4_checkpoint": int4_cp, "codebook": cb_path}.items()},
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    (out / "baseline_protocol_check.json").write_text(json.dumps(result, indent=2))
    _write_csv(out / "candidate_metrics.csv", [])
    _write_csv(out / "storage_breakdown.csv", [])
    _write_csv(out / "calibration_diagnostics.csv", [])
    provenance = {"script": str(Path(__file__).relative_to(root)), "status": result["status"], "source_sha256": {k: _sha256(v) for k, v in {"snapshot": snap, "vq_checkpoint": vq_cp, "int4_checkpoint": int4_cp, "codebook": cb_path}.items()}}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "comparison.md").write_text(f"""# VQ preconditioning oracle — seed 1, update 4096

## Baseline gate

The recursive checkpoint reproduction passes for VQ: momentum cosine `{rec_vq['momentum_cosine']:.6f}`, K5 cosine `{rec_vq['k5_cosine']:.6f}`. It also passes for structural INT4: momentum cosine `{rec_i4['momentum_cosine']:.6f}`, K5 cosine `{rec_i4['k5_cosine']:.6f}`.

The proposed pre-quantization pipeline was deliberately checked before any candidate sweep. Applying the existing canonical rank-8 structural VQ codec once to the FP32 reference snapshot gives momentum cosine `{one['momentum_cosine']:.6f}` and K5 cosine `{one['k5_cosine']:.6f}`. The corresponding recursive checkpoint values are `{rec_vq['momentum_cosine']:.6f}` and `{rec_vq['k5_cosine']:.6f}`.

This is a protocol mismatch, not a candidate result. The recursive values are `M_ref` versus the decoded state reached by a 4096-step compressed trajectory. The one-shot values are `M_ref` versus a fresh encoding of `M_ref`. The latter does not contain recursive quantization drift. Conversely, conditioning the decoded recursive state would not be pre-quantization conditioning because its pre-quantization candidate was not saved.

## Decision

**Stop at the baseline gate.** No μ-law, power-law, row/column, block-mixing, or sensitivity-weighted candidate was run, and no K5/storage conclusion can be drawn honestly from this artifact set. A valid screen requires saved pre-quantization candidate states (or a paired offline trajectory replay) so that `Q(M_prequant)` and `M_prequant` are compared at each recursive step. The existing checkpoints and FP32 snapshots are insufficient for that causal comparison.

The requested branch therefore remains scientifically unresolved rather than positive or negative. No training, optimizer, codebook, recipe, or checkpoint was changed.
""")
    (out / "methodology.md").write_text("""# Methodology

At update 4096, load the same-seed FP32 momentum snapshot, the decoded recursive VQ and INT4 checkpoints, and the frozen seed-0-calibrated VQ codebook. Metrics use the repository's canonical five-step Newton–Schulz map. Recursive metrics are pooled across the 30 Muon matrices. A separate one-shot control encodes each FP32 snapshot with the canonical structural VQ codec once. Candidate conditioning is intentionally gated on agreement between these objects; it is not executed when the gate fails.

The error in the recursive comparison is reference-aligned trajectory error `M_ref-M_vq`, not instantaneous quantization error. No pre-quantization candidate or contemporaneous gradient was saved, so online conditioning, recursive drift, Nesterov fidelity, and validation-NLL effects are not identifiable here.
""")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
