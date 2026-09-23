# Update-4096 K5-aware correction oracle

## Baseline reproduction

Seed 1, update 4096, using the same decoders and canonical K5 map as the previous correction study:

| method | total bits/value | momentum cosine | K5-map cosine |
|---|---:|---:|---:|
| raw VQ | 3.4886 | 0.4913 | 0.2345 |
| structural INT4 | 4.4882 | 0.4208 | 0.1602 |

The existing Frobenius control is reproduced in the `frobenius` rows of `pareto.csv`.

## Main comparison

| allocation | extra bits/value | total bits/value | momentum cosine | K5 cosine | K5 gap closure |
|---|---:|---:|---:|---:|---:|
| raw VQ | 0.000 | 3.4886 | 0.4913 | 0.2345 | 0.0% |
| Frobenius | 0.100 | 3.5881 | 0.7495 | 0.2384 | 0.51% |
| K5-aware | 0.099 | 3.5876 | 0.5918 | 0.2400 | 0.71% |
| Frobenius | 0.249 | 3.7376 | 0.7911 | 0.2443 | 1.28% |
| K5-aware | 0.249 | 3.7376 | 0.6248 | 0.2471 | 1.64% |
| Frobenius | 0.500 | 3.9884 | 0.8262 | 0.2541 | 2.56% |
| K5-aware | 0.500 | 3.9886 | 0.6923 | 0.2562 | 2.84% |
| Frobenius | 1.000 | 4.4882 | 0.8655 | 0.2743 | 5.20% |
| K5-aware | 1.000 | 4.4886 | 0.7741 | 0.2729 | 5.02% |

The K5-aware objective does improve K5 cosine over the Frobenius control at the sub-INT4 budgets, but only by about `+0.0016` at 0.10 bits/value, `+0.0028` at 0.25, and `+0.0021` at 0.50. At the 1-bit budget it is slightly worse. The price is a large loss of momentum cosine because the K5-aware allocator deliberately spends bytes on directions with better nonlinear-map impact rather than state-space energy.

## Direction diagnostics

The allocator selected different tensors/ranks. At the first 0.10-bit budget, Frobenius allocation begins with tensor IDs 11 and 6, while K5-aware allocation begins with 21, 16, 1, and 26 (see `allocation_history.csv`). Across all candidate tensor/rank increments, the correlation between Frobenius gain/byte and K5 gain/byte is only about `0.26`, so the objectives are not identical. The top K5-aware candidate has lower ordinary energy efficiency but higher K5 gain/byte, direct evidence that lower-energy directions can be disproportionately Muon-sensitive.

## Decision

**Outcome B — weak.** Nonlinear K5-aware allocation finds a real but very small extra K5 headroom. At the useful 3.59–3.99 bits/value range it does not substantially improve over the existing Frobenius correction, and it substantially harms momentum alignment. No corrected point approaches FP32 K5 fidelity (cosine 1.0); the best tested K5 cosine is only about 0.274.

Therefore do **not** prototype online Muon-aware additive correction yet. The evidence supports closing this correction branch and focusing on changing the residual representation itself toward K5-sensitive orientation/spectral structure. A later expansion across landmarks/seeds is not justified by this single-landmark gain unless a separate reason emerges.

`E=M_ref-M_vq` is reference-aligned trajectory error, not instantaneous quantization error. K5 is momentum-only; no contemporaneous gradient was saved, so this does not establish Nesterov update or NLL recovery. Candidate factors are BF16 and storage is charged as `2*r*(m+n)` bytes.
