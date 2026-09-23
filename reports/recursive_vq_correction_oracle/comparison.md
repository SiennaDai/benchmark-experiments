# Recursive VQ correction oracle

## Scope and provenance

This run used all five seed-1 landmarks (128, 512, 1024, 2048, 4096), 30 Muon matrices per landmark, and one checkpoint/snapshot at a time. Seed-0 recursive checkpoints are not present locally, so no seed-0 result is claimed. The exact inputs and missing-artifact status are in `provenance.json`.

For every tensor, `M_ref` is the same-seed FP32 momentum snapshot and `M_vq` is the decoded recursive VQ state. The analyzed error is `E = M_ref - M_vq`: a reference-aligned trajectory error, not the instantaneous `M_prequant - Q(M_prequant)` error. No pre-quantization candidate was saved.

The K5 metric is the real five-step Newton–Schulz spectral map applied to the momentum tensor alone. It is not the full Nesterov update because the contemporaneous gradient was not saved.

## Main seed-1 result

The global (all Muon scalars pooled) results are:

| update | total bits/value | momentum cosine | K5-map cosine |
|---:|---:|---:|---:|
| 128 | 3.489 raw VQ | 0.9053 | 0.5223 |
| 512 | 3.489 raw VQ | 0.7623 | 0.4216 |
| 1024 | 3.489 raw VQ | 0.6302 | 0.3294 |
| 2048 | 3.489 raw VQ | 0.5417 | 0.2673 |
| 4096 | 3.489 raw VQ | 0.4913 | 0.2345 |

At update 4096, the greedy MSE-gain/byte correction frontier is:

| additional bits/value | total bits/value | momentum cosine | K5-map cosine | momentum gap closure | K5 gap closure |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 3.489 | 0.4913 | 0.2345 | 0% | 0% |
| 0.100 | 3.588 | 0.7495 | 0.2384 | 50.8% | 0.5% |
| 0.249 | 3.738 | 0.7911 | 0.2443 | 58.9% | 1.3% |
| 0.500 | 3.988 | 0.8262 | 0.2541 | 65.9% | 2.6% |
| 1.000 | 4.488 | 0.8655 | 0.2743 | 73.6% | 5.3% |

The structural INT4 reference at update 4096 is 4.488 bits/value, momentum cosine 0.3759, and K5-map cosine 0.1591. Thus every corrected-VQ point shown here remains above INT4 on these reference-aligned metrics; the raw 3.489-bit VQ already does so. Correction improves FP32-aligned momentum geometry strongly, but produces only a modest K5-map improvement.

## Error structure

Mean error-energy captured by the top singular components rises from approximately rank-1/rank-2/rank-4/rank-8 = 0.187/0.256/0.341/0.437 at update 128 to 0.298/0.384/0.453/0.525 at update 4096. The drift is therefore compressible but not nearly rank-one: a small correction state has useful headroom, while a full recovery would require considerably more than 0.5 bits/value.

## Decision

1. **Does a small correction budget recover meaningful FP32 state fidelity?** Yes. At 0.10 additional bits/value, update-4096 momentum cosine rises from 0.491 to 0.750; at 0.50 bits/value it rises to 0.826.
2. **At what total budget?** The useful operating region is roughly 3.59–3.99 bits/value. It remains below structural INT4 storage while substantially improving state alignment.
3. **Does corrected VQ exceed INT4?** Yes for the available seed-1 landmarks and both momentum/K5-map proxies; the raw VQ already exceeds the INT4 K5 proxy. This is not a training-quality comparison.
4. **Does this justify online error-feedback implementation?** It justifies a carefully scoped follow-up prototype, not a production change. The oracle uses future same-seed FP32 state and exact SVD of `E`; it cannot establish causal recursive correction, validation-NLL improvement, or Nesterov update fidelity.

The budget frontier is an offline upper bound. The correction factors are not stored checkpoints and no training behavior was changed.
