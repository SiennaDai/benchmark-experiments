# Approximate low-rank structural INT4 Muon

Coverage: 10 snapshots; approximate rows: 18000; CPU runtime: 2442.5s.

The experiment compares canonical and fixed-seed randomized subspace iteration for q=(0, 2, 4), p=(0, 8), ranks=(1, 2, 4, 8, 16). Approximate factors are BF16; production INT4 dynamic b2048 and K=5 Muon are reused unchanged. See CSV files for per-tensor metrics, storage/workspace, multiply counts, danger-zone and shape analyses.

This is an offline extraction study, not a training or deployment result.

## Terminal configuration results

At q=4, p=8, randomized initialization, the state-size-weighted results were:

| rank | persistent storage / FP32 | K=5 update cosine | update relative-L2 | fraction of exact gain |
|---:|---:|---:|---:|---:|
| 4 | 0.13289 | 0.8250 | 0.5670 | 1.000 |
| 8 | 0.14027 | 0.8518 | 0.5223 | 1.000 |
| 16 | 0.15504 | 0.8864 | 0.4579 | 0.999 |

These match the exact-SVD structural reference within the reported aggregation tolerance. Canonical and randomized terminal results were effectively indistinguishable. q=2 already retained approximately 99.9%, 99.3%, and 96.0% of the exact gain for ranks 4, 8, and 16 respectively; q=0 retained only about 4–6%.

The terminal approximate/exact residual block-absmax ratios were approximately 1.0000 (rank 4), 1.0000 (rank 8), and 1.0004 (rank 16). Mean left-subspace principal-angle summaries were 0.00063, 0.00326, and 0.0320 radians for ranks 4, 8, and 16. No terminal randomized tensor had fraction-of-exact-gain below 0.5.

The deployment gate is met already by rank 1, q=2, p=0, canonical initialization: storage is 0.12735 of FP32 and weighted K=5 cosine is approximately 0.8067 while retaining about 100% of the exact rank-1 gain. Rank 4, q=2/p=0 reaches 0.8250 at 0.13289 storage; rank 8 and 16 provide higher fidelity at 0.14027 and 0.15504 storage respectively. These are offline oracle results; extraction workspace and CPU cost are not persistent-state cost, and no training prototype was launched.
