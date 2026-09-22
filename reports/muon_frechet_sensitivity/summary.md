# First-order Fréchet sensitivity of low-bit Muon states

CPU-only study: 300 matrices × four methods = 1200 instances; runtime 511.6s. Production K=5 transform and prior quantizers were not changed; training was not run.

## Production map and derivative

Production computes `X = M/(||M||F + 1e-7)` (transposing tall matrices before the iteration and restoring orientation afterward), then applies five steps `X <- aX+b(XXᵀ)X+c(XXᵀ)^2X` with `(a,b,c)=(3.4445,-4.7750,2.0315)`. There is no output rescaling. The map is spectral at a fixed scale, but production scale depends on M; the reported production derivative includes this radial normalization term. See `production_map_definition.md`.

The normalization-derivative correction itself has mean relative norm 0.022 of the FP32 update norm (median 0.017, p90 0.045); it is not the main sensitivity term on average, but is included and is necessary for exact agreement with production JVP.

## Core association with actual K=5 cosine distortion

| predictor | Pearson | Spearman | n |
|:--|--:|--:|--:|
| raw_relative_l2 | -0.156 | -0.155 | 1200 |
| right_gram_relative_fro | -0.278 | -0.281 | 1200 |
| tail_angle | 0.913 | 0.984 | 1200 |
| prior_gap_proxy | 0.126 | 0.817 | 1200 |
| sigma_sum_skew_score | 0.537 | 0.963 | 1200 |
| finite_k_skew_score | 0.821 | 0.991 | 1200 |
| frechet_relative_l2 | 0.966 | 0.998 | 1200 |
| frechet_distortion_pred | 0.995 | 0.998 | 1200 |

The production-map analytic predictor is `1-cos(O, O + DΦ[M](E))`; its relative-L2 counterpart and component energies are in `tensor_metrics.csv` / `frechet_channel_metrics.csv`. The frozen-normalization derivative is reported as an ablation. The exact production JVP is a separate predictor only on the deterministic 120-instance validation subset.

## Validation and nonlinear regime

Analytic-vs-production JVP relative-L2 mismatch: median 3.31e-06, p95 9.95e-06 over 120 rows. This confirms the implemented derivative matches the production-map JVP locally, including normalization.
At epsilon=1 (the full observed quantization residual), first-order secant mismatch has median 0.603; local error grows with epsilon (see finite-difference curve). Thus the derivative is a local sensitivity model and a strong ranking metric, not an exact finite-error reconstruction of the final update.
Exact-polar subset has 120 rows; analytic polar prediction vs actual polar distortion Pearson is 0.993. Exact-polar magnitude and symmetric channels are identically zero; skew and rectangular leakage remain.

## Generalization and channels

Mean held-out-by-seed K=5 R²: Fréchet cosine predictor 0.931, raw state error 0.036, Gram Frobenius error 0.077. Mean leave-one-method-out R²: Fréchet relative-L2 0.971, tail angle 0.684. On the smaller exact-polar subset, mean held-out-seed R² is 0.943 for the polar Fréchet prediction and 0.855 for tail angle. These are descriptive repeated-snapshot generalization checks, not independent-sample inferential statistics.
Mean share of squared derivative prediction by channel: magnitude 0.4%, symmetric 15.6%, skew 28.6%, out_of_subspace 55.4%. Energy share is not identical to explanatory power: out-of-subspace leakage can carry large predicted norm while alone being a weak across-instance predictor.

## Method comparisons

Scalar INT3→2-D VQ (paired n=300): median VQ/scalar ratios for raw error, skew-channel energy, out-of-subspace energy, total Fréchet relative-L2, actual update distortion, and polar distortion are 0.813, 0.599, 0.590, 0.772, 0.697, 0.685.

Structural INT4→2-D VQ (paired n=300): corresponding median ratios are 1.065, 0.967, 0.932, 0.978, 1.007, 1.009. Ratios below one favor VQ.

## Assessment

Overall, the evidence supports the first-order Fréchet sensitivity model as a substantially better descriptive surrogate than raw state error or global Gram norms: its predicted cosine distortion has near-monotonic pooled association and stronger held-out-seed prediction, while the analytic derivative numerically matches the actual production JVP. The finite-error linearization is not exact at epsilon=1, and some rank-deficient / high-error instances remain poorly predicted. The result supports a future optimizer-induced quantization objective as a hypothesis worth testing, but does not establish that minimizing this local metric will improve a finite-rate quantizer or training outcomes.

For the scalar INT3→2D VQ comparison, high-sensitivity skew and rectangular channels and total derivative-weighted error fall more than ordinary raw error. Structural INT4 and vector INT3 have nearly matched actual update distortion and nearly matched Fréchet metric despite VQ having somewhat larger raw state error. This cross-method consistency is positive evidence, not proof of unique causality.

No derivative-aware quantizer or new representation is implemented. Group-wise evidence is prior external context only. Because all landmarks reuse parameter identities, predictive scores are descriptive rather than independent generalization estimates.

Runtime: 511.6s CPU. Full table and all plots are in this directory.
