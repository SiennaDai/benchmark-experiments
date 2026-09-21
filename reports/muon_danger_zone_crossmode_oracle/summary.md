# Danger-zone cross-mode suppression oracle

## Scope

This is a read-only CPU oracle study over 300 eligible 2D tensors from 10 formal FP32 Muon snapshots (seeds 0/1, updates 128/512/1024/2048/4096). No training was launched and no production optimizer, quantizer, recipe, or artifact was modified. The baseline is the existing `int4-dynamic-b2048` reconstruction and the exact production Muon transform.

## Fixed danger zone

The prior continuous-spectrum interval is reused unchanged: `[-3,-2)` in `log10(sigma/sigma_max)`, i.e. approximately `sigma/sigma_max in [1e-3,1e-2)`. No retuning occurs here.

## Direct unconstrained restorations

Mean production-K=5 cosine gains were: diagonal `0.0018813211719195047`, cross-mode `0.10679927359024684`, full danger-zone `0.10955399066209794`. Exact-polar gains were diagonal `0.0017278`, cross-mode `0.0988047`, and full danger-zone `0.1015444`. These are direct restorations of actual residual coefficients; they are oracle ceilings, not deployable quantizers. Full per-tensor K=5 and exact-polar readouts are in `oracle_restoration_metrics.csv`.

The diagonal support is `i=j` within D. The cross support is `i!=j` with `i in D or j in D`; the two supports are disjoint and their union is the full danger-zone residual.

## Matched budgets

Matched-energy and matched-coefficient controls use deterministic top-magnitude actual residual coefficients. The default expensive-control cap is `6` tensors in discovery order; baseline and unconstrained restorations cover all tensors. The cap is recorded in this report rather than silently subsampling.

At the 1% `||E||_F` budget, mean cosine gain was diagonal `0.0001971` versus cross-mode `0.0000007`. At 64 selected coefficients, it was diagonal `0.0015658` versus cross-mode `0.0001883`. Under approximate raw-L2-matched corrections, it was diagonal `0.0004902` versus cross-mode `0.0000043`. Thus the unconstrained cross-mode gain is largely explained by its much larger available residual energy; under matched budgets, diagonal correction is more effective in this oracle.

For spectral-distance groups, mean production gains were local `0.03021`, medium `0.01566`, and distant `0.00078`. The mean diagonal-plus-cross interaction was `0.000873`; this is descriptive because Muon is nonlinear.

## Interpretation

The unconstrained danger-zone cross-mode restoration is much larger than the diagonal restoration, but the cross component also contains much more residual energy. At matched energy, matched coefficient count, and approximately matched raw reconstruction improvement, cross-mode correction does not outperform diagonal correction in this run. The evidence therefore supports the weaker interpretation that danger-zone cross-mode error matters in aggregate, but does not establish that each unit of cross-mode information is intrinsically more valuable than diagonal scalar information.

K=5 versus exact-polar columns separate finite-step effects from persistent polar-geometry repair. Interaction terms are descriptive and need not add because the transform is nonlinear. The result does not justify a cross-subspace representation by itself; it supports further geometry-aware work only if a larger all-tensor matched-budget study reproduces the comparison.

Runtime: `582.47` CPU seconds. Detailed CSVs and plots are the authoritative numeric outputs; no statistical significance or causal claim is made.
