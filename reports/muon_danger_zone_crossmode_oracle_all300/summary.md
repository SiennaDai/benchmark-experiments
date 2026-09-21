# Danger-zone cross-mode suppression oracle

## Scope

This is a read-only CPU oracle study over 300 eligible 2D tensors from 10 formal FP32 Muon snapshots (seeds 0/1, updates 128/512/1024/2048/4096). No training was launched and no production optimizer, quantizer, recipe, or artifact was modified. The baseline is the existing `int4-dynamic-b2048` reconstruction and the exact production Muon transform.

## Fixed danger zone

The prior continuous-spectrum interval is reused unchanged: `[-3,-2)` in `log10(sigma/sigma_max)`, i.e. approximately `sigma/sigma_max in [1e-3,1e-2)`. No retuning occurs here.

## Direct unconstrained restorations

Mean production-K=5 cosine gains were: diagonal `0.0018813211719195047`, cross-mode `0.10679927359024684`, full danger-zone `0.10955399066209794`. These are direct restorations of actual residual coefficients; they are oracle ceilings, not deployable quantizers. Exact-polar gains are in `oracle_restoration_metrics.csv`.

The diagonal support is `i=j` within D. The cross support is `i!=j` with `i in D or j in D`; the two supports are disjoint and their union is the full danger-zone residual.

## Matched budgets

Matched-energy and matched-coefficient controls use deterministic top-magnitude actual residual coefficients. The default expensive-control cap is `30` tensors in discovery order; baseline and unconstrained restorations cover all tensors. The cap is recorded in this report rather than silently subsampling.

At the 1% `||E||_F` budget, mean cosine gain was diagonal `0.0002250951` versus cross-mode `0.0000014369`. At 64 selected coefficients, it was diagonal `0.0015810` versus cross-mode `0.0001673`. Under approximate raw-L2-matched corrections, it was diagonal `0.0005042941` versus cross-mode `0.0000037087`. These matched controls are the relevant per-budget comparison; the much larger unconstrained cross-mode gain is partly explained by the cross-mode component containing substantially more residual energy.

For spectral-distance groups, mean production gains were local `0.03020913392305374`, medium `0.01565517415603002`, and distant `0.0007834161321322123`. The mean diagonal-plus-cross interaction was `0.0008733958999315898`; this is descriptive because Muon is nonlinear.

## Interpretation

Compare cross-mode versus diagonal recovery at equal energy, equal coefficient count, and approximately equal raw reconstruction improvement before making a geometry claim. K=5 versus exact-polar columns separate finite-step effects from persistent polar-geometry repair. Interaction terms are descriptive and need not add because the transform is nonlinear.

Runtime: `1180.61` CPU seconds. Detailed CSVs and plots are the authoritative numeric outputs; no statistical significance or causal claim is made.
