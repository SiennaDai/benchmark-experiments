# Fixed-codebook Fréchet-aware 2-D VQ assignment study

CPU-only, 300 formal matrix instances, k=8, frozen 64-word codebooks, p98/2048 scales and contiguous pair layout. Runtime 3603.9s. No training, codebook fitting, scale tuning, or production changes.

## Baseline and identical representation

Mean canonical MSE assignment K=5 cosine: 0.8524; seed 0 0.8532; seed 1 0.8516. Forward held-out seed-1 reproduction target from the robustness report is 0.85161. All alternatives keep the same codebook, p98 block scales, 6-bit pair indices, pair count, BF16 rank-8 factors, and metadata; `storage_check.csv` verifies exact equality.
Calibration-only normalization constants for mixed costs (12 deterministic tensors per calibration seed): pair MSE {0: 0.0086725334, 1: 0.0084985358}; pair-local Fréchet cost {0: 0.0013732856, 1: 0.0012423441}. Seed-specific λ choices from calibration-seed K=5 fidelity: {0: 0.0, 1: 0.0}; both select λ=0 (MSE), so the selected practical rule is exactly the unchanged baseline. All fixed nonzero λ values are also reported on both held-out directions using the corresponding opposite-seed calibration normalizers.

## Assignment outcomes

| assignment | mean K=5 cosine | median cosine | mean raw rel-L2 | mean Fréchet rel-L2 | changed pairs |
|:--|--:|--:|--:|--:|--:|
| mse | 0.8524 | 0.9065 | 0.1263 | 0.6326 | 0.00% |
| pair_local_frechet_hutchinson | 0.8318 | 0.8759 | 0.1429 | 0.7108 | 8.10% |
| pair_local_skew_hutchinson | 0.6570 | 0.6447 | 0.2800 | 1.4056 | 28.16% |
| mix_025 | 0.7884 | 0.8812 | 0.2217 | 1.3268 | 31.14% |
| mix_050 | 0.8022 | 0.8840 | 0.1860 | 1.0242 | 29.37% |
| mix_075 | 0.8195 | 0.8895 | 0.1581 | 0.8137 | 25.28% |

Pair-local results use Hutchinson-estimated 2×2 blocks of the exact production (J^*J) operator. These are stochastic estimates of isolated-pair curvature, not exact blocks. Exact analytic blocks are compared on the calibration audit subset in `pair_hessian_validation.csv`. They are not a global objective because omitted cross-pair terms remain.

## Local-metric vs finite-error behavior

Across all 300 tensors, pair-local Hutchinson Fréchet assignment changed 8.1% of pair labels, but mean K=5 cosine changed by -0.0206 (median -0.0112; tensor win rate 0.0%). Its full, directly recomputed Fréchet relative-L2 increased from 0.6326 to 0.7108; the estimated local pair metric did not transfer to the true total derivative metric. On held-out seed 1, its K=5 mean was 0.8299, versus MSE 0.8516.
Skew-only assignment was more damaging (mean K=5 cosine 0.6570, mean raw relative-L2 0.2800). The calibration-selected mixture was λ=0 in both directions, i.e. ordinary MSE; nonzero λ variants also lost fidelity (mean gains: λ=.25 -0.0641, λ=.50 -0.0502, λ=.75 -0.0329). Full paired tensor win/mismatch rates are in the CSVs.
The mixed-objective λ was selected on one trajectory seed and read out on the other; both directions are reported. No held-out update cosine selected the λ used for its own evaluation.

## Global coordinate-descent oracle

`global_coordinate_descent.csv` contains four tensors and 16 visited pairs per tensor. Exact global first-order objective ratios after the first sweep were 0.999520, 0.999961, 0.999881, 0.999902 (mean reduction 0.018%); actual K=5 cosine changes across the four examples were +0.000000, +0.000000, +0.000000, +0.000000. The local quadratic objective is nonincreasing, but reductions are tiny and produce no material actual K=5 recovery. This oracle therefore shows little usable assignment headroom in its limited deterministic subset; it is not evidence that exhaustive global search was performed.

## Estimator and polar checks

The pair-local 2×2 blocks used for full-coverage assignment are Hutchinson estimates from 32 deterministic probes, not exact per-pair Hessians. On 12 calibration audit tensors, the relative Frobenius discrepancy versus exact analytic pair blocks had median 0.331 and range 0.249–0.441; this is a material approximation limitation. The exact-polar subset contains 60 method/tensor rows and has the same qualitative ordering (means: MSE 0.8379, local Fréchet 0.8258, local skew 0.6557).
## Interpretation and decision

For this fixed representation, no tested Fréchet-aware assignment improves actual production K=5 fidelity. Pure pair-local Fréchet and skew rules regress consistently across both seeds; the calibration-only selected mixture is λ=0; the global coordinate-descent oracle reduces its linearized objective by less than 0.05% in the four tested cases and does not change update cosine materially. Pair-local block estimation itself is noisy (median exact-Hessian relative error about one third), so the result is a negative practical assignment study, not a proof that the exact pair-local or exhaustive global optimum has no headroom. Still, there is no evidence sufficient to justify Fréchet-aware codebook learning or training integration. Stop this optimizer-aware assignment branch unless a substantially more accurate, affordable block metric is developed independently.

Exact Hessian validation tensors: 12; exact-polar rows: 60. Runtime 3603.9 CPU seconds.
