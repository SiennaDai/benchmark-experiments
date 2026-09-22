# Does Gram / eigenspace perturbation explain Muon low-bit error?

CPU-only analysis of 300 formal matrix snapshots and 1200 selected method instances. Runtime about 7.1 minutes. No training was run; canonical quantizers and production K=5 outputs were reused.

## Methods and coverage

Compared exactly four representations: direct production dynamic INT4; BF16 top-8 structural scalar INT3 with fixed p98 scaling and the previously selected global Lloyd-Max codebook; BF16 top-8 structural INT4 using unchanged production dynamic INT4 on the residual; and BF16 top-8 structural 64-word 2-D INT3 VQ with p98/2048 normalization and the opposite-seed held-out codebook. All 300 formal 2-D matrix snapshots (2 seeds × 5 landmarks × 30 matrices) are covered. Prior update metrics are joined by seed/update/parameter ID; raw/Gram/spectrum readouts are recalculated from detached reconstructions.

## Pooled association with K=5 update distortion

See `correlation_summary.csv` for Pearson and Spearman associations; repeated matrices across landmarks and method variants are correlated observations, so these are descriptive only.

| predictor | Pearson | Spearman | n |
|:--|--:|--:|--:|
| raw_relative_fro | -0.156 | -0.155 | 1200 |
| raw_relative_spectral | -0.337 | -0.345 | 1200 |
| right_gram_relative_fro | -0.278 | -0.281 | 1200 |
| right_gram_relative_spectral | -0.028 | -0.094 | 1200 |
| right_gram_linear_relative_fro | -0.294 | -0.293 | 1200 |
| spectrum_relative_l2 | 0.602 | 0.405 | 1200 |
| tail_subspace_mean_sin | 0.913 | 0.984 | 1200 |
| tail_gap_proxy | 0.142 | 0.887 | 1200 |

## Result and hypothesis assessment

**Conclusion: partial support for eigenspace/gap geometry, but not for unnormalized global Gram perturbation magnitude as a standalone fidelity metric.** Update-distortion Spearman was -0.155 for raw relative-Frobenius error, -0.281 for right-Gram relative-Frobenius error, -0.094 for right-Gram relative-spectral error, 0.984 for tail principal-angle sine, and 0.887 for the gap-normalized Gram proxy. The raw/Gram norm associations are inverse and weak-to-modest; subspace and gap-normalized associations are strongly positive.

A one-feature log-linear raw-error model gave held-out-seed R² 0.036/0.036 (seed 1/0); the Gram-Frobenius model gave 0.079/0.076; the Gram-spectral model gave 0.001/0.004. Adding tail eigengap to Gram spectral perturbation raised held-out R² to 0.794/0.768. That combined model's leave-one-method-out R² ranges from 0.624 to 0.721. The added signal is consistent with spectral separation/eigenspace sensitivity, not Gram magnitude alone.

Among 167 pairs matched within 5% raw relative error, smaller Gram error ordered smaller update distortion in 53.3% of pairs, close to chance. This matched-pair check gives no reliable extra ordering from global Gram-Frobenius magnitude.

For structural scalar INT3 → 64-word vector INT3, median raw-error, Gram-Frobenius, tail-angle, and update-distortion ratios were 0.813, 0.923, 0.895, and 0.697. VQ improves all four in most/every paired case, but update distortion improves substantially more than Gram magnitude; this supports geometry sensitivity without establishing ΔG norm as the mediator.

Structural INT4 and vector INT3 had average update cosine 0.852 vs 0.852, and exact-polar cosine 0.837 vs 0.838. Yet the median vector-INT3/structural-INT4 Gram-Frobenius perturbation ratio was 1.328, with tail-angle ratio 1.000. Similar update fidelity despite different ΔG size argues against a one-dimensional Gram-norm explanation; similar subspace movement is compatible with an eigenspace account.

Held-out-seed and leave-one-method-out R² are in `predictive_models.csv`. These models are descriptive; repeated tensor landmarks are not independent, and the minimum-gap proxy is sensitive to clustered spectra.

## Matched reconstruction-error comparison

Found 167 within-instance method pairs whose raw relative-Frobenius errors differ by at most 5%. `matched_raw_error_pairs.csv` gives each pair and whether smaller Gram error orders smaller update distortion. Matched subsets are selected by raw-error proximity, not update results.

## Structural scalar INT3 vs 2-D vector INT3

Across 300 matched instances, median VQ/scalar ratios are: raw error 0.813, right-Gram Frobenius error 0.923, tail-subspace angle 0.895, update distortion 0.697. A ratio below one favors VQ. Exact-polar metrics and per-instance deltas are in `scalar_vs_vq.csv`.

## Structural INT4 vs vector INT3

Across 300 matched instances, median VQ/structural-INT4 ratios are: raw error 1.065, right-Gram Frobenius error 1.328, tail-subspace angle 1.000, update distortion 1.007. These methods have similar update fidelity at this rank but different residual representations; see `int4_vs_vector_int3.csv`.

## Interpretation

Overall, the evidence supports a narrower statement: singular-subspace displacement and its spectral-gap context track update distortion much more consistently than scalar raw-error or unnormalized Gram-norm summaries, but the Gram-norm matched-pair test is near chance and ΔG size is not sufficient. This does not yet justify implementing a Gram-aware compression objective. Stop the standalone-Gram-magnitude explanation branch; any next mechanism test should be specifically about mode/subspace geometry and gaps, with grouped Muon retained only as prior context. The existing group-wise study is cited in `methodology.md` but not rerun or pooled.

Only four required representative methods were evaluated; no new quantizer or Gram-aware objective was implemented. The stored head/middle/tail definition is active singular modes (sigma/sigma_max ≥ 1e-6), with each named band occupying 10% of active rank (middle centered). Gap-normalized quantities are descriptive perturbation-theory proxies, not rigorous bounds.

Total CPU wall time: 423.2 seconds. See `methodology.md` and CSVs for full definitions and subgroup outputs.
