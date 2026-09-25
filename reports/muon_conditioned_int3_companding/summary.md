# Structurally conditioned INT3 residual companding

## Executive result

This CPU-only offline study covered all 10 formal snapshots (two seeds × five updates), 300 eligible 2-D momentum tensors, and structural ranks 4 and 8. Seed 0 was used for global parameter selection; seed 1 was held out. The analysis took about 45.5 minutes on the local CPU. No training or production optimizer/quantizer behavior was changed.

Conditioned scalar INT3 has real level-allocation headroom over uniform INT3: held-out K=5 update cosine rises from **0.6355/0.6751** (k=4/8) to about **0.7326/0.7630** with calibration-selected μ-law (μ=5), or **0.7324/0.7641** with global residual-MSE Lloyd-Max. But geometry-weighted scalar codebooks and companders did not improve on those ordinary magnitude/MSE choices. Even a per-block MSE-optimal scale oracle reaches only **0.7690/0.8016**, and is not a deployable shared quantizer. Structural INT4 remains clearly better at **0.8245/0.8514**.

Interpretation: INT3 is not limited only by the uniform level placement or absmax scale; those can be improved substantially. However, the tested scalar geometry-aware objectives do not exploit additional Muon-specific headroom, and 3-bit conditioned residuals remain behind structural INT4. Evidence favors a richer residual representation over further scalar compander tuning.

## Stage A — conditioned INT3 representation headroom

Held-out seed-1 means (all eligible tensors across five updates):

| Residual representation | k=4 K=5 cosine | k=8 K=5 cosine | Notes |
|---|---:|---:|---|
| Uniform 7-level INT3 | 0.6355 | 0.6751 | absmax-scaled baseline |
| μ-law, μ=5 | 0.7326 | 0.7630 | μ selected on seed-0 MSE; same shared parameter used on held-out |
| Global Lloyd-Max | 0.7324 | 0.7641 | symmetric, 7 levels, exact zero; calibrated only on seed 0 |
| Per-block MSE-optimal scale | 0.7690 | 0.8016 | oracle scale, not a shared practical method |
| Coarse-grid global Muon-update codebook | 0.6922 | 0.7256 | six-point coarse grid; selected `a1=0.2, a2=0.6` |
| Structural INT4 reference | 0.8245 | 0.8514 | unchanged production dynamic INT4 b2048 residual |

The global Lloyd-Max positive reconstruction levels were approximately `(0.1762, 0.4152)` for k=4 and `(0.1791, 0.4198)` for k=8 (plus exact zero and symmetric negatives/endpoints). The initial geometry- and Muon-selected global codebook used `(a1,a2)=(0.2,0.6)` for both ranks. A follow-up applied a deterministic local 0.05 grid around the coarse winner, selecting `(-a2,-a1,a1,a2)=(-0.55,-0.20,0.20,0.55)` for k=4 and 8 under both Muon and geometry objectives; the danger-zone objective kept those levels at k=4 and selected `(a1,a2)=(0.25,0.55)` at k=8. Results for all 300 held-out matrices are in `refined_codebook_followup.csv`.

The refined codebook results (K=5 / exact-polar cosine) were 0.7057/0.6964 for k=4 and 0.7393/0.7298 for k=8 under the Muon and A_dir geometry objectives. The k=8 danger-zone codebook reached 0.7306/0.7212. All remain below the MSE/Magnitude-selected μ-law/Lloyd-Max means, and below structural INT4. Thus coarse-to-fine optimization does not reverse the primary conclusion. The capped per-tensor Muon-codebook search covered 20 seed-0 matrices per rank and obtained in-sample means 0.6615/0.7068; this is an in-sample upper bound within a six-codebook search and not independent held-out evidence.

The shared μ-law and Lloyd-Max methods improve over uniform by roughly **+0.097** at k=4 and **+0.088–0.089** at k=8. This meets the Stage-A “at least +0.08” headroom condition, so Stage B was run. The Stage-A threshold is a research continuation criterion, not a claim that INT3 is deployment-ready.

μ-law μ=5 behaves as a weak, useful compander here; more aggressive tested μ values did not improve held-out performance. Per-tensor Lloyd-Max reached means of about 0.7382/0.7684 on the held-out matrices where computed, but its codebook is fit separately on each tensor and therefore has more calibration freedom than a shared global codebook.

The per-block scale oracle is notably stronger than shared magnitude transforms, which indicates scale choice is a meaningful component of the INT3 gap. It still leaves a ~0.05 cosine gap to structural INT4 at k=8. The oracle chooses a separate scale per block by direct residual MSE minimization and should not be interpreted as a practical kernel-ready rule.

Selected exact-polar means are directionally consistent with finite-step results: uniform INT3 gives 0.6253/0.6653; calibrated μ-law gives 0.7245/0.7549; Lloyd-Max gives 0.7235/0.7546; structural INT4 gives 0.8135/0.8369 (k=4/8). Thus ordinary scalar allocation improves persistent polar fidelity too, but it does not close the gap.

## Stage B — geometry-aware residual companding

All global Stage-B choices were selected on seed-0 calibration tensors only. The held-out k=4/k=8 means were:

| Held-out method | k=4 | k=8 |
|---|---:|---:|
| A_dir-weighted codebook, coarse → fine | 0.7057 | 0.7393 |
| Danger-zone/local-weighted codebook, coarse → fine | 0.7057 | 0.7306 |
| Geometry-selected μ-law (μ=5) | 0.7326 | 0.7630 |
| Geometry-selected power law (γ=0.75) | 0.7040 | 0.7371 |
| Calibration Muon-cosine-selected μ-law (μ=5) | 0.7326 | 0.7630 |
| Calibration Muon-cosine-selected power law (γ=0.75) | 0.7040 | 0.7371 |

The initial coarse-grid selected non-uniform codebook was `{-1,-0.6,-0.2,0,0.2,0.6,1}`. The refined geometry-weighted and direct Muon-cosine selections converged to the same narrower inner codebook `{-1,-0.55,-0.2,0,0.2,0.55,1}` for both ranks. Both still underperformed μ-law/Lloyd-Max on held-out states. Geometry-based selection of μ and γ chose μ=5 and γ=0.75; μ=5 matched the magnitude-selected μ-law parameter and did not establish an extra geometry-specific benefit.

The best tested held-out shared geometry-aware INT3 result is below 0.80 (k=8: μ-law selected by the A_dir surrogate reaches 0.7630; refined A_dir codebook reaches 0.7393). Therefore the optional INT2 gate was **not triggered** and no INT2 analysis was run. No evaluated global geometry-derived codebook offered gain over ordinary μ-law/Lloyd-Max at matched nominal 3-bit payload. In nearest-raw-L2 pairs against value-space candidates, the refined Muon/geometry codebooks trail their nearest reference by 0.0248 cosine on average (median −0.0225; 0/300 wins); the refined danger-zone codebook trails by 0.0191 on average (median −0.0200; 13.7% wins).

μ-law changes zero allocation substantially: the mean mapped-to-zero fraction falls from 0.604/0.588 (uniform, k=4/8) to 0.293/0.281 (μ=5). Its middle-magnitude MAE decreases (about 0.00023/0.00020 to 0.00014/0.00012), while large-value MAE increases (about 0.00021/0.00018 to 0.00029/0.00025). Lloyd-Max and scale-oracle choices also reduce danger-associated and local-mixing residual fractions modestly, but these changes do not make geometry-aware level selection superior in update fidelity. Full per-quantizer small/large diagnostics are in the CSVs.

## Error allocation, matched controls, and storage

Per-tensor records in `tensor_level_results.csv`, `spectral_error_analysis.csv`, and `danger_zone_companding.csv` retain raw residual error, full-state raw metrics, K=5 update metrics, selected exact-polar metrics, zero fraction, small/middle/large-value MAE, danger-zone diagonal/cross/associated error, and local/medium/distant mixing fractions. The results show that scalar companding moves error between magnitude ranges, but the tested geometry-aware objectives did not produce a corresponding held-out update-fidelity advantage.

`matched_raw_error_controls.csv` contains true nearest-neighbor comparisons computed within the same held-out tensor/update/rank. In the initial coarse run, geometry/danger/Muon codebook selections coincided. Against the nearest raw-L2 scalar reference, the coarse codebook had median raw-error gap 0.0103 and update-cosine difference −0.0217 on average (median −0.0176), winning 0.7% of pairs. Against nearest zero-rate matches, it was lower than Lloyd-Max by 0.0394 cosine on average. Refined results are separately stored in `refined_codebook_followup.csv` and retain full tensor metrics; the refined codebook also remains below the shared magnitude-based candidates in aggregate.

All scalar residual methods use the same nominal 3-bit payload, b2048 scale metadata, and BF16 top-k factors. On held-out states, the state-size-weighted metadata-inclusive ratios are **0.1016× FP32 for k=4** and **0.1090× for k=8** (unweighted means 0.1020× / 0.1098×). Changing a global μ/codebook parameter contributes negligible storage; the per-block scale oracle is not assigned extra persistent scalar metadata beyond standard block scales in this idealized comparison. These are storage equivalents, not measured allocated memory or kernel/runtime savings.

The scalar methods' update-cosine gain over uniform is consistent across this held-out tensor set: μ-law wins on all 150 tensors for each rank, with mean gains +0.0971 (k=4) and +0.0880 (k=8); global Lloyd-Max likewise wins on all tensors, with +0.0969 and +0.0890. Gains' 10th/25th/median/75th/90th percentiles are retained in the computed summary table; seed-size-weighted means are also included so larger state matrices are not hidden by equal tensor weighting.

## Decision

The evidence supports **ordinary scalar level allocation and scale selection as useful INT3 improvements**, but it does not support a practical geometry-aware scalar compander: the mechanism-derived objectives did not outperform MSE/magnitude baselines on held-out updates, and the INT3-to-INT4 gap remains material. Do not proceed directly to a 4096-step training prototype for these scalar companders. The more justified next step is a richer structured residual representation (e.g. additional structured coefficients or subspace information), with an explicit storage-matched comparison; alternatively stop the INT3 scalar-companding branch if that richer representation is out of scope.

## Artifacts

See `methodology.md` for definitions and split details, and the CSV/PNG files in this directory for tensor-level results, calibration/evaluation selections, spectral diagnostics, failure cases, and storage accounting. The per-tensor Muon-codebook oracle was capped at the first 40 deterministic seed-0 items and is an in-sample upper bound, not held-out evidence.
