# Methodology

## Scope and selected representations

This CPU-only diagnostic uses the ten formal FP32 snapshots (seed 0/1, updates 128/512/1024/2048/4096), all 30 eligible 2-D Muon matrices per snapshot, and four canonical representations. Direct INT4 is production blockwise-dynamic b2048. Structural methods use the exact rank-8 SVD residual and BF16-factorized top-8 component. Scalar INT3 is the previously selected p98 block rule with the frozen global Lloyd-Max 7-level codebook; structural INT4 applies the existing production dynamic INT4 quantizer to the residual; 2-D VQ is the prior 64-word MSE codebook, contiguous pairing, p98 scaling in b2048 blocks, with calibration from the opposite seed. No codebook is fit on its evaluation seed. K=5 update metrics are joined from canonical reports on `(seed, update, parameter_id)`; reconstructed raw errors and all Gram/subspace diagnostics are recomputed from FP32 snapshots.

## Error and Gram metrics

For each reconstruction, `E=Mhat-M`. The right Gram perturbation is computed as `Mhat.T@Mhat - M.T@M` and independently checked against `M.T@E + E.T@M + E.T@E`; the left-Gram analogue is `Mhat@Mhat.T - M@M.T`. Frobenius/spectral relative perturbations divide by the corresponding reference Gram norm; Gram cosine is Frobenius alignment, and trace distortion is normalized absolute trace change. The linear and quadratic Gram terms are reported separately, with quadratic/Frobenius-total ratio. Spectral norms of symmetric Gram matrices use the largest absolute eigenvalue.

The active spectrum uses `sigma_i/sigma_max >= 1e-6`. Head and tail are the strongest and weakest 10% of active singular indices; middle is the centered 10%; these match the established prior convention. Singular values are paired in descending order. Top-8, head, middle, and tail left/right singular subspaces use principal-angle sines from singular values of basis overlaps. The gap proxy divides the absolute Gram perturbation spectral norm by the minimum external eigengap of the selected band in eigenvalues `sigma^2`. This is a bound-inspired diagnostic only; no theorem assumptions or equality are asserted.

## Update geometry and statistical comparisons

Production K=5 update cosine/relative-L2 are reused from prior matched-state reports, rather than recomputed with a reimplemented transform. Exact-polar readouts are recomputed from the same SVDs used for subspace diagnostics and compared to cached values when available. Update distortion is `1-update_cosine`; exact-polar distortion is analogously `1-exact_polar_cosine`.

Pearson and Spearman correlations are descriptive and reported pooled, by method, by shape, and by seed. Linear/log-linear regression compares raw-only, Gram-only, and Gram-plus-tail-gap features using seed holdout in both directions and leave-one-method-out validation. Features are standardized using training data only. No significance tests are performed; snapshot landmarks and method variants are dependent. Matched raw-error pairs are within the same `(seed, update, parameter)` and require symmetric relative raw-error difference <= 5%; their selection does not inspect Gram or update outcomes.

Scalar INT3 vs VQ and structural INT4 vs VQ are paired by exact matrix identity and report ratios below 1 as improvements for the numerator method. Group-wise Muon metrics are not included in the primary regression dataset; the previous report is used only as context: finer groups raised within-group quantization cosine while lowering `C_opt`, and a small layer-0 perturbation test found 23–27% off-group output distortion energy under full-matrix Muon.

## Numerical and scope limitations

The per-matrix spectral-gap proxy can be tiny or zero for clustered/repeated modes, making ratios large; these are descriptive instability indicators. Principal-angle comparisons pair equal index ranges even when singular values cluster, so projectors are preferable to individual vector cosine but band boundaries themselves may be unstable. The two seeds/five landmarks are not independent samples. Reports contain metric tables rather than saved reconstructions. No new quantizer, optimizer, conditioner, or training behavior is implemented.
