# Fréchet sensitivity as a finite-error monotonic coordinate

CPU-only sweep of 1200 tensor-method error directions and 10800 finite perturbations; 4 canonical errors × 300 formal matrices, t=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5). Runtime 18.8s. No quantizer, codebook, optimizer, or training behavior changed.

## Local result and exact two-mode model

The derivation in `local_cosine_derivation.md` gives the leading coefficient ½ s_perp²; acceleration/second derivative enters only at O(t³), while radial response does not affect the leading angle. For exact polar, the derivative is tangent and s_perp=s_full. The analytic 2×2 skew formula matched numerical polar factors to maximum absolute cosine-distortion error 2.22e-16. Its sensitivity ordering is strictly monotone for t>0, even though angle=atan(ts) saturates versus the linear angle ts. Diagonal and symmetric controls stay at zero polar distortion while positive definite; branch crossing is separately documented.

## Production K=5 monotonic ordering

| t | Spearman s_perp | Pearson s_perp | pair ordering accuracy | near-tie | medium gap | large gap |
|--:|--:|--:|--:|--:|--:|--:|
| 0.05 | 1.0000 | 0.9634 | 0.9995 | 0.9955 | 1.0000 | 1.0000 |
| 0.1 | 1.0000 | 0.9682 | 0.9989 | 0.9904 | 1.0000 | 1.0000 |
| 0.2 | 1.0000 | 0.9804 | 0.9977 | 0.9797 | 1.0000 | 1.0000 |
| 0.4 | 0.9998 | 0.9963 | 0.9944 | 0.9510 | 0.9998 | 1.0000 |
| 0.6 | 0.9995 | 0.9933 | 0.9910 | 0.9208 | 0.9998 | 1.0000 |
| 0.8 | 0.9989 | 0.9802 | 0.9864 | 0.8815 | 0.9996 | 1.0000 |
| 1 | 0.9981 | 0.9659 | 0.9815 | 0.8406 | 0.9988 | 1.0000 |
| 1.25 | 0.9966 | 0.9512 | 0.9760 | 0.7994 | 0.9967 | 1.0000 |
| 1.5 | 0.9950 | 0.9402 | 0.9704 | 0.7620 | 0.9937 | 1.0000 |

At t=1, mean actual K=5 cosine distortion and median first-order finite-vector mismatch by method:

| method | mean cosine distortion | mean s_perp | median vector mismatch |
|:--|--:|--:|--:|
| direct INT4 | 0.2590 | 1.294 | 0.8650 |
| structural scalar INT3 | 0.2003 | 0.8426 | 0.7115 |
| structural INT4 | 0.1482 | 0.653 | 0.5658 |
| 64-word 2D INT3 VQ | 0.1476 | 0.6326 | 0.5509 |

## Saturation response transfer

The polar-inspired common response curve calibrated on seed 0 and tested on seed 1: held-out R² 0.9595, Spearman 0.9996; reverse split R² 0.9577, Spearman 0.9996. All three monotonic model families and method-specific fits are in `response_curve_fits.csv` and `method_specific_curves.csv`. Fit quality is descriptive; repeated tensors/landmarks are not independent samples.

## Higher-order behavior and polar subset

At t=1, pooled mean remainder norm fractions are radial 0.366, along the tangent first-order direction 0.520, and remaining orthogonal 0.715. Fractions are projections onto an orthogonalized basis (O0, v_perp), not overlapping projections onto O0 and raw v. Exact polar subset: 216 rows across six deterministic matrices and four methods; see its own Spearman/ordering and curves in `polar_vs_k5.csv`.

The real-tensor isolated-mode experiment contains 135 t/pair rows, and the selected-pair interaction table contains 3 tensors. These are mechanism diagnostics; mode pairs are not independent additive causes. See `two_mode_real_tensor.csv` and `cross_mode_interactions.csv`.

The exact 2×2 local expansion coefficient was numerically checked down to t=0.001; maximum reported relative coefficient error was 1.728e-05. Across the 1,200 real tensor-method directions, 100.0% were nondecreasing at every tested t step and the mean fraction of nondecreasing steps was 1.0000. Per-direction results are in `direction_monotonicity.csv`; this is separate from pooled cross-direction ordering.

## Interpretation

The local theorem is unconditional only as a sufficiently small-t expansion at a twice-differentiable point. Finite monotonic-coordinate support must be judged from the measured t-sweep and held-out saturation fits above. Strong rank ordering can coexist with poor first-order vector prediction because curvature changes the displacement while preserving its severity order. Conversely, loss of pairwise accuracy away from ties, weak cross-seed fit, or strong method-specific curves would bound that interpretation. No global monotonic theorem is claimed.

Post-sweep aggregation runtime in the final checkpoint-resume process: 18.8 CPU seconds; the primary finite sweep was completed in two checkpointed CPU runs and its end-to-end wall time was not retained. Full output coverage: 1200 tensor-method sensitivities, 10800 K=5 finite-response rows; exact-polar subset 216 rows.
