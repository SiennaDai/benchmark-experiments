# Controlled spectral perturbation methodology

This read-only offline study used 10 existing FP32 snapshots and 300 eligible 2D tensors. Runtime was 102.59 seconds on CPU. No training or prior artifact was changed.

## Bands and perturbations

The head is the smallest leading set explaining 90% of squared singular-value energy, with at least two modes when possible. The tail is its complement. Representatives were selected before intervention from low/median/high effective condition number, low/median/high actual INT4 update error, and deterministic layer-name diversity; duplicate keys were removed.

For each epsilon in `(0.0025, 0.005, 0.01, 0.02, 0.05)`, head and tail equal-energy perturbations rotate the left singular vectors with a deterministic adjacent-pair skew-symmetric generator. Bisection solves for `||E||_F / ||M||_F = epsilon`. Singular-value-only perturbations add a deterministic positive increment to singular values in the selected band and match the same Frobenius magnitude while leaving U,V fixed. Rotation-only perturbations use the same orthogonal rotation and preserve singular values. The exact production `zeropower_newton_schulz` transform is used for every update metric.

Actual INT4 error is projected as `U^T(Q(M)-M)V`; diagonal entries are singular-value error, within-band off-diagonal entries are head/tail mixing, cross-band entries are cross mixing, and reduced-SVD residual is reported separately. No new quantizer is introduced.

All constructions are deterministic, use no RNG, and record target/actual perturbation norms and singular-value checks. Undefined or degenerate cases are marked invalid rather than replaced with zero.
