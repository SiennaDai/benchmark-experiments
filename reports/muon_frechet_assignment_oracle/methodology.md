# Methodology

## Fixed representation

The study fixes rank k=8, the existing BF16 top-8 factorized component, the existing split-specific frozen 64-word MSE codebook, contiguous row-major residual pairing, p98 shared scalar normalization for each 2048 residual scalars, and 6-bit codeword indices. Normalized residual vectors are clipped to [-1,1] exactly as in the canonical VQ implementation. Only codeword assignment changes. There is no codebook, scale, pairing, bitwidth, optimizer, or training change.

Formal snapshots cover seeds 0/1 at updates 128, 512, 1024, 2048, and 4096, with all 30 eligible 2D Muon matrices per snapshot (300 matrix instances). For held-out evaluation seed s, the canonical codebook trained on the opposite seed is used, matching the previously validated split protocol. Forward and reverse calibration use deterministic evenly spaced 12-matrix samples from the calibration trajectory only. Their MSE and local-metric normalizers and lambda choice are carried to the opposite held-out trajectory.

## Objective and assignment rules

The exact production map is the unchanged K=5 normalized Newton–Schulz implementation. The local derivative includes its state-dependent Frobenius normalization term. Per-pair full and skew 2x2 metric blocks are estimated from 32 deterministic Rademacher Hutchinson probes of J*J; they are stochastic isolated-pair block estimates and are **not** exact blocks. Candidate pair errors are evaluated in physical residual units. Mixed objectives combine block-normalized Euclidean residual error and estimated local Fréchet quadratic cost; lambda is selected by calibration-seed K=5 update cosine, never held-out data.

The global oracle starts from canonical MSE assignments and coordinate-descends the exact global first-order quadratic objective ||J[E]||² on a deterministic small subset. Its gradient includes cross-pair terms J*J[E], while per-coordinate Hessians are computed from analytic Fréchet responses. Proposed steps are checked against the exact linearized production objective. This remains an offline oracle, not a pair-separable method.

## Metrics and scope

The actual K=5 update, raw state error, Fréchet-predicted update perturbation, channel energies, and exact-polar subset are computed from unchanged prior implementations. Storage is recomputed with the existing accounting function and required to be identical across assignments. Hutchinson estimates are validated against exact analytic pair Hessians on a deterministic calibration audit. Any local-objective gain is compared directly with finite-error production K=5 fidelity because the Fréchet metric is only a local surrogate. No dense W_M is formed, no new codebook is trained, and no production or training path is modified.
