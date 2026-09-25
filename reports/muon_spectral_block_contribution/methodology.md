# Methodology

This is a read-only CPU analysis of the existing ten formal FP32 Muon momentum snapshots. Only 2D tensors are eligible. Each tensor is quantized through the production INT4 blockwise-dynamic b2048 path. No training state is modified.

The reduced FP32 SVD M=U Sigma V.T defines active modes with sigma/sigma_max >= 1e-6 and the established 10% head, centered-middle, and tail index bands. In FP32 spectral coordinates E_hat=U.T(Q(M)-M)V, each accounting block is E_ij=U_i E_hat_ij V_j.T. The nine primary 3x3 blocks and all other-associated blocks are orthogonal, so squared Frobenius energies do not double-count. Total, projected, and unresolved energy are retained.

For a direct ablation, Q_restore=Q-E_ij (or a selected group of blocks) is passed through the production K=5 Muon transform. The same candidates are also evaluated through the reduced-SVD exact polar factor. Update cosine gain and relative-L2 reduction are measured against the unquantized FP32 reference. Normalized efficiency divides cosine gain or L2 reduction by the removed residual-energy fraction; unstable zero-energy ratios are left unavailable.

The predeclared grouped tests are all diagonal, all off-diagonal, H<->M, M<->T, H<->T, left-row groups, and right-column groups. Three interaction sets test direct gain against the sum of single-block gains. These are descriptive ablations: Muon is nonlinear and no additive causal decomposition is claimed.
