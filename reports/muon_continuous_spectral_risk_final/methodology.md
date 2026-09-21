# Methodology

For each FP32 snapshot matrix M, compute a reduced FP32 SVD M=U diag(sigma) V^T and production INT4 dynamic reconstruction Q(M). Active modes satisfy sigma/sigma_max >= 1e-6. The primary continuous coordinate is log10(sigma/sigma_max), with local absolute/relative gaps, rank percentile, Newton--Schulz A_mag and A_dir descriptors.

Residual coordinates are E_hat=U^T(Q(M)-M)V. Diagonal-bin energy is the sum of |E_hat[i,i]|^2. Left/right associated energy sums full rows/columns for modes in a bin; these are parallel attributions, not an additive partition. Direct diagonal restoration removes only selected diagonal coordinates. Associated restoration removes every coordinate whose row OR column is selected; different bins overlap on cross-bin entries by design, so direct gains are not expected to add.

Controlled sensitivity uses deterministic epsilon=0.001 magnitude perturbations of selected singular values and adjacent left-vector rotations, measured through production K=5 and exact polar. Only the first 20 tensors per bin in deterministic discovery order are used for these expensive interventions. Cross-scale pairs are shortlisted by pre-intervention residual energy.

The danger zone is the narrowest contiguous fixed-bin interval reaching 50% of positive mean associated-restoration gain. This rule is declared before interpreting outcomes. Pearson and Spearman values are descriptive correlations only.
