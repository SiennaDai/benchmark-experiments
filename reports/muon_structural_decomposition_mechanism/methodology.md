# Methodology

For each FP32 matrix M, a reduced FP32 SVD defines exact truncated components M_k and residual R_k. The structural oracle is M_k + Q4(R_k), using the unchanged production INT4 blockwise-dynamic b2048 persistence path. The direct baseline is Q4(M). All matrices are detached diagnostic copies.

Fixed ranks are 1,2,4,8,16 when valid. Energy ranks are the smallest k reaching 25%, 50%, 75%, and 90% of squared Frobenius spectral energy. The original FP32 U,V basis is used for danger-zone and cross-scale measurements. Danger zone is the prior fixed [-3,-2) log10(sigma/sigma_max) interval. Post-hoc controls project the actual direct residual onto the same top-k two-sided FP32 subspace. Random controls use a fixed local seed 2026 without changing global RNG state.

Dynamic-range diagnostics use flattened b2048 absmax blocks and report matrix/residual block-scale statistics. Structural quantization error is Q4(R_k)-R_k; M_k is exact oracle side information. Production K=5 and exact-polar metrics are retained separately. This study measures representation headroom, not a practical storage design.
