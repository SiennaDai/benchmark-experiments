# Approximate low-rank extraction methodology

The approximate path uses alternating QR-orthonormalized multiplications by M and M^T, starting from either the first canonical columns or a fixed-seed (2026) Gaussian basis. q=0 is one range projection with no alternating refinement; each q>0 adds q alternating M/M^T refinements. A small projected (at most k+p square) SVD produces factors; no full matrix SVD is called by the approximate extractor. Exact SVD is used only by the evaluation/reference path.

Residual quantization is the unchanged int4-dynamic-b2048 implementation. Retained factors are BF16 round-tripped, and persistent storage uses the existing metadata-inclusive storage model. Temporary workspace is reported separately and is not persistent optimizer state.

The all-300 CPU run uses the explicitly recorded fast grid q={0,2,4}, p={0,8}, both initializations, and all requested ranks. The script also supports the complete q={0,1,2,3,4}, p={0,4,8} grid, but that expansion is substantially more expensive. Exact-polar SVD readouts are intentionally restricted to terminal q=4, p=8 configurations in the all-300 run; other rows carry explicit empty exact-polar fields rather than silently substituting a proxy.
