# Structural INT4 Muon storage--fidelity Pareto analysis

Coverage: 10 snapshots; eligible Muon tensors: 300; condition rows: 13500.

Storage uses dense INT4 residual payload plus U(mk), sigma(k), V(nk). Metadata-inclusive estimates add one FP32 b2048 absmax scale per residual block and fixed dimensions/rank/precision/quantizer identifiers; no kernel overhead is inferred.

FP32 factor fidelity is the previous oracle condition. BF16/FP16 and mixed conditions round-trip factors before reconstruction, while the residual is always Q4(M-M_k) from the original FP32 decomposition.

Highest observed weighted K=5 cosine configuration: energy_90_fp16 (0.9336277072904287), storage ratio 0.2675628473729263.

This is an offline headroom analysis; adaptive ranks and exact factors are not a deployable method.

## Weighted baselines and frontier

Across the 300 eligible Muon matrices (state-size weighted), direct INT4 is 0.12550 of FP32 Muon-state storage (7.97× compression), with K=5 update cosine 0.7640 (unweighted tensor mean 0.7410) and exact-polar cosine 0.7605 (mean 0.7329). The idealized INT8 comparison is 0.25050 of FP32 storage (3.99×) and K=5 cosine 0.9961.

The fixed-rank BF16/FP16-factor conditions are the practical low-metadata frontier: rank 1 uses 0.12735 FP32 storage and reaches cosine 0.8067; rank 4 uses 0.13289 and reaches 0.8423; rank 8 uses 0.14027 and reaches 0.8668; rank 16 uses 0.15504 and reaches 0.8979. The 90%-energy rank condition uses 0.26756 storage (3.74× compression) and reaches 0.9336 K=5 cosine / 0.9216 exact-polar cosine. FP32 factors have the same fidelity but higher storage (90%-energy ratio 0.40962, 2.44× compression).

The minimum-storage conditions reaching K=5 cosine thresholds are: 0.80 → rank 1 BF16/FP16 at 0.12735; 0.85 → rank 8 BF16/FP16 at 0.14027; 0.90 → 90%-energy BF16/FP16 at 0.26756; 0.92 → the same 90%-energy condition. Under at least 2×, 4×, and 6× compression, the best BF16/FP16-factor weighted cosines are respectively 0.9336 (90%-energy, 3.74×), 0.8979 (rank 16, 6.45×), and 0.8979 (rank 16, 6.45×).

BF16 and FP16 factor storage have identical idealized bit counts; their numerical fidelity is effectively indistinguishable at the reported precision. Keeping sigma in FP32 changes storage and fidelity only negligibly. Adaptive-rank is an oracle: for BF16, the 0.90 target policy has 120/300 tensors unreachable within ranks 1–16 and falls back to rank 16; its aggregate cosine is 0.8787 at 0.14029 storage. The 0.95 and 0.98 targets have 204 and 300 unreachable tensors respectively, so they are not achieved globally by this rank grid.

The evidence gate is positive for an offline headroom study: BF16/FP16 factors with a 90%-energy rank satisfy both a ≥0.05 gain over direct INT4 (gain ≈0.170) and ≤50% FP32 Muon-state storage. This does not establish deployability: factors are exact SVD oracle side information, residual metadata and extraction/runtime costs are idealized, and auxiliary AdamW state is excluded from the Muon-only totals.
