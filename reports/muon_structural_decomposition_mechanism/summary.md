# Structural-decomposition mechanism study

Coverage: 300 eligible 2D Muon tensors from 10 formal FP32 snapshots. Production `int4-dynamic-b2048` and the exact production Muon transform are reused unchanged. No training was launched. The exact-SVD oracle stores `M_k` in FP32 side information and quantizes only `R_k=M-M_k`; it is not a deployable quantizer or reproduction of an external method.

Fixed ranks: (1, 2, 4, 8, 16). Energy ranks: (0.25, 0.5, 0.75, 0.9), selecting the smallest deterministic k explaining each target squared-Frobenius energy. Direct INT4 rows and all rank rows are in `baseline_vs_structural.csv`; post-hoc and deterministic random-mode controls are separate.

Mean structural K=5 cosine gain versus direct INT4 across rank rows: `0.09754844388476125`. Mean recovered cosine-error fraction: `0.40904598506296797`. The direct-to-structural comparison must be read together with `residual_dynamic_range.csv`, `danger_zone_error.csv`, and `cross_scale_mixing.csv`; lower absolute residual norm is not by itself an efficiency claim.

Runtime: `1247.8` CPU seconds. This is an offline oracle/headroom study; correlations do not establish causality.
