# Methodology

For each matrix `M`, production quantization creates `Mq` and `E=Mq-M`. The FP32 reduced SVD defines `E_hat=U^T E V`. The prior danger interval is fixed at `[-3,-2)`; no threshold is fit in this study.

`danger_diagonal` contains only `E_hat[i,i]` for danger modes. `danger_cross` contains only off-diagonal entries with at least one danger index. `internal_cross`, `outside_cross`, and log-spectral `local`, `medium`, `distant` masks are disjoint subcomponents of the cross support. Candidate reconstructions are `Mq - U C V^T`, where `C` is an actual residual component.

Unconstrained direct restoration covers every eligible tensor. Matched budgets use target Frobenius norms `(0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)` times `||E||_F`; coefficients are selected by deterministic absolute magnitude and the final coefficient is scaled down only to hit the target norm. Coefficient controls use `(8, 16, 32, 64, 128)` entries. Reconstruction-matched pairs select the nearest raw-L2 improvement from the declared energy grid without tuning coefficients against update fidelity.

Every update readout uses the exact production `zeropower_newton_schulz` implementation and the exact SVD polar helper used by prior reports. All calculations are detached CPU diagnostics. The matched controls use a deterministic cap of `6` tensors; this is an explicit CPU control and not outcome-dependent.
