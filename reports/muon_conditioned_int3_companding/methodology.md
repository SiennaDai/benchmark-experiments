# Methodology

## Data and fixed representation

The analysis loads the ten existing formal FP32 Muon momentum snapshots (seeds 0/1 at updates 128, 512, 1024, 2048, 4096). It uses eligible 2-D Muon states only. Every method starts from the same deterministic truncated-SVD conditioner, `C_k`, at ranks 4 or 8, and the same FP32 residual `R_k=M-C_k`. The retained factors are independently cast to BF16 and reconstructed as `C_hat`; no method receives additional rank or factor precision. The optimizer, model, recipes, quantizer used in production, persistence, and training artifacts are untouched.

INT3 residual quantization uses flattened blocks of 2048 values, per-block absolute maximum scale, clipping to normalized `[-1,1]`, deterministic nearest-level rounding (ties select the lower sorted code), and exact dequantized zero. The uniform codebook is `[-1,-2/3,-1/3,0,1/3,2/3,1]`; a 3-bit payload has one unused codeword. The production INT4 dynamic quantizer is called unchanged only to create a structural-INT4 reference.

## Stage A methods

- Uniform INT3 delegates to the existing offline INT3 routine used in the previous conditioner study; a parity test requires bitwise-equal roundtrip output.
- μ-law maps normalized `x` to `sign(x) log(1+μ|x|)/log(1+μ)`, quantizes in transformed coordinates using the same 7 levels, applies the analytic inverse, and restores the block scale. The fixed grid is μ = 1, 5, 20, 100, 500, 2000.
- Lloyd-Max fits a symmetric seven-level scalar codebook with exact zero and fixed outer level 1. It iterates positive interior centroids deterministically from `(1/3,2/3)`. Global fitting uses an evenly spaced deterministic normalized-residual sample from seed 0 only; per-tensor fitting is explicitly an oracle and its per-tensor fit is not counted as held-out parameter selection.
- Scale oracle holds the uniform seven-level codebook fixed and searches a deterministic bounded scale grid per b2048 block for minimum block MSE. It is an oracle, not a shared deployable scale rule.
- Muon codebook oracle searches six predeclared symmetric codebooks of form `[-1,-a2,-a1,0,a1,a2,1]`, choosing mean K=5 update cosine on calibration tensors only. Global calibration uses the first three eligible matrices at each seed-0 landmark (15 matrices per k). The capped per-tensor oracle searches the same six codebooks separately on the first 40 deterministic seed-0 cases. Its reported fidelity is in-sample and optimistic by construction.

## Stage B objectives and split

Seed 0 is calibration and seed 1 is held-out evaluation. No seed-1 update metric enters parameter selection.

For the sensitivity-weighted surrogate, the original FP32 SVD basis is used. Per-mode `A_dir` comes from the exact scalar singular-value transfer corresponding to production finite-step Newton–Schulz, is divided by its median and clipped to `[0.1,20]`. The coefficient weight is `W_ij=sqrt(w_i w_j)`, and the loss is `sum_ij W_ij |(U^T E V)_ij|^2`. The spectral codebook minimizes this loss over the same six codebook candidates on calibration only.

The danger/local surrogate is fixed before evaluation: `W_ij = 1 + 4 * 1[i or j in D] + 1 * 1[d_ij < 0.5]`, with `D` the unchanged danger interval `log10(σ/σmax) in [-3,-2)` and `d_ij` spectral distance in decades. The shared codebook minimizes weighted calibration residual error. Generalized μ-law/power parameters are selected either by this geometry loss or, for the expressly oracle Muon variant, calibration K=5 update cosine. The candidate grids are the Stage-A μ grid and γ = {0.25,0.5,0.75,1,1.5,2}.

## Metrics

For quantized residual `Q(R_k)`, final state is `M_hat=C_hat+Q(R_k)`. Raw residual metrics compare `Q(R_k)` against `R_k`; full-state metrics compare `M_hat` against FP32 `M`; Muon metrics compare the exact production transform `O_5(M_hat)` against `O_5(M)`; polar metrics compare their SVD polar factors. Relative-L2 is `||estimate-reference||_F/||reference||_F`, cosine is the Frobenius dot product divided by the two Frobenius norms. Empty/zero denominator cases remain unavailable rather than being silently filled.

Spectral error diagnostics project residual quantization error through the original SVD basis. Danger diagonal error sums squared diagonal coefficients with danger-zone index; danger cross error sums squared off-diagonal coefficients with at least one danger-zone index; danger associated combines these disjoint terms. Local/medium/distant mixing are off-diagonal error-energy fractions grouped by absolute log-spectrum separation `<0.5`, `[0.5,1.5)`, and `>=1.5` decades. The zero-rate and magnitude-range MAEs are measured against block absmax-normalized residual magnitude: small `<0.05` / `<0.01`, middle `[0.05,0.5)`, large `>=0.5`.

Matched controls are formed after all held-out quantizers are evaluated. Each coarse/refined geometry-derived codebook is paired within the same tensor, landmark, and k with (a) the nearest raw residual relative-L2 among uniform, global Lloyd-Max, and tested Stage-A μ-law values, and (b) the nearest zero fraction from that value-space reference pool. These are diagnostic nearest matches, not selection criteria. Exact polar is calculated only for designated baseline/shared methods where recorded; blank fields mean not evaluated.

## Storage and limitations

Storage uses the existing metadata-inclusive accounting: 3-bit residual payload plus b2048 scale metadata plus the BF16 low-rank factor storage. All scalar quantizers have identical persistent payload and block-scale counts; a handful of shared scalar/codebook parameters is negligible. This is idealized state storage, not actual packed-kernel allocation or runtime. No INT2 probe is conducted unless the best held-out shared geometry-aware INT3 reaches K=5 cosine 0.80. In this run it did not, so no `int2_probe.csv` is expected.

All statements are descriptive, based on two trajectories and this fixed calibration/evaluation split. No significance or causal claim is made. The 40-case per-tensor Muon codebook is an in-sample oracle upper bound only.
