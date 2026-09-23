# Corrected recursive Muon mechanism decomposition

## Protocol integrity

Both runs passed shared-metadata validation: seed `1`, data seed `1337`, algorithm seed `2026`, dataset fingerprint `30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18`, schedule horizon `4096`, and `8192` tokens/update. The FP32 and VQ snapshots also agree on Muon momentum, Nesterov setting, K5 iteration count, coefficients, and epsilon. The paired summaries agree on target-token budget and sequence length, and all landmark tensor names/shapes match. Both summaries report `paused_staged` at update 512 with no divergence status. Raw snapshots pair at updates `1, 8, 32, 128, 512` with 30 matching tensor names/shapes each. All 512 train events have matching update/token positions, tokens/update, and learning rates. Batch IDs/hashes were not saved, so data-window alignment is supported by the deterministic protocol and aligned positions but is not independently hash-verified.

Snapshot fields were mapped using `MuonMechanismObserver` plus the optimizer step implementation: `gradient` is current `g_t`; `momentum_prev_decoded` is the entering persistent/decoded `M_(t-1)`; VQ `momentum_prev_candidate` is the prior step's unquantized candidate before persistence; `momentum_candidate` is current `M_t`; and VQ `momentum_persisted_decoded` is `decode(Q(M_t))` for the next step. FP32 persistence equals the candidate. At update 1, entering states and the prior candidate are explicitly zero.

## Corrected temporal table

Directions were rebuilt as `D_t = (1+μ)g_t + μ²M_(t-1)` and checked against `g_t + μM_t` using the saved candidate. Metrics are globally pooled over tensor values by summed dot products and squared norms. Persistence relative-L2 is normalized by the VQ candidate; trajectory/gradient/direction/K5 relative-L2 values use the FP32 counterpart as reference. Local relative-L2 uses the actual VQ K5 norm; state-only and gradient-only use FP32 K5 norm.

| Update | Persistence cos / rel-L2 | Momentum cos / rel-L2 | Gradient cos / rel-L2 | Direction cos / rel-L2 | Full K5 cos / rel-L2 | Local K5 cos / rel-L2 | State-only K5 cos / rel-L2 | Gradient-only K5 cos / rel-L2 | Val NLL FP32 / VQ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.99751 / 0.07062 | 1.00000 / 0.00000 | 1.00000 / 0.00000 | 1.00000 / 0.00000 | 1.00000 / 0.00000 | 1.00000 / 0.00000 | 1.00000 / 0.00000 | 1.00000 / 0.00000 | — |
| 8 | 0.99951 / 0.03144 | 0.99787 / 0.06525 | 1.00000 / 0.00243 | 0.99844 / 0.05578 | 0.76229 / 0.70785 | 0.94219 / 0.33922 | 0.76228 / 0.70787 | 0.99999 / 0.00366 | — |
| 32 | 0.99981 / 0.01928 | 0.99688 / 0.07927 | 0.99765 / 0.06853 | 0.99677 / 0.08060 | 0.61394 / 0.93820 | 0.96501 / 0.26392 | 0.61421 / 0.93794 | 0.99839 / 0.05668 | — |
| 128 | 0.99684 / 0.07949 | 0.90881 / 0.41748 | 0.94629 / 0.32789 | 0.91774 / 0.39772 | 0.57081 / 0.93389 | 0.94954 / 0.31794 | 0.58965 / 0.91324 | 0.94242 / 0.33952 | 6.36257 / 6.39744 |
| 512 | 0.99493 / 0.10067 | 0.76642 / 0.66198 | 0.84230 / 0.56237 | 0.79142 / 0.63237 | 0.45786 / 1.05484 | 0.94774 / 0.32329 | 0.51490 / 0.99755 | 0.85523 / 0.53941 | 5.55547 / 5.58253 |

## K5 distortion and local-versus-accumulated comparison

Distortion is `1 - pooled cosine`. State-only, gradient-only, and local previous-persistence comparisons are separate local substitutions; they are nonlinear diagnostics and must not be summed to obtain full trajectory distortion.

| Update | Full paired K5 distortion | Local previous-persistence distortion | State-only distortion | Gradient-only distortion | Full / local distortion ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00× |
| 8 | 0.23771 | 0.05781 | 0.23772 | 0.00001 | 4.11× |
| 32 | 0.38606 | 0.03499 | 0.38579 | 0.00161 | 11.03× |
| 128 | 0.42919 | 0.05046 | 0.41035 | 0.05758 | 8.51× |
| 512 | 0.54214 | 0.05226 | 0.48510 | 0.14477 | 10.37× |

## Reading the endpoint

At update 512, the full paired K5 cosine is `0.4579` versus `0.9477` for the local previous-persistence substitution. The latter holds the current VQ gradient fixed and isolates only the preceding persistence decision; the much larger full discrepancy therefore reflects accumulated trajectory differences, not just that one quantization event.
At the same endpoint, state-only K5 cosine is `0.5149` and gradient-only is `0.8552` (both compared with the FP32 K5 output). This indicates the current state mismatch is more damaging than substituting the current gradient alone in these local tests, while gradient divergence is also material. These nonlinear substitutions are not additive, so they do not quantify causal shares.


## Tensor-level concentration at updates 128 and 512

### Update 128

| Diagnostic | Most affected tensors (metric) |
|---|---|
| Persistence relative-L2 | `transformer.h.1.mlp.c_proj.weight` (0.102); `transformer.h.2.mlp.c_proj.weight` (0.100); `transformer.h.0.mlp.c_proj.weight` (0.097); `transformer.h.0.mlp.w2.weight` (0.091); `transformer.h.3.mlp.c_proj.weight` (0.088) |
| Gradient relative-L2 | `transformer.h.1.attn.c_attn.weight` (0.461); `transformer.h.2.attn.c_attn.weight` (0.422); `transformer.h.0.attn.c_attn.weight` (0.402); `transformer.h.1.mlp.w1.weight` (0.364); `transformer.h.3.attn.c_attn.weight` (0.362) |
| Full K5 update relative-L2 | `transformer.h.5.attn.c_attn.weight` (1.190); `transformer.h.5.attn.c_proj.weight` (1.183); `transformer.h.4.attn.c_attn.weight` (1.183); `transformer.h.3.attn.c_proj.weight` (1.180); `transformer.h.4.attn.c_proj.weight` (1.179) |
| State-only K5 distortion | `transformer.h.3.attn.c_attn.weight` (0.595); `transformer.h.3.attn.c_proj.weight` (0.594); `transformer.h.4.attn.c_proj.weight` (0.592); `transformer.h.2.attn.c_attn.weight` (0.591); `transformer.h.4.attn.c_attn.weight` (0.587) |
| Gradient-only K5 distortion | `transformer.h.2.attn.c_attn.weight` (0.104); `transformer.h.3.attn.c_attn.weight` (0.099); `transformer.h.1.attn.c_attn.weight` (0.098); `transformer.h.4.attn.c_attn.weight` (0.086); `transformer.h.2.attn.c_proj.weight` (0.085) |

Across 30 tensors, median full-K5 rel-L2 is `0.863`; the table therefore shows tail severity, not a claim that only these tensors diverge.

### Update 512

| Diagnostic | Most affected tensors (metric) |
|---|---|
| Persistence relative-L2 | `transformer.h.2.mlp.c_proj.weight` (0.132); `transformer.h.1.mlp.c_proj.weight` (0.131); `transformer.h.3.mlp.c_proj.weight` (0.130); `transformer.h.2.mlp.w2.weight` (0.119); `transformer.h.3.mlp.w2.weight` (0.119) |
| Gradient relative-L2 | `transformer.h.1.attn.c_attn.weight` (0.801); `transformer.h.2.attn.c_attn.weight` (0.787); `transformer.h.3.attn.c_attn.weight` (0.769); `transformer.h.4.attn.c_attn.weight` (0.708); `transformer.h.0.attn.c_attn.weight` (0.665) |
| Full K5 update relative-L2 | `transformer.h.5.attn.c_attn.weight` (1.285); `transformer.h.4.attn.c_attn.weight` (1.281); `transformer.h.3.attn.c_attn.weight` (1.269); `transformer.h.3.attn.c_proj.weight` (1.266); `transformer.h.2.attn.c_proj.weight` (1.259) |
| State-only K5 distortion | `transformer.h.5.attn.c_attn.weight` (0.705); `transformer.h.4.attn.c_attn.weight` (0.703); `transformer.h.3.attn.c_attn.weight` (0.702); `transformer.h.2.attn.c_attn.weight` (0.695); `transformer.h.3.attn.c_proj.weight` (0.691) |
| Gradient-only K5 distortion | `transformer.h.3.attn.c_attn.weight` (0.233); `transformer.h.2.attn.c_attn.weight` (0.231); `transformer.h.1.attn.c_attn.weight` (0.225); `transformer.h.4.attn.c_attn.weight` (0.213); `transformer.h.2.attn.c_proj.weight` (0.202) |

Across 30 tensors, median full-K5 rel-L2 is `0.958`; the table therefore shows tail severity, not a claim that only these tensors diverge.

## Mechanism interpretation

The corrected diagnostics support H2 (recursive closed-loop drift) and H3 (state-driven local discrepancy), especially early. At update 8 the state-only K5 cosine is nearly the full paired cosine while gradient-only remains essentially 1; by 128/512 state-only remains more damaging than gradient-only, but gradient-only divergence is substantial by 512. The local previous-persistence K5 distortion stays far below the full paired distortion, consistent with accumulated history rather than one immediately preceding encode/decode event. H4 contributes later but is not the dominant single local substitution in this run. H5 remains possible because nonlinear state-gradient interaction was not isolated as an additive quantity. This is descriptive evidence from one seed and five raw landmarks, not a causal proof.

Validation NLL remains close despite large update-space divergence: at update 512 the values are `5.55547` (FP32) and `5.58253` (VQ), a gap of `+0.02706`. K5 fidelity should not be treated as a one-to-one predictor of NLL.

## Next-method implication

The justified next focus is the recursive persistence/state-feedback path, not another static quantizer sweep. The local state-only comparison makes entering-state divergence the leading local signal; current-gradient feedback becomes material later. Do not choose a specific intervention from this single-seed screen alone; validate the state-versus-gradient ordering at a longer horizon and/or another paired seed first.

## Interpretation boundaries

State-only and gradient-only values are one-step substitutions, not additive causal effects or alternate trajectories. The local previous-persistence comparison isolates only the immediately preceding persistence decision while holding the current VQ gradient fixed. Full paired K5 divergence includes accumulated state and parameter/gradient trajectory differences.

See `landmark_metrics.csv` for pooled state, gradient, direction, update, and counterfactual metrics; `tensor_metrics.csv` retains per-tensor diagnostics.
