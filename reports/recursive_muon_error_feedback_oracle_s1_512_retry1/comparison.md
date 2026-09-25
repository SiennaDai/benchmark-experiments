# Recursive Muon full-FP32 error-feedback oracle

No training was run by this analysis. α=0 and FP32 are reused paired trajectories.

## Protocol integrity

Compatible: compatible; raw updates: [1, 8, 32, 128, 512]; paired training events: 512.
The event evidence matches update/token/LR positions under the deterministic data protocol; explicit batch IDs were not saved.

## K5 trajectory fidelity vs FP32

| update | method | K5 cosine | K5 rel-L2 | state-only cosine | gradient-only cosine | local feedback-only cosine |
|---:|---|---:|---:|---:|---:|---:|
| 1 | alpha0 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1 | alpha05 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1 | alpha1 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| 8 | alpha0 | 0.762290 | 0.707854 | 0.762283 | 0.999993 | 1.000000 |
| 8 | alpha05 | 0.902086 | 0.448445 | 0.902084 | 0.999998 | 0.981629 |
| 8 | alpha1 | 1.000000 | 0.000002 | 1.000000 | 1.000000 | 0.888986 |
| 32 | alpha0 | 0.613940 | 0.938200 | 0.614212 | 0.998394 | 1.000000 |
| 32 | alpha05 | 0.775883 | 0.695516 | 0.776111 | 0.999361 | 0.987037 |
| 32 | alpha1 | 1.000000 | 0.000004 | 1.000000 | 1.000000 | 0.876460 |
| 128 | alpha0 | 0.570806 | 0.933892 | 0.589655 | 0.942422 | 1.000000 |
| 128 | alpha05 | 0.729114 | 0.746324 | 0.740386 | 0.969719 | 0.981111 |
| 128 | alpha1 | 1.000000 | 0.000005 | 1.000000 | 1.000000 | 0.862609 |
| 512 | alpha0 | 0.457860 | 1.054844 | 0.514902 | 0.855228 | 1.000000 |
| 512 | alpha05 | 0.620106 | 0.885269 | 0.656901 | 0.902360 | 0.980870 |
| 512 | alpha1 | 1.000000 | 0.000008 | 1.000000 | 1.000000 | 0.868691 |

## Validation NLL

| update | fp32 | alpha0 | alpha05 | alpha1 |
|---:|---:|---:|---:|---:|
| 0 | 10.863900 | 10.863900 | 10.863900 | 10.863900 |
| 128 | 6.362571 | 6.397438 | 6.377009 | 6.362571 |
| 256 | 5.832529 | 5.846570 | 5.840166 | 5.832529 |
| 384 | 5.678909 | 5.703422 | 5.696013 | 5.678909 |
| 512 | 5.555469 | 5.582526 | 5.570630 | 5.555469 |

## Interpretation boundary

α=1 is an oracle because its full FP32 error buffer restores the information discarded by VQ at the previous persistence step. It is not evidence for a practical low-bit error-feedback state. Compare full trajectory metrics separately from the local one-step correction-only counterfactual.

## Causal interpretation at update 512

| method | momentum cosine | gradient cosine | Nesterov cosine | actual K5 cosine | actual K5 rel-L2 | validation NLL |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | 5.555469 |
| VQ, α=0 | 0.766419 | 0.842301 | 0.791417 | 0.457860 | 1.054844 | 5.582526 |
| VQ + FP32 feedback, α=0.5 | 0.867341 | 0.910492 | 0.879130 | 0.620106 | 0.885269 | 5.570630 |
| VQ + FP32 feedback, α=1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000008 | 5.555469 |

The α=0.5 intervention improves K5 cosine by 0.16225 over α=0, closing about 29.9% of the α=0-to-FP32 cosine gap; K5 relative-L2 falls by about 16.1%. Its validation-NLL gap falls from 0.02706 to 0.01516 (about 44.0% closure), but remains nonzero. These are descriptive results from one seed, not significance estimates.

At α=1, all paired landmark momentum, gradient, direction, and K5 metrics remain essentially equal to FP32: at update 512 their relative-L2 errors are respectively 4.34e-6, 3.77e-6, 4.20e-6, and 8.38e-6. Validation NLL differs from FP32 by only 2.4e-8. This is the expected result of restoring the omitted persistence residual, not a practical low-bit result.

The local counterfactual is different from the full trajectory comparison. At update 512, removing only the immediately preceding feedback injection while holding the current VQ gradient fixed yields K5 cosine 0.98087 for α=0.5 and 0.86869 for α=1, versus the FP32-aligned full-run cosines 0.62011 and 1.00000. Thus a single local correction changes the current K5 map modestly for α=0.5 and materially for α=1; the much larger α=0.5 full-run gain reflects the repeated effect of persistence corrections across the trajectory. The α=0 local comparison is 1.0 by construction because no correction is injected.

The state-only versus gradient-only substitutions at update 512 are also informative: for α=0, their K5 cosines are 0.51490 and 0.85523; for α=0.5 they are 0.65690 and 0.90236. Current entering-state divergence is therefore the more damaging isolated substitution at this landmark, while the full discrepancy is worse than either isolated substitution, consistent with accumulated nonlinear interaction. By α=1 both substitutions are FP32-aligned because both the state and gradient trajectories have remained aligned.

The current-step persistence error itself does not disappear: its pooled relative-L2 at update 512 is 0.1007 (α=0), 0.1062 (α=0.5), and 0.1309 (α=1), measured against each method's current candidate. The α=1 error buffer reconstructs the previous unquantized candidate with relative-L2 1.58e-9, so the quantization error can remain sizable while its temporal information loss is canceled in the next recurrence.

## Protocol, storage, and limits

All four runs report seed 1, data seed 1337, algorithm seed 2026, frozen-data fingerprint `30152c9b80e86cadbc9215f83794d92011bbbf5b827a2aedb31aa5d50c78fe18`, 4096 total/scheduler updates, and a clean staged stop at update 512. All 512 train events align on update, processed tokens, tokens/update, and learning rate; batch IDs were not persisted, so pairing is supported by the deterministic protocol and event alignment rather than explicit batch-ID evidence. The VQ codebook key/path is shared. Baseline snapshots came from source `81ed8bbae95b0aea286bae081b901a257d83e8ce`; the two feedback runs came from `66fc77c2f5bca86eaa879eab4858e048c486fa5b`.

The full FP32 error buffer is exactly 42,467,328 bytes for 10,616,832 Muon scalars (32 additional bits/value). The compressed VQ Muon payload plus this buffer is 35.4886 effective bits/original Muon value. Full optimizer-state storage, including non-Muon optimizer state, is 201,670,880 bytes for either feedback run, versus 159,203,552 bytes for α=0 and 197,041,152 bytes for the FP32 run. The state manifest marks the error buffer present and FP32 momentum absent; α=1 is an information-restoration oracle, not a storage-efficient optimizer.

**Conclusion:** this intervention strongly supports persistence quantization as a causal source of the recursive drift observed in α=0: restoring the discarded residual at each step recovers the FP32 trajectory to numerical tolerance, and half restoration yields consistent intermediate recovery. The evidence justifies a narrowly scoped follow-up to characterize/compress the persistent error buffer. It does not establish the behavior of any compressed buffer, nor does it generalize beyond seed 1 and 512 updates. No training was performed during this analysis.
