# VQ preconditioning oracle — seed 1, update 4096

## Baseline gate

The recursive checkpoint reproduction passes for VQ: momentum cosine `0.491336`, K5 cosine `0.234496`. It also passes for structural INT4: momentum cosine `0.420797`, K5 cosine `0.160150`.

The proposed pre-quantization pipeline was deliberately checked before any candidate sweep. Applying the existing canonical rank-8 structural VQ codec once to the FP32 reference snapshot gives momentum cosine `0.991810` and K5 cosine `0.881015`. The corresponding recursive checkpoint values are `0.491336` and `0.234496`.

This is a protocol mismatch, not a candidate result. The recursive values are `M_ref` versus the decoded state reached by a 4096-step compressed trajectory. The one-shot values are `M_ref` versus a fresh encoding of `M_ref`. The latter does not contain recursive quantization drift. Conversely, conditioning the decoded recursive state would not be pre-quantization conditioning because its pre-quantization candidate was not saved.

## Decision

**Stop at the baseline gate.** No μ-law, power-law, row/column, block-mixing, or sensitivity-weighted candidate was run, and no K5/storage conclusion can be drawn honestly from this artifact set. A valid screen requires saved pre-quantization candidate states (or a paired offline trajectory replay) so that `Q(M_prequant)` and `M_prequant` are compared at each recursive step. The existing checkpoints and FP32 snapshots are insufficient for that causal comparison.

The requested branch therefore remains scientifically unresolved rather than positive or negative. No training, optimizer, codebook, recipe, or checkpoint was changed.
