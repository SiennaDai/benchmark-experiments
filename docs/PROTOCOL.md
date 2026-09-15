# Experiment protocol

Recipes are complete JSON objects with no inheritance. Duplicate keys, unknown fields, bad types, unsupported precision/backend combinations, incorrect FFN/head dimensions, and token budgets not divisible by effective batch are errors. The resolved recipe and its SHA-256 scientific fingerprint are saved with each run.

Token files are little-endian uint16 and are checked against manifest byte length and SHA-256 before model allocation. For length `T`, window `j` reads tokens `[jT, jT+T]`; inputs are the first `T`, targets the last `T`. Adjacent targets do not overlap. Training shuffles window IDs with a private NumPy generator; validation/test use ascending IDs. Sampler permutation, offset, epoch and RNG state are checkpointed.

For update `k`, warmup uses `peak*k/W`. Cosine decay uses `u=(k-W)/(N-W)` after warmup and reaches the configured final ratio at update `N`; `W=0` spans updates 1 through N and `N=1` uses peak LR. `N` is `schedule.total_updates` when supplied, otherwise the actual updates derived from `train.target_tokens`; training always terminates at that latter target-token-derived count. Gradients are accumulated from token-weighted microbatch losses, checked for finiteness, globally clipped once, and followed by exactly one optimizer step.

FP32 disables autocast. BF16 requires CUDA BF16 support and never falls back to FP16. Parameters and gradients remain FP32. Strict numerical recipes use math attention, TF32=false and compile=false.

Evaluation sums per-batch mean CE times target count in an FP64 host scalar and reports nats/token and perplexity. It preserves model train/eval state and consumes no sampler or RNG. Training reads validation only; test requires the independent evaluate command.

Checkpoints are written at complete update boundaries via temporary file plus atomic replace. They include model/optimizer, named parameter order, scientific/data fingerprints, counters, sampler, Python/NumPy/Torch RNG, elapsed time, run ID and next log segment. Resume rejects changed scientific/data fingerprints and appends a new segment.

Reference AdamW computes FP32 moments and updates with epsilon outside the square root. `bf16_roundtrip` uses each unrounded new moment for the current update, then stores BF16-to-FP32 rounded moments for the next update. It is a numerical persistence simulation and has FP32 memory footprint; it is not a bitsandbytes emulation.

## INT8 linear persisted-state simulation

`int8_linear_first_moment`, `int8_linear_second_moment`, and
`int8_linear_all_moments` apply to Reference AdamW's `exp_avg`, `exp_avg_sq`,
or both, respectively. `int8_linear_momentum` applies only to a Reference Muon
hidden-matrix momentum buffer; its auxiliary AdamW state remains FP32. Each
selected state tensor is independently simulated after its current FP32 update
and after it has been used for that step's parameter update:

\[
s=\|x\|_\infty/127,\qquad Q(x)=\operatorname{clamp}(\operatorname{round}(x/s),-127,127)s.
\]

The all-zero tensor remains zero. Rounding is deterministic nearest rounding;
there is no stochastic rounding, clipping policy, nonlinear codebook, or fused
kernel. Dequantized results remain FP32 in this platform, so this tests
post-update persistence numerics—not optimizer-state memory saving or BF16/INT8
training kernels. `precision.json` records the selected and intentionally FP32
state tensors for every run.
