# Experiment protocol

Recipes are complete JSON objects with no inheritance. Duplicate keys, unknown fields, bad types, unsupported precision/backend combinations, incorrect FFN/head dimensions, and token budgets not divisible by effective batch are errors. The resolved recipe and its SHA-256 scientific fingerprint are saved with each run.

Token files are little-endian uint16 and are checked against manifest byte length and SHA-256 before model allocation. For length `T`, window `j` reads tokens `[jT, jT+T]`; inputs are the first `T`, targets the last `T`. Adjacent targets do not overlap. Training shuffles window IDs with a private NumPy generator; validation/test use ascending IDs. Sampler permutation, offset, epoch and RNG state are checkpointed.

For update `k`, warmup uses `peak*k/W`. Cosine decay uses `u=(k-W)/(N-W)` after warmup and reaches the configured final ratio at update `N`; `W=0` spans updates 1 through N and `N=1` uses peak LR. Gradients are accumulated from token-weighted microbatch losses, checked for finiteness, globally clipped once, and followed by exactly one optimizer step.

FP32 disables autocast. BF16 requires CUDA BF16 support and never falls back to FP16. Parameters and gradients remain FP32. Strict numerical recipes use math attention, TF32=false and compile=false.

Evaluation sums per-batch mean CE times target count in an FP64 host scalar and reports nats/token and perplexity. It preserves model train/eval state and consumes no sampler or RNG. Training reads validation only; test requires the independent evaluate command.

Checkpoints are written at complete update boundaries via temporary file plus atomic replace. They include model/optimizer, named parameter order, scientific/data fingerprints, counters, sampler, Python/NumPy/Torch RNG, elapsed time, run ID and next log segment. Resume rejects changed scientific/data fingerprints and appends a new segment.

Reference AdamW computes FP32 moments and updates with epsilon outside the square root. `bf16_roundtrip` uses each unrounded new moment for the current update, then stores BF16-to-FP32 rounded moments for the next update. It is a numerical persistence simulation and has FP32 memory footprint; it is not a bitsandbytes emulation.
