# Upstream provenance

- Repository: `https://github.com/epfml/llm-optimizer-benchmark.git`
- Frozen commit: `fdecd5d7dcba228f15b623a96b3346a5281c8734`
- Commit date: 2025-12-30
- License: Apache-2.0; the original `LICENSE` is retained unchanged.
- A pristine copy of this commit for local comparison is retained in the ignored `work/upstream-reference/` directory.

The platform directly imports `src/models/llama.py` and its base attention/model classes. Llama attention, RoPE, RMSNorm, SwiGLU FFN, tied embeddings, initialization, and forward loss remain upstream code. Local model changes are limited to delaying tokenizer construction, allowing explicit `math` attention, and returning all-position logits when requested without targets. These remove optional data dependencies and expose the numerical test path without changing the training forward definition.

Tests fix a shared state dict/input and check deterministic logits/loss, causal behavior, CE agreement, gradient accumulation, and tied-parameter grouping. A pristine independent-process parity harness was not completed; T11 is therefore partial rather than claimed as full upstream parity.
