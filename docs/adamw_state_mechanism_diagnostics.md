# AdamW state-mechanism diagnostics

`logging.state_diagnostics: true` enables a read-only stream only for
`reference_adamw`. It writes `state_diagnostics.jsonl` without changing the
normal `metrics.jsonl` schema.

For each optimizer step, `exp_avg` and `exp_avg_sq` are observed after their
FP32 moment updates. The current parameter update is calculated from these
unquantized values. The observer then compares them with the state after the
configured persistence simulation. Thus `denom_pre / denom_post` is a
next-step persistence-denominator amplification proxy; it is not claimed to
be the current-step denominator or a causal measurement.

All readings are detached reductions. Counts/extrema/L2 error are exact over
each state tensor; percentile summaries use bounded deterministic evenly
spaced samples per tensor to avoid material GPU-memory overhead. Non-finite
diagnostic scalars are encoded as `{"nonfinite":"nan"}`, `+inf`, or `-inf`,
so strict JSON remains enabled.
