# AdamW blockwise dynamic INT8 persistence

This treatment is a numerical persistence simulation.  It leaves parameters,
gradients, the current FP32 AdamW moment/update arithmetic, and current-step
denominator unchanged.  Only after the parameter update it dynamic-quantizes
and dequantizes selected state for the following step.

The codebook constructor is a pinned pure-PyTorch transcription of
`bitsandbytes.functional.create_dynamic_map(signed=True|False,
max_exponent_bits=7, total_bits=8)` from the bitsandbytes reference source.
It intentionally does not import bitsandbytes or use its kernels.  For each
state tensor separately, contiguous blocks of 2048 values have independent
`absmax` scales.  A partial final block has its own scale and blocks never span
state tensors.

- `exp_avg` uses the signed 256-level dynamic map.
- `exp_avg_sq` uses the unsigned 256-level dynamic map.
- Assignment is nearest codebook value; exact ties select the lower value
  deterministically.
- A zero block remains exactly zero and persistence remains FP32 after
  dequantization.

The source and parameters are recorded in each run's `precision.json` under
`state_persistence.dynamic_map_provenance`.  This is not a bitsandbytes kernel
or optimizer-memory-saving benchmark.
