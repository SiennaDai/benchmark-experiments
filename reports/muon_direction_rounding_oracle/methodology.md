# Muon INT4 direction-aware rounding oracle

The fixed quantizer is the existing signed dynamic-map INT4 blockwise roundtrip with block size 2048 and absmax scale. Production nearest is delegated to `persist_state`; lower/upper candidates use the same codebook, clamp, scale, and block partition.

The raw-direction oracle starts at nearest and performs deterministic block-local grouped coordinate descent (two passes; up to 32 flips per accepted group). Candidates are eligible when normalized distance to the neighboring-level midpoint is at most the configured midpoint margin (default 0.25). The Muon-update oracle uses the same eligible set, ranks candidates by a deterministic first-order raw-direction proxy, considers at most 32 candidates per tensor, then tests groups of 8 with the exact production `zeropower_newton_schulz` (up to 4 groups); a group is accepted only on strict post-Muon cosine improvement. Candidate, considered, accepted, and exact-evaluation counts are recorded. This is an offline oracle/headroom study with access to FP32 M and is not a deployable quantizer.

The search is deterministic and bounded; only final metrics use exact production Muon transforms.
