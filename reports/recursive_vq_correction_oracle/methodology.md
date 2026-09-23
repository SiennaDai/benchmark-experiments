# Methodology

For each available seed and landmark, the script loads exactly one FP32 momentum snapshot and one recursive checkpoint, decodes the 30 Muon states, computes `E=M_ref-M_vq`, and releases them before moving to the next landmark. The correction is an offline oracle: an exact SVD of E followed by BF16 factor storage `A=U sqrt(S)`, `B=sqrt(S) V^T`. Fixed ranks are evaluated independently. Budget points greedily select prefix singular components by squared singular-value gain per BF16-factor byte.

The K5 statistic is the canonical five-step Newton--Schulz spectral map applied to the momentum tensor alone. The training checkpoint did not save the contemporaneous gradient or pre-quantization candidate, so this is not Nesterov update fidelity and cannot identify instantaneous quantization error or causal online error feedback.

Storage includes the decoded VQ payload and shared FP32 64x2 codebook. Correction bytes are `2*r*(m+n)` per matrix; checkpoint files are read-only inputs and are never copied to the report.
