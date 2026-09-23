# Methodology

At update 4096, load the same-seed FP32 momentum snapshot, the decoded recursive VQ and INT4 checkpoints, and the frozen seed-0-calibrated VQ codebook. Metrics use the repository's canonical five-step Newton–Schulz map. Recursive metrics are pooled across the 30 Muon matrices. A separate one-shot control encodes each FP32 snapshot with the canonical structural VQ codec once. Candidate conditioning is intentionally gated on agreement between these objects; it is not executed when the gate fails.

The error in the recursive comparison is reference-aligned trajectory error `M_ref-M_vq`, not instantaneous quantization error. No pre-quantization candidate or contemporaneous gradient was saved, so online conditioning, recursive drift, Nesterov fidelity, and validation-NLL effects are not identifiable here.
