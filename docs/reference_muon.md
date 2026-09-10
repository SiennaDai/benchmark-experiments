# Reference Muon benchmark implementation

`reference_muon` follows [Keller Jordan's reference Muon implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py) and the [PyTorch Muon API](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html), using their defaults: momentum `0.95`, Nesterov enabled, five Newton–Schulz steps, coefficients `(3.4445, -4.7750, 2.0315)`, and normalization safeguard `eps=1e-7`.  The benchmark uses FP32 throughout, including Newton–Schulz arithmetic; it intentionally does not use a fused or BF16 orthogonalization kernel.

For a hidden matrix gradient \(g_t\), \(B_t=0.95B_{t-1}+g_t\), the Nesterov direction is \(g_t+0.95B_t\), and the update is its five-step approximate polar factor.  The iteration normalizes by Frobenius norm (plus epsilon), transposes tall matrices before multiplying, then applies \(X \leftarrow aX+(bXX^T+c(XX^T)^2)X\).  Decoupled weight decay is applied as \(W\leftarrow(1-\eta\lambda)W\) before \(W\leftarrow W-\eta O\).

Only unique, 2D parameters named under `transformer.h.*` are Muon eligible.  Embedding/LM head (including their tied shared tensor), all RMSNorm gains, and every other non-eligible parameter are deduplicated and use the auxiliary FP32 ReferenceAdamW rule.  This makes grouping inspectable in `parameters.json` and `precision.json`.

`bf16_roundtrip` changes only persisted Muon `muon_momentum`, after the current FP32 update: `momentum.to(torch.bfloat16).float()`.  Auxiliary AdamW `exp_avg` and `exp_avg_sq` remain FP32 in both conditions.  Thus this is a persistence-numerics comparison, not a compressed-memory or hardware-BF16 benchmark.
