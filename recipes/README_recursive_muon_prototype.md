# Recursive Muon staged 4096-update Kaggle prototype

Do not rerun the completed FP32 Muon reference. Run only the two compressed
trajectories, stopping explicitly at each gate:

```bash
# GPU 0: structural INT4
CUDA_VISIBLE_DEVICES=0 python src/main.py --recipe recipes/recursive_muon_structural_int4_4096_s0.json \
  --run-dir runs/recursive_int4_4096_s0 --stop-at-update 128 --to-device cuda
CUDA_VISIBLE_DEVICES=0 python src/main.py --recipe recipes/recursive_muon_structural_int4_4096_s0.json \
  --run-dir runs/recursive_int4_4096_s0 --resume runs/recursive_int4_4096_s0/checkpoints/latest.pt --stop-at-update 512 --to-device cuda
```

Repeat the resume command with `--stop-at-update 1024`, `2048`, and `4096`.
Run the VQ trajectory on GPU 1 by replacing `int4` with `vq_int3` and using
`runs/recursive_vq_int3_4096_s0`.

After each gate, compare it with the existing FP32 run artifact:

```bash
python scripts/check_recursive_gate.py \
  --fp32-recipe recipes/mini_fp32_reference_muon_4x_s0.json \
  --candidate-recipe recipes/recursive_muon_structural_int4_4096_s0.json \
  --candidate-recipe recipes/recursive_muon_structural_vq_int3_4096_s0.json \
  --fp32-run runs/mini_fp32_reference_muon_4x_s0 \
  --candidate-run runs/recursive_int4_4096_s0 \
  --landmark 128 --output reports/recursive_gate_compatibility
```

Use the same command with the VQ candidate run and each later landmark.

Validate protocol compatibility before starting:

```bash
python scripts/check_recursive_gate.py \
  --fp32-recipe recipes/mini_fp32_reference_muon_4x_s0.json \
  --candidate-recipe recipes/recursive_muon_structural_int4_4096_s0.json \
  --candidate-recipe recipes/recursive_muon_structural_vq_int3_4096_s0.json \
  --output reports/recursive_gate_compatibility
```

Both compressed recipes are one 4096-update scientific protocol: 8192
tokens/update, cosine scheduler horizon 4096, warmup 20, seed 0, data seed
1337, algorithm seed 2026, and frozen SlimPajama data. `--stop-at-update` is a
runtime gate only; it does not alter the scheduler or recipe fingerprint.
Every stage exits intentionally paused and requires an explicit resume. The
structural extractor is explicitly `exact_svd_oracle`, so its runtime is a
prototype cost rather than a deployment claim. The VQ codebook is frozen and
calibrated on the opposite formal trajectory (`s1_k8_w64_t8_v1200`).
