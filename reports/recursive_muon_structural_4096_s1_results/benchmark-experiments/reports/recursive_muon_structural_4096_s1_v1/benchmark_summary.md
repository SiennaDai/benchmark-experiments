# Benchmark summary

## Benchmark

- Path: `/kaggle/working/benchmark-experiments/reports/recursive_muon_structural_4096_s1_v1`
- Protocol: `slimpajama-hash-split-v1`
- Varied fields: optimizer.recursive_representation, optimizer.recursive_codebook_path, optimizer.recursive_codebook_key
- Runs: 2

## Scientific consistency

All non-varied scientific conditions matched.

## Results

| Run | Optimizer | Status | Final val NLL | Best val NLL | Tokens/s | Peak GPU MiB | Opt state MiB |
|---|---|---|---|---|---|---|---|
| recursive_int4_4096_s1 | recursive_muon | completed | 4.794689869508147 | 4.794689869508147 | 5963.873045823321 | 1204.3759765625 | 153.09347534179688 |
| recursive_vq_int3_4096_s1 | recursive_muon | completed | 4.786886706482619 | 4.786886706482619 | 6156.0090435514885 | 1203.1103515625 | 151.82833862304688 |

## Paired trajectory

Observed validation-NLL deltas are treatment − control; null denotes a missing evaluation event and no interpolation or interpretation is performed.

Group `[1]`, treatment `vq_int3`:

| Update | Control NLL | Treatment NLL | Paired Δ |
|---|---|---|---|
| 128 | 6.441212950274348 | 6.397437922656536 | -0.04377502761781216 |
| 512 | 5.591307728551328 | 5.582526474259794 | -0.008781254291534424 |
| 1024 | 5.3705981047824025 | 5.362903876230121 | -0.0076942285522818565 |
| 2048 | 5.006218456663191 | 5.007708823308349 | 0.0014903666451573372 |
| 4096 | 4.794689869508147 | 4.786886706482619 | -0.0078031630255281925 |

## Key deltas

- recursive_vq_int3_4096_s1 vs recursive_int4_4096_s1: final val NLL Δ -0.00780316
- recursive_vq_int3_4096_s1 vs recursive_int4_4096_s1: best val NLL Δ -0.00780316
- recursive_vq_int3_4096_s1 vs recursive_int4_4096_s1: optimizer state bytes Δ -1.32659e+06
- recursive_vq_int3_4096_s1 vs recursive_int4_4096_s1: tokens/s Δ 192.136

## Artifacts

- `comparison.csv`, `comparison.md`, `optimizer_lowp_summary.json`, `optimizer_lowp_summary.csv`, `optimizer_absolute_nll.png`, `optimizer_lowp_sensitivity.png`, `loss_vs_tokens.png`, `memory_comparison.png`

## Caveats

- Single seed; values are descriptive, not a multi-seed estimate.
