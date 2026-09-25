# Muon conditioned residual vector INT3 study

CPU-only offline evaluation. Seed 0 supplied all learned codebooks and the small Muon-aware angle-offset calibration; seed 1 is the held-out evaluation split. Codebook training uses an evenly spaced deterministic sample of 8 tensors at each seed-0 landmark and up to 1200 vectors per tensor, with exact 2048-scalar block p98 normalization. The primary 64-codeword pairing comparison covers every eligible tensor instance (300 total); 32/128-codeword budget endpoints and the more expensive polar/PCA/spectral/oracle/scale diagnostics use a deterministic held-out subset. No training or production code was changed.

## Held-out seed 1

| rank | method | n | mean K=5 cosine | median [p25,p75] | size-weighted | mean update rel-L2 | exact-polar cosine | scalar INT3 delta | storage / FP32 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | mse128_contiguous | 20 | 0.88311 | 0.89731 [0.79699,0.97158] | 0.89974 | 0.44662 | — | 0.09667 | 0.11736 |
| 8 | mse128_contiguous | 20 | 0.90655 | 0.91861 [0.83713,0.97517] | 0.91934 | 0.40071 | — | 0.09884 | 0.12481 |
| 4 | mse32_contiguous | 20 | 0.77632 | 0.78909 [0.63360,0.91913] | 0.80139 | 0.63855 | — | -0.01012 | 0.08608 |
| 8 | mse32_contiguous | 20 | 0.80518 | 0.81897 [0.68194,0.92825] | 0.82668 | 0.59691 | — | -0.00253 | 0.09353 |
| 4 | mse64_checkerboard | 150 | 0.82522 | 0.88440 [0.70047,0.93201] | 0.84221 | 0.56767 | — | 0.04724 | 0.10164 |
| 8 | mse64_checkerboard | 150 | 0.85120 | 0.90394 [0.74047,0.94167] | 0.86584 | 0.52434 | — | 0.05170 | 0.10902 |
| 4 | mse64_column | 150 | 0.81083 | 0.88218 [0.66734,0.93187] | 0.82224 | 0.58733 | — | 0.03286 | 0.10164 |
| 8 | mse64_column | 150 | 0.83625 | 0.90223 [0.71287,0.94073] | 0.84541 | 0.54636 | — | 0.03675 | 0.10902 |
| 4 | mse64_contiguous | 150 | 0.82578 | 0.88516 [0.69850,0.93264] | 0.84277 | 0.56667 | 0.81452 (n=150) | 0.04780 | 0.10164 |
| 8 | mse64_contiguous | 150 | 0.85161 | 0.90515 [0.73973,0.94183] | 0.86627 | 0.52346 | 0.83702 (n=150) | 0.05211 | 0.10902 |
| 4 | mse64_contiguous_absmax | 20 | 0.80439 | 0.81518 [0.67507,0.93486] | 0.82755 | 0.59430 | — | 0.01796 | 0.10172 |
| 8 | mse64_contiguous_absmax | 20 | 0.83252 | 0.84578 [0.72690,0.94116] | 0.85147 | 0.55195 | — | 0.02482 | 0.10916 |
| 4 | mse64_contiguous_perdim | 20 | 0.83371 | 0.84881 [0.71938,0.95214] | 0.85527 | 0.54056 | — | 0.04727 | 0.10221 |
| 8 | mse64_contiguous_perdim | 20 | 0.86224 | 0.87593 [0.76456,0.95856] | 0.87957 | 0.49318 | — | 0.05453 | 0.10965 |
| 4 | mse64_contiguous_rms2 | 20 | 0.81892 | 0.83461 [0.70155,0.94165] | 0.84105 | 0.56974 | — | 0.03249 | 0.10172 |
| 8 | mse64_contiguous_rms2 | 20 | 0.84683 | 0.86063 [0.74515,0.94772] | 0.86504 | 0.52549 | — | 0.03913 | 0.10916 |
| 4 | mse64_row | 150 | 0.82578 | 0.88516 [0.69850,0.93264] | 0.84277 | 0.56667 | — | 0.04780 | 0.10164 |
| 8 | mse64_row | 150 | 0.85161 | 0.90515 [0.73973,0.94183] | 0.86627 | 0.52346 | — | 0.05211 | 0.10902 |
| 4 | muon_aware_polar_oracle | 20 | 0.75719 | 0.76782 [0.61040,0.90454] | 0.78269 | 0.67023 | 0.75312 (n=20) | -0.02925 | 0.10171 |
| 8 | muon_aware_polar_oracle | 20 | 0.78664 | 0.79930 [0.65740,0.91510] | 0.80887 | 0.62869 | 0.78147 (n=20) | -0.02107 | 0.10916 |
| 4 | pca_rotated_mse64 | 20 | 0.83475 | 0.84970 [0.72125,0.95253] | 0.85620 | 0.53884 | 0.82570 (n=20) | 0.04832 | 0.10173 |
| 8 | pca_rotated_mse64 | 20 | 0.86184 | 0.87628 [0.76158,0.95831] | 0.87915 | 0.49407 | 0.84876 (n=20) | 0.05414 | 0.10917 |
| 4 | polar_16x4_uniform | 20 | 0.80134 | 0.81405 [0.66903,0.93352] | 0.82489 | 0.59892 | — | 0.01491 | 0.10171 |
| 8 | polar_16x4_uniform | 20 | 0.83035 | 0.84465 [0.71675,0.94159] | 0.84993 | 0.55444 | — | 0.02264 | 0.10916 |
| 4 | polar_32x2_log | 20 | 0.61921 | 0.61858 [0.49097,0.73975] | 0.63661 | 0.80134 | — | -0.16723 | 0.10171 |
| 8 | polar_32x2_log | 20 | 0.63391 | 0.63634 [0.51011,0.75086] | 0.65087 | 0.80643 | — | -0.17379 | 0.10915 |
| 4 | polar_8x8_sqrt | 20 | 0.75763 | 0.76772 [0.61060,0.90460] | 0.78291 | 0.66986 | 0.75354 (n=20) | -0.02881 | 0.10171 |
| 8 | polar_8x8_sqrt | 20 | 0.78683 | 0.79963 [0.65595,0.91547] | 0.80901 | 0.62855 | 0.78118 (n=20) | -0.02087 | 0.10916 |
| 4 | spectral_local_mse64 | 20 | 0.09639 | 0.09669 [0.08649,0.09897] | 0.09510 | 0.99550 | — | — | 1.53029 |
| 8 | spectral_local_mse64 | 20 | 0.13370 | 0.13306 [0.12653,0.14182] | 0.13139 | 0.99119 | — | — | 1.53773 |

## Main interpretation

The held-out 64-word MSE vector quantizer at the matched 3-bit/value residual budget reaches K=5 update cosine 0.8258 (k=4) and 0.8516 (k=8), on all 150 seed-1 tensor instances per rank. Against the matched scalar p98 INT3 baseline (0.7780/0.7995), gains are +0.0478/+0.0521; the vector method beats scalar on 100.0% of these instances. It approximately matches the prior structural INT4 means (0.8245/0.8514), without exceeding the residual's nominal 3-bit/value payload. This is evidence that 2D residual representation can break the scalar INT3 ceiling in this offline matched-state study, not evidence about training trajectory behavior.

Pairing is secondary but not entirely irrelevant: row and contiguous are identical for the present even-width matrices, checkerboard is close (means 0.8252/0.8512), while column pairing is lower. The fixed polar grids and the small polar-offset Muon-aware oracle do not match the MSE-trained codebook; that oracle is restricted to a tiny calibration set and one polar family, so it is not a general upper bound over vector codebooks. PCA/per-dimension scaling diagnostics use only 20 held-out instances and should not be compared as full-coverage winners. Spectral-local pairing is an explicitly non-deployable control and performs poorly here.

The strongest supported interpretation is that a shared 2D MSE codebook adds useful local residual representation capacity, and on this held-out trajectory closes most or all of the observed structural INT4 cosine gap at a similar total storage ratio. Follow-up should first repeat the result with an independently trained codebook/alternate calibration split and validate scale/packing/runtime behavior before any training prototype. Spatial-vs-spectral comparisons do not support a claim that spectral pairing adds value.


Full CPU analysis runtime: 1650.7s (27.5 min); current report aggregation: 4.0s. Evaluation phase in this invocation: 3.4s. Eligible matrix instances: 300. Diagnostic tensors/update: 4.
