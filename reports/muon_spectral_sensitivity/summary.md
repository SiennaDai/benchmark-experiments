# Summary

Analyzed 10 formal snapshots (seeds 0/1, updates [128, 512, 1024, 2048, 4096])
and 300 eligible 2D tensors. Non-2D snapshot entries were
excluded from SVD and post-Muon analysis because production Muon only
orthogonalizes 2D matrices. Runtime was 193.59 seconds on CPU.

See `tensor_spectral_metrics.csv`, `quantized_spectral_metrics.csv`,
`spectral_error_decomposition.csv`, `subspace_distortion.csv`,
`spectral_update_correlations.csv`, and `conditioning_intervention.csv`.
Correlations are descriptive only. The controlled tau intervention is the
primary mechanism check; it keeps singular-vector orientation fixed and uses
the unchanged production quantizers and Muon transform.
