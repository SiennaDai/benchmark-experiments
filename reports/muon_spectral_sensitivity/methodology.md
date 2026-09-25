# Muon spectral-sensitivity analysis

This is a read-only offline mechanism study over 10 existing FP32
Muon snapshots and 300 eligible 2D Muon tensors. No training,
optimizer mutation, or existing report modification was performed. Runtime on
this CPU run was 193.59 seconds.

## Definitions

For each FP32 matrix M, the reduced SVD is M=U diag(sigma) V^T in FP32.
`effective_rank` counts sigma_i/sigma_max >= 1e-06; `effective_condition_number`
is sigma_max divided by the smallest value meeting that threshold. The naive
condition number is sigma_max/sigma_min and is left empty when sigma_min is
zero. Effective-rank entropy is exp(-sum p log p), p=sigma/sum(sigma), and
stable rank is sum(sigma^2)/sigma_max^2. Percentiles are ordinary singular
value quantiles. Tail energy fractions use the last ceil(10%) and ceil(25%)
index modes.

The exact existing blockwise production quantizers are applied with block size
2048, absmax scale, existing codebooks, clipping, and nearest rounding:
`int8-linear-b2048`, `int4-linear-b2048`, and `int4-dynamic-b2048`. The exact
production `zeropower_newton_schulz` transform is used for post-Muon metrics,
with its steps/coefficients/epsilon read from each snapshot provenance.

For E=Q(M)-M, `E_hat=U^T E V`; diagonal terms are relative to sigma_i when
sigma_i/sigma_max >= 1e-06. Top, centered-middle, and tail bands are each
10% of the singular-index range (at least one mode). Projection/subspace
metrics use principal angles and normalized Frobenius distance between the
left and right singular subspace projectors; this avoids one-to-one vector
comparisons for clustered singular values. Any reduced-basis residual is
reported as `unresolved_error_energy`.

## Controlled intervention

Representatives were selected *before* intervention: lowest, median, and
highest effective condition number plus lowest and highest nearest INT4
dynamic update-direction error, deduplicated. The tau grid is
`tau/sigma_max = 0, 0.0001, 0.0003, 0.001, 0.003, 0.01`. Each M_tau keeps U,V
fixed and replaces sigma_i by max(sigma_i,tau). INT8 linear is a healthy
low-distortion control. This is an oracle/mechanism intervention, not a
deployable quantizer and not tuned against training loss.

## Correlations and interpretation

Pearson/Spearman values in `spectral_update_correlations.csv` are descriptive
associations over tensor/landmark pairs; they do not establish causality or
statistical significance. The strongest dynamic-INT4 Spearman associations
were: [('tail_subspace_distortion', 0.960812897921088, 300), ('tail_relative_spectral_perturbation', 0.9242124912499028, 300), ('log_effective_condition_number', 0.9148474983055367, 300)].

Intervention dynamic-INT4 means (tau, update cosine, update relative L2,
effective condition) were: [(0.0, 0.7474130511283874, 0.6750473320484162, 193670.42857818602), (0.0001, 0.7474161505699157, 0.6750148475170136, 4033.846156311035), (0.0003, 0.7472692549228668, 0.6749757587909698, 1367.022035217285), (0.001, 0.7469253838062286, 0.6713939607143402, 433.6482192993164), (0.003, 0.7486541330814361, 0.6386710584163666, 166.9756607055664), (0.01, 0.8512223720550537, 0.5477613270282745, 71.55914916992188)]. Use these controls to
decide whether conditioning changes fidelity rather than inferring from
correlation alone. Non-finite or undefined metrics are empty fields.
