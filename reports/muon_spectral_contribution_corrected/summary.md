# Offline spectral-band contribution analysis

Coverage: 10 formal snapshots, 300 eligible 2D Muon tensors. Runtime: 640.61s CPU.

The production INT4 dynamic b2048 quantizer and production Muon transform are reused unchanged.
Active modes use sigma/sigma_max >= 1e-06; head, centered-middle, and tail are the established 10%-of-active-index bands. An explicit other band accounts for the remaining active modes.
Residual coordinates are E_hat=U.T(Q(M)-M)V. The nine named row/column blocks plus explicit other blocks are orthogonal; high-level named-band attribution uses disjoint row-associated blocks, while cross-band mixing and other-mode energy are reported separately.
Controlled sensitivity uses epsilon=0.001. Magnitude perturbations change only selected singular values; orientation perturbations rotate left singular vectors with a deterministic skew generator. The sensitivity is empirical update relative-L2 / epsilon, not a derivative.

## Aggregate band summaries

- head: {'mean_residual_fraction': 0.09747524090111255, 'median_residual_fraction': 0.09286477416753769, 'mean_sensitivity': 1.1282368487445638, 'median_sensitivity': 1.10359525680542, 'mean_restoration_gain': 0.0011590735117594402, 'median_restoration_gain': 0.000970304012298584}
- middle: {'mean_residual_fraction': 0.061321469818552334, 'median_residual_fraction': 0.0443892739713192, 'mean_sensitivity': 4.115062816611801, 'median_sensitivity': 3.4396872520446777, 'mean_restoration_gain': 0.019946931799252828, 'median_restoration_gain': 0.004630923271179199}
- tail: {'mean_residual_fraction': 0.049237439079831045, 'median_residual_fraction': 0.03652819246053696, 'mean_sensitivity': 14.69636294602727, 'median_sensitivity': 8.567646026611328, 'mean_restoration_gain': 0.01001040796438853, 'median_restoration_gain': 0.006150245666503906}

## Interpretation

Sensitivity and actual contribution are intentionally separate. A small high-sensitivity band can have low absolute contribution if its actual INT4 residual is small; restoration ablations are the direct descriptive contribution test.
No causal or additive claim is made: the Muon transform can couple spectral components, and reduced-SVD unresolved residual is kept explicit.

## Per-seed summaries

### seed 0

- head: residual fraction mean/median 0.096105/0.087461; sensitivity mean/median 1.131983/1.106089; restoration gain mean/median 0.001146/0.000946
- middle: residual fraction mean/median 0.061524/0.044123; sensitivity mean/median 4.100469/3.438433; restoration gain mean/median 0.019867/0.004492
- tail: residual fraction mean/median 0.049180/0.036462; sensitivity mean/median 14.722298/8.587788; restoration gain mean/median 0.010016/0.005911
### seed 1

- head: residual fraction mean/median 0.098846/0.101837; sensitivity mean/median 1.124491/1.101884; restoration gain mean/median 0.001172/0.001044
- middle: residual fraction mean/median 0.061119/0.044999; sensitivity mean/median 4.129657/3.458915; restoration gain mean/median 0.020027/0.004857
- tail: residual fraction mean/median 0.049295/0.036650; sensitivity mean/median 14.670428/8.305607; restoration gain mean/median 0.010005/0.006182
