# Continuous spectral risk analysis

Coverage: 10 formal snapshots, 300 eligible 2D Muon tensors, 115193 active modes. Baseline processing was run one landmark at a time to stay within the CPU execution watchdog; the declared intervention sample is one deterministic tensor per represented bin from the first landmark. No GPU training was launched.

Production INT4 dynamic b2048 and the production Muon transform are reused unchanged. The continuous coordinate is `log10(sigma_i / sigma_max)` for active modes (`sigma_i/sigma_max >= 1e-6`). Fixed bins are [-6,-5.5), [-5.5,-5), [-5,-4.5), [-4.5,-4), [-4,-3.5), [-3.5,-3), [-3,-2.5), [-2.5,-2), [-2,-1.5), [-1.5,-1), [-1,-0.5), [-0.5,0); all were retained because sparse-bin support is reported explicitly.

The direct intervention cap is deterministic: the first 1 tensor in snapshot/parameter discovery order per bin is used for bin restoration and sensitivity. Baseline residual allocation covers every eligible tensor. The final directory is a clean, deduplicated reconstruction; the earlier working directory remains separate and is not used for conclusions.

## Danger zone

The zone is selected from associated-bin restoration gains by the narrowest contiguous interval reaching 50% of total positive gain. It is `-3` to `-2` in log10 normalized singular value, with 0.140679 of 0.265478 positive gain in the selected interval. This is a deterministic descriptive rule, not a fitted threshold.

Sensitivity alone identifies where perturbations are dangerous; residual energy alone identifies where INT4 error is large; direct restoration measures where realized error matters. The report keeps these quantities separate and does not claim causal mediation. Cross-bin associated restorations overlap on cross-bin entries, so their gains are not additive.
