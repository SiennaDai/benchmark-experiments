# Reporting

`metrics.jsonl` is an audit log, not the primary day-to-day reading interface.

Read benchmark output in this order:

1. `benchmark_summary.md`
2. `comparison.csv`
3. `loss_vs_tokens.png`, `loss_vs_time.png` (when available), and `memory_comparison.png`
4. each run's `summary.json`
5. `metrics.jsonl` only for diagnosis

Create a compact run summary:

```bash
.venv/bin/python scripts/summarize_run.py runs/mini_adamw_s0
.venv/bin/python scripts/summarize_run.py runs/mini_adamw_s0 --output reports/mini_adamw_s0.md
```

Compare scientifically matched runs:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/compare_runs.py \
  --runs runs/mini_adamw_s0 runs/mini_bnb32_s0 \
  --vary optimizer.name --output reports/mini_optimizer_v1
```

The script refuses to compare any non-varied scientific configuration difference and writes
`differences.json`. `tokens_per_second` is processed training target tokens divided by the
sum of recorded update durations. Optimizer-state bytes are deduplicated PyTorch tensor
storage bytes reported by the optimizer state only; they exclude parameters and gradients.
CUDA peak values are PyTorch allocator peaks, not total board memory from `nvidia-smi`.
