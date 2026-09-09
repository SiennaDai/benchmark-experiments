# Low-precision optimizer pretraining experiments

This repository is an incremental, single-process experiment layer over EPFL's `llm-optimizer-benchmark`. It keeps the upstream Llama implementation and initialization while adding strict recipes, frozen token streams, deterministic training/evaluation/resume, a readable AdamW reference, BF16 persisted-state simulation, and lazy bitsandbytes adapters.

## Environment

Python 3.11/3.12 is supported. The verified CPU environment used Python 3.12.3 and PyTorch 2.8.0+cpu. Recreate it without CUDA packages:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy==2.3.2 tiktoken==0.11.0 matplotlib==3.10.5 pytest==8.4.1
```

GPU runs need a CUDA-compatible PyTorch build and `bitsandbytes==0.48.1`; these were not installed or verified on the CPU-only validation host.

## Data and training

Commands run from the repository root. Synthetic data is offline and deterministic:

```bash
.venv/bin/python scripts/prepare_data.py --kind synthetic --output data/diagnostic
.venv/bin/python scripts/prepare_data.py --kind synthetic --output data/overfit --overfit --train-tokens 65537 --validation-tokens 4097 --test-tokens 4097
.venv/bin/python scripts/prepare_data.py --kind slimpajama --output data/slimpajama_small --plan-only
.venv/bin/python scripts/prepare_data.py --kind slimpajama --output data/slimpajama_small --source-max-bytes 536870912
```

The last command uses the frozen revision and selects only complete files within the 512 MiB source budget. If the service or local cache is unavailable it writes the plan and exits with the reason; it never silently exceeds the budget.

Relative manifest paths in recipes are resolved below the runtime-only `--data-root`
(the repository root by default). The selected mount path is not part of the
scientific fingerprint, while the manifest's content fingerprint is still checked
when a run starts. This lets the same recipe use a different mount without changing
the experiment definition:

```bash
python src/main.py --recipe recipes/mini_bf16_adamw.json --data-root /kaggle/input --dry-run --to-device cuda:0
```

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --dry-run --to-device cpu
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --run-dir runs/diagnostic_01 --to-device cpu
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --resume runs/diagnostic_01/checkpoints/latest.pt --to-device cpu
.venv/bin/python scripts/evaluate.py --checkpoint runs/diagnostic_01/checkpoints/final.pt --split validation --to-device cpu
```

Each run contains resolved configuration, environment/source/data/parameter/precision metadata, append-only JSONL events, summary, and atomic checkpoints. A wall-clock boundary is checked between complete updates:

```bash
.venv/bin/python src/main.py --recipe recipes/diagnostic_cpu.json --run-dir runs/bounded --max-wall-seconds 30
```

## Diagnostics, comparison, and capacity planning

```bash
.venv/bin/python scripts/optimizer_diagnostics.py --recipe recipes/diagnostic_state_sim.json --output reports/optimizer_diagnostics
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/compare_runs.py --runs runs/runA runs/runB --vary optimizer.name optimizer.state_simulation --output reports/comparison
.venv/bin/python scripts/profile_run.py --recipe recipes/reference124m_pilot.json --max-wall-seconds 900 --output reports/profile124m --to-device cuda:0
.venv/bin/python -m pytest tests/
```

`compare_runs.py` refuses ranking when non-varied scientific fields differ and retains failed/paused statuses. `profile_run.py` records a validated dry-run on unsupported hardware; a CUDA host is required for a real capacity measurement. GPU recipes are supplied but are intentionally not auto-run.

`--to-device` accepts `auto`, `cpu`, `cuda`, or `cuda:N`; `--to_device` is an alias. Unsupported CUDA/BF16 combinations fail explicitly and never fall back to another precision.

## Local and Kaggle launchers

Both launchers call the same `src/main.py` entry point. Locally:

```bash
DATA_ROOT="$PWD" OUTPUT_ROOT="$PWD/runs" DEVICE=cpu \
  bash scripts/run_local.sh recipes/diagnostic_cpu.json diagnostic_01
```

On Kaggle, mount token data so the recipe's relative manifest exists below
`DATA_ROOT`, enable a GPU, clone a pinned commit, then run:

```bash
DATA_ROOT=/kaggle/input OUTPUT_ROOT=/kaggle/working/runs \
MAX_WALL_SECONDS=39600 \
  bash scripts/run_kaggle.sh recipes/mini_bf16_adamw.json adamw_trial_01
```

`run_kaggle.sh` exposes exactly one physical GPU (selected by
`KAGGLE_GPU_INDEX`, default `0`) as `cuda:0`, performs a strict CUDA/BF16/data
dry-run, then starts training. It never silently selects DDP or changes precision.
Set `DRY_RUN_ONLY=1` for preflight only. Resume from a mounted checkpoint with:

```bash
RESUME_CHECKPOINT=/kaggle/input/my-resume/run/checkpoints/latest.pt \
  bash scripts/run_kaggle.sh recipes/mini_bf16_adamw.json adamw_trial_01
```

Kaggle provides the CUDA-matched PyTorch build. The optional pinned add-ons in
`requirements/kaggle.txt` deliberately omit Torch; set `KAGGLE_INSTALL_DEPS=1`
only when the image lacks one of them. `WANDB_MODE` defaults to `offline`, though
the current training platform records JSONL locally and does not yet emit W&B
runs. Preserve `/kaggle/working/runs` with a saved Notebook version or publish
large checkpoints as a private Kaggle Dataset before the runtime ends.

The minimal notebook at `notebooks/kaggle_launcher.ipynb` clones
`https://github.com/SiennaDai/benchmark-experiments.git`; formal experiments
should replace its `GIT_REF` with an immutable commit SHA or tag.

## Recipe contract

Recipes are strict, complete JSON objects. They do not support inheritance,
overrides, environment-variable interpolation, or Hydra syntax. The loader
rejects duplicate or unknown keys, invalid types, unsupported combinations, and
inconsistent model/token dimensions. Every recipe must contain these groups:

| Group | Contents |
|---|---|
| `experiment` | `name`, `protocol_id`, and the model/data/algorithm seeds |
| `model` | Llama dimensions, vocabulary/sequence length, initialization and normalization settings |
| `data` | relative `manifest`, train/validation split names, and repeated-epoch policy |
| `train` | target tokens, micro-batch size, accumulation steps, and gradient clipping |
| `optimizer` | optimizer name and numerical settings (`lr`, `betas`, `eps`, weight decay, backend flags, state simulation) |
| `schedule` | cosine/constant schedule, warmup updates, and final learning-rate ratio |
| `precision` | compute/parameter/gradient dtypes, attention backend, determinism, TF32 and compile flags |
| `eval` | evaluation interval, token budget, batch size, compute and attention backend |
| `logging` | logging interval, diagnostics, and the currently inactive `wandb` flag |
| `checkpoint` | checkpoint interval and whether initial/final checkpoints are saved |

The scientific fingerprint covers the complete recipe. Runtime-only controls
such as `--data-root`, `--run-dir`, `--resume`, `--to-device`, and
`--max-wall-seconds` are not recipe fields and do not change that fingerprint.
The manifest content fingerprint is still checked at startup. Copy an existing
recipe, change its `experiment.name` and only the intended scientific fields,
and give the file a descriptive name. Validate it before training:

```bash
.venv/bin/python src/main.py --recipe recipes/my_recipe.json \
  --data-root "$PWD" --dry-run --to-device cpu
.venv/bin/python -m json.tool recipes/my_recipe.json >/dev/null
```

The smallest practical starting point is `recipes/diagnostic_cpu.json`; use
`recipes/mini_bf16_adamw.json` for the single-GPU BF16 path. Keep the original
recipe with the run so that resume and run comparison use exactly the same
scientific definition.

See the [Chinese operation manual](docs/OPERATION_MANUAL.zh-CN.md), [general Kaggle experiment guide](docs/KAGGLE_EXPERIMENT_GUIDE.zh-CN.md), [protocol](docs/PROTOCOL.md), [validation evidence](docs/VALIDATION.md), [upstream provenance](docs/UPSTREAM.md), and [optimizer extension notes](docs/ADDING_AN_OPTIMIZER.md).
