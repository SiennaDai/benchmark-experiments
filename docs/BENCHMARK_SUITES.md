# Benchmark suites

A benchmark suite is a small JSON inventory of independent, complete recipes. It
orchestrates the fixed lifecycle below on one machine, one device, and in declared
order:

```text
define suite -> validate comparability -> preflight -> run/resume -> summarize -> compare -> report
```

It is intentionally not a sweep language, scheduler, inheritance system, or a
mechanism for changing recipes. A recipe remains the only scientific configuration
source of truth.

## Schema

```json
{
  "name": "example_benchmark_v1",
  "runs": [
    {"run_id": "baseline_s0", "recipe": "../recipes/baseline.json"},
    {"run_id": "variant_s0", "recipe": "../recipes/variant.json"}
  ],
  "vary": ["optimizer.name"],
  "runtime": {"device": "cuda:0", "max_wall_seconds": 3300},
  "report": {"output": "reports/example_benchmark_v1"},
  "replication": {
    "group_by": ["experiment.seed", "experiment.data_seed"],
    "treatment_field": "optimizer.state_simulation",
    "control_value": "none",
    "treatment_values": ["bf16_roundtrip"]
  }
}
```

The top-level and nested keys are strict. `run_id` is unique and the listed order
is execution and report order. List each seed as another complete recipe/run entry;
there are no variables, interpolation, loops, inherited fields, device fallbacks,
or automatic allowed differences. `max_wall_seconds` may be `null`.

Recipe references are resolved relative to the suite file. The report path is
resolved relative to the repository root. Run artifacts are written at
`<output-root>/<run_id>`.

## Field semantics and comparability

The comparison implementation in `reporting.py` is shared by `compare_runs.py`
and the suite runner.

- **Scientific fields** are all declared recipe fields except identity fields.
  They include model, data protocol, training, optimizer, schedule, precision,
  evaluation, logging/checkpoint protocol, `experiment.protocol_id`, and all
  seeds. A field may differ only when its exact dot path is in `vary`.
- **Identity/label fields**: `experiment.name` identifies a recipe/run for people;
  it is ignored in scientific comparison and cannot be put in `vary`.
- **Runtime-only fields** are suite `runtime` controls and CLI mount/output paths.
  They are deliberately absent from recipes and recipe fingerprints. They must not
  be used to make experiments comparable.

Generated `fingerprint` and `derived.recipe_path` values are also ignored by the
comparison guard. A recipe fingerprint still identifies the complete recipe for
artifact/checkpoint safety; it is not a treatment label.

`experiment.protocol_id` names the stable experimental protocol (for example, the
model/data/train/eval procedure), not an optimizer/treatment. If old recipes use
it for treatment names, give every otherwise matched recipe one common protocol
ID and move treatment naming to `experiment.name`; retain each recipe as a complete
auditable JSON document. Do not bulk-rewrite a historical artifact in place.

Before training, the suite loads every recipe, validates `vary`, rejects every
non-varied difference, runs the platform's existing dry-run (device/resource and
data-capacity validation), and checks the frozen data-manifest fingerprints agree.
No training starts unless the whole preflight passes.

## Run, resume, and report

Once per benchmark, create complete recipes, create the suite JSON, and freeze its
data. On each execution environment, checkout the intended immutable Git commit,
mount that data, and verify the environment. Then the normal command is:

```bash
python scripts/run_benchmark.py \
  --suite benchmarks/example_benchmark_v1.json \
  --data-root /mounted/data \
  --output-root runs
```

Useful non-training modes are:

```bash
python scripts/run_benchmark.py --suite benchmarks/example_benchmark_v1.json --data-root /mounted/data --output-root runs --preflight-only
python scripts/run_benchmark.py --suite benchmarks/example_benchmark_v1.json --data-root /mounted/data --output-root runs --report-only
```

Completed artifacts are skipped. A `paused_budget` (or interrupted) artifact is
resumed only when `checkpoints/latest.pt`, recipe fingerprint, data-manifest
fingerprint, and source commit all match. A failed, partially written, missing-
checkpoint, source-mismatched, or fingerprint-mismatched artifact stops the suite;
the runner never deletes artifacts, retries failures, invents a run ID, edits a
recipe, or downgrades device/precision.

`--report-only` is read-only with respect to run artifacts. It applies the same
recipe comparison guard and delegates aggregation/figures to `compare_runs.py` and
`reporting.py`; it does not require a mounted device or rerun data preflight.

## Suite artifacts

The report directory contains `suite_resolved.json`, `preflight.json`,
`run_status.json`, `benchmark_summary.md`, `comparison.csv`, `comparison.md`,
`differences.json`, and the generic comparison figures. The resolved record keeps
the suite definition, checkout commit/dirty state, data and recipe fingerprints,
runtime settings, status records, and timestamps. Runtime metadata is never added
to a scientific recipe fingerprint.

`replication` is optional. When present, it declares field-based pairing rather than
inferring semantics from run names. Reporting writes descriptive per-treatment `n`,
mean, sample standard deviation, and paired treatment-minus-control deltas to
`replication.json` and `benchmark_summary.md`; it does not compute p-values,
confidence intervals, or significance claims. For CPU-only preflight of a GPU suite,
use `--preflight-device cpu` with `--preflight-only`; the suite runtime remains the
authoritative training device.
