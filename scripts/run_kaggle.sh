#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 RECIPE RUN_ID" >&2
  echo "Environment: DATA_ROOT OUTPUT_ROOT RESUME_CHECKPOINT MAX_WALL_SECONDS KAGGLE_GPU_INDEX KAGGLE_INSTALL_DEPS DRY_RUN_ONLY" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 64
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
recipe="$1"
run_id="$2"
[[ "$recipe" = /* ]] || recipe="$repo_root/$recipe"
data_root="${DATA_ROOT:-/kaggle/input}"
output_root="${OUTPUT_ROOT:-/kaggle/working/runs}"
gpu_index="${KAGGLE_GPU_INDEX:-0}"
python_bin="${PYTHON_BIN:-python}"

[[ -f "$recipe" ]] || { echo "Recipe not found: $recipe" >&2; exit 66; }
[[ -d "$data_root" ]] || { echo "DATA_ROOT directory not found: $data_root" >&2; exit 66; }
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "RUN_ID must contain only letters, digits, dot, underscore, and hyphen." >&2; exit 64; }
[[ "$gpu_index" =~ ^[0-9]+$ ]] || { echo "KAGGLE_GPU_INDEX must be a single non-negative GPU index." >&2; exit 64; }
if [[ -n "${MAX_WALL_SECONDS:-}" ]]; then
  [[ "$MAX_WALL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "MAX_WALL_SECONDS must be a non-negative number." >&2; exit 64; }
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "Resume checkpoint not found: $RESUME_CHECKPOINT" >&2; exit 66; }
fi

mkdir -p -- "$output_root"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$output_root/$run_id/wandb}"

if [[ "${KAGGLE_INSTALL_DEPS:-0}" == "1" ]]; then
  "$python_bin" -m pip install --requirement "$repo_root/requirements/kaggle.txt"
fi

preflight=("$python_bin" "$repo_root/src/main.py" --recipe "$recipe" --data-root "$data_root" --to-device cuda:0 --dry-run)
echo "Kaggle preflight (physical GPU $gpu_index exposed as cuda:0)..."
"${preflight[@]}"

if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

args=("$python_bin" "$repo_root/src/main.py" --recipe "$recipe" --data-root "$data_root" --run-dir "$output_root/$run_id" --to-device cuda:0)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  args+=(--resume "$RESUME_CHECKPOINT")
fi
if [[ -n "${MAX_WALL_SECONDS:-}" ]]; then
  args+=(--max-wall-seconds "$MAX_WALL_SECONDS")
fi

exec "${args[@]}"
