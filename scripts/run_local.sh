#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 RECIPE RUN_ID" >&2
  echo "Environment: DATA_ROOT OUTPUT_ROOT DEVICE RESUME_CHECKPOINT MAX_WALL_SECONDS DRY_RUN_ONLY" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 64
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
recipe="$1"
run_id="$2"
[[ "$recipe" = /* ]] || recipe="$repo_root/$recipe"

if [[ ! -f "$recipe" ]]; then
  echo "Recipe not found: $recipe" >&2
  exit 66
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_ID must contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 64
fi

data_root="${DATA_ROOT:-$repo_root}"
output_root="${OUTPUT_ROOT:-$repo_root/runs}"
device="${DEVICE:-auto}"
python_bin="${PYTHON_BIN:-python}"
mkdir -p -- "$output_root"

args=("$python_bin" "$repo_root/src/main.py" --recipe "$recipe" --data-root "$data_root" --run-dir "$output_root/$run_id" --to-device "$device")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "Resume checkpoint not found: $RESUME_CHECKPOINT" >&2; exit 66; }
  args+=(--resume "$RESUME_CHECKPOINT")
fi
if [[ -n "${MAX_WALL_SECONDS:-}" ]]; then
  [[ "$MAX_WALL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "MAX_WALL_SECONDS must be a non-negative number." >&2; exit 64; }
  args+=(--max-wall-seconds "$MAX_WALL_SECONDS")
fi
if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

exec "${args[@]}"
