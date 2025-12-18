#!/usr/bin/env bash
set -euo pipefail

# Run collect_tot.py for hendrycks_math levels 1/2/4/5.
# This is a thin wrapper around YOUR command format.

DATASET_PATH="/data/jsg_data/hendrycks_math"
SPLIT="train"
LEVELS=(1 2 4 5)
MAX_SAMPLES=500
GPUS="1,2"
OUT_BASE="datas"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_collect_tot_math_levels_simple.sh [options]

Options:
  --dataset-path PATH   (default: ${DATASET_PATH})
  --split NAME          train|test (default: ${SPLIT})
  --max-samples N       per level (default: ${MAX_SAMPLES})
  --gpus CSV            e.g. 1,2 (default: ${GPUS})
  --out-base DIR        base output dir (default: ${OUT_BASE})

Example (matches your format):
  python collect_tot.py --dataset-name hendrycks_math --dataset-path ${DATASET_PATH} --split ${SPLIT} --level 2 --max-samples 300 --gpus 1,2 --output-dir datas/tot_math_l2 --output-prefix math_l2 --merge
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --out-base) OUT_BASE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

for lvl in "${LEVELS[@]}"; do
  python collect_tot.py \
    --dataset-name hendrycks_math \
    --dataset-path "${DATASET_PATH}" \
    --split "${SPLIT}" \
    --level "${lvl}" \
    --max-samples "${MAX_SAMPLES}" \
    --gpus "${GPUS}" \
    --output-dir "${OUT_BASE}/tot_math_l${lvl}" \
    --output-prefix "math_l${lvl}" \
    --merge
done

