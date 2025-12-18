#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dpsk"
DATASET_ROOT="/data/jsg_data/hendrycks_math"
SPLIT="train"
LEVELS="1,2,4,5"
K=500
SEED=42
GPUS="0"
OUT_BASE="datas/tot_math_levels"
RUN_ID=""

usage() {
  cat <<'EOF'
Run collect_tot.py on MATH (hendrycks_math) for multiple difficulty levels.

Usage:
  bash scripts/collect_tot_math_levels.sh [options] -- [extra collect_tot.py args]

Options:
  --env NAME            Conda env name (default: dpsk)
  --dataset-root PATH   hendrycks_math root dir (default: /data/jsg_data/hendrycks_math)
  --split NAME          Dataset split: train|test (default: train)
  --levels CSV          Levels to run (default: 1,2,4,5)
  --k N                 Max samples per level (default: 500)
  --seed N              Sampling seed (default: 42)
  --gpus CSV            GPU ids (default: 0)
  --out-base PATH       Base output dir (default: datas/tot_math_levels)
  --run-id ID           Shared run id across levels (default: timestamp)
  -h, --help            Show this help

Example:
  bash scripts/collect_tot_math_levels.sh --gpus 1,2 --k 500 -- \\
    --no-vllm --model-dir /data/jsg_data/Qwen/Qwen2.5-Math-7B-Instruct --branches 5 --rollouts-per-candidate 5
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --out-base) OUT_BASE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please install/miniconda or ensure conda is available." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

IFS=',' read -r -a LEVEL_ARR <<<"${LEVELS}"
for lvl in "${LEVEL_ARR[@]}"; do
  lvl="$(echo "${lvl}" | tr -d '[:space:]')"
  [[ -z "${lvl}" ]] && continue

  out_dir="${OUT_BASE}/level${lvl}"
  prefix="math_l${lvl}"

  echo "[collect_tot_math_levels] level=${lvl} k=${K} gpus=${GPUS} out=${out_dir} run_id=${RUN_ID}"
  python collect_tot.py \
    --dataset-name hendrycks_math \
    --dataset-path "${DATASET_ROOT}" \
    --split "${SPLIT}" \
    --level "${lvl}" \
    --max-samples "${K}" \
    --seed "${SEED}" \
    --gpus "${GPUS}" \
    --output-dir "${out_dir}" \
    --output-prefix "${prefix}" \
    --run-id "${RUN_ID}" \
    --merge --merge-sort \
    "${EXTRA_ARGS[@]}"
done

