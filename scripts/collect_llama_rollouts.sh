#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="../model/Meta-Llama-3-8B"
GPUS="0,1,2,3,4,5,6,7"
OUT_ROOT="datas/llama"

mkdir -p "${OUT_ROOT}"

python collect_tot.py \
  --dataset-name gsm8k \
  --split train \
  --gpus "${GPUS}" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUT_ROOT}/gsm8k" \
  --output-prefix gsm8k_tot \
  --merge

python collect_tot.py \
  --dataset-name svamp \
  --gpus "${GPUS}" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUT_ROOT}/svamp" \
  --output-prefix svamp_tot \
  --merge
