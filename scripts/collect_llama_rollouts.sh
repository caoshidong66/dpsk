#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="../model/Meta-Llama-3-8B"
GPUS="0,1,2,3,4,5,6,7"
OUT_ROOT="datas/llama"
GSM8K_PATH="../data/GSM8K"
SVAMP_PATH="../data/SVAMP/SVAMP.json"

mkdir -p "${OUT_ROOT}"

python collect_tot.py \
  --dataset-name gsm8k \
  --dataset-path "${GSM8K_PATH}" \
  --split train \
  --gpus "${GPUS}" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUT_ROOT}/gsm8k" \
  --output-prefix gsm8k_tot \
  --sample-batch-size 8 \
  --merge

python collect_tot.py \
  --dataset-name svamp \
  --dataset-path "${SVAMP_PATH}" \
  --gpus "${GPUS}" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUT_ROOT}/svamp" \
  --output-prefix svamp_tot \
  --sample-batch-size 8 \
  --merge
