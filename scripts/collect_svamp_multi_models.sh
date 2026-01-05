#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="../model"
MODELS=(
  "Qwen3-8B"
)

GPUS="0,1,2,3,4,5,6,7"
DATASET_PATH="../data/GSM8K"
OUT_ROOT="datas_gsm8k_retest"

for model_name in "${MODELS[@]}"; do
  model_dir="${MODEL_ROOT}/${model_name}"
  output_dir="${OUT_ROOT}/${model_name}/svamp"
  mkdir -p "${output_dir}"

  python collect_tot.py \
    --dataset-name gsm8k \
    --dataset-path "${DATASET_PATH}" \
    --split test \
    --gpus "${GPUS}" \
    --model-dir "${model_dir}" \
    --output-dir "${output_dir}" \
    --output-prefix svamp_tot \
    --sample-batch-size 16 \
    --rollouts-per-candidate 8 \
    --rollout-batch-size 125 \
    --max-samples 200 \
    --log-per-sample \
    --merge
done
