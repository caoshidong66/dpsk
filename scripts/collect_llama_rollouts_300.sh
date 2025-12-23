#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="../model"
MODELS=(
  "Qwen3-4B-Instruct-2507"
  "Qwen3-8B"
)
GPUS="5,6,7,8"
VLLM_TP_SIZE=4
OUT_ROOT="datas_300"
export VLLM_GPU_MEMORY_UTILIZATION=0.9 
GSM8K_PATH="../data/GSM8K"
SVAMP_PATH="../data/SVAMP/SVAMP.json"
MATH_PATH="../data/hendrycks_math"

mkdir -p "${OUT_ROOT}"

for model_name in "${MODELS[@]}"; do
  model_dir="${MODEL_ROOT}/${model_name}"
  model_out="${OUT_ROOT}/${model_name}"
  mkdir -p "${model_out}"
  use_vllm_args=()

  python collect_tot.py \
    --dataset-name gsm8k \
    --dataset-path "${GSM8K_PATH}" \
    --split train \
    --gpus "${GPUS}" \
    --vllm-tp-size "${VLLM_TP_SIZE}" \
    --model-dir "${model_dir}" \
    --output-dir "${model_out}/gsm8k" \
    --output-prefix gsm8k_tot \
    --sample-batch-size 16 \
    --rollouts-per-candidate 8 \
    --rollout-batch-size 200 \
    --max-samples 300 \
    --log-per-sample \
    "${use_vllm_args[@]}" \
    --merge

  python collect_tot.py \
    --dataset-name svamp \
    --dataset-path "${SVAMP_PATH}" \
    --gpus "${GPUS}" \
    --vllm-tp-size "${VLLM_TP_SIZE}" \
    --model-dir "${model_dir}" \
    --output-dir "${model_out}/svamp" \
    --output-prefix svamp_tot \
    --sample-batch-size 16 \
    --rollouts-per-candidate 8 \
    --rollout-batch-size 200 \
    --max-samples 300 \
    --log-per-sample \
    "${use_vllm_args[@]}" \
    --merge

  python collect_tot.py \
    --dataset-name hendrycks_math \
    --dataset-path "${MATH_PATH}" \
    --split train \
    --gpus "${GPUS}" \
    --vllm-tp-size "${VLLM_TP_SIZE}" \
    --model-dir "${model_dir}" \
    --output-dir "${model_out}/math" \
    --output-prefix math_tot \
    --sample-batch-size 16 \
    --rollouts-per-candidate 8 \
    --rollout-batch-size 200 \
    --max-samples 300 \
    --log-per-sample \
    "${use_vllm_args[@]}" \
    --merge
done
