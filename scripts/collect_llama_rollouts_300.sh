#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="../model"
MODELS=(
  "Meta-Llama-3-8B"
  "Meta-Llama-3-8B-Instruct"
  "Qwen3-4B-Instruct-2507"
  "Qwen3-8B"
)
GPUS="0,1,2,3,4,5,6,7"
OUT_ROOT="datas_svamp_retest"
export TORCH_COMPILE_DISABLE=1
export VLLM_GPU_MEMORY_UTILIZATION=0.8
export VLLM_DISABLE_PROGRESS_BAR=1
export VLLM_LOGGING_LEVEL=ERROR
SVAMP_PATH="../data/SVAMP/SVAMP.json"

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
    --model-dir "${model_dir}" \
    --output-dir "${model_out}/gsm8k" \
    --output-prefix gsm8k_tot \
    --sample-batch-size 32 \
    --rollouts-per-candidate 8 \
    --rollout-batch-size 200 \
    --max-samples 300 \
    --log-per-sample \
    "${use_vllm_args[@]}"

  python collect_tot.py \
    --dataset-name svamp \
    --dataset-path "${SVAMP_PATH}" \
    --gpus "${GPUS}" \
    --model-dir "${model_dir}" \
    --output-dir "${model_out}/svamp" \
    --output-prefix svamp_tot \
    --sample-batch-size 32 \
    --rollouts-per-candidate 8 \
    --rollout-batch-size 200 \
    --max-samples 300 \
    --log-per-sample \
    "${use_vllm_args[@]}"

done
