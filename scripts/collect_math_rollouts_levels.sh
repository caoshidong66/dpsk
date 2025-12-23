#!/usr/bin/env bash

export TORCHINDUCTOR_CACHE_DIR=/local/$USER/torchinductor_$RANDOM
export VLLM_CACHE_DIR=/local/$USER/vllm_$RANDOM
export HF_HOME=/local/$USER/hf_$RANDOM

set -euo pipefail

MODEL_ROOT="../model"
MODELS=(
  "Meta-Llama-3-8B"
  "Meta-Llama-3-8B-Instruct"
  "Qwen3-4B-Instruct-2507"
  "Qwen3-8B"
)
GPUS="2,3,4,5"
VLLM_TP_SIZE=4
OUT_ROOT="datas_math"
export TORCH_COMPILE_DISABLE=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_DISABLE_PROGRESS_BAR=1
export VLLM_LOGGING_LEVEL=ERROR
MATH_PATH="../data/hendrycks_math"

mkdir -p "${OUT_ROOT}"

for model_name in "${MODELS[@]}"; do
  model_dir="${MODEL_ROOT}/${model_name}"
  model_out="${OUT_ROOT}/${model_name}"
  mkdir -p "${model_out}"

  for level in 1 2 3 4 5; do
    python collect_tot.py \
      --dataset-name hendrycks_math \
      --dataset-path "${MATH_PATH}" \
      --split train \
      --level "${level}" \
      --gpus "${GPUS}" \
      --vllm-tp-size "${VLLM_TP_SIZE}" \
      --model-dir "${model_dir}" \
      --output-dir "${model_out}/math_l${level}" \
      --output-prefix math_l${level}_tot \
      --sample-batch-size 16 \
      --rollouts-per-candidate 8 \
      --rollout-batch-size 200 \
      --max-samples 300 \
      --log-per-sample
  done
done
