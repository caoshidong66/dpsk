#!/usr/bin/env bash

export TORCHINDUCTOR_CACHE_DIR=/local/$USER/torchinductor_$RANDOM
export VLLM_CACHE_DIR=/local/$USER/vllm_$RANDOM
export HF_HOME=/local/$USER/hf_$RANDOM

set -euo pipefail

MODEL_ROOT="../model"
MODELS=(
  "Qwen3-8B"
  "Meta-Llama-3-8B-Instruct"
  "Qwen3-4B-Instruct-2507"

)
GPUS="2"
OUT_ROOT="datas/all300_case"
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

  for level in 4 5; do
    python collect_tot.py \
      --dataset-name hendrycks_math \
      --dataset-path "${MATH_PATH}" \
      --split test \
      --level "${level}" \
      --gpus "${GPUS}" \
      --model-dir "${model_dir}" \
      --output-dir "${model_out}/math_l${level}" \
      --output-prefix math_l${level}_tot \
      --sample-batch-size 8 \
      --rollouts-per-candidate 8 \
      --rollout-batch-size  50 \
      --max-samples 50 \
      --log-per-sample
  done
done
