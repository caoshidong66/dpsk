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
GPUS="0,1,2,3,4,5,6,7"
OUT_ROOT="datas_svamp_all"
export TORCH_COMPILE_DISABLE=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_DISABLE_PROGRESS_BAR=1
export VLLM_LOGGING_LEVEL=ERROR
SVAMP_PATH="../data/SVAMP/SVAMP.json"

mkdir -p "${OUT_ROOT}"

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NPROC="${#GPU_ARR[@]}"

for model_name in "${MODELS[@]}"; do
  model_dir="${MODEL_ROOT}/${model_name}"
  model_out="${OUT_ROOT}/${model_name}"
  mkdir -p "${model_out}"

  CUDA_VISIBLE_DEVICES="${GPUS}" \
  torchrun --nproc_per_node "${NPROC}" collect_tot.py \
    --dataset-name svamp \
    --dataset-path "${SVAMP_PATH}" \
    --gpus "${GPUS}" \
    --model-dir "${model_dir}" \
    --output-dir "${model_out}/svamp" \
    --output-prefix svamp_tot \
    --sample-batch-size 16 \
    --max-samples 300 \
    --rollouts-per-candidate 8 \
    --rollout-batch-size 125 \
    --log-per-sample
done
