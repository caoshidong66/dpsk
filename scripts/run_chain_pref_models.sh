#!/usr/bin/env bash
set -euo pipefail

# Paths you likely need to customize.
MODEL_ROOT="../model"
NUM_SAMPLES=500
TEST_ID_DIR="datas/test_id"

# Multi-GPU training (DDP).
TRAIN_GPU_IDS="0,1,2,3,4,5,6,7"
TRAIN_NUM_GPUS=8

DATASETS=(
  "gsm8k"
  "svamp"
)

MODELS=(
  "Meta-Llama-3-8B"
  "Meta-Llama-3-8B-Instruct"
  "Qwen3-4B-Instruct-2507"
  "Qwen3-8B"
)

RESULTS_DIR="outputs/chain_pref_eval"
mkdir -p "${TEST_ID_DIR}"
mkdir -p "${RESULTS_DIR}"

for model_name in "${MODELS[@]}"; do
  model_dir="${MODEL_ROOT}/${model_name}"
  results_file="${RESULTS_DIR}/${model_name}.jsonl"

  for dataset in "${DATASETS[@]}"; do
    data_dir="datas/${model_name}/${dataset}"
    shopt -s nullglob
    jsonl_files=("${data_dir}"/*.jsonl)
    shopt -u nullglob
    if [[ ${#jsonl_files[@]} -ne 1 ]]; then
      echo "[run] expected 1 jsonl in ${data_dir}, found ${#jsonl_files[@]}" >&2
      exit 1
    fi
    tot_path="${jsonl_files[0]}"

    case "${dataset}" in
      gsm8k) eval_root="../data/GSM8K" ;;
      svamp) eval_root="../data/SVAMP/SVAMP.json" ;;
      *) echo "[run] unknown dataset ${dataset}" >&2; exit 1 ;;
    esac

    out_dir="outputs/chain_pref_${model_name}_${dataset}"

    echo "[run] model ${model_name} dataset ${dataset}: training -> ${out_dir}"
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" torchrun --nproc_per_node="${TRAIN_NUM_GPUS}" training/train_chain_preference.py \
      --tot-jsonl "${tot_path}" \
      --model-name-or-path "${model_dir}" \
      --output-dir "${out_dir}" \
      --batch-size 8 \
      --grad-accum-steps 1 \
      --bf16 \
      --device-map none \
      --use-lora \
      --flatten-steps \
      --learning-rate 1e-4 \
      --eval-after-train \
      --eval-num-samples "${NUM_SAMPLES}" \
      --eval-id-cache "${TEST_ID_DIR}/${dataset}_test_${NUM_SAMPLES}.json" \
      --eval-dataset-root "${eval_root}"
  done
done
