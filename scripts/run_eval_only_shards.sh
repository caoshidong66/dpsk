#!/usr/bin/env bash
set -euo pipefail

# Edit these as needed.
MODEL_DIR="../model/Meta-Llama-3-8B"
LORA_DIR="weights/outputs/chain_pref_llama_gsm8k_epoch1"
DATASET_NAME="gsm8k"
DATASET_ROOT="../data/GSM8K"
SPLIT="test"
NUM_SAMPLES=500
GPU_IDS="0,2,6,7"
ID_DIR="datas/test_id"
OUT_DIR="outputs/eval_only"
MAX_NEW_TOKENS=256

mkdir -p "${ID_DIR}"
mkdir -p "${OUT_DIR}"

ID_CACHE="${ID_DIR}/${DATASET_NAME}_${SPLIT}_${NUM_SAMPLES}.json"

echo "[eval] building id cache -> ${ID_CACHE}"
CUDA_VISIBLE_DEVICES=0 python eval_only.py \
  --dataset-name "${DATASET_NAME}" \
  --dataset-root "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --num-samples "${NUM_SAMPLES}" \
  --id-cache "${ID_CACHE}" \
  --model-dir "${MODEL_DIR}" \
  --lora-dir "${LORA_DIR}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-json "${OUT_DIR}/${DATASET_NAME}_${SPLIT}_shard0.json" \
  --shard-id 0 \
  --num-shards 1

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
NUM_SHARDS="${#GPU_LIST[@]}"

echo "[eval] running ${NUM_SHARDS} shards on GPUs: ${GPU_IDS}"
for shard in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python eval_only.py \
    --dataset-name "${DATASET_NAME}" \
    --dataset-root "${DATASET_ROOT}" \
    --split "${SPLIT}" \
    --num-samples "${NUM_SAMPLES}" \
    --id-cache "${ID_CACHE}" \
    --model-dir "${MODEL_DIR}" \
    --lora-dir "${LORA_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --output-json "${OUT_DIR}/${DATASET_NAME}_${SPLIT}_shard${shard}.json" \
    --shard-id "${shard}" \
    --num-shards "${NUM_SHARDS}" &
done
wait

echo "[eval] done -> ${OUT_DIR}"
