#!/usr/bin/env bash
set -euo pipefail

# Paths you likely need to customize.
DATA_ROOT="../data/hendrycks_math"
MODEL_DIR="../model/Meta-Llama-3-8B"
NUM_SAMPLES=300
GPU_ID="0"

declare -A TOT_JSONL
TOT_JSONL[1]="datas/tot_math_l1/math_l1.20251215_152302.all.jsonl"
TOT_JSONL[2]="datas/tot_math_l2/math_l2.20251215_175641.all.jsonl"
TOT_JSONL[3]="datas/tot_math_l3/math_l3.20251214_073429.all.jsonl"
TOT_JSONL[4]="datas/tot_math_l4/math_l4.20251215_203600.all.jsonl"
TOT_JSONL[5]="datas/tot_math_l5/math_l5.20251216_011244.all.jsonl"

for level in 1 2 3 4 5; do
  tot_path="${TOT_JSONL[$level]}"
  if [[ ! -f "${tot_path}" ]]; then
    echo "[run] missing ToT JSONL for level ${level}: ${tot_path}" >&2
    exit 1
  fi

  out_dir="outputs/chain_pref_l${level}"
  lora_dir="weights/${out_dir}"

  echo "[run] level ${level}: training -> ${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python training/train_chain_preference.py \
    --tot-jsonl "${tot_path}" \
    --model-name-or-path "${MODEL_DIR}" \
    --output-dir "${out_dir}" \
    --batch-size 2 \
    --grad-accum-steps 1 \
    --bf16 \
    --device-map none \
    --use-lora

  echo "[run] level ${level}: eval base (CoT only)"
  python eval_llama_cot_tot.py \
    --dataset-root "${DATA_ROOT}" \
    --split test \
    --level "${level}" \
    --num-samples "${NUM_SAMPLES}" \
    --model-dir "${MODEL_DIR}" \
    --gpus "${GPU_ID}" \
    --cot-batch-size 8 \
    --only-cot

  echo "[run] level ${level}: eval LoRA (CoT only)"
  python eval_llama_cot_tot.py \
    --dataset-root "${DATA_ROOT}" \
    --split test \
    --level "${level}" \
    --num-samples "${NUM_SAMPLES}" \
    --model-dir "${MODEL_DIR}" \
    --lora-dir "${lora_dir}" \
    --reuse-merged \
    --gpus "${GPU_ID}" \
    --cot-batch-size 8 \
    --only-cot
done
