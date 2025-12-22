#!/usr/bin/env bash
set -euo pipefail

# Paths you likely need to customize.
DATA_ROOT="../data/hendrycks_math"
MODEL_DIR="../model/Meta-Llama-3-8B"
NUM_SAMPLES=300
GPU_IDS="1"
NUM_GPUS=1

declare -A TOT_JSONL
TOT_JSONL[1]="datas/tot_math_l1/math_l1.20251215_152302.all.jsonl"
TOT_JSONL[2]="datas/tot_math_l2/math_l2.20251215_175641.all.jsonl"
TOT_JSONL[3]="datas/tot_math_l3/math_l3.20251214_073429.all.jsonl"
TOT_JSONL[4]="datas/tot_math_l4/math_l4.20251215_203600.all.jsonl"
TOT_JSONL[5]="datas/tot_math_l5/math_l5.20251216_011244.all.jsonl"

RESULTS_FILE="outputs/chain_pref_eval_results.jsonl"
mkdir -p "$(dirname "${RESULTS_FILE}")"

for level in 1 2 3 4 5; do
  tot_path="${TOT_JSONL[$level]}"
  if [[ ! -f "${tot_path}" ]]; then
    echo "[run] missing ToT JSONL for level ${level}: ${tot_path}" >&2
    exit 1
  fi

  out_dir="outputs/chain_pref_l${level}"
  lora_dir="weights/${out_dir}"

  echo "[run] level ${level}: training -> ${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --nproc_per_node="${NUM_GPUS}" training/train_chain_preference.py \
    --tot-jsonl "${tot_path}" \
    --model-name-or-path "${MODEL_DIR}" \
    --output-dir "${out_dir}" \
    --batch-size 8 \
    --grad-accum-steps 1 \
    --bf16 \
    --device-map none \
    --use-lora \
    --flatten-steps \
    --learning-rate 1e-5 \
    --eval-after-train \
    --eval-num-samples "${NUM_SAMPLES}" \
    --eval-id-cache "datas/eval_ids/hendrycks_level${level}_test_${NUM_SAMPLES}.json" \
    --eval-dataset-root "${DATA_ROOT}"

  echo "[run] level ${level}: eval base (CoT only)"
  base_out="$(python eval_llama_cot_tot.py \
    --dataset-root "${DATA_ROOT}" \
    --split test \
    --level "${level}" \
    --num-samples "${NUM_SAMPLES}" \
    --model-dir "${MODEL_DIR}" \
    --gpus "${GPU_IDS}" \
    --cot-batch-size 4 \
    --only-cot \
    --no-vllm-for-cot)"
  echo "${base_out}"
  base_json="$(python - <<'PY'
import json, sys
lines = sys.stdin.read().splitlines()
last = None
for line in lines:
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        try:
            obj = json.loads(line)
            last = line
        except Exception:
            continue
if last is None:
    last = "{}"
print(last)
PY
<<< "${base_out}")"
  echo "{\"level\":${level},\"model\":\"base\",\"result\":${base_json}}" >> "${RESULTS_FILE}"

  echo "[run] level ${level}: eval LoRA (CoT only)"
  lora_out="$(python eval_llama_cot_tot.py \
    --dataset-root "${DATA_ROOT}" \
    --split test \
    --level "${level}" \
    --num-samples "${NUM_SAMPLES}" \
    --model-dir "${MODEL_DIR}" \
    --lora-dir "${lora_dir}" \
    --gpus "${GPU_IDS}" \
    --cot-batch-size 4 \
    --only-cot \
    --no-vllm-for-cot \
    --lora-no-merge)"
  echo "${lora_out}"
  lora_json="$(python - <<'PY'
import json, sys
lines = sys.stdin.read().splitlines()
last = None
for line in lines:
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        try:
            obj = json.loads(line)
            last = line
        except Exception:
            continue
if last is None:
    last = "{}"
print(last)
PY
<<< "${lora_out}")"
  echo "{\"level\":${level},\"model\":\"lora\",\"result\":${lora_json}}" >> "${RESULTS_FILE}"
done
