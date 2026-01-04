#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,4}"
NPROC="${NPROC:-2}"

MODEL_ROOT="/home/comp/25480758/model"
DATA_ROOT="/home/comp/25480758/dpsk/datas"
OUT_ROOT="outputs"

MODELS=(
  "Meta-Llama-3-8B-Instruct"
  "Meta-Llama-3-8B"
  "Qwen3-4B-Instruct-2507"
  "Qwen3-8B"
)

latest_jsonl() {
  local pattern="$1"
  local latest
  latest="$(ls -t ${pattern} 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf "%s" "${latest}"
}

for model_name in "${MODELS[@]}"; do
  model_dir="${MODEL_ROOT}/${model_name}"
  data_dir="${DATA_ROOT}/${model_name}/gsm8k"
  jsonl_path="$(latest_jsonl "${data_dir}/gsm8k.jsonl"
  if [[ -z "${jsonl_path}" ]]; then
    echo "[chain_pref] skip ${model_name}: no gsm8k_tot jsonl under ${data_dir}" >&2
    continue
  fi

  out_dir="${OUT_ROOT}/chain_pref_${model_name//\//_}_gsm8k"
  echo "[chain_pref] model=${model_name} data=${jsonl_path} out=${out_dir}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node "${NPROC}" training/train_chain_preference.py \
    --tot-jsonl "${jsonl_path}" \
    --model-name-or-path "${model_dir}" \
    --output-dir "${out_dir}"
done
