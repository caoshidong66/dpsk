#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dpsk"
DATASET_ROOT="/data/jsg_data/hendrycks_math"
SPLIT="train"
OUT_DIR="datas/math_samples"
LEVELS="1,2,4,5"
K=500
SEED=42

usage() {
  cat <<'EOF'
Generate MATH (hendrycks_math) samples by difficulty level.

Usage:
  bash scripts/generate_math_samples.sh [options]

Options:
  --env NAME            Conda env name (default: dpsk)
  --dataset-root PATH   $ bash scripts/generate_math_samples.sh
/data/jsg_data/anaconda3/envs/dpsk/lib/python3.11/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[generate_math_samples] Wrote level=1: 500 -> datas/math_samples/math_level1_500.jsonl
[generate_math_samples] Wrote level=2: 500 -> datas/math_samples/math_level2_500.jsonl
[generate_math_samples] Wrote level=4: 500 -> datas/math_samples/math_level4_500.jsonl
[generate_math_samples] Wrote level=5: 500 -> datas/math_samples/math_level5_500.jsonlhendrycks_math root dir (default: /data/jsg_data/hendrycks_math)
  --split NAME          Dataset split: train|test (default: train)
  --out-dir PATH        Output directory for jsonl files (default: datas/math_samples)
  --levels CSV          Levels to export (default: 1,2,4,5)
  --k N                 Samples per level (default: 500)
  --seed N              Random seed (default: 42)
  -h, --help            Show this help

Outputs:
  <out-dir>/math_level<level>_<k>.jsonl
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please install/miniconda or ensure conda is available." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p "${OUT_DIR}"

DATASET_ROOT="${DATASET_ROOT}" \
SPLIT="${SPLIT}" \
OUT_DIR="${OUT_DIR}" \
LEVELS="${LEVELS}" \
K="${K}" \
SEED="${SEED}" \
python - <<'PY'
import json
import os
import random
import re
from pathlib import Path

from dataset_utils import iter_samples, normalize_sample

dataset_root = os.environ["DATASET_ROOT"]
split = os.environ["SPLIT"]
out_dir = Path(os.environ["OUT_DIR"])
levels_csv = os.environ["LEVELS"]
k = int(os.environ["K"])
seed = int(os.environ["SEED"])

levels = []
for part in (p.strip() for p in levels_csv.split(",") if p.strip()):
    m = re.search(r"(\d+)", part)
    if not m:
        raise SystemExit(f"Invalid level in --levels: {part!r}")
    levels.append(int(m.group(1)))
levels = sorted(set(levels))

def parse_level(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else None

rngs = {lvl: random.Random(seed + lvl * 10007) for lvl in levels}
reservoirs = {lvl: [] for lvl in levels}  # list[{index:int, sample:dict}]
seen = {lvl: 0 for lvl in levels}

for idx, raw in enumerate(iter_samples("hendrycks_math", dataset_root, split=split)):
    lvl = parse_level(raw.get("level"))
    if lvl not in reservoirs:
        continue
    seen[lvl] += 1
    entry = {"index": idx, "sample": normalize_sample("hendrycks_math", raw)}
    res = reservoirs[lvl]
    if len(res) < k:
        res.append(entry)
        continue
    j = rngs[lvl].randint(1, seen[lvl])
    if j <= k:
        res[j - 1] = entry

out_dir.mkdir(parents=True, exist_ok=True)
for lvl in levels:
    res = reservoirs[lvl]
    if len(res) < k:
        print(f"[generate_math_samples] Only found {len(res)} samples for level={lvl} (requested {k}).")
    rngs[lvl].shuffle(res)
    out_path = out_dir / f"math_level{lvl}_{k}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in res[:k]:
            out = {
                "dataset_name": "hendrycks_math",
                "dataset_path": dataset_root,
                "split": split,
                "index": item["index"],
                "sample": item["sample"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\\n")
    print(f"[generate_math_samples] Wrote level={lvl}: {min(len(res), k)} -> {out_path}")
PY
