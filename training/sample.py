#!/usr/bin/env python3
"""
Print a ToT sample in a paper-friendly format:
  - show the prefix used at each step
  - list candidates sorted by success_rate with scores
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cot_math import build_completion_prompt
from tool import steps_for_dataset, steps_for_level


def _infer_dataset_from_path(path: str) -> Optional[str]:
    lower = path.lower()
    if "gsm8k" in lower:
        return "gsm8k"
    if "svamp" in lower:
        return "svamp"
    if "hendrycks" in lower or "math" in lower:
        return "hendrycks_math"
    return None


def _infer_num_steps(prompt: Optional[str], dataset_name: Optional[str], level: Optional[object]) -> int:
    if prompt:
        match = re.search(r"exactly\s+(\d+)\s+reasoning steps", prompt, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
    if dataset_name in {"gsm8k", "svamp"}:
        return steps_for_dataset(dataset_name)
    return steps_for_level(level)


def _ensure_step_prefix(prefix: str, step_index: int) -> str:
    label = f"Step {step_index}:"
    stripped = prefix.rstrip()
    if stripped.endswith(label):
        return stripped + " "
    if not stripped.endswith("\n"):
        stripped += "\n"
    return stripped + label + " "


def _load_record(path: Path, index: int) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == index:
                return json.loads(line)
    raise IndexError(f"index {index} out of range for {path}")


def _sorted_candidates(candidates: List[Dict[str, object]]) -> List[Tuple[int, float, str]]:
    rows: List[Tuple[int, float, str]] = []
    for c in candidates:
        idx = c.get("candidate_index")
        sr = c.get("success_rate")
        text = c.get("step_text")
        if not isinstance(idx, int) or not isinstance(sr, (int, float)) or not isinstance(text, str):
            continue
        rows.append((idx, float(sr), text))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a ToT sample with sorted candidates.")
    parser.add_argument("--tot-jsonl", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.tot_jsonl)
    record = _load_record(path, int(args.index))
    base = record.get("tot") if isinstance(record, dict) and isinstance(record.get("tot"), dict) else record

    problem = base.get("problem") if isinstance(base, dict) else None
    prompt = base.get("prompt") if isinstance(base, dict) else None
    level = base.get("level") if isinstance(base, dict) else None
    steps_trace = base.get("steps_trace") if isinstance(base, dict) else None
    if not isinstance(problem, str) or not isinstance(steps_trace, list):
        raise ValueError("Sample missing problem/steps_trace.")

    dataset_name = _infer_dataset_from_path(str(path))
    num_steps = _infer_num_steps(prompt, dataset_name, level)
    if not prompt:
        prompt = build_completion_prompt(problem, num_steps=num_steps)

    print(f"[sample] index={args.index} dataset={dataset_name} level={level}")
    print("Problem:")
    print(problem)
    print("\n---\n")

    current_prefix = prompt
    for entry in steps_trace:
        step_idx = entry.get("step_index")
        candidates = entry.get("candidates")
        if not isinstance(step_idx, int) or not isinstance(candidates, list):
            continue
        prefix = _ensure_step_prefix(current_prefix, step_idx)
        print(f"Step {step_idx} Prefix:")
        print(prefix)
        print("Candidates (sorted by success_rate):")
        sorted_rows = _sorted_candidates(candidates)
        if args.max_candidates is not None:
            sorted_rows = sorted_rows[: int(args.max_candidates)]
        for cand_idx, sr, text in sorted_rows:
            out_text = text
            if args.max_chars is not None and len(out_text) > int(args.max_chars):
                out_text = out_text[: int(args.max_chars)] + "..."
            print(f"  [{cand_idx}] success={sr:.3f} :: {out_text}")
        chosen = entry.get("chosen_candidate_index")
        chosen_text = entry.get("chosen_step_text")
        if isinstance(chosen, int):
            print(f"Chosen: {chosen}")
        if isinstance(chosen_text, str):
            current_prefix = _ensure_step_prefix(current_prefix, step_idx)
            current_prefix = current_prefix + chosen_text.strip() + "\n"
        print("\n---\n")


if __name__ == "__main__":
    main()
