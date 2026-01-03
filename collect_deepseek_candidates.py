from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from cot_math import build_completion_prompt
from tool import steps_for_dataset, steps_for_level


CONFIG: Dict[str, object] = {
    "model": "deepseek-v3-2-251201",
    "temperature": 0.0,
    "max_new_tokens": 1024,
    "max_samples": None,
    "no_full_solution": False,
    "no_prefix_candidate": False,
    "log_candidates": False,
}


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


def _infer_dataset_from_path(path: str) -> Optional[str]:
    lower = path.lower()
    if "gsm8k" in lower:
        return "gsm8k"
    if "svamp" in lower:
        return "svamp"
    if "hendrycks" in lower or "math" in lower:
        return "hendrycks_math"
    return None


def _extract_steps_from_completion(text: str, num_steps: int) -> List[str]:
    steps: List[str] = []
    if not text:
        return [""] * num_steps

    lower = text.lower()
    for i in range(1, num_steps + 1):
        start_match = re.search(rf"step\s*{i}\s*:", lower)
        if not start_match:
            steps.append("")
            continue
        start = start_match.end()
        next_match = re.search(rf"step\s*{i + 1}\s*:", lower[start:]) if i < num_steps else None
        answer_match = re.search(r"answer\s*:", lower[start:])
        candidates = []
        if next_match:
            candidates.append(start + next_match.start())
        if answer_match:
            candidates.append(start + answer_match.start())
        end = min(candidates) if candidates else len(text)
        steps.append(text[start:end].strip())
    return steps


def _ensure_step_prefix(prefix: str, step_index: int) -> str:
    label = f"Step {step_index}:"
    stripped = prefix.rstrip()
    if stripped.endswith(label):
        return stripped + " "
    if not stripped.endswith("\n"):
        stripped += "\n"
    return stripped + label + " "


def _next_candidate_index(candidates: List[Dict[str, object]]) -> int:
    max_idx = -1
    for cand in candidates:
        idx = cand.get("candidate_index")
        if isinstance(idx, int) and idx > max_idx:
            max_idx = idx
    return max_idx + 1


def _deepseek_chat_completion(
    prompt: str,
    *,
    model: str,
    temperature: float,
    stop_tokens: Optional[List[str]] = None,
    max_tokens: int = 1024,
) -> str:
    from api import ark_chat_with_stop, client

    stop_token = None
    stops = stop_tokens or []
    if stops:
        stop_token = stops[0]
    # ark_chat_with_stop does not expose max_tokens; keep arg for API compatibility.
    return ark_chat_with_stop(
        client=client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stop_token=stop_token or "</END>",
        thinking="disabled",
        temperature=temperature,
    )


def _build_prefix_from_chosen(prompt: str, steps_trace: List[Dict[str, object]], upto_step: int) -> str:
    prefix = prompt
    for entry in steps_trace:
        step_idx = entry.get("step_index")
        if not isinstance(step_idx, int) or step_idx >= upto_step:
            break
        chosen = entry.get("chosen_step_text")
        if not isinstance(chosen, str):
            break
        prefix = _ensure_step_prefix(prefix, step_idx)
        prefix = prefix + (chosen.strip() + "\n")
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment ToT JSONL with DeepSeek candidates.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--model", type=str, default=str(CONFIG["model"]))
    parser.add_argument("--temperature", type=float, default=float(CONFIG["temperature"]))
    parser.add_argument("--max-new-tokens", type=int, default=int(CONFIG["max_new_tokens"]))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-full-solution", action="store_true")
    parser.add_argument("--no-prefix-candidate", action="store_true")
    parser.add_argument("--log-candidates", action="store_true")
    args = parser.parse_args()

    if args.max_samples is None and CONFIG.get("max_samples") is not None:
        args.max_samples = int(CONFIG["max_samples"])
    if not args.no_full_solution and CONFIG.get("no_full_solution"):
        args.no_full_solution = True
    if not args.no_prefix_candidate and CONFIG.get("no_prefix_candidate"):
        args.no_prefix_candidate = True
    if not args.log_candidates and CONFIG.get("log_candidates"):
        args.log_candidates = True

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    dataset_name = _infer_dataset_from_path(str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line_idx, line in enumerate(f_in):
            if args.max_samples is not None and written >= int(args.max_samples):
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            base = record.get("tot") if isinstance(record, dict) and isinstance(record.get("tot"), dict) else record
            steps_trace = base.get("steps_trace") if isinstance(base, dict) else None
            if not isinstance(steps_trace, list) or not steps_trace:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                continue

            problem = base.get("problem") if isinstance(base, dict) else None
            prompt = base.get("prompt") if isinstance(base, dict) else None
            level = base.get("level") if isinstance(base, dict) else None
            if not prompt and isinstance(problem, str):
                num_steps = _infer_num_steps(None, dataset_name, level)
                prompt = build_completion_prompt(problem, num_steps=num_steps)
            num_steps = _infer_num_steps(prompt, dataset_name, level)

            if not args.no_full_solution and isinstance(prompt, str):
                completion = _deepseek_chat_completion(
                    prompt,
                    model=args.model,
                    temperature=float(args.temperature),
                    stop_tokens=["</END>"],
                    max_tokens=int(args.max_new_tokens),
                )
                full_steps = _extract_steps_from_completion(completion, num_steps)
                for entry in steps_trace:
                    step_idx = entry.get("step_index")
                    if not isinstance(step_idx, int) or step_idx < 1 or step_idx > num_steps:
                        continue
                    candidates = entry.get("candidates")
                    if not isinstance(candidates, list):
                        candidates = []
                        entry["candidates"] = candidates
                    new_idx = _next_candidate_index(candidates)
                    step_text = full_steps[step_idx - 1]
                    candidates.append(
                        {
                            "candidate_index": new_idx,
                            "step_text": step_text,
                            "success_count": 0,
                            "success_rate": 0.0,
                            "source": "deepseek_full",
                        }
                    )
                    if args.log_candidates:
                        print(
                            f"[deepseek] sample={line_idx} step={step_idx} add={new_idx} source=deepseek_full",
                            flush=True,
                        )

            if not args.no_prefix_candidate and isinstance(prompt, str):
                for entry in steps_trace:
                    step_idx = entry.get("step_index")
                    if not isinstance(step_idx, int) or step_idx < 1 or step_idx > num_steps:
                        continue
                    prefix = _build_prefix_from_chosen(prompt, steps_trace, step_idx)
                    step_prompt = _ensure_step_prefix(prefix, step_idx)
                    completion = _deepseek_chat_completion(
                        step_prompt,
                        model=args.model,
                        temperature=float(args.temperature),
                        stop_tokens=[f"\nStep {step_idx + 1}:", "\nAnswer:"],
                        max_tokens=min(512, int(args.max_new_tokens)),
                    )
                    step_text = completion.strip()
                    candidates = entry.get("candidates")
                    if not isinstance(candidates, list):
                        candidates = []
                        entry["candidates"] = candidates
                    new_idx = _next_candidate_index(candidates)
                    candidates.append(
                        {
                            "candidate_index": new_idx,
                            "step_text": step_text,
                            "success_count": 0,
                            "success_rate": 0.0,
                            "source": "deepseek_prefix",
                        }
                    )
                    if args.log_candidates:
                        print(
                            f"[deepseek] sample={line_idx} step={step_idx} add={new_idx} source=deepseek_prefix",
                            flush=True,
                        )

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[deepseek] wrote {written} records -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
