#!/usr/bin/env python3
import ast
from pathlib import Path
from typing import Any, Dict, List

# Allow JSON-style literals in pasted data blocks.
null = None
true = True
false = False
data = {"dataset_name": "gsm8k", "dataset_path": "../data/GSM8K", "split": "train", "index": 17, "gpu": "0,1,2,3,4,5,6,7", "rank": 0, "world_size": 1, "elapsed_sec": 591.2172243595123, "tot": {"problem": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.", "solution": "If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43", "level": null, "prompt": "You are an expert math problem solver. You must reason step by step and avoid logical or arithmetic mistakes.\n\nSolve the following math problem.\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`.\n\nProblem: In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.\n\nReasoning step by step:\nStep 1:", "steps_trace": [{"step_index": 1, "candidates": [{"candidate_index": 0, "step_text": "26 pink hard hats - 4 pink hard hats = 22 pink hard hats", "success_count": 0, "success_rate": 0.0}, {"candidate_index": 1, "step_text": "26 - 4 = ? (number of pink hats after Carl's removal)", "success_count": 1, "success_rate": 0.125}, {"candidate_index": 2, "step_text": "We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.", "success_count": 6, "success_rate": 0.75}, {"candidate_index": 3, "step_text": "Initial number of pink hard hats is 26.", "success_count": 4, "success_rate": 0.5}, {"candidate_index": 4, "step_text": "(Please write your reasoning in this line)", "success_count": 0, "success_rate": 0.0}], "chosen_candidate_index": 2, "chosen_step_text": "We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.", "chosen_success_rate": 0.75}, {"step_index": 2, "candidates": [{"candidate_index": 0, "step_text": "6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 1, "step_text": "6 pink hard hats are taken away by John, which means 6 pink hard hats are removed from the truck. We are left with 22 - 6 = 16 pink hard hats.", "success_count": 5, "success_rate": 0.625}, {"candidate_index": 2, "step_text": "6 pink hard hats were removed by John, and he also removed twice as many green hard hats as the number of pink hard hats he removed. So, 6 green hard hats were removed, leaving us with 15 - 6 = 9 green hard hats.", "success_count": 0, "success_rate": 0.0}, {"candidate_index": 3, "step_text": "6 pink hard hats are taken away by John, leaving us with 22 - 6 = 16 pink hard hats. Since John takes away twice as many green hard hats as the number of pink hard hats that he removed (i.e., 6 green hard hats), we have 15 - 6 = 9 green hard hats left.", "success_count": 1, "success_rate": 0.125}, {"candidate_index": 4, "step_text": "6 pink hard hats were taken away by John, which means 6 green hard hats were taken away by John, since John took away twice as many green hard hats as the number of pink hard hats he removed. Therefore, we have 15 - 6 = 9 green hard hats remaining.", "success_count": 1, "success_rate": 0.125}], "chosen_candidate_index": 0, "chosen_step_text": "6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.", "chosen_success_rate": 1.0}, {"step_index": 3, "candidates": [{"candidate_index": 0, "step_text": "24 yellow hard hats remain unchanged.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 1, "step_text": "24 yellow hard hats remain unchanged, so we still have 24 yellow hard hats.", "success_count": 6, "success_rate": 0.75}, {"candidate_index": 2, "step_text": "24 yellow hard hats remain unchanged, so we still have 24 yellow hard hats.", "success_count": 7, "success_rate": 0.875}, {"candidate_index": 3, "step_text": "24 yellow hard hats are left untouched. So, we have 24 yellow hard hats.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 4, "step_text": "24 yellow hard hats remain in the truck and are not affected by the removals.", "success_count": 7, "success_rate": 0.875}], "chosen_candidate_index": 0, "chosen_step_text": "24 yellow hard hats remain unchanged.", "chosen_success_rate": 1.0}, {"step_index": 4, "candidates": [{"candidate_index": 0, "step_text": "16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 1, "step_text": "16 pink hard hats and 3 green hard hats are left. 16 + 3 = 19.", "success_count": 7, "success_rate": 0.875}, {"candidate_index": 2, "step_text": "16 pink hard hats and 3 green hard hats are left. In total, we have 16 + 3 + 24 = 43 hard hats remaining.", "success_count": 7, "success_rate": 0.875}, {"candidate_index": 3, "step_text": "16 pink hard hats, 3 green hard hats, and 24 yellow hard hats are left in the truck.", "success_count": 6, "success_rate": 0.75}, {"candidate_index": 4, "step_text": "16 pink hard hats and 3 green hard hats are left, so the total number of hard hats left is 16 + 3 + 24 = 43.", "success_count": 7, "success_rate": 0.875}], "chosen_candidate_index": 0, "chosen_step_text": "16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.", "chosen_success_rate": 1.0}, {"step_index": 5, "candidates": [{"candidate_index": 0, "step_text": "24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 1, "step_text": "19 hats of these two colors and 24 yellow hard hats are left. So, the total number of hard hats remaining is 19 + 24 = 43.", "success_count": 7, "success_rate": 0.875}, {"candidate_index": 2, "step_text": "19 hats of these two colors are combined with the unchanged 24 yellow hard hats, giving us a total of 19 + 24 = 43 hard hats.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 3, "step_text": "3 green hard hats, 19 hats of pink and green, and 24 yellow hard hats are left in the truck, making a total of 3 + 19 + 24 = ?", "success_count": 0, "success_rate": 0.0}, {"candidate_index": 4, "step_text": "19 hats of these two colors and 24 yellow hard hats remain. So, the total number of hard hats left in the truck is 19 + 24 = 43.", "success_count": 8, "success_rate": 1.0}], "chosen_candidate_index": 0, "chosen_step_text": "24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43.", "chosen_success_rate": 1.0}], "final_prefix": "You are an expert math problem solver. You must reason step by step and avoid logical or arithmetic mistakes.\n\nSolve the following math problem.\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`.\n\nProblem: In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.\n\nReasoning step by step:\nStep 1: We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.\nStep 2: 6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.\nStep 3: 24 yellow hard hats remain unchanged.\nStep 4: 16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.\nStep 5: 24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43.\n", "final_completion": "Answer: 43\n```\n\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`. \n\nHere is the solution:\nStep 1: We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.\nStep 2: 6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.\nStep 3: 24 yellow hard hats remain unchanged.\nStep 4: 16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.\nStep 5: 24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43", "final_full_output": "You are an expert math problem solver. You must reason step by step and avoid logical or arithmetic mistakes.\n\nSolve the following math problem.\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`.\n\nProblem: In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.\n\nReasoning step by step:\nStep 1: We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.\nStep 2: 6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.\nStep 3: 24 yellow hard hats remain unchanged.\nStep 4: 16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.\nStep 5: 24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43.\nAnswer: 43\n```\n\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`. \n\nHere is the solution:\nStep 1: We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.\nStep 2: 6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.\nStep 3: 24 yellow hard hats remain unchanged.\nStep 4: 16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.\nStep 5: 24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43", "final_answer": "43", "final_is_correct": true, "raw_sample": {"question": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.", "answer": "If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43", "problem": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.", "solution": "If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43"}}}


def _get_data(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    idx = text.find("data")
    if idx == -1:
        raise ValueError(f"No 'data' assignment found in {path}")
    eq = text.find("=", idx)
    if eq == -1:
        raise ValueError(f"No '=' found for data assignment in {path}")
    start = text.find("{", eq)
    if start == -1:
        raise ValueError(f"No '{{' found for data assignment in {path}")

    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f"Unbalanced braces in data assignment in {path}")

    data_str = text[start:end]
    try:
        data = ast.literal_eval(data_str)
    except Exception as exc:
        raise ValueError(f"Failed to parse data dict in {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Parsed data is not a dict in {path}")
    return data


def _print_steps(data: Dict[str, Any]) -> None:
    def _norm(text: object) -> str:
        if not isinstance(text, str):
            return ""
        # Handle strings that contain literal "\n" sequences.
        return text.replace("\\n", "\n")

    tot = data.get("tot") if isinstance(data.get("tot"), dict) else data
    problem = tot.get("problem") if isinstance(tot, dict) else None
    prompt = tot.get("prompt") if isinstance(tot, dict) else None
    if not isinstance(problem, str):
        raise ValueError("No problem found in data")

    print("Question:")
    print(problem)
    print()
    if isinstance(prompt, str) and prompt.strip():
        print("Prompt:")
        print(_norm(prompt))
        print()

    steps_trace = tot.get("steps_trace") if isinstance(tot, dict) else None
    if not isinstance(steps_trace, list):
        print("No steps_trace found.")
        return

    for step in steps_trace:
        if not isinstance(step, dict):
            continue
        step_index = step.get("step_index")
        print(f"Step {step_index}")
        candidates: List[Dict[str, Any]] = step.get("candidates") or []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            idx = cand.get("candidate_index")
            text = _norm(cand.get("step_text", ""))
            rate = cand.get("success_rate")
            print(f"[{idx}] {text} | success_rate={rate}")
        print()


def main() -> int:
    if isinstance(globals().get("data"), dict):
        data = globals()["data"]
    else:
        path = Path("eval_only.py")
        if not path.exists():
            raise FileNotFoundError("eval_only.py not found in current directory.")
        data = _get_data(path)
    _print_steps(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
