import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataset_utils import default_dataset_path, load_one_sample
from llama_api import llama_completion
from tool import extract_model_answer, steps_for_dataset, steps_for_level


def _extract_section(text: str, header: str) -> Optional[str]:
    pattern = rf"{header}\s*:(.*?)(?=\n(?:Refined Step|Draft Step|Final Step|Answer:)|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _format_steps_context(
    steps: Dict[int, Dict[str, Optional[str]]],
    upto: Optional[int] = None,
) -> str:
    lines = []
    for idx in sorted(steps.keys()):
        if upto is not None and idx > upto:
            continue
        entry = steps[idx]
        text = entry.get("refined") or entry.get("draft")
        if text:
            lines.append(f"Step {idx}:\n{text}")
    return "\n".join(lines)


def _build_prompt(
    problem: str,
    step_idx: int,
    num_steps: int,
    progress_text: str,
) -> str:
    system = (
        f"You are an expert math tutor. You work in exactly {num_steps} reasoning steps.\n"
        "Each round you must follow the requested output format verbatim."
    )
    if step_idx == 1:
        instruction = (
            "Draft the initial Step 1 for solving the problem. "
            "Only produce one section named 'Draft Step 1:'."
        )
    elif step_idx <= num_steps:
        instruction = (
            f"Refine the previous step (Step {step_idx - 1}) to fix mistakes and improve clarity.\n"
            f"Then propose a draft for Step {step_idx}.\n"
            f"Use the exact format:\n"
            f"Refined Step {step_idx - 1}:\n...\nDraft Step {step_idx}:\n..."
        )
    else:
        instruction = (
            f"Only refine the final step (Step {num_steps}). "
            f"Do not generate Step {num_steps + 1}. "
            f"Format:\nRefined Step {num_steps}:\n..."
        )

    progress_section = ""
    if progress_text.strip():
        progress_section = f"\nCurrent Progress:\n{progress_text}\n"

    prompt = (
        f"{system}\n\nProblem:\n{problem}\n"
        f"{progress_section}\nInstruction:\n{instruction}\n"
        "Begin your answer now.\n"
    )
    return prompt


def run_refine_chain(
    sample: Dict[str, str],
    model_dir: Optional[str],
    num_steps: int = 5,
    temperature: float = 0.3,
    verbose: bool = False,
) -> Dict[str, object]:
    problem = sample.get("problem") or sample.get("question") or sample.get("prompt")
    if not problem:
        raise ValueError("Sample missing problem/question field.")

    steps: Dict[int, Dict[str, Optional[str]]] = {}

    # Round 1: draft Step 1
    prompt = _build_prompt(problem, 1, num_steps, progress_text="")
    completion = llama_completion(
        prompt=prompt,
        model_dir=model_dir,
        temperature=temperature,
        max_new_tokens=256,
        generation_prefix=None,
    )
    if verbose:
        print("\n" + "=" * 40)
        print("Round 1 Prompt:\n", prompt)
        print("-" * 20)
        print("Round 1 Completion:\n", completion)
        print("=" * 40 + "\n")
    draft_step1 = _extract_section(completion, "Draft Step 1") or completion.strip()
    steps[1] = {"draft": draft_step1, "refined": None}

    prev_text = draft_step1
    history = [
        {
            "round": 1,
            "prompt": prompt,
            "completion": completion,
            "draft_step": draft_step1,
        }
    ]

    # Rounds 2..num_steps: refine previous, draft next
    for step_idx in range(2, num_steps + 1):
        progress = _format_steps_context(steps, upto=step_idx - 1)
        prompt = _build_prompt(problem, step_idx, num_steps, progress)
        completion = llama_completion(
            prompt=prompt,
            model_dir=model_dir,
            temperature=temperature,
            max_new_tokens=256,
            generation_prefix=None,
        )
        if verbose:
            print("\n" + "=" * 40)
            print(f"Round {step_idx} Prompt:\n", prompt)
            print("-" * 20)
            print(f"Round {step_idx} Completion:\n", completion)
            print("=" * 40 + "\n")
        refined = (
            _extract_section(completion, f"Refined Step {step_idx - 1}") or prev_text
        )
        draft = _extract_section(completion, f"Draft Step {step_idx}") or completion.strip()

        steps[step_idx - 1]["refined"] = refined
        steps[step_idx] = {"draft": draft, "refined": None}
        prev_text = draft

        history.append(
            {
                "round": step_idx,
                "prompt": prompt,
                "completion": completion,
                "refined_step": refined,
                "draft_step": draft,
            }
        )

    # Final refinement round for Step num_steps
    progress = _format_steps_context(steps, upto=num_steps)
    final_prompt = _build_prompt(problem, num_steps + 1, num_steps, progress)
    final_completion = llama_completion(
        prompt=final_prompt,
        model_dir=model_dir,
        temperature=temperature,
        max_new_tokens=256,
        generation_prefix=None,
    )
    if verbose:
        print("\n" + "=" * 40)
        print(f"Round {num_steps + 1} Prompt:\n", final_prompt)
        print("-" * 20)
        print(f"Round {num_steps + 1} Completion:\n", final_completion)
        print("=" * 40 + "\n")
    final_refined = (
        _extract_section(final_completion, f"Refined Step {num_steps}") or prev_text
    )
    steps[num_steps]["refined"] = final_refined
    history.append(
        {
            "round": num_steps + 1,
            "prompt": final_prompt,
            "completion": final_completion,
            "refined_step": final_refined,
        }
    )

    return {
        "problem": problem,
        "steps": steps,
        "history": history,
    }


def _build_final_answer_prompt(problem: str, refined_steps: Dict[int, Dict[str, Optional[str]]]) -> str:
    lines = []
    for idx in sorted(refined_steps.keys()):
        entry = refined_steps[idx]
        txt = (entry.get("refined") or entry.get("draft") or "").strip()
        if txt:
            lines.append(f"Step {idx}:\n{txt}")
    steps_block = "\n".join(lines)
    return (
        "You are an expert math problem solver.\n"
        "Given the problem and the reasoning steps, output the final answer.\n"
        "Output MUST end with a line of the form: Answer: <final_answer>\n\n"
        f"Problem:\n{problem}\n\n"
        f"Reasoning:\n{steps_block}\n\n"
        "Now provide the final answer.\n"
    )


def run_diffcot_on_single(
    sample: Dict[str, str],
    model_dir: Optional[str],
    num_steps: int = 5,
    temperature: float = 0.3,
    answer_temperature: float = 0.0,
    answer_max_new_tokens: int = 64,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    用 diffcot 的 refine-chain 先生成/修正 steps，再额外生成一次最终 Answer 行。
    返回的 answer_completion 可以直接用 tool.is_model_correct(answer_completion, solution) 判题。
    """
    chain = run_refine_chain(
        sample=sample,
        model_dir=model_dir,
        num_steps=num_steps,
        temperature=temperature,
        verbose=verbose,
    )
    problem = chain.get("problem") or ""
    steps = chain.get("steps") or {}
    if not isinstance(problem, str) or not isinstance(steps, dict):
        raise ValueError("Unexpected diffcot output structure.")

    answer_prompt = _build_final_answer_prompt(problem, steps)
    answer_completion = llama_completion(
        prompt=answer_prompt,
        model_dir=model_dir,
        temperature=answer_temperature,
        max_new_tokens=answer_max_new_tokens,
        top_p=0.95,
        generation_prefix=None,
    )
    return {
        "problem": problem,
        "steps": steps,
        "history": chain.get("history"),
        "answer_prompt": answer_prompt,
        "answer_completion": answer_completion,
        "answer": extract_model_answer(answer_completion),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the iterative refine-then-draft prompting scheme."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hendrycks_math",
        choices=["hendrycks_math", "gsm8k", "svamp"],
        help="Dataset name (default: hendrycks_math).",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Dataset path; if omitted uses built-in defaults for the dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split for gsm8k (default: train).",
    )
    parser.add_argument(
        "--model-dir",
        default="/data/jsg_data/model/meta-llama/llama3-8b",
        help="Local model directory (default: /data/jsg_data/model/meta-llama/llama3-8b).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="How many reasoning steps to iterate (default: svamp=3, gsm8k=5, hendrycks_math=5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3).",
    )
    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = default_dataset_path(args.dataset_name)
    sample = load_one_sample(args.dataset_name, args.dataset_path, split=args.split)
    if args.steps is None:
        if args.dataset_name == "hendrycks_math":
            args.steps = steps_for_level(sample.get("level"))
        else:
            args.steps = steps_for_dataset(args.dataset_name)
    out = run_refine_chain(
        sample=sample,
        model_dir=args.model_dir,
        num_steps=int(args.steps),
        temperature=args.temperature,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
