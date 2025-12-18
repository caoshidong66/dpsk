import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from api import client, ark_chat_with_stop
from cot_math import build_completion_prompt
from dataset_utils import default_dataset_path, load_one_sample
from tool import extract_model_answer, extract_gt_answer, is_model_correct, steps_for_dataset, steps_for_level


def run_tot_on_single_sample(
    dataset_name: str = "hendrycks_math",
    dataset_path: Optional[str | Path] = None,
    split: str = "train",
    model: str = "deepseek-v3-2-251201",
    num_branches: int = 5,
    num_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    在一条 hendrycks_math 样本上跑简单的 Tree-of-Thought（多分支采样 + 多数表决）。
    """
    sample = load_one_sample(dataset_name=dataset_name, dataset_path=dataset_path, split=split)

    problem = (
        sample.get("problem")
        or sample.get("question")
        or sample.get("prompt")
    )
    if not problem:
        raise ValueError("Sample does not contain a 'problem' / 'question' field.")

    solution = sample.get("solution") or sample.get("answer")
    level = sample.get("level")
    if num_steps is None:
        if dataset_name.strip().lower() in {"gsm8k", "svamp"}:
            num_steps = steps_for_dataset(dataset_name)
        else:
            num_steps = steps_for_level(level)

    prompt = build_completion_prompt(problem, num_steps=num_steps)
    stop_token = "</END>"

    branches: List[Dict[str, Any]] = []
    for i in range(num_branches):
        completion = ark_chat_with_stop(
            client=client,
            model=model,
            messages=[{"role": "assistant", "content": prompt}],
            stop_token=stop_token,
            thinking="disabled",
            temperature=0.7,
        )

        answer = extract_model_answer(completion)
        correct = False
        if solution is not None:
            correct = is_model_correct(completion, solution)

        branches.append(
            {
                "index": i,
                "completion": completion,
                "answer": answer,
                "is_correct": correct,
            }
        )

    # 多数表决：统计非空答案的出现频次
    answers = [b["answer"] for b in branches if b["answer"]]
    majority_answer: Optional[str] = None
    majority_count = 0
    majority_correct: Optional[bool] = None
    if answers:
        counter = Counter(answers)
        majority_answer, majority_count = counter.most_common(1)[0]
        if solution is not None:
            # 用同样的判定逻辑检查多数答案是否正确
            majority_correct = is_model_correct(f"Answer: {majority_answer}", solution)

    result: Dict[str, Any] = {
        "problem": problem,
        "solution": solution,
        "level": level,
        "prompt": prompt,
        "branches": branches,
        "majority_answer": majority_answer,
        "majority_count": majority_count,
        "majority_is_correct": majority_correct,
        "raw_sample": sample,
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="在单个样本上运行 Tree-of-Thought 推理（支持切换数据集）"
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
        "--num-steps",
        type=int,
        default=None,
        help="Override reasoning steps; defaults: svamp=3, gsm8k=5, hendrycks_math=by level.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v3-2-251201",
        help="Ark 模型名称（默认：deepseek-v3-2-251201）",
    )
    parser.add_argument(
        "--branches",
        type=int,
        default=5,
        help="Tree-of-Thought 的分支数（默认：5）",
    )

    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = default_dataset_path(args.dataset_name)

    tot_result = run_tot_on_single_sample(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        model=args.model,
        num_branches=args.branches,
        num_steps=args.num_steps,
    )
    print(json.dumps(tot_result, ensure_ascii=False, indent=2))
