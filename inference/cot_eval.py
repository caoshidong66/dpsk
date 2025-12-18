import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from cot_math import _load_one_hendrycks_sample  # type: ignore
from llama_cot_math import run_llama_cot_on_single
from tool import is_model_correct


def evaluate_cot_model(
    dataset_root: Path,
    model_dir: Optional[str],
    num_samples: int,
    temperature: float,
    use_vllm: bool = False,
) -> Dict[str, object]:
    correct = 0
    results = []

    for idx in range(num_samples):
        sample = _load_one_hendrycks_sample(dataset_root)
        out = run_llama_cot_on_single(
            dataset_root=None,
            model_dir=model_dir,
            sample=sample,
            temperature=temperature,
            use_vllm=use_vllm,
        )
        completion = out.get("completion") or ""
        solution = out.get("raw_sample", {}).get("solution") or out.get("raw_sample", {}).get("answer")
        is_correct = False
        if completion and solution:
            is_correct = is_model_correct(completion, solution)
        if is_correct:
            correct += 1

        results.append(
            {
                "index": idx,
                "problem": sample.get("problem"),
                "completion": completion,
                "is_correct": is_correct,
            }
        )
        print(f"[cot_eval] sample {idx + 1}/{num_samples}, correct={is_correct}")

    accuracy = correct / num_samples if num_samples > 0 else 0.0
    return {
        "summary": {
            "num_samples": num_samples,
            "num_correct": correct,
            "accuracy": accuracy,
        },
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CoT model (trained via training scripts) on hendrycks_math samples."
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/jsg_data/hendrycks_math",
        help="Root directory of hendrycks_math dataset.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Checkpoint directory of the SFT/CPO CoT model to evaluate.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for evaluation (default: 0.3).",
    )
    args = parser.parse_args()

    results = evaluate_cot_model(
        dataset_root=Path(args.dataset_root),
        model_dir=args.model_dir,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
