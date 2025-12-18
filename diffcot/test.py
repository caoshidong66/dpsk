import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataset_utils import default_dataset_path, iter_samples, normalize_sample
from llama_api import llama_completion
from tool import is_model_correct, steps_for_dataset, steps_for_level

from diffcot.gen import run_refine_chain


def _reservoir_samples(
    dataset_name: str,
    dataset_path: Union[str, Path],
    split: str,
    k: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, object]] = []
    for i, raw in enumerate(iter_samples(dataset_name, dataset_path, split=split), start=1):
        sample = normalize_sample(dataset_name, raw)
        if len(reservoir) < k:
            reservoir.append(sample)
            continue
        j = rng.randint(1, i)
        if j <= k:
            reservoir[j - 1] = sample
    return reservoir


def evaluate_model_on_samples(
    dataset_name: str,
    dataset_path: Path,
    split: str,
    model_dir: Optional[str],
    num_samples: int,
    temperature: float,
    steps: Optional[int],
    seed: int,
) -> Dict[str, object]:
    correct = 0
    results: List[Dict[str, object]] = []

    samples = _reservoir_samples(dataset_name, dataset_path, split, num_samples, seed=seed)
    for idx, sample in enumerate(samples):
        if steps is None:
            if dataset_name == "hendrycks_math":
                steps_eff = steps_for_level(sample.get("level"))
            else:
                steps_eff = steps_for_dataset(dataset_name)
        else:
            steps_eff = steps
        out = run_refine_chain(
            sample=sample,  # type: ignore[arg-type]
            model_dir=model_dir,
            num_steps=steps_eff,
            temperature=temperature,
        )
        final_text = out["steps"][steps_eff]["refined"]
        solution = sample.get("solution") or sample.get("answer")
        is_correct = False
        if isinstance(final_text, str) and solution:
            is_correct = is_model_correct(final_text, solution)
        if is_correct:
            correct += 1

        out_record = {
            "index": idx,
            "problem": sample.get("problem"),
            "final_step": final_text,
            "is_correct": is_correct,
            "raw": out,
        }
        results.append(out_record)
        print(f"[diffcot/test] Sample {idx + 1}/{num_samples} -> correct={is_correct}")

    accuracy = correct / num_samples if num_samples > 0 else 0.0
    summary = {
        "num_samples": num_samples,
        "num_correct": correct,
        "accuracy": accuracy,
    }
    return {"summary": summary, "details": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate refined-step model on random samples (supports dataset switch)."
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
        default="test",
        choices=["train", "test"],
        help="Split for gsm8k (default: test).",
    )
    parser.add_argument(
        "--model-dir",
        default="/data/jsg_data/model/meta-llama/llama3-8b",
        help="Local model checkpoint (default: LLaMA).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Evaluation sample count (default: 50).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of reasoning steps (default: svamp=3, gsm8k=5, hendrycks_math=by level).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = default_dataset_path(args.dataset_name)

    results = evaluate_model_on_samples(
        dataset_name=args.dataset_name,
        dataset_path=Path(args.dataset_path),
        split=args.split,
        model_dir=args.model_dir,
        num_samples=args.num_samples,
        temperature=args.temperature,
        steps=args.steps,
        seed=args.seed,
    )
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
