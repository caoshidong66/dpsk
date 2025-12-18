import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tool import is_model_correct

from diffcot.gen import run_diffcot_on_single


def _parse_level(level: object) -> Optional[int]:
    if level is None:
        return None
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        import re

        m = re.search(r"(\d+)", level)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _iter_hendrycks_samples(dataset_root: Path) -> Iterable[Dict[str, Any]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    jsonl_files = sorted(dataset_root.rglob("*.jsonl"))
    if jsonl_files:
        for path in jsonl_files:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        return

    json_files = sorted(dataset_root.rglob("*.json"))
    if json_files:
        for path in json_files:
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(data, dict):
                items = data.get("data")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            yield item
                else:
                    yield data
        return

    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    if parquet_files:
        try:
            from datasets import Dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Parquet dataset requires `datasets` (pip install datasets)."
            ) from exc

        for path in parquet_files:
            ds = Dataset.from_parquet(str(path))
            for row in ds:
                if isinstance(row, dict):
                    yield row
        return

    raise FileNotFoundError(
        f"No .jsonl/.json/.parquet files found under {dataset_root}"
    )


def _reservoir_sample_level(
    dataset_root: Path,
    level: int,
    k: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    for sample in _iter_hendrycks_samples(dataset_root):
        lvl = _parse_level(sample.get("level"))
        if lvl != level:
            continue
        seen += 1
        if len(reservoir) < k:
            reservoir.append(sample)
            continue
        j = rng.randrange(seen)
        if j < k:
            reservoir[j] = sample

    return reservoir, seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate diffcot/gen (refine chain) on MATH level-3 samples."
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/jsg_data/hendrycks_math",
        help="hendrycks_math dataset root.",
    )
    parser.add_argument(
        "--model-dir",
        default="/data/jsg_data/model/meta-llama/llama3-8b",
        help="Local model checkpoint directory.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="MATH level to evaluate (default: 3).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reservoir sampling seed (default: 42).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="How many refine steps to run (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generation temperature (default: 0.3).",
    )
    parser.add_argument(
        "--verbose-gen",
        action="store_true",
        help="Print diffcot/gen prompts & completions (very verbose).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Optional path to write per-sample results as JSONL.",
    )
    args = parser.parse_args()

    samples, total_level = _reservoir_sample_level(
        dataset_root=Path(args.dataset_root),
        level=args.level,
        k=args.num_samples,
        seed=args.seed,
    )
    if not samples:
        raise RuntimeError(f"No samples found for level={args.level}")
    if total_level < args.num_samples:
        print(f"[diffcot/testcode] Only found {total_level} level-{args.level} samples.")

    out_f = None
    if args.output_jsonl:
        out_f = Path(args.output_jsonl).open("w", encoding="utf-8")

    correct = 0
    for i, sample in enumerate(samples):
        problem = sample.get("problem") or sample.get("question") or sample.get("prompt")
        solution = sample.get("solution") or sample.get("answer")

        out = run_diffcot_on_single(
            sample=sample,
            model_dir=args.model_dir,
            num_steps=args.steps,
            temperature=args.temperature,
            verbose=args.verbose_gen,
        )
        is_ok = False
        if isinstance(solution, str) and solution:
            is_ok = is_model_correct(str(out.get("answer_completion") or ""), solution)
        if is_ok:
            correct += 1

        record = {
            "index": i,
            "level": sample.get("level"),
            "problem": problem,
            "model_answer": out.get("answer"),
            "answer_completion": out.get("answer_completion"),
            "is_correct": is_ok,
        }
        if out_f is not None:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

        print(f"[diffcot/testcode] {i + 1}/{len(samples)} correct={is_ok}")

    if out_f is not None:
        out_f.close()

    accuracy = correct / len(samples) if samples else 0.0
    summary = {"num_samples": len(samples), "num_correct": correct, "accuracy": accuracy}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
