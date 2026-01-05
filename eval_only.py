#!/usr/bin/env python3
"""
Standalone evaluation script (no DDP, no torch.distributed).

Runs deterministic decoding on a single GPU with optional data sharding:
  if sample_index % num_shards != shard_id: skip
"""
data = {"dataset_name": "gsm8k", "dataset_path": "../data/GSM8K", "split": "train", "index": 17, "gpu": "0,1,2,3,4,5,6,7", "rank": 0, "world_size": 1, "elapsed_sec": 591.2172243595123, "tot": {"problem": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.", "solution": "If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43", "level": null, "prompt": "You are an expert math problem solver. You must reason step by step and avoid logical or arithmetic mistakes.\n\nSolve the following math problem.\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`.\n\nProblem: In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.\n\nReasoning step by step:\nStep 1:", "steps_trace": [{"step_index": 1, "candidates": [{"candidate_index": 0, "step_text": "26 pink hard hats - 4 pink hard hats = 22 pink hard hats", "success_count": 0, "success_rate": 0.0}, {"candidate_index": 1, "step_text": "26 - 4 = ? (number of pink hats after Carl's removal)", "success_count": 1, "success_rate": 0.125}, {"candidate_index": 2, "step_text": "We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.", "success_count": 6, "success_rate": 0.75}, {"candidate_index": 3, "step_text": "Initial number of pink hard hats is 26.", "success_count": 4, "success_rate": 0.5}, {"candidate_index": 4, "step_text": "(Please write your reasoning in this line)", "success_count": 0, "success_rate": 0.0}], "chosen_candidate_index": 2, "chosen_step_text": "We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.", "chosen_success_rate": 0.75}, {"step_index": 2, "candidates": [{"candidate_index": 0, "step_text": "6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 1, "step_text": "6 pink hard hats are taken away by John, which means 6 pink hard hats are removed from the truck. We are left with 22 - 6 = 16 pink hard hats.", "success_count": 5, "success_rate": 0.625}, {"candidate_index": 2, "step_text": "6 pink hard hats were removed by John, and he also removed twice as many green hard hats as the number of pink hard hats he removed. So, 6 green hard hats were removed, leaving us with 15 - 6 = 9 green hard hats.", "success_count": 0, "success_rate": 0.0}, {"candidate_index": 3, "step_text": "6 pink hard hats are taken away by John, leaving us with 22 - 6 = 16 pink hard hats. Since John takes away twice as many green hard hats as the number of pink hard hats that he removed (i.e., 6 green hard hats), we have 15 - 6 = 9 green hard hats left.", "success_count": 1, "success_rate": 0.125}, {"candidate_index": 4, "step_text": "6 pink hard hats were taken away by John, which means 6 green hard hats were taken away by John, since John took away twice as many green hard hats as the number of pink hard hats he removed. Therefore, we have 15 - 6 = 9 green hard hats remaining.", "success_count": 1, "success_rate": 0.125}], "chosen_candidate_index": 0, "chosen_step_text": "6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.", "chosen_success_rate": 1.0}, {"step_index": 3, "candidates": [{"candidate_index": 0, "step_text": "24 yellow hard hats remain unchanged.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 1, "step_text": "24 yellow hard hats remain unchanged, so we still have 24 yellow hard hats.", "success_count": 6, "success_rate": 0.75}, {"candidate_index": 2, "step_text": "24 yellow hard hats remain unchanged, so we still have 24 yellow hard hats.", "success_count": 7, "success_rate": 0.875}, {"candidate_index": 3, "step_text": "24 yellow hard hats are left untouched. So, we have 24 yellow hard hats.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 4, "step_text": "24 yellow hard hats remain in the truck and are not affected by the removals.", "success_count": 7, "success_rate": 0.875}], "chosen_candidate_index": 0, "chosen_step_text": "24 yellow hard hats remain unchanged.", "chosen_success_rate": 1.0}, {"step_index": 4, "candidates": [{"candidate_index": 0, "step_text": "16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 1, "step_text": "16 pink hard hats and 3 green hard hats are left. 16 + 3 = 19.", "success_count": 7, "success_rate": 0.875}, {"candidate_index": 2, "step_text": "16 pink hard hats and 3 green hard hats are left. In total, we have 16 + 3 + 24 = 43 hard hats remaining.", "success_count": 7, "success_rate": 0.875}, {"candidate_index": 3, "step_text": "16 pink hard hats, 3 green hard hats, and 24 yellow hard hats are left in the truck.", "success_count": 6, "success_rate": 0.75}, {"candidate_index": 4, "step_text": "16 pink hard hats and 3 green hard hats are left, so the total number of hard hats left is 16 + 3 + 24 = 43.", "success_count": 7, "success_rate": 0.875}], "chosen_candidate_index": 0, "chosen_step_text": "16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.", "chosen_success_rate": 1.0}, {"step_index": 5, "candidates": [{"candidate_index": 0, "step_text": "24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 1, "step_text": "19 hats of these two colors and 24 yellow hard hats are left. So, the total number of hard hats remaining is 19 + 24 = 43.", "success_count": 7, "success_rate": 0.875}, {"candidate_index": 2, "step_text": "19 hats of these two colors are combined with the unchanged 24 yellow hard hats, giving us a total of 19 + 24 = 43 hard hats.", "success_count": 8, "success_rate": 1.0}, {"candidate_index": 3, "step_text": "3 green hard hats, 19 hats of pink and green, and 24 yellow hard hats are left in the truck, making a total of 3 + 19 + 24 = ?", "success_count": 0, "success_rate": 0.0}, {"candidate_index": 4, "step_text": "19 hats of these two colors and 24 yellow hard hats remain. So, the total number of hard hats left in the truck is 19 + 24 = 43.", "success_count": 8, "success_rate": 1.0}], "chosen_candidate_index": 0, "chosen_step_text": "24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43.", "chosen_success_rate": 1.0}], "final_prefix": "You are an expert math problem solver. You must reason step by step and avoid logical or arithmetic mistakes.\n\nSolve the following math problem.\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`.\n\nProblem: In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.\n\nReasoning step by step:\nStep 1: We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.\nStep 2: 6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.\nStep 3: 24 yellow hard hats remain unchanged.\nStep 4: 16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.\nStep 5: 24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43.\n", "final_completion": "Answer: 43\n```\n\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`. \n\nHere is the solution:\nStep 1: We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.\nStep 2: 6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.\nStep 3: 24 yellow hard hats remain unchanged.\nStep 4: 16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.\nStep 5: 24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43", "final_full_output": "You are an expert math problem solver. You must reason step by step and avoid logical or arithmetic mistakes.\n\nSolve the following math problem.\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`.\n\nProblem: In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.\n\nReasoning step by step:\nStep 1: We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.\nStep 2: 6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.\nStep 3: 24 yellow hard hats remain unchanged.\nStep 4: 16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.\nStep 5: 24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43.\nAnswer: 43\n```\n\nYou MUST use exactly 5 reasoning steps, After the reasoning, output the final answer in the last line using the format: `Answer: <final_answer>`. \n\nHere is the solution:\nStep 1: We have 26 pink hard hats initially. Carl takes away 4 pink hard hats, leaving us with 26 - 4 = 22 pink hard hats.\nStep 2: 6 pink hard hats are taken away by John. So, we have 22 - 6 = 16 pink hard hats left. John also takes away twice as many green hard hats as the number of pink hard hats he removed, which is 2 x 6 = 12 green hard hats. So, we have 15 - 12 = 3 green hard hats left.\nStep 3: 24 yellow hard hats remain unchanged.\nStep 4: 16 pink hard hats and 3 green hard hats are left. So, we have 16 + 3 = 19 hats of these two colors.\nStep 5: 24 yellow hard hats and 19 hats of pink and green colors are left. So, the total number of hard hats remaining in the truck is 24 + 19 = 43", "final_answer": "43", "final_is_correct": true, "raw_sample": {"question": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.", "answer": "If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43", "problem": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats.  If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.", "solution": "If there were 26 pink hard hats and Carl took away 4 pink hard hats, the number of pink hard hats that remained is 26-4 = <<26-4=22>>22\nJohn also took away 6 pink hard hats, leaving 22-6 = <<22-6=16>>16 pink hard hats in the truck.\nIf John also took twice as many green hard hats as pink hard hats, he took 2*6 = <<6*2=12>>12 green hard hats.\nThe total number of green hard hats that remained in the truck is 15-12 = <<15-12=3>>3\nIn the truck, after some are taken, there were 3 green hard hats + 16 pink hard hats = <<3+16=19>>19 hard hats in the truck.\nAltogether, 19 green and pink hard hats + 24 yellow hards hats = <<19+24=43>>43 hard hats remained in the truck\n#### 43"}}}
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from dataset_utils import default_dataset_path, iter_samples, normalize_sample
from tool import is_model_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone deterministic eval (single GPU).")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=["gsm8k", "svamp", "hendrycks_math"],
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-cache", type=str, default=None)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--lora-dir", type=str, default=None)
    parser.add_argument("--lora-name", type=str, default="eval_lora")
    parser.add_argument("--lora-scaling", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output.",
    )
    return parser.parse_args()


def _load_or_create_indices(
    *,
    id_cache: Optional[str],
    dataset_name: str,
    dataset_root: str,
    split: str,
    level: Optional[int],
    num_samples: Optional[int],
    seed: int,
) -> Optional[List[int]]:
    if id_cache is None and num_samples is None:
        return None

    cache_path = Path(id_cache) if id_cache else None
    if cache_path is None:
        cache_path = None
    if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        indices = data.get("indices") if isinstance(data, dict) else None
        if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
            return indices[: num_samples] if num_samples is not None else indices

    if num_samples is None:
        raise FileNotFoundError(f"--id-cache not found: {cache_path}")

    indices: List[int] = []
    for idx, raw in enumerate(iter_samples(dataset_name, dataset_root, split=split)):
        sample = normalize_sample(dataset_name, raw)
        if dataset_name == "hendrycks_math" and level is not None:
            lvl = sample.get("level")
            if isinstance(lvl, str):
                if str(level) not in lvl:
                    continue
            elif isinstance(lvl, int):
                if int(lvl) != level:
                    continue
            else:
                continue
        indices.append(idx)

    rng = random.Random(int(seed))
    rng.shuffle(indices)
    indices = indices[: int(num_samples)]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump({"indices": indices}, f, ensure_ascii=False, indent=2)
    return indices


def _iter_samples(
    *,
    dataset_name: str,
    dataset_root: str,
    split: str,
    level: Optional[int],
    indices: Optional[List[int]],
) -> Iterable[Dict[str, object]]:
    if indices is not None:
        wanted = set(indices)
        for idx, raw in enumerate(iter_samples(dataset_name, dataset_root, split=split)):
            if idx not in wanted:
                continue
            yield normalize_sample(dataset_name, raw)
        return

    for raw in iter_samples(dataset_name, dataset_root, split=split):
        sample = normalize_sample(dataset_name, raw)
        if dataset_name == "hendrycks_math" and level is not None:
            lvl = sample.get("level")
            if isinstance(lvl, str):
                if str(level) not in lvl:
                    continue
            elif isinstance(lvl, int):
                if int(lvl) != level:
                    continue
            else:
                continue
        yield sample


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    dataset_root = args.dataset_root or default_dataset_path(args.dataset_name)
    indices = _load_or_create_indices(
        id_cache=args.id_cache,
        dataset_name=args.dataset_name,
        dataset_root=dataset_root,
        split=args.split,
        level=args.level,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if indices is not None:
        total_expected = sum(
            1 for i in range(len(indices)) if i % args.num_shards == args.shard_id
        )
    else:
        total_expected = None

    progress = None
    if not args.no_progress:
        try:
            from tqdm import tqdm  # type: ignore

            progress = tqdm(total=total_expected, desc="Eval", unit="sample")
        except Exception:
            progress = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enable_lora = args.lora_dir is not None
    llm = LLM(
        model=args.model_dir,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_lora=enable_lora,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    lora_request = None
    if args.lora_dir:
        try:
            lora_request = LoRARequest(
                lora_name=args.lora_name,
                lora_id=1,
                lora_path=args.lora_dir,
                lora_scale=args.lora_scaling,
            )
        except TypeError:
            try:
                lora_request = LoRARequest(
                    lora_name=args.lora_name,
                    lora_id=1,
                    lora_path=args.lora_dir,
                    lora_scaling=args.lora_scaling,
                )
            except TypeError:
                lora_request = LoRARequest(args.lora_name, 1, args.lora_dir)

    total = 0
    correct = 0
    processed = 0
    batch_prompts: List[str] = []
    batch_solutions: List[Optional[str]] = []

    def _flush_batch() -> None:
        nonlocal total, correct, processed, batch_prompts, batch_solutions
        if not batch_prompts:
            return
        outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        for out, solution in zip(outputs, batch_solutions):
            completion = out.outputs[0].text if out.outputs else ""
            if solution is not None and is_model_correct(completion, solution):
                correct += 1
            total += 1
            processed += 1
            if progress is not None:
                progress.update(1)
            else:
                print(f"[eval] shard {args.shard_id} processed {processed}", flush=True)
        batch_prompts = []
        batch_solutions = []

    for sample_idx, sample in enumerate(
        _iter_samples(
            dataset_name=args.dataset_name,
            dataset_root=dataset_root,
            split=args.split,
            level=args.level,
            indices=indices,
        )
    ):
        if sample_idx % args.num_shards != args.shard_id:
            continue
        problem = sample.get("problem") or sample.get("question") or sample.get("prompt")
        if not problem:
            continue
        solution = sample.get("solution") or sample.get("answer")
        prompt = f"Solve the problem and give the final answer.\nProblem: {problem}\nAnswer:"
        batch_prompts.append(prompt)
        batch_solutions.append(solution)
        if len(batch_prompts) >= int(args.batch_size):
            _flush_batch()

    _flush_batch()

    acc = (correct / total) if total > 0 else None
    result = {"num_samples": total, "num_correct": correct, "accuracy": acc}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if progress is not None:
        progress.close()

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
