#!/usr/bin/env python3
"""
Standalone evaluation script (no DDP, no torch.distributed).

Runs deterministic decoding on a single GPU with optional data sharding:
  if sample_index % num_shards != shard_id: skip
"""

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
        outputs = llm.generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        completion = outputs[0].outputs[0].text
        if solution is not None and is_model_correct(completion, solution):
            correct += 1
        total += 1
        processed += 1
        if progress is not None:
            progress.update(1)
        else:
            print(f"[eval] shard {args.shard_id} processed {processed}", flush=True)

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
