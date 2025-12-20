from __future__ import annotations

import argparse
import json
import os
import re
import random
import time
from datetime import datetime
from multiprocessing import get_context, Manager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_utils import default_dataset_path, iter_samples, normalize_sample
from llama_tot_math import run_llama_tot_on_batch, run_llama_tot_on_single


def _parse_gpus(value: str) -> List[str]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--gpus is empty")
    return parts


def _parse_level(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"(\d+)", value)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _normalize_type(value: object) -> str:
    if value is None:
        return "unknown"
    s = str(value).strip()
    return s or "unknown"


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _reservoir_sample_hendrycks_level(
    dataset_path: str,
    *,
    split: str,
    level: int,
    k: int,
    seed: int,
    start_index: int = 0,
    end_index: Optional[int] = None,
    allowed_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0
    for idx, raw in enumerate(iter_samples("hendrycks_math", dataset_path, split=split)):
        if idx < start_index:
            continue
        if end_index is not None and idx >= end_index:
            break
        lvl = _parse_level(raw.get("level"))
        if lvl != level:
            continue
        typ = _normalize_type(raw.get("type"))
        if allowed_types is not None and typ not in allowed_types:
            continue
        seen += 1
        sample = normalize_sample("hendrycks_math", raw)
        if len(reservoir) < k:
            reservoir.append({"index": idx, "sample": sample})
            continue
        j = rng.randint(1, seen)
        if j <= k:
            reservoir[j - 1] = {"index": idx, "sample": sample}
    if len(reservoir) < k:
        print(f"[collect_tot] Only found {len(reservoir)} samples for level={level} (requested {k}).")
    return reservoir


def _balanced_sample_hendrycks_level_by_type(
    dataset_path: str,
    *,
    split: str,
    level: int,
    k: int,
    seed: int,
    start_index: int = 0,
    end_index: Optional[int] = None,
    allowed_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    # Pass 1: count available per type for the given level
    counts: Dict[str, int] = {}
    for idx, raw in enumerate(iter_samples("hendrycks_math", dataset_path, split=split)):
        if idx < start_index:
            continue
        if end_index is not None and idx >= end_index:
            break
        lvl = _parse_level(raw.get("level"))
        if lvl != level:
            continue
        typ = _normalize_type(raw.get("type"))
        if allowed_types is not None and typ not in allowed_types:
            continue
        counts[typ] = counts.get(typ, 0) + 1

    if not counts:
        print(f"[collect_tot] No samples found for level={level}.")
        return []

    types_sorted = sorted(counts.keys())
    t = len(types_sorted)
    base = k // t
    rem = k % t
    quotas: Dict[str, int] = {}
    for i, typ in enumerate(types_sorted):
        quotas[typ] = base + (1 if i < rem else 0)

    # Pass 2: reservoir-sample within each type up to its quota
    rng = random.Random(seed)
    reservoirs: Dict[str, List[Dict[str, Any]]] = {typ: [] for typ in types_sorted}
    seen_per_type: Dict[str, int] = {typ: 0 for typ in types_sorted}

    for idx, raw in enumerate(iter_samples("hendrycks_math", dataset_path, split=split)):
        if idx < start_index:
            continue
        if end_index is not None and idx >= end_index:
            break
        lvl = _parse_level(raw.get("level"))
        if lvl != level:
            continue
        typ = _normalize_type(raw.get("type"))
        if typ not in quotas:
            continue

        quota = quotas[typ]
        if quota <= 0:
            continue

        seen_per_type[typ] += 1
        sample = normalize_sample("hendrycks_math", raw)
        bucket = reservoirs[typ]
        if len(bucket) < quota:
            bucket.append({"index": idx, "sample": sample})
            continue
        j = rng.randint(1, seen_per_type[typ])
        if j <= quota:
            bucket[j - 1] = {"index": idx, "sample": sample}

    selected: List[Dict[str, Any]] = []
    for typ in types_sorted:
        bucket = reservoirs[typ]
        if len(bucket) < quotas[typ]:
            print(
                f"[collect_tot] Only found {len(bucket)} samples for type={typ}, level={level} "
                f"(requested {quotas[typ]})."
            )
        selected.extend(bucket)

    rng.shuffle(selected)
    if len(selected) < k:
        print(f"[collect_tot] Only selected {len(selected)} total samples (requested {k}).")
    return selected


def _worker_main(
    *,
    rank: int,
    world_size: int,
    gpu_id: str,
    dataset_name: str,
    dataset_path: str,
    split: str,
    output_path: str,
    selected: Optional[List[Dict[str, Any]]],
    start_index: int,
    end_index: Optional[int],
    max_samples: Optional[int],
    model_dir: Optional[str],
    branches: int,
    rollouts_per_candidate: int,
    temperature: float,
    use_vllm: bool,
    rollout_batch_size: int,
    num_steps: Optional[int],
    sample_batch_size: int,
    progress_total: Optional[int],
    progress_counter,
    progress_lock,
    progress_rank0: bool,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    out_fp = Path(output_path)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    def _progress_line(processed_count: int, total_count: Optional[int]) -> None:
        if not progress_rank0:
            return
        if total_count is None or total_count <= 0:
            msg = f"[collect_tot] processed={processed_count}"
        else:
            msg = (
                f"[collect_tot] processed={processed_count}/{total_count} "
                f"({processed_count / total_count:.1%})"
            )
        print("\r" + msg, end="", flush=True)

    def _finish_progress() -> None:
        if progress_rank0:
            print("", flush=True)

    def _bump_progress(delta: int) -> None:
        if progress_counter is None:
            return
        with progress_lock:
            progress_counter.value += delta
            _progress_line(progress_counter.value, progress_total)

    def _estimate_total_samples() -> Optional[int]:
        if max_samples is not None:
            max_samples_i = int(max_samples)
            if max_samples_i > 0:
                return max_samples_i
        try:
            if dataset_name == "gsm8k":
                base = Path(dataset_path)
                file_path = base / f"{split}.jsonl" if base.is_dir() else base
                with file_path.open("r", encoding="utf-8") as f_count:
                    return sum(1 for _ in f_count)
            if dataset_name == "svamp":
                path = Path(dataset_path)
                with path.open("r", encoding="utf-8") as f_count:
                    data = json.load(f_count)
                if isinstance(data, list):
                    return len(data)
                if isinstance(data, dict) and isinstance(data.get("data"), list):
                    return len(data["data"])
            if dataset_name == "hendrycks_math":
                root = Path(dataset_path)
                jsonl_files = list(root.rglob("*.jsonl"))
                json_files = list(root.rglob("*.json")) if not jsonl_files else []
                if jsonl_files:
                    total = 0
                    for fp in jsonl_files:
                        with fp.open("r", encoding="utf-8") as f_count:
                            total += sum(1 for _ in f_count)
                    return total
                if json_files:
                    total = 0
                    for fp in json_files:
                        with fp.open("r", encoding="utf-8") as f_count:
                            data = json.load(f_count)
                        if isinstance(data, list):
                            total += len(data)
                        elif isinstance(data, dict) and isinstance(data.get("data"), list):
                            total += len(data["data"])
                        else:
                            total += 1
                    return total
        except Exception:
            return None
        return None

    with out_fp.open("a", encoding="utf-8") as f:
        processed = 0
        if selected is not None:
            batch_entries: List[Dict[str, Any]] = []
            for pos, entry in enumerate(selected):
                if pos % world_size != rank:
                    continue
                batch_entries.append(entry)
                if len(batch_entries) < sample_batch_size:
                    continue

                t0 = time.time()
                outputs = run_llama_tot_on_batch(
                    samples=[e["sample"] for e in batch_entries],
                    model_dir=model_dir,
                    num_step_candidates=branches,
                    rollouts_per_candidate=rollouts_per_candidate,
                    temperature=temperature,
                    use_vllm=use_vllm,
                    rollout_batch_size=rollout_batch_size,
                    num_steps=num_steps,
                )
                elapsed = time.time() - t0
                for entry_item, out in zip(batch_entries, outputs):
                    idx = int(entry_item["index"])
                    record = {
                        "dataset_name": dataset_name,
                        "dataset_path": str(dataset_path),
                        "split": split,
                        "index": idx,
                        "gpu": str(gpu_id),
                        "rank": rank,
                        "world_size": world_size,
                        "elapsed_sec": elapsed,
                        "tot": out,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += 1
                f.flush()
                _bump_progress(len(batch_entries))
                batch_entries = []
            return

        total_all = _estimate_total_samples()

        batch_samples: List[Dict[str, Any]] = []
        batch_indices: List[int] = []
        for idx, raw in enumerate(iter_samples(dataset_name, dataset_path, split=split)):
            if idx < start_index:
                continue
            if end_index is not None and idx >= end_index:
                break
            if max_samples is not None and idx >= start_index + max_samples:
                break
            if (idx - start_index) % world_size != rank:
                continue

            sample = normalize_sample(dataset_name, raw)
            batch_samples.append(sample)
            batch_indices.append(idx)
            if len(batch_samples) < sample_batch_size:
                continue

            t0 = time.time()
            outputs = run_llama_tot_on_batch(
                samples=batch_samples,
                model_dir=model_dir,
                num_step_candidates=branches,
                rollouts_per_candidate=rollouts_per_candidate,
                temperature=temperature,
                use_vllm=use_vllm,
                rollout_batch_size=rollout_batch_size,
                num_steps=num_steps,
            )
            elapsed = time.time() - t0
            for out_idx, out in zip(batch_indices, outputs):
                record: Dict[str, Any] = {
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "split": split,
                    "index": out_idx,
                    "gpu": str(gpu_id),
                    "rank": rank,
                    "world_size": world_size,
                    "elapsed_sec": elapsed,
                    "tot": out,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
            f.flush()
            _bump_progress(len(batch_indices))
            batch_samples = []
            batch_indices = []

        if batch_samples:
            t0 = time.time()
            outputs = run_llama_tot_on_batch(
                samples=batch_samples,
                model_dir=model_dir,
                num_step_candidates=branches,
                rollouts_per_candidate=rollouts_per_candidate,
                temperature=temperature,
                use_vllm=use_vllm,
                rollout_batch_size=rollout_batch_size,
                num_steps=num_steps,
            )
            elapsed = time.time() - t0
            for out_idx, out in zip(batch_indices, outputs):
                record = {
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "split": split,
                    "index": out_idx,
                    "gpu": str(gpu_id),
                    "rank": rank,
                    "world_size": world_size,
                    "elapsed_sec": elapsed,
                    "tot": out,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
            f.flush()
            _bump_progress(len(batch_indices))


def merge_jsonl(
    input_paths: List[Path],
    output_path: Path,
    *,
    sort_by_index: bool = False,
) -> None:
    records: List[Dict[str, Any]] = []
    if sort_by_index:
        for p in input_paths:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        records.sort(key=lambda r: (r.get("index", -1), str(r.get("gpu", ""))))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as out_f:
            for r in records:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for p in input_paths:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    out_f.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU ToT data collection (per-GPU JSONL with optional merge)."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="gsm8k",
        choices=["hendrycks_math", "gsm8k", "svamp"],
        help="Dataset name (default: gsm8k).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
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
        "--level",
        type=int,
        default=None,
        help="Only for hendrycks_math: sample questions with this difficulty level (e.g. 3).",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="Only for hendrycks_math: comma-separated list of problem types to include (e.g. Algebra,Geometry).",
    )
    parser.add_argument(
        "--balance-by-type",
        action="store_true",
        help="Only for hendrycks_math: balance sampled questions across `type` categories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override reasoning steps; defaults: svamp=3, gsm8k=5, hendrycks_math=by level.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local model directory; if omitted `llama_tot_math.py` defaults apply.",
    )

    parser.add_argument(
        "--branches",
        type=int,
        default=5,
        help="Candidates per step (default: 4).",
    )
    parser.add_argument(
        "--rollouts-per-candidate",
        type=int,
        default=5,
        help="Rollouts per candidate (default: 4).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.5).",
    )
    parser.add_argument(
        "--no-vllm",
        dest="use_vllm",
        action="store_false",
        help="Disable vLLM and use transformers backend.",
    )
    parser.set_defaults(use_vllm=True)
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=16,
        help="vLLM rollout stage batch size (default: 16).",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help="How many samples to process in parallel per GPU (default: 1).",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU ids to use (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datas",
        help="Directory to write per-GPU JSONL files (default: datas).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="tot",
        help="Output prefix for per-GPU files (default: tot).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier appended to filenames; default is a timestamp like 20251213_235959.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index (inclusive) in the dataset stream (default: 0).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index (exclusive) in the dataset stream (default: none).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to process from start-index (default: none).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge per-GPU outputs into a single JSONL (auto-named under output-dir).",
    )
    parser.add_argument(
        "--merge-out",
        type=str,
        default=None,
        help="If set, merge all per-GPU outputs into this JSONL after completion.",
    )
    parser.add_argument(
        "--merge-sort",
        action="store_true",
        help="When merging, sort by `index` (loads all records into memory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gpus = _parse_gpus(args.gpus)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset_path or default_dataset_path(args.dataset_name)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{args.output_prefix}.{run_id}"
    selected: Optional[List[Dict[str, Any]]] = None
    if args.level is not None:
        if args.dataset_name != "hendrycks_math":
            raise ValueError("--level is only supported when --dataset-name=hendrycks_math")
        if args.max_samples is None:
            raise ValueError("When using --level, please also set --max-samples (e.g. 300).")
        allowed_types = _parse_csv(args.types)
        if args.balance_by_type:
            selected = _balanced_sample_hendrycks_level_by_type(
                dataset_path,
                split=args.split,
                level=int(args.level),
                k=int(args.max_samples),
                seed=int(args.seed),
                start_index=int(args.start_index),
                end_index=args.end_index,
                allowed_types=allowed_types,
            )
        else:
            selected = _reservoir_sample_hendrycks_level(
                dataset_path,
                split=args.split,
                level=int(args.level),
                k=int(args.max_samples),
                seed=int(args.seed),
                start_index=int(args.start_index),
                end_index=args.end_index,
                allowed_types=allowed_types,
            )

    ctx = get_context("spawn")
    manager = Manager()
    progress_counter = manager.Value("i", 0)
    progress_lock = manager.Lock()
    progress_total = None
    if selected is not None:
        progress_total = len(selected)
    else:
        try:
            if args.max_samples is not None:
                progress_total = int(args.max_samples)
        except Exception:
            progress_total = None
    procs = []
    for rank, gpu_id in enumerate(gpus):
        out_path = output_dir / f"{output_prefix}.gpu{gpu_id}.jsonl"
        p = ctx.Process(
            target=_worker_main,
            kwargs={
                "rank": rank,
                "world_size": len(gpus),
                "gpu_id": gpu_id,
                "dataset_name": args.dataset_name,
                "dataset_path": dataset_path,
                "split": args.split,
                "output_path": str(out_path),
                "selected": selected,
                "start_index": args.start_index,
                "end_index": args.end_index,
                "max_samples": args.max_samples,
                "model_dir": args.model_dir,
                "branches": args.branches,
                "rollouts_per_candidate": args.rollouts_per_candidate,
                "temperature": args.temperature,
                "use_vllm": args.use_vllm,
                "rollout_batch_size": args.rollout_batch_size,
                "num_steps": args.num_steps,
                "sample_batch_size": args.sample_batch_size,
                "progress_total": progress_total,
                "progress_counter": progress_counter,
                "progress_lock": progress_lock,
                "progress_rank0": rank == 0,
            },
        )
        p.start()
        procs.append(p)
        print(f"[collect_tot] started rank={rank} gpu={gpu_id} -> {out_path}")

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"Worker exited with code {p.exitcode}")

    merge_out = args.merge_out
    if merge_out is None and args.merge:
        merge_out = str(output_dir / f"{output_prefix}.all.jsonl")
    if merge_out:
        inputs = [output_dir / f"{output_prefix}.gpu{gpu}.jsonl" for gpu in gpus]
        merge_jsonl(inputs, Path(merge_out), sort_by_index=args.merge_sort)
        print(f"[collect_tot] merged -> {merge_out}")


if __name__ == "__main__":
    main()
