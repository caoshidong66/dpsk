from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union


def default_dataset_path(dataset_name: str) -> str:
    name = dataset_name.strip().lower()
    if name in {"hendrycks", "hendrycks_math", "math"}:
        return "/data/jsg_data/hendrycks_math"
    if name == "gsm8k":
        return "/data/jsg_data/GSM8K"
    if name == "svamp":
        return "/data/jsg_data/SVAMP.json"
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _iter_json(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if isinstance(item, dict):
                    yield item
            return
        yield data


def iter_samples(
    dataset_name: str,
    dataset_path: Optional[Union[str, Path]] = None,
    *,
    split: str = "train",
) -> Iterator[Dict[str, Any]]:
    """
    Yield raw samples for supported datasets.

    - hendrycks_math: dataset_path is a directory; will rglob for *.jsonl/*.json/*.parquet.
    - gsm8k: dataset_path is a directory containing train.jsonl/test.jsonl.
    - svamp: dataset_path is a JSON file (default /data/jsg_data/SVAMP.json).
    """
    name = dataset_name.strip().lower()
    path = Path(dataset_path) if dataset_path is not None else Path(default_dataset_path(name))

    if name in {"hendrycks", "hendrycks_math", "math"}:
        if not path.exists():
            raise FileNotFoundError(f"hendrycks_math root not found: {path}")
        # If the dataset root has train/test subfolders, honor `split`.
        if path.is_dir() and split in {"train", "test"}:
            train_dir = path / "train"
            test_dir = path / "test"
            if train_dir.exists() and test_dir.exists():
                path = train_dir if split == "train" else test_dir
        # Some local dumps mix .jsonl and per-sample .json files; read both.
        jsonl_files = sorted(path.rglob("*.jsonl"))
        json_files = sorted(path.rglob("*.json"))
        if jsonl_files:
            for fp in jsonl_files:
                yield from _iter_jsonl(fp)
        if json_files:
            for fp in json_files:
                yield from _iter_json(fp)
        if jsonl_files or json_files:
            return
        parquet_files = sorted(path.rglob("*.parquet"))
        if parquet_files:
            try:
                from datasets import Dataset, concatenate_datasets  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Reading Parquet requires `datasets` (pip install datasets)."
                ) from exc

            # Common local dumps place files like:
            #   train-00000-of-00001.parquet / test-00000-of-00001.parquet
            # at the dataset root; respect the requested split when possible.
            if split in {"train", "test"}:
                split_parquets = [fp for fp in parquet_files if fp.name.startswith(f"{split}-") or fp.name == f"{split}.parquet"]
            else:
                split_parquets = []

            if split in {"train", "test"} and not split_parquets:
                raise FileNotFoundError(
                    f"No {split} parquet shards found under {path} (found {len(parquet_files)} parquet files)."
                )

            files_to_read = split_parquets or parquet_files
            datasets_list = [Dataset.from_parquet(str(fp)) for fp in files_to_read]
            ds = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]

            for row in ds:
                if isinstance(row, dict):
                    yield row
            return
        raise FileNotFoundError(f"No .jsonl/.json/.parquet found under {path}")

    if name == "gsm8k":
        if path.is_dir():
            file_path = path / f"{split}.jsonl"
        else:
            file_path = path
        if not file_path.exists():
            raise FileNotFoundError(f"GSM8K file not found: {file_path}")
        yield from _iter_jsonl(file_path)
        return

    if name == "svamp":
        if path.is_dir():
            raise ValueError("SVAMP expects a JSON file path, not a directory.")
        if not path.exists():
            raise FileNotFoundError(f"SVAMP file not found: {path}")
        yield from _iter_json(path)
        return

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def normalize_sample(dataset_name: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw sample to a common schema:
      - problem: str
      - solution: str | None
      - level: optional (hendrycks_math)
    """
    name = dataset_name.strip().lower()
    if name in {"hendrycks", "hendrycks_math", "math"}:
        problem = sample.get("problem") or sample.get("question") or sample.get("prompt")
        solution = sample.get("solution") or sample.get("answer")
        out = dict(sample)
        if problem is not None:
            out["problem"] = problem
        if solution is not None:
            out["solution"] = solution
        return out

    if name == "gsm8k":
        problem = sample.get("question") or sample.get("problem") or sample.get("prompt")
        solution = sample.get("answer") or sample.get("solution")
        out = dict(sample)
        if problem is not None:
            out["problem"] = problem
        if solution is not None:
            out["solution"] = solution
        return out

    if name == "svamp":
        body = sample.get("Body") or sample.get("body") or ""
        question = sample.get("Question") or sample.get("question") or sample.get("problem") or ""
        problem = (str(body).strip() + "\n" + str(question).strip()).strip()
        solution = sample.get("Answer") or sample.get("answer") or sample.get("solution")
        out = dict(sample)
        if problem:
            out["problem"] = problem
        if solution is not None:
            out["solution"] = str(solution)
        return out

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def load_one_sample(
    dataset_name: str,
    dataset_path: Optional[Union[str, Path]] = None,
    *,
    split: str = "train",
) -> Dict[str, Any]:
    for raw in iter_samples(dataset_name, dataset_path, split=split):
        return normalize_sample(dataset_name, raw)
    raise ValueError(f"No samples found for dataset_name={dataset_name}")
