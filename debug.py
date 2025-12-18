#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TotStats:
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    unknown: int = 0
    unique_indices: int = 0
    duplicate_indices: int = 0
    min_index: Optional[int] = None
    max_index: Optional[int] = None
    per_gpu_total: Optional[Dict[str, int]] = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


_TS_RE = re.compile(r"\.(\d{8}_\d{6})\.")
_RUN_FILE_RE = re.compile(r"^math_l(?P<level>\d+)\.(?P<run_id>\d{8}_\d{6})\.(?P<kind>gpu(?P<gpu>\d+)|all)\.jsonl$")


def _parse_levels(value: str) -> list[int]:
    levels: list[int] = []
    for part in [p.strip() for p in value.split(",") if p.strip()]:
        m = re.search(r"(\d+)", part)
        if not m:
            raise ValueError(f"Invalid level: {part!r}")
        levels.append(int(m.group(1)))
    levels = sorted(set(levels))
    if not levels:
        raise ValueError("No levels specified")
    return levels


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _extract_timestamp(path: Path) -> str:
    m = _TS_RE.search(path.name)
    return m.group(1) if m else ""


def _collect_runs_for_level(level_dir: Path, level: int) -> Dict[str, Dict[str, object]]:
    runs: Dict[str, Dict[str, object]] = {}
    for p in level_dir.glob(f"math_l{level}.*.jsonl"):
        m = _RUN_FILE_RE.match(p.name)
        if not m:
            continue
        run_id = m.group("run_id")
        entry = runs.setdefault(run_id, {"all": None, "gpus": []})
        if m.group("kind") == "all":
            entry["all"] = p
        else:
            entry["gpus"].append(p)
    for run_id, entry in runs.items():
        entry["gpus"] = sorted(entry["gpus"])  # type: ignore[assignment]
    return runs


def _pick_tot_math_run_files(
    data_root: Path, level: int, run_id: Optional[str]
) -> Tuple[str, Sequence[Path]]:
    level_dir = data_root / f"tot_math_l{level}"
    if not level_dir.exists():
        raise FileNotFoundError(f"Missing level dir: {level_dir}")

    runs = _collect_runs_for_level(level_dir, level)
    if not runs:
        raise FileNotFoundError(f"No files found under {level_dir} for math_l{level}")

    chosen_run_id: str
    if run_id is not None:
        if run_id not in runs:
            known = ", ".join(sorted(runs.keys()))
            raise FileNotFoundError(
                f"No files matched run_id={run_id!r} under {level_dir}; available: {known}"
            )
        chosen_run_id = run_id
    else:
        # Heuristic: prefer the run with the most records (more "complete"),
        # tie-break by latest run_id.
        best: Tuple[int, str] | None = None  # (line_count, run_id)
        for rid, entry in runs.items():
            all_fp = entry.get("all")
            if isinstance(all_fp, Path):
                paths = [all_fp]
            else:
                gpu_files = entry.get("gpus")
                paths = gpu_files if isinstance(gpu_files, list) else []
            line_count = 0
            for p in paths:
                try:
                    with p.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                line_count += 1
                except FileNotFoundError:
                    continue
            cand = (line_count, rid)
            if best is None or cand > best:
                best = cand
        chosen_run_id = best[1] if best is not None else max(runs.keys())

    entry = runs[chosen_run_id]
    all_fp = entry.get("all")
    if isinstance(all_fp, Path):
        return chosen_run_id, [all_fp]
    gpu_files = entry.get("gpus")
    if isinstance(gpu_files, list) and gpu_files:
        return chosen_run_id, gpu_files
    raise FileNotFoundError(f"Run {chosen_run_id} under {level_dir} has no readable jsonl files")

def _summarize_runs(data_root: Path, level: int) -> list[tuple[str, int, int, bool]]:
    """
    Returns: [(run_id, non_empty_lines, file_count, has_all)]
    """
    level_dir = data_root / f"tot_math_l{level}"
    runs = _collect_runs_for_level(level_dir, level)
    out: list[tuple[str, int, int, bool]] = []
    for rid, entry in runs.items():
        all_fp = entry.get("all")
        if isinstance(all_fp, Path):
            paths = [all_fp]
            has_all = True
        else:
            gpu_files = entry.get("gpus")
            paths = gpu_files if isinstance(gpu_files, list) else []
            has_all = False
        line_count = 0
        for p in paths:
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            line_count += 1
            except FileNotFoundError:
                continue
        out.append((rid, line_count, len(paths), has_all))
    out.sort(key=lambda x: (x[1], x[0]))
    return out


def _is_correct_from_record(record: dict) -> Optional[bool]:
    tot = record.get("tot")
    if not isinstance(tot, dict):
        return None
    v = tot.get("final_is_correct")
    if isinstance(v, bool):
        return v
    v2 = tot.get("is_correct")
    if isinstance(v2, bool):
        return v2
    return None


def compute_tot_accuracy(paths: Sequence[Path]) -> TotStats:
    total = correct = incorrect = unknown = 0
    index_counts: Dict[int, int] = {}
    per_gpu_total: Dict[str, int] = {}
    min_index: Optional[int] = None
    max_index: Optional[int] = None
    for path in paths:
        for record in _iter_jsonl(path):
            total += 1
            idx = record.get("index")
            if isinstance(idx, int):
                index_counts[idx] = index_counts.get(idx, 0) + 1
                if min_index is None or idx < min_index:
                    min_index = idx
                if max_index is None or idx > max_index:
                    max_index = idx
            gpu = record.get("gpu")
            if gpu is not None:
                g = str(gpu)
                per_gpu_total[g] = per_gpu_total.get(g, 0) + 1
            ok = _is_correct_from_record(record)
            if ok is True:
                correct += 1
            elif ok is False:
                incorrect += 1
            else:
                unknown += 1
    unique_indices = len(index_counts)
    duplicate_indices = sum(1 for c in index_counts.values() if c > 1)
    return TotStats(
        total=total,
        correct=correct,
        incorrect=incorrect,
        unknown=unknown,
        unique_indices=unique_indices,
        duplicate_indices=duplicate_indices,
        min_index=min_index,
        max_index=max_index,
        per_gpu_total=per_gpu_total,
    )


def _try_import_dataset_utils() -> object:
    """
    Import dataset_utils from repo root.
    This keeps debug.py runnable even if executed from subdirectories.
    """
    try:
        import dataset_utils  # type: ignore

        return dataset_utils
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_root))
        import dataset_utils  # type: ignore

        return dataset_utils


@dataclass(frozen=True)
class DatasetLevelCounts:
    total: int
    per_level: Dict[int, int]


def count_hendrycks_levels(dataset_root: Path, split: str, levels: Sequence[int]) -> DatasetLevelCounts:
    dataset_utils = _try_import_dataset_utils()
    iter_samples = getattr(dataset_utils, "iter_samples")

    def parse_level(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        m = re.search(r"(\d+)", str(value))
        return int(m.group(1)) if m else None

    wanted = set(levels)
    counts: Dict[int, int] = {lvl: 0 for lvl in levels}
    total = 0
    for raw in iter_samples("hendrycks_math", str(dataset_root), split=split):
        total += 1
        lvl = parse_level(raw.get("level")) if isinstance(raw, dict) else None
        if lvl in wanted:
            counts[int(lvl)] += 1
    return DatasetLevelCounts(total=total, per_level=counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug helpers: compute ToT accuracy from datas outputs.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="datas",
        help="Data root directory (default: datas).",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="1,2,3,4",
        help="Comma-separated levels to report (default: 1,2,3,4).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id (e.g. 20251215_175641) to select per-level .all.jsonl.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="If set, also count how many raw hendrycks_math samples exist per level under this dataset root.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split used when counting raw dataset levels (default: train).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-level selected files and run_id.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    levels = _parse_levels(args.levels)

    rows: list[tuple[int, str, Sequence[Path], TotStats]] = []
    for lvl in levels:
        chosen_run_id, fps = _pick_tot_math_run_files(data_root, lvl, args.run_id)
        stats = compute_tot_accuracy(fps)
        rows.append((lvl, chosen_run_id, fps, stats))

    print("ToT accuracy (hendrycks_math / MATH)")
    print(f"data_root={data_root}")
    if args.run_id:
        print(f"run_id={args.run_id}")
    print()

    if args.dataset_root:
        ds_root = Path(args.dataset_root)
        ds_counts = count_hendrycks_levels(ds_root, args.dataset_split, levels)
        print("Raw dataset counts (hendrycks_math)")
        print(f"dataset_root={ds_root} split={args.dataset_split} total={ds_counts.total}")
        for lvl in levels:
            print(f"  level {lvl}: {ds_counts.per_level.get(lvl, 0)}")
        print()

    header = (
        f"{'level':>5}  {'run_id':>15}  {'total':>6}  {'uniq':>6}  {'dup':>5}  "
        f"{'min':>6}  {'max':>6}  {'correct':>7}  {'acc':>7}  {'unknown':>7}  file(s)"
    )
    print(header)
    print("-" * len(header))
    for lvl, run_id, fps, st in rows:
        acc = f"{st.accuracy*100:6.2f}%"
        files_str = ", ".join(str(p) for p in fps)
        min_idx = "-" if st.min_index is None else str(st.min_index)
        max_idx = "-" if st.max_index is None else str(st.max_index)
        print(
            f"{lvl:>5}  {run_id:>15}  {st.total:>6}  {st.unique_indices:>6}  {st.duplicate_indices:>5}  "
            f"{min_idx:>6}  {max_idx:>6}  {st.correct:>7}  {acc:>7}  {st.unknown:>7}  {files_str}"
        )

    if args.verbose:
        print()
        for lvl, run_id, fps, _ in rows:
            summaries = _summarize_runs(data_root, lvl)
            if summaries:
                print(f"[level {lvl}] available runs (records/files/merged):")
                for rid, n, fc, has_all in summaries:
                    merged = "all" if has_all else "gpu"
                    mark = "*" if rid == run_id else " "
                    print(f"  {mark} {rid}  {n}  {fc}  {merged}")
            print(f"[level {lvl}] selected run_id={run_id} files={len(fps)}")
            for p in fps:
                print(f"  - {p}")
            if st.per_gpu_total:
                gpu_str = ", ".join(f"gpu{g}={n}" for g, n in sorted(st.per_gpu_total.items()))
                print(f"  per-gpu records: {gpu_str}")

            if args.dataset_root and st.max_index is not None:
                # A hint for common failure mode: running with --end-index or only scanning a prefix.
                total_raw = ds_counts.total
                if total_raw and st.max_index < total_raw * 0.2:
                    print(
                        "  hint: max(index) is low vs raw dataset size; "
                        "this often means the collection run used --end-index or scanned only a prefix."
                    )


if __name__ == "__main__":
    main()
