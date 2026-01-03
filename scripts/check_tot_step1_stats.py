#!/usr/bin/env python3
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _get_step1_candidates(record: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    tot = record.get("tot") if isinstance(record.get("tot"), dict) else record
    steps_trace = tot.get("steps_trace") if isinstance(tot, dict) else None
    if not steps_trace or not isinstance(steps_trace, list):
        return None
    step1 = steps_trace[0] if steps_trace else None
    if not isinstance(step1, dict):
        return None
    candidates = step1.get("candidates")
    if not candidates or not isinstance(candidates, list):
        return None
    return candidates


def _candidate_rate(candidate: Dict[str, Any]) -> float:
    rate = candidate.get("success_rate")
    if isinstance(rate, (int, float)):
        return float(rate)
    count = candidate.get("success_count")
    if isinstance(count, (int, float)):
        # Fallback when only counts are present.
        return float(count)
    return 0.0


def _stats_for_file(path: Path, rng: random.Random) -> Tuple[int, int, int, float, float, float]:
    total = 0
    usable = 0
    missing = 0
    rand_sum = 0.0
    best_sum = 0.0
    worst_sum = 0.0

    for record in _iter_jsonl(path):
        total += 1
        candidates = _get_step1_candidates(record)
        if not candidates:
            missing += 1
            continue
        rates = [_candidate_rate(c) for c in candidates]
        if not rates:
            missing += 1
            continue
        rand_sum += _candidate_rate(rng.choice(candidates))
        best_sum += max(rates)
        worst_sum += min(rates)
        usable += 1

    if usable == 0:
        return total, usable, missing, 0.0, 0.0, 0.0
    return (
        total,
        usable,
        missing,
        rand_sum / usable,
        best_sum / usable,
        worst_sum / usable,
    )


def _print_stats(path: Path, stats: Tuple[int, int, int, float, float, float]) -> None:
    total, usable, missing, rand_avg, best_avg, worst_avg = stats
    print(f"\nFile: {path}")
    print(f"Total JSONL rows: {total}")
    print(f"Usable rows (has step1 candidates): {usable}")
    print(f"Missing/invalid step1: {missing}")
    print(f"Random candidate success rate (avg): {rand_avg:.4f}")
    print(f"Best-case success rate (avg max): {best_avg:.4f}")
    print(f"Worst-case success rate (avg min): {worst_avg:.4f}")


def main() -> int:
    rng = random.Random(42)
    print("Enter JSONL path(s). Empty line to quit.")
    while True:
        try:
            raw = input("Path: ").strip()
        except EOFError:
            break
        if not raw:
            break
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1].strip()
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"Not found: {path}")
            continue
        stats = _stats_for_file(path, rng)
        _print_stats(path, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
