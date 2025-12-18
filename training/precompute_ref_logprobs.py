#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running from any CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.train_chain_preference import (  # noqa: E402
    _infer_input_device,
    compute_logprob_sum,
    convert_tot_to_chain_records,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute reference-model per-step logprobs for step-DPO training."
    )
    p.add_argument("--tot-jsonl", type=str, required=True, help="collect_tot.py merged .all.jsonl")
    p.add_argument("--model-name-or-path", type=str, required=True, help="Reference/base model checkpoint.")
    p.add_argument("--output", type=str, required=True, help="Output JSONL with ref logprobs embedded.")
    p.add_argument("--seed", type=int, default=42, help="Seed for negative-candidate sampling (default: 42).")
    p.add_argument("--max-length", type=int, default=1024, help="Max length used for truncation (default: 1024).")
    p.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per example.")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples.")
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["none", "auto"],
        help="Load placement for ref model (default: auto).",
    )
    p.add_argument("--gpus", type=str, default=None, help="CUDA_VISIBLE_DEVICES override, e.g. 0 or 0,1,2.")
    p.add_argument(
        "--disable-p2p",
        action="store_true",
        help="Set NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1 (useful on older RTX 4000 drivers).",
    )
    p.add_argument("--bf16", action="store_true", help="Load ref model in bf16.")
    p.add_argument("--fp16", action="store_true", help="Load ref model in fp16.")
    return p.parse_args()


def _cap_steps(record: Dict[str, object], max_steps: int) -> Dict[str, object]:
    out = dict(record)
    for k in ["good_steps", "bad_steps", "good_rewards", "bad_rewards", "ref_good_logprobs", "ref_bad_logprobs"]:
        v = out.get(k)
        if isinstance(v, list):
            out[k] = v[:max_steps]
    return out


def main() -> None:
    args = parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    if args.disable_p2p:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")

    tot_path = Path(args.tot_jsonl)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = convert_tot_to_chain_records(tot_path, seed=int(args.seed))
    if args.max_steps is not None:
        records = [_cap_steps(r, int(args.max_steps)) for r in records]
    if args.limit is not None:
        records = records[: int(args.limit)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        device_map=("auto" if args.device_map == "auto" else None),
    )
    if args.device_map != "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ref_model.to(device)
    ref_model.eval()
    device = _infer_input_device(ref_model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with out_path.open("w", encoding="utf-8") as f:
        with torch.inference_mode():
            for n, rec in enumerate(records, start=1):
                problem = rec.get("problem")
                prompt = rec.get("prompt")
                good_steps = rec.get("good_steps")
                bad_steps = rec.get("bad_steps")
                if not isinstance(problem, str):
                    continue
                if not isinstance(good_steps, list) or not isinstance(bad_steps, list):
                    continue

                context_good = prompt if isinstance(prompt, str) and prompt.strip() else problem + "\n"
                context_bad = prompt if isinstance(prompt, str) and prompt.strip() else problem + "\n"

                ref_good_logprobs: List[float] = []
                ref_bad_logprobs: List[float] = []

                for step_w, step_l in zip(good_steps, bad_steps):
                    if not isinstance(step_w, str) or not isinstance(step_l, str):
                        break
                    lp_w = compute_logprob_sum(
                        ref_model, tokenizer, context_good, step_w, device, max_length=int(args.max_length)
                    )
                    lp_l = compute_logprob_sum(
                        ref_model, tokenizer, context_bad, step_l, device, max_length=int(args.max_length)
                    )
                    ref_good_logprobs.append(float(lp_w.detach().cpu().item()))
                    ref_bad_logprobs.append(float(lp_l.detach().cpu().item()))
                    context_good = (context_good + step_w).rstrip() + "\n"
                    context_bad = (context_bad + step_l).rstrip() + "\n"

                out: Dict[str, object] = dict(rec)
                out["ref_model_name_or_path"] = args.model_name_or_path
                out["ref_seed"] = int(args.seed)
                out["ref_max_length"] = int(args.max_length)
                out["ref_good_logprobs"] = ref_good_logprobs
                out["ref_bad_logprobs"] = ref_bad_logprobs

                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                if n % 50 == 0:
                    print(f"[precompute_ref] processed {n}/{len(records)}")

    print(f"[precompute_ref] wrote -> {out_path}")


if __name__ == "__main__":
    main()

