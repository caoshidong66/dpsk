import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)  # type: ignore
from trl import CPOTrainer  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_utils import add_lora_args, apply_lora_from_args, freeze_model  # noqa: E402


def load_tot_jsonl(jsonl_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def choose_pair_for_step(
    candidates: List[Dict[str, object]],
) -> Optional[Tuple[Dict[str, object], Dict[str, object]]]:
    """
    Return (chosen, rejected) by comparing success_rate.
    If not enough candidates, return None.
    """
    if len(candidates) < 2:
        return None
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c.get("success_rate") or 0.0, c.get("success_count") or 0),
        reverse=True,
    )
    chosen = sorted_cands[0]
    rejected = sorted_cands[-1]
    if (
        (chosen.get("success_rate") is None)
        or (rejected.get("success_rate") is None)
        or chosen["success_rate"] == rejected["success_rate"]
    ):
        return None
    return chosen, rejected


def build_prompt_for_step(
    problem: str,
    existing_steps: Dict[int, Dict[str, Optional[str]]],
    focus_step: int,
    step_text: str,
) -> str:
    lines = [f"Problem:\n{problem}\n"]
    lines.append("Current Progress:")
    for idx in sorted(existing_steps.keys()):
        entry = existing_steps[idx]
        text = entry.get("refined") or entry.get("draft")
        if text:
            lines.append(f"Step {idx}:\n{text}")
    lines.append(
        f"\nYou must refine the draft of Step {focus_step} shown below "
        "and produce a better version. Format:\n"
        f"Refined Step {focus_step}:\n...\n"
    )
    lines.append(f"Draft Step {focus_step} (current):\n{step_text}\n")
    lines.append("Begin your answer now.")
    return "\n".join(lines)


def convert_tot_records_to_pairs(
    records: List[Dict[str, object]],
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for rec in records:
        problem = rec.get("problem")
        steps_trace = rec.get("steps_trace") or []
        if not problem or not steps_trace:
            continue

        steps: Dict[int, Dict[str, Optional[str]]] = {}
        for entry in steps_trace:
            step_idx = entry.get("step_index")
            candidates = entry.get("candidates")
            if not isinstance(step_idx, int) or not isinstance(candidates, list):
                continue
            pair = choose_pair_for_step(candidates)
            if pair is None:
                continue
            chosen, rejected = pair
            prompt = build_prompt_for_step(
                problem=problem,
                existing_steps=steps,
                focus_step=step_idx,
                step_text=rejected.get("step_text") or "",
            )
            chosen_text = chosen.get("step_text") or ""
            rejected_text = rejected.get("step_text") or ""
            if not chosen_text or not rejected_text:
                continue
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": f"Refined Step {step_idx}:\n{chosen_text}",
                    "rejected": f"Refined Step {step_idx}:\n{rejected_text}",
                }
            )
            steps[step_idx] = {"draft": chosen_text, "refined": chosen_text}
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train refinement model using CPO on ToT rollout pairs."
    )
    parser.add_argument(
        "--tot-jsonl",
        type=str,
        required=True,
        help="Path to JSONL produced by llama_tot_math (with steps_trace).",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Base model checkpoint for refinement.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--ref-model-name-or-path",
        type=str,
        default=None,
        help="Reference model checkpoint (optional).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of training pairs for quick tests.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Training epochs (default: 1).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size (default: 1).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="CPO beta parameter (default: 0.1).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "ipo", "hinge", "kto_pair"],
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum total sequence length (prompt + response).",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=768,
        help="Maximum prompt length (tokens).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 training if supported.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 training.",
    )
    add_lora_args(parser, default_r=8, default_target_modules="q_proj,v_proj")
    args = parser.parse_args()

    tot_records = load_tot_jsonl(Path(args.tot_jsonl))
    pair_examples = convert_tot_records_to_pairs(tot_records)
    if args.max_samples is not None:
        pair_examples = pair_examples[: args.max_samples]
    print(f"[diffcot] Loaded {len(pair_examples)} paired refinement examples.")

    dataset = Dataset.from_list(pair_examples)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model = apply_lora_from_args(model, args)
    if args.ref_model_name_or_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        freeze_model(ref_model)
    else:
        if args.use_lora:
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=dtype,
                device_map="auto",
            )
            freeze_model(ref_model)
        else:
            ref_model = None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
    )

    trainer = CPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        loss_type=args.loss_type,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
