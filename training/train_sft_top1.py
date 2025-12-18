"""
Simple SFT script that uses the highest-success-rate step from ToT data as supervision.

Usage example:
  python training/train_sft_top1.py \
      --tot-jsonl /path/to/llama_tot_outputs.jsonl \
      --model-name-or-path /data/jsg_data/model/meta-llama/llama3-8b \
      --output-dir /data/jsg_data/checkpoints/llama3-sft-top1 \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 8 \
      --num-train-epochs 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)  # type: ignore

from lora_utils import add_lora_args, apply_lora_from_args


def load_tot_jsonl(jsonl_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_best_candidate(candidates: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not candidates:
        return None
    sorted_cands = sorted(
        candidates,
        key=lambda c: (
            c.get("success_rate") or 0.0,
            c.get("success_count") or 0,
        ),
        reverse=True,
    )
    best = sorted_cands[0]
    if best.get("step_text"):
        return best
    return None


def build_prompt_with_context(
    problem: str,
    steps_context: Dict[int, str],
    step_idx: int,
) -> str:
    lines = [f"Problem:\n{problem}\n"]
    if steps_context:
        lines.append("Current Progress:")
        for idx in sorted(steps_context.keys()):
            lines.append(f"Step {idx}:\n{steps_context[idx]}")
    lines.append(
        f"\nContinue solving the problem. "
        f"Write the reasoning for Step {step_idx} clearly.\n"
        f"Format:\nStep {step_idx}:\n..."
    )
    lines.append("Begin your answer now.")
    return "\n".join(lines)


def convert_records_to_sft_examples(records: List[Dict[str, object]]) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for rec in records:
        problem = rec.get("problem")
        steps_trace = rec.get("steps_trace") or []
        if not problem:
            continue

        context: Dict[int, str] = {}
        for step_entry in steps_trace:
            step_idx = step_entry.get("step_index")
            candidates = step_entry.get("candidates")
            if not isinstance(step_idx, int) or not isinstance(candidates, list):
                continue
            best = select_best_candidate(candidates)
            if best is None:
                continue

            prompt = build_prompt_with_context(problem, context, step_idx)
            target_text = best.get("step_text") or ""
            if not target_text.strip():
                continue
            completion = f"Step {step_idx}:\n{target_text}"
            examples.append({"text": prompt + "\n\n" + completion})

            context[step_idx] = target_text
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tune a step-by-step model using top-1 ToT candidates."
    )
    parser.add_argument(
        "--tot-jsonl",
        type=str,
        required=True,
        help="Path to JSONL file produced by llama_tot_math.py (with steps_trace).",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Base model checkpoint (e.g., /data/.../llama3-8b).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save fine-tuned model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of SFT examples for quick tests.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of training epochs (default: 1).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization (default: 1024).",
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

    records = load_tot_jsonl(Path(args.tot_jsonl))
    examples = convert_records_to_sft_examples(records)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    print(f"[train_sft_top1] Prepared {len(examples)} SFT training examples.")

    dataset = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=args.max_length,
            return_tensors=None,
        )

    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
