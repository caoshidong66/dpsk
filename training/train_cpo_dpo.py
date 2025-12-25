"""
Simple training script that runs Constrained Preference Optimization (CPO)
for step-by-step datasets that contain paired (chosen vs. rejected) completions.

This file uses HuggingFace Transformers + TRL.  It expects the dataset to have
three text columns:
  - prompt:        the instruction / question (e.g. math problem with "Step 1:" prefix)
  - chosen:        the preferred reasoning (should end with the correct answer)
  - rejected:      the lower-quality reasoning

Example JSONL record:
{
  "prompt": "Reasoning step by step...\nStep 1:",
  "chosen": "Step 1: ...\nStep 2: ...\nAnswer: 42",
  "rejected": "Step 1: ...\nStep 2: ...\nAnswer: 13"
}

Usage:
  python training/train_cpo_dpo.py \
      --dataset data/math_tot_pairs.jsonl \
      --model-name-or-path /data/jsg_data/model/meta-llama/llama3-8b \
      --output-dir /data/jsg_data/checkpoints/llama3-cpo \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 8 \
      --num-train-epochs 1 \
      --beta 0.1

Make sure `pip install trl datasets accelerate transformers` first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from datasets import Dataset, load_dataset  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)  # type: ignore
from trl import CPOTrainer  # type: ignore

from lora_utils import add_lora_args, apply_lora_from_args, freeze_model

LOCAL_BASE = os.environ.get(
    "LOCAL_SCRATCH", os.path.join(os.path.expanduser("~"), "local")
)
os.environ["TMPDIR"] = f"{LOCAL_BASE}/tmp"
os.environ["TEMP"] = f"{LOCAL_BASE}/tmp"
os.environ["TMP"] = f"{LOCAL_BASE}/tmp"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{LOCAL_BASE}/torchinductor"
os.environ["TORCH_COMPILE_DEBUG_DIR"] = f"{LOCAL_BASE}/torchcompile"
os.environ["TRITON_CACHE_DIR"] = f"{LOCAL_BASE}/triton"
os.environ["CUDA_CACHE_PATH"] = f"{LOCAL_BASE}/cuda"
os.environ["XDG_CACHE_HOME"] = f"{LOCAL_BASE}/xdg_cache"
for cache_dir in [
    os.environ["TMPDIR"],
    os.environ["TORCHINDUCTOR_CACHE_DIR"],
    os.environ["TORCH_COMPILE_DEBUG_DIR"],
    os.environ["TRITON_CACHE_DIR"],
    os.environ["CUDA_CACHE_PATH"],
    os.environ["XDG_CACHE_HOME"],
]:
    os.makedirs(cache_dir, exist_ok=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a step-by-step reasoning model with CPO / DPO style loss."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a JSON/JSONL/Arrow dataset that contains prompt/chosen/rejected columns.",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Name of the column that stores the prompt text (default: prompt).",
    )
    parser.add_argument(
        "--chosen-column",
        type=str,
        default="chosen",
        help="Name of the preferred response column (default: chosen).",
    )
    parser.add_argument(
        "--rejected-column",
        type=str,
        default="rejected",
        help="Name of the rejected response column (default: rejected).",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Base policy model checkpoint (e.g. /data/.../llama3-8b).",
    )
    parser.add_argument(
        "--ref-model-name-or-path",
        type=str,
        default=None,
        help=(
            "Reference model checkpoint. "
            "If omitted, a frozen copy of the policy model will be used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where trained model / tokenizer will be saved.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of training examples (useful for quick debugging).",
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
        help="Per-device training batch size (default: 1).",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Per-device evaluation batch size (default: 1).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps for the scheduler (default: 100).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Interval (in steps) for logging (default: 10).",
    )
    parser.add_argument(
        "--save-per-epoch",
        action="store_true",
        help="Save checkpoints at the end of each epoch.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Inverse temperature beta used by CPO (default: 0.1).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "ipo", "hinge", "kto_pair"],
        help="CPO loss type / objective (default: sigmoid).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional label smoothing for DPO/CPO losses (default: 0.0).",
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
        help="Enable bfloat16 training if supported.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable float16 training (mutually exclusive with --bf16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    add_lora_args(parser, default_r=8, default_target_modules="q_proj,v_proj")
    return parser.parse_args()


def load_pairwise_dataset(
    dataset_path: Union[str, Path],
    prompt_col: str,
    chosen_col: str,
    rejected_col: str,
    max_samples: Optional[int] = None,
) -> Dataset:
    data_files = str(dataset_path)
    ds = load_dataset("json", data_files=data_files)["train"]

    # Optionally subsample for quick experiments
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Rename to the column names expected by TRL's DPO/CPO trainers
    rename_map: Dict[str, str] = {}
    for src, dst in [
        (prompt_col, "prompt"),
        (chosen_col, "chosen"),
        (rejected_col, "rejected"),
    ]:
        if src != dst and src in ds.column_names:
            rename_map[src] = dst
    if rename_map:
        ds = ds.rename_columns(rename_map)

    missing = [col for col in ("prompt", "chosen", "rejected") if col not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return ds


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    using_ddp = local_rank >= 0

    dataset = load_pairwise_dataset(
        dataset_path=args.dataset,
        prompt_col=args.prompt_column,
        chosen_col=args.chosen_column,
        rejected_col=args.rejected_column,
        max_samples=args.max_samples,
    )

    # Split into train/eval (10% eval by default)
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=("auto" if not using_ddp else None),
    )
    model.config.use_cache = False
    model = apply_lora_from_args(model, args)

    if args.ref_model_name_or_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=("auto" if not using_ddp else None),
        )
        freeze_model(ref_model)
    else:
        if args.use_lora:
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=("auto" if not using_ddp else None),
            )
            freeze_model(ref_model)
        else:
            ref_model = None

    if using_ddp and model.device.type != "cuda":
        model.to(torch.device(f"cuda:{local_rank}"))
    if using_ddp and ref_model is not None and ref_model.device.type != "cuda":
        ref_model.to(torch.device(f"cuda:{local_rank}"))

    save_strategy = "epoch" if args.save_per_epoch else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy="no",
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        run_name="cpo_step_training",
    )

    trainer = CPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        generate_during_eval=False,
    )

    trainer.train()

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
