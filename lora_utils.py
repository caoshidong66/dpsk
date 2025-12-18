from __future__ import annotations

import argparse
from typing import Iterable, List, Optional


def add_lora_args(
    parser: argparse.ArgumentParser,
    *,
    default_r: int = 8,
    default_dropout: float = 0.05,
    default_target_modules: str = "q_proj,v_proj",
) -> None:
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA fine-tuning (requires `peft`).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=default_r,
        help=f"LoRA rank (default: {default_r}).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=0,
        help="LoRA alpha; 0 means auto (= 2 * r).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=default_dropout,
        help=f"LoRA dropout (default: {default_dropout}).",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=default_target_modules,
        help=(
            "Comma-separated module names to apply LoRA to "
            f"(default: {default_target_modules})."
        ),
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="Which biases to train (default: none).",
    )


def _parse_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def apply_lora_from_args(model, args: argparse.Namespace):
    if not getattr(args, "use_lora", False):
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "LoRA requested but `peft` is not available. Install with: pip install peft"
        ) from e

    target_modules = _parse_csv(getattr(args, "lora_target_modules", "q_proj,v_proj"))
    if not target_modules:
        raise ValueError("--lora-target-modules resolved to an empty list.")

    r = int(getattr(args, "lora_r", 8))
    alpha = int(getattr(args, "lora_alpha", 0)) or (2 * r)
    dropout = float(getattr(args, "lora_dropout", 0.05))
    bias = str(getattr(args, "lora_bias", "none"))

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
    )
    model = get_peft_model(model, config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def freeze_model(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def trainable_parameters(model) -> Iterable:
    return (p for p in model.parameters() if p.requires_grad)
