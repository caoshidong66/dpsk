"""
Chain-of-Preference training script.

Given ToT-style data with per-step rewards (r_i = success_rate), compute soft
step weights and optimize a step-wise DPO loss:

  alpha_w = softmax(+gamma * r_w)
  alpha_l = softmax(-gamma * r_l)
  logit = beta * (good_term - bad_term)
  loss = -log(sigmoid(logit))

You can either supply a preprocessed JSONL via --data, or pass a raw ToT JSONL
via --tot-jsonl; in the latter case the script will automatically build
good/bad step sequences and use each candidate's success_rate as r_i.

The --tot-jsonl input can be either:
  1) a "tot object" JSONL where each line is {"problem": ..., "steps_trace": ...}
  2) a "collect_tot.py record" JSONL where each line is {"tot": {...}, ...}

For each ToT step, we pick:
  - positive: the best candidate (highest success_rate; tie-break by success_count)
  - negative: a random candidate among the remaining candidates

Expected JSONL format for --data:
{
  "problem": "...",
  "good_steps": ["Step 1 ...", "Step 2 ...", ...],
  "good_rewards": [0.8, 0.3, ...],     # success rates
  "bad_steps": ["...", "..."],
  "bad_rewards": [...]
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

TRAIN_CONFIG: Dict[str, Optional[object]] = {
    "batch_size": 16,
    "grad_accum_steps": 1,
    "learning_rate": 2e-5,
    "epochs": 1,
    "max_length": 1024,
    "bf16": True,
    "fp16": False,
    "save_per_epoch": True,
    "eval_after_train": False,
    "eval_num_samples": 64,
}

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


# Allow running as `python training/train_chain_preference.py` from any CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lora_utils import (
    add_lora_args,
    apply_lora_from_args,
    freeze_model,
    trainable_parameters,
)
from dataset_utils import iter_samples, normalize_sample, default_dataset_path
from tool import is_model_correct, steps_for_dataset, steps_for_level


class ChainPrefDataset(Dataset):
    def __init__(self, records: List[Dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.records[idx]


def load_chain_data(jsonl_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def convert_tot_to_chain_records(
    jsonl_path: Path,
    *,
    seed: int,
    max_steps: Optional[int] = None,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    raw = load_chain_data(jsonl_path)
    rng = random.Random(int(seed))
    for rec in raw:
        # Support both direct tot jsonl and collect_tot.py outputs (nested under "tot")
        tot = rec.get("tot") if isinstance(rec, dict) else None
        if isinstance(tot, dict):
            base = tot
        else:
            base = rec

        problem = base.get("problem") if isinstance(base, dict) else None
        prompt = base.get("prompt") if isinstance(base, dict) else None
        steps_trace = (base.get("steps_trace") if isinstance(base, dict) else None) or []
        if not problem or not steps_trace:
            continue

        good_steps: List[str] = []
        bad_steps: List[str] = []
        good_rewards: List[float] = []
        bad_rewards: List[float] = []

        for entry in steps_trace:
            candidates = entry.get("candidates")
            if not isinstance(candidates, list) or len(candidates) < 2:
                continue
            sorted_cands = sorted(
                candidates,
                key=lambda c: (c.get("success_rate") or 0.0, c.get("success_count") or 0),
                reverse=True,
            )
            best = sorted_cands[0]
            others = sorted_cands[1:]
            neg = rng.choice(others)

            step_text_best = best.get("step_text")
            step_text_neg = neg.get("step_text")
            sr_best = best.get("success_rate")
            sr_neg = neg.get("success_rate")
            if (
                step_text_best
                and step_text_neg
                and isinstance(sr_best, (int, float))
                and isinstance(sr_neg, (int, float))
            ):
                good_steps.append(step_text_best)
                bad_steps.append(step_text_neg)
                good_rewards.append(float(sr_best))
                bad_rewards.append(float(sr_neg))

        if good_steps and bad_steps:
            length = min(len(good_steps), len(bad_steps))
            if max_steps is not None:
                length = min(length, int(max_steps))

            out = {
                "problem": problem,
                "good_steps": good_steps[:length],
                "good_rewards": good_rewards[:length],
                "bad_steps": bad_steps[:length],
                "bad_rewards": bad_rewards[:length],
            }
            if isinstance(prompt, str) and prompt.strip():
                out["prompt"] = prompt
            records.append(out)
    return records


def flatten_chain_records(
    records: List[Dict[str, object]],
    *,
    gamma: float,
) -> List[Dict[str, object]]:
    flat: List[Dict[str, object]] = []
    for rec in records:
        if "good_step" in rec and "bad_step" in rec:
            flat.append(rec)
            continue

        problem = rec.get("problem")
        prompt = rec.get("prompt")
        good_steps = rec.get("good_steps")
        bad_steps = rec.get("bad_steps")
        good_rewards = rec.get("good_rewards")
        bad_rewards = rec.get("bad_rewards")
        ref_good_lp = rec.get("ref_good_logprobs")
        ref_bad_lp = rec.get("ref_bad_logprobs")

        if not (problem and good_steps and bad_steps and good_rewards and bad_rewards):
            continue

        if not (
            isinstance(good_steps, list)
            and isinstance(bad_steps, list)
            and isinstance(good_rewards, list)
            and isinstance(bad_rewards, list)
        ):
            continue

        length = min(len(good_steps), len(bad_steps), len(good_rewards), len(bad_rewards))
        if length == 0:
            continue

        alpha_w = softmax_weights(good_rewards[:length], gamma=gamma, device=torch.device("cpu"))
        alpha_l = softmax_weights(bad_rewards[:length], gamma=-gamma, device=torch.device("cpu"))

        if isinstance(prompt, str) and prompt.strip():
            context_good = prompt
            context_bad = prompt
        else:
            context_good = str(problem) + "\n"
            context_bad = str(problem) + "\n"

        for i in range(length):
            out: Dict[str, object] = {
                "problem": problem,
                "step_index": i + 1,
                "total_steps": length,
                "prefix_good": context_good,
                "prefix_bad": context_bad,
                "good_step": good_steps[i],
                "bad_step": bad_steps[i],
                "good_reward": good_rewards[i],
                "bad_reward": bad_rewards[i],
                "good_weight": float(alpha_w[i].item()),
                "bad_weight": float(alpha_l[i].item()),
            }
            if isinstance(prompt, str) and prompt.strip():
                out["prompt"] = prompt
            if isinstance(ref_good_lp, list) and i < len(ref_good_lp):
                out["ref_good_logprob"] = ref_good_lp[i]
            if isinstance(ref_bad_lp, list) and i < len(ref_bad_lp):
                out["ref_bad_logprob"] = ref_bad_lp[i]
            flat.append(out)

            context_good = (context_good + str(good_steps[i])).rstrip() + "\n"
            context_bad = (context_bad + str(bad_steps[i])).rstrip() + "\n"

    return flat


def encode_step(
    tokenizer,
    context: str,
    step: str,
    device: torch.device,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    step_ids = tokenizer.encode(step, add_special_tokens=False)
    if len(context_ids) + len(step_ids) > max_length:
        keep_for_context = max_length - len(step_ids)
        if keep_for_context <= 0:
            context_ids = []
            step_ids = step_ids[-max_length:]
        else:
            context_ids = context_ids[-keep_for_context:]

    input_ids = torch.tensor([context_ids + step_ids], device=device)
    labels = torch.tensor(
        [[-100] * len(context_ids) + step_ids],
        device=device,
    )
    return {"input_ids": input_ids, "labels": labels, "step_len": len(step_ids)}


def _build_batch_inputs(
    tokenizer,
    contexts: List[str],
    steps: List[str],
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_enc = tokenizer(contexts, add_special_tokens=False)
    step_enc = tokenizer(steps, add_special_tokens=False)
    context_ids_batch = context_enc["input_ids"]
    step_ids_batch = step_enc["input_ids"]

    merged: List[List[int]] = []
    labels: List[List[int]] = []
    for context_ids, step_ids in zip(context_ids_batch, step_ids_batch):
        if len(context_ids) + len(step_ids) > max_length:
            keep_for_context = max_length - len(step_ids)
            if keep_for_context <= 0:
                context_ids = []
                step_ids = step_ids[-max_length:]
            else:
                context_ids = context_ids[-keep_for_context:]
        merged.append(context_ids + step_ids)
        labels.append([-100] * len(context_ids) + step_ids)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    max_len = max(len(x) for x in merged) if merged else 0
    input_ids = torch.full((len(merged), max_len), pad_id, device=device, dtype=torch.long)
    label_ids = torch.full((len(labels), max_len), -100, device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(merged), max_len), device=device, dtype=torch.long)
    for i, (seq, lab) in enumerate(zip(merged, labels)):
        if not seq:
            continue
        input_ids[i, : len(seq)] = torch.tensor(seq, device=device)
        label_ids[i, : len(lab)] = torch.tensor(lab, device=device)
        attention_mask[i, : len(seq)] = 1
    return input_ids, label_ids, attention_mask


def _compute_logprob_sums_batch(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=1)


def compute_logprob_sum(
    model,
    tokenizer,
    context: str,
    step: str,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    encoded = encode_step(tokenizer, context, step, device, max_length=max_length)
    outputs = model(
        input_ids=encoded["input_ids"],
        labels=encoded["labels"],
    )
    # loss is mean over unmasked positions; convert to sum log-prob
    step_len = encoded["step_len"]
    logprob = -outputs.loss * step_len
    return logprob


def compute_ref_logprob_sum(
    model,
    ref_model,
    tokenizer,
    context: str,
    step: str,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    """
    Reference logprob computation.

    - If ref_model is provided: use it.
    - Else (common LoRA case): disable adapters on `model` (PEFT) and use base model as reference,
      which avoids loading a second full checkpoint.
    """
    with torch.no_grad():
        if ref_model is not None:
            return compute_logprob_sum(
                ref_model, tokenizer, context, step, device, max_length=max_length
            )

        # When using DeepSpeed/Accelerate, the wrapped model may expose the PEFT methods under `.module`.
        raw = model
        while hasattr(raw, "module"):
            raw = getattr(raw, "module")
        disable_ctx = getattr(raw, "disable_adapter", None)
        if callable(disable_ctx):
            with disable_ctx():
                return compute_logprob_sum(
                    model, tokenizer, context, step, device, max_length=max_length
                )
        return compute_logprob_sum(model, tokenizer, context, step, device, max_length=max_length)


def _infer_input_device(model, fallback: torch.device) -> torch.device:
    """
    With accelerate device_map sharding, inputs should be on the same device as the embedding layer.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def softmax_weights(rewards: List[float], gamma: float, device: torch.device) -> torch.Tensor:
    r = torch.tensor(rewards, dtype=torch.float32, device=device)
    return F.softmax(gamma * r, dim=0)


def compute_stepwise_loss(
    batch: List[Dict[str, object]],
    model,
    ref_model,
    tokenizer,
    beta: float,
    gamma: float,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    if batch and all("good_step" in ex and "bad_step" in ex for ex in batch):
        contexts_good: List[str] = []
        contexts_bad: List[str] = []
        steps_good: List[str] = []
        steps_bad: List[str] = []
        good_weights: List[float] = []
        bad_weights: List[float] = []
        ref_good_vals: List[Optional[float]] = []
        ref_bad_vals: List[Optional[float]] = []

        for example in batch:
            problem = example["problem"]
            prompt = example.get("prompt")
            prefix_good = example.get("prefix_good")
            prefix_bad = example.get("prefix_bad")
            good_step = example["good_step"]
            bad_step = example["bad_step"]
            good_weight = float(example.get("good_weight", 1.0))
            bad_weight = float(example.get("bad_weight", 1.0))
            ref_good_lp = example.get("ref_good_logprob")
            ref_bad_lp = example.get("ref_bad_logprob")

            if isinstance(prefix_good, str) and prefix_good.strip():
                context_good = prefix_good
            elif isinstance(prompt, str) and prompt.strip():
                context_good = prompt
            else:
                context_good = str(problem) + "\n"

            if isinstance(prefix_bad, str) and prefix_bad.strip():
                context_bad = prefix_bad
            elif isinstance(prompt, str) and prompt.strip():
                context_bad = prompt
            else:
                context_bad = str(problem) + "\n"

            contexts_good.append(context_good)
            contexts_bad.append(context_bad)
            steps_good.append(str(good_step))
            steps_bad.append(str(bad_step))
            good_weights.append(good_weight)
            bad_weights.append(bad_weight)
            ref_good_vals.append(float(ref_good_lp) if isinstance(ref_good_lp, (int, float)) else None)
            ref_bad_vals.append(float(ref_bad_lp) if isinstance(ref_bad_lp, (int, float)) else None)

        good_input, good_labels, good_mask = _build_batch_inputs(
            tokenizer, contexts_good, steps_good, device, max_length
        )
        bad_input, bad_labels, bad_mask = _build_batch_inputs(
            tokenizer, contexts_bad, steps_bad, device, max_length
        )

        logpi_w = _compute_logprob_sums_batch(model, good_input, good_labels, good_mask)
        logpi_l = _compute_logprob_sums_batch(model, bad_input, bad_labels, bad_mask)

        use_precomputed_ref = all(v is not None for v in ref_good_vals) and all(
            v is not None for v in ref_bad_vals
        )
        if use_precomputed_ref:
            logpi_ref_w = torch.tensor(ref_good_vals, device=device)
            logpi_ref_l = torch.tensor(ref_bad_vals, device=device)
        else:
            if ref_model is not None:
                logpi_ref_w = _compute_logprob_sums_batch(ref_model, good_input, good_labels, good_mask)
                logpi_ref_l = _compute_logprob_sums_batch(ref_model, bad_input, bad_labels, bad_mask)
            else:
                raw = model
                while hasattr(raw, "module"):
                    raw = getattr(raw, "module")
                disable_ctx = getattr(raw, "disable_adapter", None)
                if callable(disable_ctx):
                    with disable_ctx():
                        logpi_ref_w = _compute_logprob_sums_batch(
                            model, good_input, good_labels, good_mask
                        )
                        logpi_ref_l = _compute_logprob_sums_batch(
                            model, bad_input, bad_labels, bad_mask
                        )
                else:
                    logpi_ref_w = _compute_logprob_sums_batch(model, good_input, good_labels, good_mask)
                    logpi_ref_l = _compute_logprob_sums_batch(model, bad_input, bad_labels, bad_mask)

        good_weight_t = torch.tensor(good_weights, device=device)
        bad_weight_t = torch.tensor(bad_weights, device=device)
        logit = beta * (good_weight_t * (logpi_w - logpi_ref_w) - bad_weight_t * (logpi_l - logpi_ref_l))
        loss = F.softplus(-logit)
        return loss.mean()

    total_loss = torch.tensor(0.0, device=device)
    for example in batch:
        if "good_step" in example and "bad_step" in example:
            problem = example["problem"]
            prompt = example.get("prompt")
            prefix_good = example.get("prefix_good")
            prefix_bad = example.get("prefix_bad")
            good_step = example["good_step"]
            bad_step = example["bad_step"]
            good_weight = example.get("good_weight", 1.0)
            bad_weight = example.get("bad_weight", 1.0)
            ref_good_lp = example.get("ref_good_logprob")
            ref_bad_lp = example.get("ref_bad_logprob")

            if not (problem and good_step and bad_step):
                continue

            if isinstance(prefix_good, str) and prefix_good.strip():
                context_good = prefix_good
            elif isinstance(prompt, str) and prompt.strip():
                context_good = prompt
            else:
                context_good = str(problem) + "\n"

            if isinstance(prefix_bad, str) and prefix_bad.strip():
                context_bad = prefix_bad
            elif isinstance(prompt, str) and prompt.strip():
                context_bad = prompt
            else:
                context_bad = str(problem) + "\n"

            logpi_w = compute_logprob_sum(
                model, tokenizer, context_good, str(good_step), device, max_length=max_length
            )
            if isinstance(ref_good_lp, (int, float)):
                logpi_ref_w = torch.tensor(float(ref_good_lp), device=device)
            else:
                logpi_ref_w = compute_ref_logprob_sum(
                    model, ref_model, tokenizer, context_good, str(good_step), device, max_length=max_length
                )

            logpi_l = compute_logprob_sum(
                model, tokenizer, context_bad, str(bad_step), device, max_length=max_length
            )
            if isinstance(ref_bad_lp, (int, float)):
                logpi_ref_l = torch.tensor(float(ref_bad_lp), device=device)
            else:
                logpi_ref_l = compute_ref_logprob_sum(
                    model, ref_model, tokenizer, context_bad, str(bad_step), device, max_length=max_length
                )

            good_weight_t = torch.tensor(float(good_weight), device=device)
            bad_weight_t = torch.tensor(float(bad_weight), device=device)
            logit = beta * (good_weight_t * (logpi_w - logpi_ref_w) - bad_weight_t * (logpi_l - logpi_ref_l))
            loss = F.softplus(-logit)
            total_loss = total_loss + loss
            continue

        problem = example["problem"]
        prompt = example.get("prompt")
        ref_good_lp = example.get("ref_good_logprobs")
        ref_bad_lp = example.get("ref_bad_logprobs")
        good_steps = example["good_steps"]
        bad_steps = example["bad_steps"]
        good_rewards = example["good_rewards"]
        bad_rewards = example["bad_rewards"]

        if not (problem and good_steps and bad_steps and good_rewards and bad_rewards):
            continue

        good_steps = list(good_steps)
        bad_steps = list(bad_steps)
        good_rewards = list(good_rewards)
        bad_rewards = list(bad_rewards)

        if len(good_steps) != len(good_rewards) or len(bad_steps) != len(bad_rewards):
            continue

        if isinstance(prompt, str) and prompt.strip():
            context_good = prompt
            context_bad = prompt
        else:
            context_good = str(problem) + "\n"
            context_bad = str(problem) + "\n"

        good_term = torch.tensor(0.0, device=device)
        bad_term = torch.tensor(0.0, device=device)

        alpha_w = softmax_weights(good_rewards, gamma, device)
        alpha_l = softmax_weights(bad_rewards, -gamma, device)

        use_precomputed_ref = (
            isinstance(ref_good_lp, list)
            and isinstance(ref_bad_lp, list)
            and len(ref_good_lp) == len(good_steps)
            and len(ref_bad_lp) == len(bad_steps)
        )

        for i, (step_w, step_l) in enumerate(zip(good_steps, bad_steps), start=1):
            logpi_w = compute_logprob_sum(
                model, tokenizer, context_good, step_w, device, max_length=max_length
            )
            if use_precomputed_ref and isinstance(ref_good_lp[i - 1], (int, float)):
                logpi_ref_w = torch.tensor(float(ref_good_lp[i - 1]), device=device)
            else:
                logpi_ref_w = compute_ref_logprob_sum(
                    model, ref_model, tokenizer, context_good, step_w, device, max_length=max_length
                )
            good_term += alpha_w[i - 1] * (logpi_w - logpi_ref_w)

            logpi_l = compute_logprob_sum(
                model, tokenizer, context_bad, step_l, device, max_length=max_length
            )
            if use_precomputed_ref and isinstance(ref_bad_lp[i - 1], (int, float)):
                logpi_ref_l = torch.tensor(float(ref_bad_lp[i - 1]), device=device)
            else:
                logpi_ref_l = compute_ref_logprob_sum(
                    model, ref_model, tokenizer, context_bad, step_l, device, max_length=max_length
                )
            bad_term += alpha_l[i - 1] * (logpi_l - logpi_ref_l)

            # Keep the original step text formatting (often already includes "Step i:" markers).
            context_good = (context_good + step_w).rstrip() + "\n"
            context_bad = (context_bad + step_l).rstrip() + "\n"

        logit = beta * (good_term - bad_term)
        loss = F.softplus(-logit)  # = -log(sigmoid(logit))
        total_loss = total_loss + loss

    batch_size = max(len(batch), 1)
    return total_loss / batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train step-wise Chain-of-Preference DPO without TRL."
    )
    parser.add_argument("--data", type=str, help="JSONL with good/bad steps + rewards.")
    parser.add_argument(
        "--tot-jsonl",
        type=str,
        help="Raw ToT JSONL (produced by llama_tot_math). success_rate will be used as rewards.",
    )
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Base model checkpoint.")
    parser.add_argument("--ref-model-name-or-path", type=str, default=None, help="Reference model (optional).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save fine-tuned model.")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids to expose via CUDA_VISIBLE_DEVICES (e.g. 0,1,2). "
            "Applied before any CUDA init."
        ),
    )
    parser.add_argument(
        "--disable-p2p",
        action="store_true",
        help="Workaround for older RTX 4000-series drivers: set NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="none",
        choices=["none", "auto"],
        help="Model placement strategy: 'none' loads on a single device; 'auto' shards across GPUs (default: none).",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Enable DeepSpeed ZeRO via Accelerate using this JSON config file.",
    )
    parser.add_argument("--beta", type=float, default=0.1, help="Inverse temperature for DPO loss.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Reward sharpening parameter.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument(
        "--save-per-epoch",
        action="store_true",
        help="Save checkpoints (LoRA adapters or full model) at the end of each epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (num examples).")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Max number of reasoning steps to use per example (default: no limit). "
            "In --flatten-steps mode, steps beyond this index are skipped."
        ),
    )
    parser.add_argument(
        "--flatten-steps",
        action="store_true",
        help=(
            "Flatten each step into a separate (prefix, win, lose) training record to reduce memory. "
            "This changes the per-example loss into per-step loss."
        ),
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1).",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42).")
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum total length for context+step (default: 1024).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory.",
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 if available.")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="When using LoRA, save adapter weights under this directory (default: weights).",
    )
    parser.add_argument(
        "--require-precomputed-ref",
        action="store_true",
        help="Require ref_good_logprobs/ref_bad_logprobs in the loaded records; errors if missing.",
    )
    parser.add_argument(
        "--eval-after-train",
        action="store_true",
        help="Run an evaluation pass immediately after training using the in-memory model.",
    )
    parser.add_argument(
        "--eval-num-samples",
        type=int,
        default=None,
        help="If set, evaluate on this many samples; otherwise evaluate the full split.",
    )
    parser.add_argument(
        "--eval-id-cache",
        type=str,
        default=None,
        help="Optional JSON cache of eval indices (e.g. datas/eval_ids/hendrycks_level1_test_300.json).",
    )
    parser.add_argument(
        "--eval-dataset-root",
        type=str,
        default=None,
        help="Override evaluation dataset root path (defaults to dataset_utils defaults).",
    )
    add_lora_args(parser, default_r=8, default_target_modules="q_proj,v_proj")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key, value in TRAIN_CONFIG.items():
        if value is not None and hasattr(args, key):
            setattr(args, key, value)
    args.flatten_steps = True

    # GPU selection must happen before the first CUDA init (e.g., torch.cuda.is_available()).
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    using_ddp = local_rank >= 0
    if args.gpus and not using_ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    if args.disable_p2p:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")

    torch.manual_seed(int(args.seed))
    if using_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    accelerator = None
    if args.deepspeed_config:
        # Resolve relative DeepSpeed config path against repo root for robustness.
        ds_cfg = str(args.deepspeed_config)
        ds_cfg_path = Path(ds_cfg)
        if not ds_cfg_path.exists():
            candidate = _REPO_ROOT / ds_cfg
            if candidate.exists():
                ds_cfg_path = candidate
        if not ds_cfg_path.exists():
            raise FileNotFoundError(
                f"--deepspeed-config not found: {args.deepspeed_config!r}. "
                f"Tried: {Path(ds_cfg).resolve()} and {_REPO_ROOT / ds_cfg}"
            )
        ds_zero_stage = None
        try:
            with ds_cfg_path.open("r", encoding="utf-8") as f:
                ds_cfg_obj = json.load(f)
            if isinstance(ds_cfg_obj, dict):
                zo = ds_cfg_obj.get("zero_optimization")
                if isinstance(zo, dict) and "stage" in zo:
                    ds_zero_stage = int(zo["stage"])
        except Exception:
            ds_zero_stage = None

        # Work around older DeepSpeed versions importing `numpy.BUFSIZE`, removed in newer NumPy.
        try:  # pragma: no cover
            import numpy as np  # type: ignore

            if not hasattr(np, "BUFSIZE"):
                np.BUFSIZE = 8192  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            from accelerate import Accelerator  # type: ignore
            from accelerate.utils import DeepSpeedPlugin  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Requested --deepspeed-config but accelerate/deepspeed is not available in this environment."
            ) from exc

        mixed_precision = "no"
        if args.bf16:
            mixed_precision = "bf16"
        elif args.fp16:
            mixed_precision = "fp16"

        ds_plugin = DeepSpeedPlugin(hf_ds_config=str(ds_cfg_path))
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=int(args.grad_accum_steps),
            deepspeed_plugin=ds_plugin,
        )

    if args.data:
        records = load_chain_data(Path(args.data))
    elif args.tot_jsonl:
        records = convert_tot_to_chain_records(
            Path(args.tot_jsonl),
            seed=int(args.seed),
            max_steps=args.max_steps,
        )
    else:
        raise ValueError("Either --data or --tot-jsonl must be provided.")

    if args.max_steps is not None:
        capped: List[Dict[str, object]] = []
        max_steps = int(args.max_steps)
        for r in records:
            if "step_index" in r:
                step_index = r.get("step_index")
                if isinstance(step_index, int) and step_index > max_steps:
                    continue
            good_steps = r.get("good_steps")
            bad_steps = r.get("bad_steps")
            good_rewards = r.get("good_rewards")
            bad_rewards = r.get("bad_rewards")
            ref_good_lp = r.get("ref_good_logprobs")
            ref_bad_lp = r.get("ref_bad_logprobs")
            if (
                isinstance(good_steps, list)
                and isinstance(bad_steps, list)
                and isinstance(good_rewards, list)
                and isinstance(bad_rewards, list)
            ):
                r = dict(r)
                r["good_steps"] = good_steps[:max_steps]
                r["bad_steps"] = bad_steps[:max_steps]
                r["good_rewards"] = good_rewards[:max_steps]
                r["bad_rewards"] = bad_rewards[:max_steps]
                if isinstance(ref_good_lp, list):
                    r["ref_good_logprobs"] = ref_good_lp[:max_steps]
                if isinstance(ref_bad_lp, list):
                    r["ref_bad_logprobs"] = ref_bad_lp[:max_steps]
            capped.append(r)
        records = capped

    if args.flatten_steps:
        records = flatten_chain_records(records, gamma=float(args.gamma))

    if args.require_precomputed_ref:
        missing = 0
        for r in records:
            if "good_step" in r and "bad_step" in r:
                a = r.get("ref_good_logprob")
                b = r.get("ref_bad_logprob")
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    missing += 1
            else:
                a = r.get("ref_good_logprobs")
                b = r.get("ref_bad_logprobs")
                if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and len(a) > 0):
                    missing += 1
        if missing:
            raise ValueError(
                f"--require-precomputed-ref set but {missing} records are missing ref_*_logprobs. "
                "Run training/precompute_ref_logprobs.py first and pass its output via --data."
            )

    dataset = ChainPrefDataset(records)
    sampler = None
    if using_ddp:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda x: x,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    # DeepSpeed handles placement; device_map sharding is incompatible with it.
    if accelerator is not None and args.device_map != "none":
        raise ValueError("When using --deepspeed-config, please set --device-map none.")
    if using_ddp and args.device_map != "none":
        raise ValueError("When using DDP, please set --device-map none.")

    if accelerator is not None:
        device = accelerator.device
    else:
        if using_ddp:
            device = torch.device(f"cuda:{local_rank}")
        else:
            if args.device_map == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        device_map=("auto" if (accelerator is None and args.device_map == "auto") else None),
    )
    if accelerator is None and args.device_map != "auto":
        model.to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # HF gradient checkpointing uses torch.utils.checkpoint and can trigger known ZeRO-3 assertions
        # in some DeepSpeed versions; when running under DeepSpeed, rely on DS activation checkpointing
        # from the deepspeed config instead.
        if accelerator is None:
            model.gradient_checkpointing_enable()
        else:
            if "ds_zero_stage" in locals() and ds_zero_stage == 3:
                if accelerator.is_main_process:
                    print(
                        "[chain_pref] NOTE: disabling HF --gradient-checkpointing under DeepSpeed ZeRO-3; "
                        "use activation_checkpointing in the deepspeed config instead."
                    )
            else:
                model.gradient_checkpointing_enable()
                if accelerator.is_main_process:
                    print("[chain_pref] enabled HF --gradient-checkpointing under DeepSpeed (ZeRO-1/2).")
            if accelerator.is_main_process and ("ds_zero_stage" not in locals() or ds_zero_stage is None):
                print(
                    "[chain_pref] NOTE: could not detect ZeRO stage from config; "
                    "if you see DeepSpeed ZeRO-3 assertions, disable HF --gradient-checkpointing."
                )
    model = apply_lora_from_args(model, args)
    # If no ref model is provided, we will use adapter-disabled base model as reference (LoRA case).
    ref_model = None
    if args.ref_model_name_or_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_name_or_path,
            dtype=dtype,
            device_map=("auto" if (accelerator is None and args.device_map == "auto") else None),
        )
        if accelerator is None and args.device_map != "auto":
            ref_model.to(device)
        freeze_model(ref_model)

    # Use the embedding device for inputs/tensors (important for device_map=auto).
    if accelerator is None and not using_ddp:
        device = _infer_input_device(model, device)

    # DeepSpeed ZeRO-3 can assert if the same Parameter appears multiple times in the optimizer param list.
    uniq_params = []
    seen_param_ids = set()
    for p in trainable_parameters(model):
        pid = id(p)
        if pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)
        uniq_params.append(p)
    optimizer = AdamW(uniq_params, lr=args.learning_rate)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    if accelerator is not None:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
    elif using_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    progress = None
    if accelerator is not None:
        if accelerator.is_main_process:
            try:
                from tqdm import tqdm  # type: ignore

                progress = tqdm(total=total_steps, desc="Train", unit="step")
            except Exception:
                progress = None
    elif not using_ddp or dist.get_rank() == 0:
        try:
            from tqdm import tqdm  # type: ignore

            progress = tqdm(total=total_steps, desc="Train", unit="step")
        except Exception:
            progress = None

    def _unwrap_ddp(m):
        return m.module if isinstance(m, DDP) else m

    def _run_eval_phase(phase_label: str) -> None:
        is_rank0 = True
        if using_ddp and dist.is_initialized():
            is_rank0 = dist.get_rank() == 0
        if using_ddp and dist.is_initialized() and not is_rank0 and accelerator is None:
            dist.barrier()
            return
        if accelerator is None:
            if is_rank0:
                print(f"[chain_pref] running {phase_label}...")
        elif accelerator.is_main_process:
            print(f"[chain_pref] running {phase_label}...")
        if accelerator is None and is_rank0:
            print("[rank0] eval: start", flush=True)

        eval_dataset_name = "hendrycks_math"
        eval_split = "test"
        eval_level = None
        eval_id_cache = args.eval_id_cache
        eval_num_samples = args.eval_num_samples if args.eval_num_samples is not None else 64
        if args.tot_jsonl:
            lower = str(args.tot_jsonl).lower()
            if "gsm8k" in lower:
                eval_dataset_name = "gsm8k"
            elif "svamp" in lower:
                eval_dataset_name = "svamp"
            elif "hendrycks" in lower or "math" in lower:
                eval_dataset_name = "hendrycks_math"
                for lvl in ["l1", "l2", "l3", "l4", "l5"]:
                    if f"_{lvl}" in lower or f"{lvl}" in lower:
                        try:
                            eval_level = int(lvl[-1])
                        except ValueError:
                            eval_level = None
                        break

        eval_root = args.eval_dataset_root or default_dataset_path(eval_dataset_name)
        if accelerator is None and is_rank0:
            print(
                f"[rank0] eval: root={eval_root} cache={eval_id_cache}",
                flush=True,
            )

        def _load_eval_indices() -> Optional[List[int]]:
            cache_path = None
            if eval_id_cache:
                cache_path = Path(eval_id_cache)
            elif eval_dataset_name == "hendrycks_math" and eval_level is not None and eval_num_samples:
                default_cache = (
                    _REPO_ROOT
                    / "datas"
                    / "eval_ids"
                    / f"hendrycks_level{eval_level}_{eval_split}_{int(eval_num_samples)}.json"
                )
                if default_cache.exists():
                    cache_path = default_cache
            if cache_path is None:
                return None
            is_rank0_local = True
            if using_ddp and dist.is_initialized():
                is_rank0_local = dist.get_rank() == 0
            if not cache_path.exists() or cache_path.stat().st_size == 0:
                if eval_num_samples is None:
                    raise FileNotFoundError(f"--eval-id-cache not found: {cache_path}")
                if is_rank0_local:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    indices: List[int] = []
                    for idx, _raw in enumerate(
                        iter_samples(eval_dataset_name, eval_root, split=eval_split)
                    ):
                        indices.append(idx)
                    rng = random.Random(int(args.seed))
                    rng.shuffle(indices)
                    indices = indices[: int(eval_num_samples)]
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump({"indices": indices}, f, ensure_ascii=False, indent=2)
                if using_ddp and dist.is_initialized():
                    dist.barrier()
            with cache_path.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    if not is_rank0_local:
                        if using_ddp and dist.is_initialized():
                            dist.barrier()
                    with cache_path.open("r", encoding="utf-8") as retry_f:
                        data = json.load(retry_f)
            indices = data.get("indices") if isinstance(data, dict) else None
            if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
                raise ValueError(f"Invalid eval id cache format: {cache_path}")
            if eval_num_samples is not None:
                return indices[: int(eval_num_samples)]
            return indices

        eval_indices = _load_eval_indices()
        if accelerator is None and is_rank0:
            print("[rank0] eval: load indices done", flush=True)

        def _iter_eval_samples():
            count = 0
            if eval_indices is not None:
                wanted = set(eval_indices)
                collected: Dict[int, Dict[str, object]] = {}
                for idx, raw in enumerate(iter_samples(eval_dataset_name, eval_root, split=eval_split)):
                    if idx not in wanted:
                        continue
                    sample = normalize_sample(eval_dataset_name, raw)
                    collected[idx] = sample
                    if len(collected) >= len(wanted):
                        break
                for idx in eval_indices:
                    sample = collected.get(idx)
                    if sample is None:
                        continue
                    yield sample
                    count += 1
                    if eval_num_samples is not None and count >= int(eval_num_samples):
                        break
                return

            for raw in iter_samples(eval_dataset_name, eval_root, split=eval_split):
                sample = normalize_sample(eval_dataset_name, raw)
                if eval_dataset_name == "hendrycks_math" and eval_level is not None:
                    lvl = sample.get("level")
                    if isinstance(lvl, str):
                        if str(eval_level) not in lvl:
                            continue
                    elif isinstance(lvl, int):
                        if int(lvl) != eval_level:
                            continue
                    else:
                        continue
                yield sample
                count += 1
                if eval_num_samples is not None and count >= int(eval_num_samples):
                    break

        def _run_eval():
            correct = 0
            total = 0
            gen_model = _unwrap_ddp(model)
            was_training = gen_model.training
            prev_use_cache = getattr(gen_model.config, "use_cache", None)
            gen_model.eval()
            if prev_use_cache is not None:
                gen_model.config.use_cache = True
            print("[rank0] eval: begin generate loop", flush=True)
            for sample in _iter_eval_samples():
                print(f"[rank0] eval sample {total + 1} begin", flush=True)
                problem = (
                    sample.get("problem")
                    or sample.get("question")
                    or sample.get("prompt")
                )
                if not problem:
                    continue
                solution = sample.get("solution") or sample.get("answer")
                level = sample.get("level")
                if eval_dataset_name in {"gsm8k", "svamp"}:
                    num_steps = steps_for_dataset(eval_dataset_name)
                else:
                    num_steps = steps_for_level(level)
                prompt = (
                    "You are an expert math problem solver. "
                    "You must reason step by step and avoid logical or arithmetic mistakes.\n\n"
                    "Solve the following math problem.\n"
                    f"You MUST use exactly {num_steps} reasoning steps, "
                    "After the reasoning, output the final answer in the last line "
                    "using the format: `Answer: <final_answer>`.\n\n"
                    f"Problem: {problem}\n\n"
                    "Reasoning step by step:\n"
                    "Step 1:"
                )
                encoded = tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                with torch.inference_mode():
                    output_ids = gen_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.2,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = output_ids[0, input_ids.shape[1] :]
                completion = tokenizer.decode(generated, skip_special_tokens=True)
                is_correct_flag = False
                if solution is not None:
                    is_correct_flag = is_model_correct(completion, solution)
                if is_correct_flag:
                    correct += 1
                total += 1
                if total % 10 == 0:
                    print(f"[chain_pref][eval] processed {total}")
            if was_training:
                gen_model.train()
            if prev_use_cache is not None:
                gen_model.config.use_cache = prev_use_cache
            acc = (correct / total) if total > 0 else None
            return {"num_samples": total, "num_correct": correct, "accuracy": acc}

        if accelerator is not None:
            if accelerator.is_main_process:
                eval_summary = _run_eval()
                print(json.dumps({"eval": eval_summary}, ensure_ascii=False, indent=2))
            accelerator.wait_for_everyone()
        else:
            if is_rank0:
                eval_summary = _run_eval()
                print(json.dumps({"eval": eval_summary}, ensure_ascii=False, indent=2))
            if using_ddp and dist.is_initialized():
                dist.barrier()

    if args.eval_after_train and not using_ddp:
        _run_eval_phase("eval-before-train")

    out_dir = Path(args.output_dir)
    if getattr(args, "use_lora", False):
        base = Path(args.weights_dir)
        if out_dir.is_absolute():
            out_dir = base / out_dir.name
        else:
            out_dir = base / out_dir

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        if using_ddp and sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader, start=1):
            ctx = accelerator.accumulate(model) if accelerator is not None else nullcontext()
            with ctx:
                loss = compute_stepwise_loss(
                    batch=batch,
                    model=model,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    beta=args.beta,
                    gamma=args.gamma,
                    device=(accelerator.device if accelerator is not None else device),
                    max_length=args.max_length,
                )
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if progress is not None:
                progress.update(1)
            elif (not using_ddp or dist.get_rank() == 0) and global_step % 10 == 0:
                print(f"[chain_pref] progress {global_step}/{total_steps}")

            if step % 10 == 0:
                if accelerator is None:
                    if not using_ddp or dist.get_rank() == 0:
                        print(f"[chain_pref] epoch {epoch + 1}, step {step}, loss={loss.item():.4f}")
                elif accelerator.is_main_process:
                    print(f"[chain_pref] epoch {epoch + 1}, step {step}, loss={loss.item():.4f}")

        if args.save_per_epoch:
            epoch_dir = Path(f"{out_dir}_epoch{epoch + 1}")
            epoch_dir.mkdir(parents=True, exist_ok=True)
            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(str(epoch_dir))
                    print(f"[chain_pref] saved epoch {epoch + 1} -> {epoch_dir}")
                accelerator.wait_for_everyone()
            else:
                if not using_ddp or dist.get_rank() == 0:
                    to_save = _unwrap_ddp(model)
                    to_save.save_pretrained(str(epoch_dir))
                    print(f"[chain_pref] saved epoch {epoch + 1} -> {epoch_dir}")
                if using_ddp and dist.is_initialized():
                    dist.barrier()

    if progress is not None:
        progress.close()

    if args.eval_after_train and not using_ddp:
        _run_eval_phase("eval-after-train")

    out_dir.mkdir(parents=True, exist_ok=True)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(str(out_dir))
            print(f"[chain_pref] saved -> {out_dir}")
    else:
        if is_rank0:
            to_save = _unwrap_ddp(model)
            to_save.save_pretrained(str(out_dir))
            print(f"[chain_pref] saved -> {out_dir}")
        if using_ddp and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
