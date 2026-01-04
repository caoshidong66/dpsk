"""
Minimal Chain-of-Preference (flattened) trainer.

Only supports flattened (prefix, win, lose) triples.
Uses DDP if launched via torchrun.
No eval, no DeepSpeed, no extra CLI args.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from lora_utils import freeze_model, trainable_parameters


TRAIN_CONFIG: Dict[str, Optional[object]] = {
    "batch_size":4,
    "grad_accum_steps": 1,
    "learning_rate": 2e-5,
    "epochs": 3000,
    "max_length": 1024,
    "beta": 0.1,
    "gamma": 2.0,
    "bf16": True,
    "fp16": False,
    "seed": 42,
    "log_every": 10,
    "save_per_epoch": True,
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 0,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,v_proj",
    "lora_bias": "none",
    "gpus": "1,4,5,6",  # e.g. "0,1,2,3"
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


def _apply_lora_from_config(model, cfg: Dict[str, Optional[object]]):
    if not cfg.get("use_lora"):
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("LoRA requested but peft is not available.") from exc

    r = int(cfg.get("lora_r", 8))
    alpha = int(cfg.get("lora_alpha", 0)) or (2 * r)
    dropout = float(cfg.get("lora_dropout", 0.05))
    target_modules = [
        x.strip() for x in str(cfg.get("lora_target_modules", "q_proj,v_proj")).split(",") if x.strip()
    ]
    bias = str(cfg.get("lora_bias", "none"))

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


def softmax_weights(rewards: List[float], gamma: float) -> List[float]:
    if not rewards:
        return []
    r = torch.tensor(rewards, dtype=torch.float32)
    weights = F.softmax(gamma * r, dim=0)
    return [float(x) for x in weights.tolist()]


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
        if length <= 0:
            continue
        alpha_w = softmax_weights([float(x) for x in good_rewards[:length]], gamma=float(gamma))
        alpha_l = softmax_weights([float(x) for x in bad_rewards[:length]], gamma=-float(gamma))

        for i in range(length):
            out: Dict[str, object] = {
                "problem": problem,
                "good_step": good_steps[i],
                "bad_step": bad_steps[i],
                "good_weight": alpha_w[i],
                "bad_weight": alpha_l[i],
            }
            if isinstance(prompt, str) and prompt.strip():
                out["prompt"] = prompt
            flat.append(out)
    return flat


def convert_tot_to_chain_records(
    jsonl_path: Path,
    *,
    seed: int,
    gamma: float,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    raw = load_chain_data(jsonl_path)
    rng = random.Random(int(seed))
    for rec in raw:
        tot = rec.get("tot") if isinstance(rec, dict) else None
        base = tot if isinstance(tot, dict) else rec

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
            length = min(len(good_steps), len(bad_steps), len(good_rewards), len(bad_rewards))
            if length <= 0:
                continue
            alpha_w = softmax_weights(good_rewards[:length], gamma=float(gamma))
            alpha_l = softmax_weights(bad_rewards[:length], gamma=-float(gamma))
            for i in range(length):
                out = {
                    "problem": problem,
                    "good_step": good_steps[i],
                    "bad_step": bad_steps[i],
                    "good_weight": alpha_w[i],
                    "bad_weight": alpha_l[i],
                }
                if isinstance(prompt, str) and prompt.strip():
                    out["prompt"] = prompt
                records.append(out)
    return records


def infer_dataset_meta(path: str) -> Dict[str, Optional[object]]:
    lower = path.lower()
    dataset_name: Optional[str] = None
    level: Optional[int] = None
    if "gsm8k" in lower:
        dataset_name = "gsm8k"
    elif "svamp" in lower:
        dataset_name = "svamp"
    elif "hendrycks" in lower or "math" in lower:
        dataset_name = "hendrycks_math"
        for lvl in ["l1", "l2", "l3", "l4", "l5"]:
            if f"_{lvl}" in lower or f"{lvl}" in lower:
                try:
                    level = int(lvl[-1])
                except ValueError:
                    level = None
                break
    return {"dataset_name": dataset_name, "level": level}


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


def compute_stepwise_loss(
    batch: List[Dict[str, object]],
    model,
    ref_model,
    tokenizer,
    beta: float,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    contexts_good: List[str] = []
    contexts_bad: List[str] = []
    steps_good: List[str] = []
    steps_bad: List[str] = []
    good_weights: List[float] = []
    bad_weights: List[float] = []

    for example in batch:
        problem = example["problem"]
        prompt = example.get("prompt")
        good_step = example["good_step"]
        bad_step = example["bad_step"]
        good_weight = float(example.get("good_weight", 1.0))
        bad_weight = float(example.get("bad_weight", 1.0))

        context = prompt if isinstance(prompt, str) and prompt.strip() else str(problem) + "\n"
        contexts_good.append(context)
        contexts_bad.append(context)
        steps_good.append(str(good_step))
        steps_bad.append(str(bad_step))
        good_weights.append(good_weight)
        bad_weights.append(bad_weight)

    good_input, good_labels, good_mask = _build_batch_inputs(
        tokenizer, contexts_good, steps_good, device, max_length
    )
    bad_input, bad_labels, bad_mask = _build_batch_inputs(
        tokenizer, contexts_bad, steps_bad, device, max_length
    )

    logpi_w = _compute_logprob_sums_batch(model, good_input, good_labels, good_mask)
    logpi_l = _compute_logprob_sums_batch(model, bad_input, bad_labels, bad_mask)

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
                logpi_ref_w = _compute_logprob_sums_batch(model, good_input, good_labels, good_mask)
                logpi_ref_l = _compute_logprob_sums_batch(model, bad_input, bad_labels, bad_mask)
        else:
            logpi_ref_w = _compute_logprob_sums_batch(model, good_input, good_labels, good_mask)
            logpi_ref_l = _compute_logprob_sums_batch(model, bad_input, bad_labels, bad_mask)

    good_weight_t = torch.tensor(good_weights, device=device)
    bad_weight_t = torch.tensor(bad_weights, device=device)
    logit = float(beta) * (
        good_weight_t * (logpi_w - logpi_ref_w) - bad_weight_t * (logpi_l - logpi_ref_l)
    )
    loss = F.softplus(-logit)
    return loss.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal chain preference trainer (flattened only).")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--tot-jsonl", type=str, default=None)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--ref-model-name-or-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key, value in TRAIN_CONFIG.items():
        setattr(args, key, value)
    if args.beta is not None:
        setattr(args, "beta", float(args.beta))
    if args.gamma is not None:
        setattr(args, "gamma", float(args.gamma))

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if isinstance(getattr(args, "gpus", None), str) and args.gpus.strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    using_ddp = local_rank >= 0
    if using_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    torch.manual_seed(int(args.seed))

    data_path = args.data or args.tot_jsonl
    if not data_path:
        raise ValueError("Either --data or --tot-jsonl must be provided.")
    meta = infer_dataset_meta(str(data_path))
    if not using_ddp or dist.get_rank() == 0:
        print(
            f"[chain_pref] dataset={meta.get('dataset_name')} level={meta.get('level')}",
            flush=True,
        )

    if args.data:
        records = load_chain_data(Path(args.data))
        records = flatten_chain_records(records, gamma=float(args.gamma))
    else:
        records = convert_tot_to_chain_records(
            Path(args.tot_jsonl),
            seed=int(args.seed),
            gamma=float(args.gamma),
        )
    dataset = ChainPrefDataset(records)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if using_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda x: x,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    device = torch.device(f"cuda:{local_rank}" if using_ddp else "cuda:0")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    model = _apply_lora_from_config(model, TRAIN_CONFIG)

    ref_model = None
    if args.ref_model_name_or_path:
        ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name_or_path, torch_dtype=dtype)
        ref_model.to(device)
        freeze_model(ref_model)
    elif args.use_lora:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
        ref_model.to(device)
        freeze_model(ref_model)

    if using_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(list(trainable_parameters(model)), lr=float(args.learning_rate))
    total_steps = len(dataloader) * int(args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )

    progress = None
    if not using_ddp or dist.get_rank() == 0:
        try:
            from tqdm import tqdm  # type: ignore

            progress = tqdm(total=total_steps, desc="Train", unit="step")
        except Exception:
            progress = None

    global_step = 0
    model.train()
    for epoch in range(int(args.epochs)):
        if using_ddp and sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader, start=1):
            loss = compute_stepwise_loss(
                batch=batch,
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                beta=float(args.beta),
                device=device,
                max_length=int(args.max_length),
            )
            loss = loss / int(args.grad_accum_steps)
            loss.backward()
            if step % int(args.grad_accum_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if (not using_ddp or dist.get_rank() == 0) and global_step % int(args.log_every) == 0:
                print(f"[chain_pref] epoch {epoch + 1}, step {step}, loss={loss.item():.4f}")
            if progress is not None:
                progress.update(1)

        if args.save_per_epoch:
            out_dir = Path(args.output_dir)
            if args.use_lora:
                out_dir = Path("weights") / out_dir.name
            epoch_dir = Path(f"{out_dir}_epoch{epoch + 1}")
            epoch_dir.mkdir(parents=True, exist_ok=True)
            if not using_ddp or dist.get_rank() == 0:
                to_save = model.module if isinstance(model, DDP) else model
                to_save.save_pretrained(str(epoch_dir))
                print(f"[chain_pref] saved epoch {epoch + 1} -> {epoch_dir}")
            if using_ddp:
                dist.barrier()

    if progress is not None:
        progress.close()

    out_dir = Path(args.output_dir)
    if args.use_lora:
        out_dir = Path("weights") / out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not using_ddp or dist.get_rank() == 0:
        to_save = model.module if isinstance(model, DDP) else model
        to_save.save_pretrained(str(out_dir))
        print(f"[chain_pref] saved -> {out_dir}")
    if using_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
