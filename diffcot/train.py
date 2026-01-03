"""
Minimal DiffCoT trainer (preference-style, sliding window).

This script supports two-loss training:
  - refine loss (Refined Step k-1)
  - draft loss  (Draft Step k)

Data format (JSONL) supports two sources:
1) ToT-style rollouts (preferred):
  - "steps_trace": list of steps
  - each step has "candidates" with success_rate
   We train to refine step k-1 (top2 -> top1) and draft step k (top2 -> top1).
2) DiffCoT refine logs (fallback):
   - "steps": list/dict with "draft" + "refined"

Modes:
- preference: DPO-style loss between refined (good) and draft (bad)
- sft: standard next-token loss on refined only

We convert each step into a training triple with a sliding window prompt:
  Prompt includes problem + recent refined steps + current draft step.
  The model is trained to prefer "Refined Step k: <refined>" over the draft.

No TRL, no eval, no DeepSpeed. DDP works via torchrun.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    "batch_size": 8,
    "grad_accum_steps": 1,
    "learning_rate": 2e-5,
    "epochs": 1,
    "max_length": 1024,
    "beta": 0.1,
    "window_size": 3,  # number of previous refined steps to include
    "refine_weight": 0.7,
    "draft_weight": 0.3,
    "draft_prev_top2_prob": 0.2,
    "mode": "preference",  # "preference" or "sft"
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
    "gpus": "0,1,2,3",  # e.g. "0,1,2,3"
}


# Allow running from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


class PairDataset(Dataset):
    def __init__(self, records: List[Dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.records[idx]


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _iter_steps(steps_field: object) -> Iterable[Tuple[int, Dict[str, object]]]:
    """
    Normalize steps into (step_index, entry) tuples.

    Supports:
      - dict: {"1": {...}, 2: {...}}
      - list: [{...}, {...}] (index is 1-based)
    """
    if isinstance(steps_field, dict):
        items: List[Tuple[int, Dict[str, object]]] = []
        for key, val in steps_field.items():
            try:
                step_idx = int(key)
            except Exception:
                continue
            if isinstance(val, dict):
                items.append((step_idx, val))
        for step_idx, entry in sorted(items, key=lambda x: x[0]):
            yield step_idx, entry
        return
    if isinstance(steps_field, list):
        for i, entry in enumerate(steps_field, start=1):
            if isinstance(entry, dict):
                yield i, entry


def _build_refine_prompt(
    problem: str,
    previous_steps: List[str],
    step_idx: int,
    draft_prev: str,
) -> str:
    """
    DiffCoT-style refine prompt:
      - refine Step (k-1) only
    """
    lines = [
        f"You are an expert math tutor. You work in step {step_idx} now.",
        "Follow the requested output format verbatim.",
        "",
        "Problem:",
        problem,
        "",
    ]
    if previous_steps:
        lines.append("Current Progress:")
        for idx, text in enumerate(previous_steps, start=1):
            lines.append(f"Step {idx}:\n{text}")
        lines.append("")

    lines.append(
        f"Draft Step {step_idx - 1} (current):\n{draft_prev}\n"
        f"Refine Step {step_idx - 1}.\n"
        f"Format:\nRefined Step {step_idx - 1}:\n..."
    )
    return "\n".join(lines)


def _build_draft_prompt(
    problem: str,
    previous_steps: List[str],
    step_idx: int,
    refined_prev: str,
) -> str:
    """
    DiffCoT-style draft prompt:
      - given refined Step (k-1), draft Step k only
    """
    lines = [
        f"You are an expert math tutor. You work in step {step_idx} now.",
        "Follow the requested output format verbatim.",
        "",
        "Problem:",
        problem,
        "",
    ]
    if previous_steps:
        lines.append("Current Progress:")
        for idx, text in enumerate(previous_steps, start=1):
            lines.append(f"Step {idx}:\n{text}")
        lines.append("")

    lines.append(
        f"Refined Step {step_idx - 1}:\n{refined_prev}\n"
        f"Draft Step {step_idx}.\n"
        f"Format:\nDraft Step {step_idx}:\n..."
    )
    return "\n".join(lines)


def _pick_top2_candidates(
    candidates: List[Dict[str, object]],
) -> Optional[Tuple[Dict[str, object], Dict[str, object]]]:
    """
    Pick (best, second_best) by success_rate/success_count.
    Skip if not enough candidates or same success_rate.
    """
    if len(candidates) < 2:
        return None
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c.get("success_rate") or 0.0, c.get("success_count") or 0),
        reverse=True,
    )
    best = sorted_cands[0]
    second = sorted_cands[1]
    sr_best = best.get("success_rate")
    sr_second = second.get("success_rate")
    if not isinstance(sr_best, (int, float)) or not isinstance(sr_second, (int, float)):
        return None
    if sr_best == sr_second:
        return None
    return best, second


def _build_pairs_from_steps_trace(
    problem: str,
    steps_trace: List[Dict[str, object]],
    window_size: int,
    rng: random.Random,
    draft_prev_top2_prob: float,
) -> List[Dict[str, object]]:
    """
    Build pairs from ToT-style candidates:
      - refine loss: good=top1, bad=top2 for Step (k-1)
      - draft  loss: good=top1, bad=top2 for Step k
    Context uses previous top-1 steps (sliding window, up to k-2).
    """
    pairs: List[Dict[str, object]] = []
    # Map step_index -> (top1, top2, chosen)
    top_map: Dict[int, Tuple[str, str, Optional[str]]] = {}
    for entry in steps_trace:
        step_idx = entry.get("step_index")
        candidates = entry.get("candidates")
        if not isinstance(step_idx, int) or not isinstance(candidates, list):
            continue
        picked = _pick_top2_candidates(candidates)
        if picked is None:
            continue
        best, second = picked
        best_text = best.get("step_text")
        second_text = second.get("step_text")
        if not isinstance(best_text, str) or not isinstance(second_text, str):
            continue
        chosen_text = entry.get("chosen_step_text")
        if not isinstance(chosen_text, str):
            chosen_text = None
        top_map[step_idx] = (best_text, second_text, chosen_text)

    refined_history: List[str] = []
    max_step = max(top_map.keys(), default=0)
    for step_idx in range(2, max_step + 1):
        if step_idx - 1 not in top_map or step_idx not in top_map:
            continue
        prev_best, prev_second, prev_chosen = top_map[step_idx - 1]
        curr_best, curr_second, _curr_chosen = top_map[step_idx]

        if window_size > 0:
            context_steps = refined_history[-window_size:]
        else:
            context_steps = refined_history[:]

        prompt_refine = _build_refine_prompt(
            problem=problem,
            previous_steps=context_steps,
            step_idx=step_idx,
            draft_prev=prev_second,
        )
        refined_prev_for_context = prev_chosen or prev_best
        if draft_prev_top2_prob > 0 and rng.random() < draft_prev_top2_prob:
            refined_prev_for_context = prev_second

        prompt_draft = _build_draft_prompt(
            problem=problem,
            previous_steps=context_steps,
            step_idx=step_idx,
            refined_prev=refined_prev_for_context,
        )
        pairs.append(
            {
                "problem": problem,
                "prompt_refine": prompt_refine,
                "good_refine": f"Refined Step {step_idx - 1}:\n{prev_best}",
                "bad_refine": f"Refined Step {step_idx - 1}:\n{prev_second}",
                "prompt_draft": prompt_draft,
                "good_draft": f"Draft Step {step_idx}:\n{curr_best}",
                "bad_draft": f"Draft Step {step_idx}:\n{curr_second}",
                "good_weight": 1.0,
                "bad_weight": 1.0,
            }
        )
        refined_history.append(refined_prev_for_context)
    return pairs


def _build_pairs_from_refine_logs(
    problem: str,
    steps_field: object,
    window_size: int,
) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []
    # Build step map first so we can access step k and k-1.
    step_map: Dict[int, Dict[str, object]] = {}
    for step_idx, entry in _iter_steps(steps_field):
        step_map[step_idx] = entry

    refined_history: List[str] = []
    max_step = max(step_map.keys(), default=0)
    for step_idx in range(2, max_step + 1):
        prev_entry = step_map.get(step_idx - 1)
        curr_entry = step_map.get(step_idx)
        if not isinstance(prev_entry, dict) or not isinstance(curr_entry, dict):
            continue
        prev_draft = prev_entry.get("draft")
        prev_refined = prev_entry.get("refined")
        curr_draft = curr_entry.get("draft")
        if not isinstance(prev_draft, str) or not isinstance(prev_refined, str) or not isinstance(curr_draft, str):
            continue

        if window_size > 0:
            context_steps = refined_history[-window_size:]
        else:
            context_steps = refined_history[:]
        prompt_refine = _build_refine_prompt(
            problem=problem,
            previous_steps=context_steps,
            step_idx=step_idx,
            draft_prev=prev_draft,
        )
        prompt_draft = _build_draft_prompt(
            problem=problem,
            previous_steps=context_steps,
            step_idx=step_idx,
            refined_prev=prev_refined,
        )
        pairs.append(
            {
                "problem": problem,
                "prompt_refine": prompt_refine,
                "good_refine": f"Refined Step {step_idx - 1}:\n{prev_refined}",
                "bad_refine": f"Refined Step {step_idx - 1}:\n{prev_draft}",
                "prompt_draft": prompt_draft,
                "good_draft": f"Draft Step {step_idx}:\n{curr_draft}",
                "bad_draft": "",
                "good_weight": 1.0,
                "bad_weight": 1.0,
            }
        )
        refined_history.append(prev_refined)
    return pairs


def build_pairs_from_records(
    records: List[Dict[str, object]],
    *,
    window_size: int,
    seed: int,
    draft_prev_top2_prob: float,
) -> List[Dict[str, object]]:
    """
    Build (prompt, good_step, bad_step) pairs from:
      - ToT rollouts (steps_trace candidates)
      - fallback refine logs (steps draft/refined)
    """
    pairs: List[Dict[str, object]] = []
    rng = random.Random(int(seed))
    for rec in records:
        base = rec
        if isinstance(rec, dict) and isinstance(rec.get("tot"), dict):
            base = rec["tot"]
        problem = base.get("problem") or base.get("question") or base.get("prompt")
        if not isinstance(problem, str):
            continue

        steps_trace = base.get("steps_trace")
        if isinstance(steps_trace, list) and steps_trace:
            pairs.extend(
                _build_pairs_from_steps_trace(
                    problem,
                    steps_trace,
                    window_size,
                    rng,
                    draft_prev_top2_prob,
                )
            )
            continue

        steps_field = base.get("steps")
        if steps_field:
            pairs.extend(_build_pairs_from_refine_logs(problem, steps_field, window_size))
    return pairs


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


def compute_preference_loss(
    batch: List[Dict[str, object]],
    model,
    ref_model,
    tokenizer,
    beta: float,
    refine_weight: float,
    draft_weight: float,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    def _dpo_loss_for_fields(
        prompt_key: str,
        good_key: str,
        bad_key: str,
    ) -> Optional[torch.Tensor]:
        contexts: List[str] = []
        good_steps: List[str] = []
        bad_steps: List[str] = []
        good_weights: List[float] = []
        bad_weights: List[float] = []

        for example in batch:
            prompt = example.get(prompt_key)
            good = example.get(good_key)
            bad = example.get(bad_key)
            if (
                not isinstance(prompt, str)
                or not isinstance(good, str)
                or not isinstance(bad, str)
                or good == bad
                or not bad.strip()
            ):
                continue
            contexts.append(prompt)
            good_steps.append(good)
            bad_steps.append(bad)
            good_weights.append(float(example.get("good_weight", 1.0)))
            bad_weights.append(float(example.get("bad_weight", 1.0)))

        if not contexts:
            return None

        good_input, good_labels, good_mask = _build_batch_inputs(
            tokenizer, contexts, good_steps, device, max_length
        )
        bad_input, bad_labels, bad_mask = _build_batch_inputs(
            tokenizer, contexts, bad_steps, device, max_length
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

    refine_loss = _dpo_loss_for_fields("prompt_refine", "good_refine", "bad_refine")
    draft_loss = _dpo_loss_for_fields("prompt_draft", "good_draft", "bad_draft")

    if refine_loss is None and draft_loss is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if refine_loss is None:
        return draft_loss
    if draft_loss is None:
        return refine_loss
    return float(refine_weight) * refine_loss + float(draft_weight) * draft_loss


def compute_sft_loss(
    batch: List[Dict[str, object]],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    def _sft_loss_for_fields(prompt_key: str, good_key: str) -> Optional[torch.Tensor]:
        contexts: List[str] = []
        good_steps: List[str] = []
        for example in batch:
            prompt = example.get(prompt_key)
            good = example.get(good_key)
            if not isinstance(prompt, str) or not isinstance(good, str):
                continue
            contexts.append(prompt)
            good_steps.append(good)
        if not contexts:
            return None
        input_ids, labels, mask = _build_batch_inputs(
            tokenizer, contexts, good_steps, device, max_length
        )
        outputs = model(input_ids=input_ids, attention_mask=mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
        return loss

    refine_loss = _sft_loss_for_fields("prompt_refine", "good_refine")
    draft_loss = _sft_loss_for_fields("prompt_draft", "good_draft")

    if refine_loss is None and draft_loss is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if refine_loss is None:
        return draft_loss
    if draft_loss is None:
        return refine_loss
    return float(TRAIN_CONFIG["refine_weight"]) * refine_loss + float(TRAIN_CONFIG["draft_weight"]) * draft_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DiffCoT refinement trainer.")
    parser.add_argument("--data-jsonl", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--ref-model-name-or-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["preference", "sft"])
    parser.add_argument("--refine-weight", type=float, default=None)
    parser.add_argument("--draft-weight", type=float, default=None)
    parser.add_argument("--draft-prev-top2-prob", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key, value in TRAIN_CONFIG.items():
        setattr(args, key, value)
    if args.beta is not None:
        setattr(args, "beta", float(args.beta))
    if args.window_size is not None:
        setattr(args, "window_size", int(args.window_size))
    if args.mode is not None:
        setattr(args, "mode", str(args.mode))
    if args.refine_weight is not None:
        setattr(args, "refine_weight", float(args.refine_weight))
    if args.draft_weight is not None:
        setattr(args, "draft_weight", float(args.draft_weight))
    if args.draft_prev_top2_prob is not None:
        setattr(args, "draft_prev_top2_prob", float(args.draft_prev_top2_prob))

    if isinstance(args.gpus, str) and args.gpus.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    using_ddp = local_rank >= 0
    if using_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    torch.manual_seed(int(args.seed))

    records = load_jsonl(Path(args.data_jsonl))
    pair_records = build_pairs_from_records(
        records,
        window_size=int(args.window_size),
        seed=int(args.seed),
        draft_prev_top2_prob=float(args.draft_prev_top2_prob),
    )
    if not using_ddp or dist.get_rank() == 0:
        print(f"[diffcot] loaded {len(pair_records)} pairs", flush=True)

    dataset = PairDataset(pair_records)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if using_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
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
            if str(args.mode) == "sft":
                loss = compute_sft_loss(
                    batch=batch,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=int(args.max_length),
                )
            else:
                loss = compute_preference_loss(
                    batch=batch,
                    model=model,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    beta=float(args.beta),
                    refine_weight=float(args.refine_weight),
                    draft_weight=float(args.draft_weight),
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
                print(f"[diffcot] epoch {epoch + 1}, step {step}, loss={loss.item():.4f}")
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
                print(f"[diffcot] saved epoch {epoch + 1} -> {epoch_dir}")
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
        print(f"[diffcot] saved -> {out_dir}")
    if using_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
