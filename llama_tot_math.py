import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cot_math import build_completion_prompt
from dataset_utils import default_dataset_path, load_one_sample
from llama_api import llama_completion as hf_llama_completion
from local_llama_backend import get_vllm_engine, resolve_model_dir
from tool import extract_model_answer, is_model_correct, steps_for_dataset, steps_for_level


def _truncate_by_stop(text: str, stop_tokens: Optional[List[str]]) -> str:
    """
    若提供 stop tokens，则对生成文本进行截断。
    """
    if not stop_tokens:
        return text

    earliest: Optional[int] = None
    for token in stop_tokens:
        idx = text.find(token)
        if idx != -1:
            if earliest is None or idx < earliest:
                earliest = idx
    if earliest is not None:
        return text[:earliest]
    return text


def _extract_step_block(raw_completion: str, step_index: int) -> str:
    """
    从一次 step 生成的文本中提取当前 step 的内容（不含 'Step k:' 标签行）。

    说明：
      - prompt 本身会以 'Step k:' 结尾作为生成前缀；
      - 为避免重复标签，step_text 只保存标签后面的内容；
      - 若模型在输出中又重复生成了 'Step k:'，这里会把它剥离掉。
    """
    if not raw_completion:
        return ""

    # 保留原始换行，但去掉首尾空行
    raw_lines = raw_completion.splitlines()
    lines = [ln.rstrip() for ln in raw_lines]
    # 去掉前导空行
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    first = lines[0].strip()
    expected = f"step {step_index}:"
    if first.lower() == expected:
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    elif first.lower().startswith("step ") and first.lower().startswith(f"step {step_index}:"):
        # 容忍 "Step k: ..." 同行的格式
        rest = first[len(f"Step {step_index}:") :].lstrip()
        lines = ([rest] if rest else []) + lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)

    return "\n".join(lines).strip()


def _ensure_step_prefix(prefix: str, step_index: int) -> str:
    label = f"Step {step_index}:"
    # Make generation start *after* the label, not by emitting another label.
    # A trailing space helps many models continue the current step content.
    stripped = prefix.rstrip()
    if stripped.endswith(label):
        return stripped + " "
    if not stripped.endswith("\n"):
        stripped += "\n"
    return stripped + label + " "


def _generate_batch(
    prompts: List[str],
    model_dir: Optional[Union[str, Path]],
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    n: int,
    use_vllm: bool,
    stop_tokens: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
) -> List[List[str]]:
    """
    根据 use_vllm 选择后端：
      - True: 使用 vLLM 并行生成
      - False: 使用 transformers (hf_llama_completion) 逐条生成
    """
    if use_vllm:
        from vllm import SamplingParams  # type: ignore

        model_path = resolve_model_dir(model_dir)
        engine = get_vllm_engine(model_path)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=n,
            stop=stop_tokens,
        )
        if chunk_size is None or chunk_size <= 0:
            chunk_size = len(prompts) if prompts else 1

        all_texts: List[List[str]] = []
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i : i + chunk_size]
            outputs = engine.generate(chunk, sampling_params)
            for out in outputs:
                texts = [o.text for o in out.outputs]
                all_texts.append(texts)
        return all_texts

    # Fallback: sequential generation via transformers
    results: List[List[str]] = []
    for prompt in prompts:
        samples: List[str] = []
        for _ in range(n):
            text = hf_llama_completion(
                prompt=prompt,
                model_dir=model_dir,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
            )
            text = _truncate_by_stop(text, stop_tokens)
            samples.append(text)
        results.append(samples)
    return results


def run_llama_tot_on_single(
    dataset_name: str = "hendrycks_math",
    dataset_path: Optional[Union[str, Path]] = None,
    split: str = "train",
    model_dir: Optional[Union[str, Path]] = None,
    # 每个 step 要评估的候选步数
    num_step_candidates: int = 1,
    # 每个候选步 rollout 次数
    rollouts_per_candidate: int = 1,
    temperature: float = 0.5,
    use_vllm: bool = True,
    rollout_batch_size: int = 16,
    sample: Optional[Dict[str, Any]] = None,
    num_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    使用本地 LLaMA3-8B 在一条样本上运行「用于数据采集的 ToT」：
      - 按 step 逐步生成推理
      - 在每个 step 上先生成若干候选 step
      - 对每个候选 step 做多次 rollout，计算成功率
      - 选择成功率最高的候选 step 作为该 step 的「执行步」
    最终返回每个 step 的候选/rollout 详细信息，方便做离线分析与训练。
    """
    if sample is None:
        sample = load_one_sample(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
        )

    problem = (
        sample.get("problem")
        or sample.get("question")
        or sample.get("prompt")
    )
    if not problem:
        raise ValueError("Sample does not contain a 'problem' / 'question' field.")

    solution = sample.get("solution") or sample.get("answer")
    level = sample.get("level")
    if num_steps is None:
        if dataset_name.strip().lower() in {"gsm8k", "svamp"}:
            num_steps = steps_for_dataset(dataset_name)
        else:
            num_steps = steps_for_level(level)
    prompt = build_completion_prompt(problem, num_steps=num_steps)
    current_prefix = prompt

    steps_trace: List[Dict[str, Any]] = []

    # 逐 step 进行 ToT 搜索 + 数据采集
    for step_idx in range(1, num_steps + 1):
        candidates: List[Dict[str, Any]] = []
        candidate_prefixes: List[str] = []

        # 1) 使用 vLLM 一次性生成当前 step 的多个候选步
        step_prompt = _ensure_step_prefix(current_prefix, step_idx)
        step_samples_nested = _generate_batch(
            prompts=[step_prompt],
            model_dir=model_dir,
            temperature=temperature,
            max_new_tokens=256,
            top_p=0.95,
            n=num_step_candidates,
            use_vllm=use_vllm,
            stop_tokens=[f"\nStep {step_idx + 1}:", "\nAnswer:"],
        )
        step_samples = step_samples_nested[0] if step_samples_nested else []

        # 构造每个候选步的前缀（包含完整的 Step 段落）
        for cand_idx, step_completion in enumerate(step_samples):
            step_text = _extract_step_block(step_completion, step_idx)
            cand_prefix = step_prompt + (step_text + "\n" if step_text else "\n")

            candidates.append(
                {
                    "candidate_index": cand_idx,
                    "step_text": step_text,
                    # Only keep aggregate metrics (no rollout traces to save space)
                    "success_count": 0,
                    "success_rate": 0.0,
                }
            )
            candidate_prefixes.append(cand_prefix)

        # 2) 对所有候选步，使用 vLLM 并行做多次 rollout
        rollout_requests: List[tuple[int, int, str]] = []
        for cand_idx, cand_prefix in enumerate(candidate_prefixes):
            for r in range(rollouts_per_candidate):
                rollout_requests.append((cand_idx, r, cand_prefix))

        if rollout_requests:
            rollout_prompts = [req[2] for req in rollout_requests]
            rollout_outputs_nested = _generate_batch(
                prompts=rollout_prompts,
                model_dir=model_dir,
                temperature=temperature,
                max_new_tokens=256,
                top_p=0.95,
                n=1,
                use_vllm=use_vllm,
                chunk_size=rollout_batch_size,
            )
            rollout_texts = [
                texts[0] if texts else "" for texts in rollout_outputs_nested
            ]

            for (cand_idx, r, cand_prefix), rollout_completion in zip(
                rollout_requests, rollout_texts
            ):
                full_output = cand_prefix + rollout_completion
                is_correct_flag = False
                if solution is not None:
                    is_correct_flag = is_model_correct(full_output, solution)

                cand_entry = candidates[cand_idx]
                if is_correct_flag:
                    cand_entry["success_count"] += 1

        for cand_entry in candidates:
            if rollouts_per_candidate > 0:
                cand_entry["success_rate"] = (
                    cand_entry["success_count"] / rollouts_per_candidate
                )

        # 3) 选出当前 step 成功率最高的候选步
        if candidates:
            best_cand = max(
                candidates, key=lambda c: (c["success_rate"], -c["candidate_index"])
            )
        else:
            best_cand = None

        steps_trace.append(
            {
                "step_index": step_idx,
                "candidates": candidates,
                "chosen_candidate_index": best_cand["candidate_index"] if best_cand else None,
                "chosen_step_text": best_cand["step_text"] if best_cand else None,
                "chosen_success_rate": best_cand["success_rate"] if best_cand else None,
            }
        )

        # 更新前缀，只沿着评分最高的候选步继续向后推理
        if best_cand is not None:
            # Reconstruct the prefix from the step prompt + extracted step text.
            best_idx = int(best_cand["candidate_index"])
            current_prefix = candidate_prefixes[best_idx]
        else:
            # 如果某一步完全失败，后续就不再继续扩展
            break

    # 在最后一步之后，再做一次完整 rollout 作为「执行路径」的参考
    final_outputs_nested = _generate_batch(
        prompts=[current_prefix],
        model_dir=model_dir,
        temperature=temperature,
        max_new_tokens=256,
        top_p=0.95,
        n=1,
        use_vllm=use_vllm,
    )
    final_completion = (
        final_outputs_nested[0][0] if final_outputs_nested and final_outputs_nested[0] else ""
    )
    final_full_output = current_prefix + final_completion
    final_answer = extract_model_answer(final_full_output)
    final_is_correct: Optional[bool] = None
    if solution is not None:
        final_is_correct = is_model_correct(final_full_output, solution)

    result: Dict[str, Any] = {
        "problem": problem,
        "solution": solution,
        "level": level,
        "prompt": prompt,
        "steps_trace": steps_trace,
        "final_prefix": current_prefix,
        "final_completion": final_completion,
        "final_full_output": final_full_output,
        "final_answer": final_answer,
        "final_is_correct": final_is_correct,
        "raw_sample": sample,
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="使用本地模型对单条样本运行 ToT（支持切换数据集）"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hendrycks_math",
        choices=["hendrycks_math", "gsm8k", "svamp"],
        help="Dataset name (default: hendrycks_math).",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Dataset path. For gsm8k: directory with train.jsonl/test.jsonl; "
            "for svamp: SVAMP.json file; for hendrycks_math: root directory."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split for gsm8k (default: train).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override reasoning steps; defaults: svamp=3, gsm8k=5, hendrycks_math=by level.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="本地模型目录，若不提供则由 --model-type 决定默认路径",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["llama", "qwen"],
        default="llama",
        help="选择使用的本地模型类型（llama 或 qwen），仅在未显式提供 --model-dir 时生效",
    )
    parser.add_argument(
        "--branches",
        type=int,
        default=4,
        help="每个 step 的候选步数（默认：1）",
    )
    parser.add_argument(
        "--rollouts-per-candidate",
        type=int,
        default=4,
        help="每个候选步的 rollout 次数（默认：1）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="采样 temperature（默认：0.5）",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="指定 CUDA_VISIBLE_DEVICES，例如 '0' 或 '0,1'（默认：不修改环境变量）",
    )
    parser.add_argument(
        "--no-vllm",
        dest="use_vllm",
        action="store_false",
        help="禁用 vLLM，改用 transformers 顺序生成",
    )
    parser.set_defaults(use_vllm=True)
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=16,
        help="vLLM rollout 阶段每批并行的 prompt 数（默认：16）",
    )

    args = parser.parse_args()

    # 若显式指定 GPU，则通过环境变量控制 vLLM / PyTorch 使用的设备
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 根据模型类型设置默认路径（除非用户显式传入 --model-dir）
    if args.model_dir is None:
        if args.model_type == "llama":
            args.model_dir = "/data/jsg_data/model/meta-llama/llama3-8b"
        elif args.model_type == "qwen":
            args.model_dir = "/data/jsg_data/Qwen/Qwen2.5-Math-7B-Instruct"
    if args.dataset_path is None:
        args.dataset_path = default_dataset_path(args.dataset_name)

    backend = "vLLM" if args.use_vllm else "transformers"
    print(f"[llama_tot_math] Using backend: {backend}, model_type: {args.model_type}, model_dir: {args.model_dir}")

    out = run_llama_tot_on_single(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        model_dir=args.model_dir,
        num_step_candidates=args.branches,
        rollouts_per_candidate=args.rollouts_per_candidate,
        temperature=args.temperature,
        use_vllm=args.use_vllm,
        rollout_batch_size=args.rollout_batch_size,
        num_steps=args.num_steps,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
