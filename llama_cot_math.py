import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cot_math import build_completion_prompt
from dataset_utils import default_dataset_path, load_one_sample
from llama_api import llama_completion
from local_llama_backend import get_vllm_engine, resolve_model_dir
from tool import extract_model_answer, is_model_correct, steps_for_dataset, steps_for_level


def _cot_completion_with_vllm(
    prompt: str,
    model_dir: Optional[Union[str, Path]],
    temperature: float,
    max_new_tokens: int,
) -> str:
    """
    仅在 use_vllm=True 时调用，使用共享的 vLLM 引擎执行一次 completion。
    """
    from vllm import SamplingParams  # type: ignore

    model_path = resolve_model_dir(model_dir)
    engine = get_vllm_engine(model_path)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=1,
    )
    outputs = engine.generate([prompt], sampling_params, use_tqdm=False)
    if not outputs:
        return ""
    first = outputs[0]
    if not first.outputs:
        return ""
    return first.outputs[0].text


def _cot_completion_batch_with_vllm(
    prompts: List[str],
    model_dir: Optional[Union[str, Path]],
    temperature: float,
    max_new_tokens: int,
) -> List[str]:
    from vllm import SamplingParams  # type: ignore

    model_path = resolve_model_dir(model_dir)
    engine = get_vllm_engine(model_path)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=1,
    )
    outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
    results: List[str] = []
    for out in outputs:
        if not out.outputs:
            results.append("")
        else:
            results.append(out.outputs[0].text)
    return results


def run_llama_cot_on_single(
    dataset_name: str = "hendrycks_math",
    dataset_path: Optional[Union[str, Path]] = None,
    split: str = "train",
    model_dir: Optional[Union[str, Path]] = None,
    sample: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    use_vllm: bool = False,
    num_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    使用本地 LLaMA3-8B，在一条 hendrycks_math 样本上做一次 COT completion。

    若提供 sample，则直接使用该样本；否则从 dataset_root 中加载一条样本。
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

    if use_vllm:
        completion = _cot_completion_with_vllm(
            prompt=prompt,
            model_dir=model_dir,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    else:
        completion = llama_completion(
            prompt=prompt,
            model_dir=model_dir,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    answer = extract_model_answer(completion)
    is_correct_flag = False
    if solution is not None:
        is_correct_flag = is_model_correct(completion, solution)

    result: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "answer": answer,
        "is_correct": is_correct_flag,
        "problem": problem,
        "level": level,
        "raw_sample": sample,
    }
    return result


def run_llama_cot_on_batch(
    samples: List[Dict[str, Any]],
    model_dir: Optional[Union[str, Path]] = None,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    use_vllm: bool = False,
    num_steps: Optional[int] = None,
    dataset_name: str = "hendrycks_math",
) -> List[Dict[str, Any]]:
    if not samples:
        return []

    prompts: List[str] = []
    steps_list: List[int] = []
    for sample in samples:
        problem = (
            sample.get("problem")
            or sample.get("question")
            or sample.get("prompt")
        )
        if not problem:
            raise ValueError("Sample does not contain a 'problem' / 'question' field.")
        level = sample.get("level")
        if num_steps is None:
            if dataset_name.strip().lower() in {"gsm8k", "svamp"}:
                steps_eff = steps_for_dataset(dataset_name)
            else:
                steps_eff = steps_for_level(level)
        else:
            steps_eff = num_steps
        steps_list.append(steps_eff)
        prompts.append(build_completion_prompt(problem, num_steps=steps_eff))

    if use_vllm:
        completions = _cot_completion_batch_with_vllm(
            prompts=prompts,
            model_dir=model_dir,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    else:
        completions = [
            llama_completion(
                prompt=prompt,
                model_dir=model_dir,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            for prompt in prompts
        ]

    results: List[Dict[str, Any]] = []
    for sample, prompt, completion, steps_eff in zip(samples, prompts, completions, steps_list):
        answer = extract_model_answer(completion)
        solution = sample.get("solution") or sample.get("answer")
        is_correct_flag = False
        if solution is not None:
            is_correct_flag = is_model_correct(completion, solution)
        results.append(
            {
                "prompt": prompt,
                "completion": completion,
                "answer": answer,
                "is_correct": is_correct_flag,
                "problem": sample.get("problem") or sample.get("question") or sample.get("prompt"),
                "level": sample.get("level"),
                "raw_sample": sample,
                "num_steps": steps_eff,
            }
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="使用本地模型在单条样本上跑 COT（支持切换数据集）"
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
        help="Dataset path; if omitted uses built-in defaults for the dataset.",
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
        default="/data/jsg_data/model/meta-llama/llama3-8b",
        help="Local model directory (default: /data/jsg_data/model/meta-llama/llama3-8b).",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM backend (default: off).",
    )

    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = default_dataset_path(args.dataset_name)

    out = run_llama_cot_on_single(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split=args.split,
        model_dir=args.model_dir,
        use_vllm=args.use_vllm,
        num_steps=args.num_steps,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
