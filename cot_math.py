import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tool import steps_for_level


def build_completion_prompt(problem: str, num_steps: Optional[int] = None) -> str:
    """
    构造一个用于 completion 风格的数据采集的 prompt。

    返回的字符串会以 'Step 1:' 结尾，模型的输出可以视为这个前缀的续写。
    """
    if num_steps is not None:
        step_instruction = (
            f"You MUST use exactly {num_steps} reasoning steps, "
        )
    else:
        step_instruction = (
            "You MUST use detailed step-by-step chain-of-thought reasoning.\n"
        )

    prompt = (
        "You are an expert math problem solver. "
        "You must reason step by step and avoid logical or arithmetic mistakes.\n\n"
        "Solve the following math problem.\n"
        + step_instruction +
        "After the reasoning, output the final answer in the last line "
        "using the format: `Answer: <final_answer>`.\n\n"
        f"Problem: {problem}\n\n"
        "Reasoning step by step:\n"
        "Step 1:"
    )
    return prompt


# 默认认为 hendrycks_math 是正确的 MATH 数据集
HENDRYCKS_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "hendrycks_math"


def _load_one_hendrycks_sample(
    dataset_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    从 hendrycks_math 数据集中读取一条样本。

    这里不假设具体文件名，只是：
      1. 在目录下递归找 *.jsonl，如果有就取第一个文件的第一条记录；
      2. 否则在目录下递归找 *.json，取第一个文件中的第一条记录。
    """
    root = Path(dataset_root) if dataset_root is not None else HENDRYCKS_DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"hendrycks_math root not found: {root}")

    jsonl_files = list(root.rglob("*.jsonl"))
    json_files = list(root.rglob("*.json")) if not jsonl_files else []

    if jsonl_files:
        path = sorted(jsonl_files)[0]
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return json.loads(line)
        raise ValueError(f"No non-empty JSONL lines found in {path}")

    if json_files:
        path = sorted(json_files)[0]
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list) and data["data"]:
                return data["data"][0]
            return data
        raise ValueError(f"Unsupported JSON structure in {path}")

    # 3. 尝试读取 Parquet（适配你现在的 hendrycks_math 结构）
    parquet_files = list(root.rglob("*.parquet"))
    if parquet_files:
        try:
            # 优先使用 Hugging Face datasets 来读取本地 parquet
            from datasets import Dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖环境
            raise ImportError(
                "Reading hendrycks_math Parquet files requires the `datasets` library "
                "to be installed (pip install datasets)."
            ) from exc

        path = sorted(parquet_files)[0]
        ds = Dataset.from_parquet(str(path))
        if len(ds) == 0:
            raise ValueError(f"No rows found in Parquet file {path}")
        # 假设一行就是一个样本，字段中包含 problem / question / solution 等
        return ds[0]

    raise FileNotFoundError(f"No .jsonl, .json, or .parquet files found under {root}")


def run_single_cot_from_hendrycks(
    dataset_root: Optional[Union[str, Path]] = None,
    model: str = "deepseek-v3-2-251201",
) -> Dict[str, Any]:
    """
    选用 hendrycks_math 数据集，读取一条样本并跑一次 CoT，并返回结果。
    """
    sample = _load_one_hendrycks_sample(dataset_root=dataset_root)

    problem = (
        sample.get("problem")
        or sample.get("question")
        or sample.get("prompt")
    )
    if not problem:
        raise ValueError("Sample does not contain a 'problem' / 'question' field.")

    # 根据题目难度 level 决定需要多少步推理，并构造 completion 风格前缀
    level = sample.get("level")
    num_steps = steps_for_level(level)
    prompt = build_completion_prompt(problem, num_steps=num_steps)

    # 为避免在纯本地 LLaMA 流程中强依赖 Ark，这里在函数内部按需导入
    from api import client, ark_chat_with_stop

    stop_token = "</END>"
    completion = ark_chat_with_stop(
        client=client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stop_token=stop_token,
        thinking="disabled",
    )

    result: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "problem": problem,
        "raw_sample": sample,
    }
    return result


def run_cot_on_math(
    dataset_path: str,
    output_path: str = "math_cot_outputs.jsonl",
    model: str = "deepseek-v3-2-251201",
    limit: Optional[int] = 100,
) -> None:
    """
    使用 ark_chat_with_stop 在 math 数据集上做 CoT 推理。

    约定输入数据集为 JSONL，每行一个样本，字段至少包含：
      - problem / question: 题目文本
      - solution / answer: 标准答案（可选，用于后处理评估）
    """
    ds_path = Path(dataset_path)
    if not ds_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    out_path = Path(output_path)
    out_f = out_path.open("w", encoding="utf-8")

    # 为避免在纯本地 LLaMA 流程中强依赖 Ark，这里在函数内部按需导入
    from api import client, ark_chat_with_stop

    stop_token = "</END>"  # 非常规标记，基本不会被模型自然生成

    with ds_path.open("r", encoding="utf-8") as f_in, out_f:
        for idx, line in enumerate(f_in):
            if limit is not None and idx >= limit:
                break

            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            problem = (
                item.get("problem")
                or item.get("question")
                or item.get("prompt")
            )
            if not problem:
                continue

            level = item.get("level")
            num_steps = steps_for_level(level) if level is not None else None
            prompt = build_completion_prompt(problem, num_steps=num_steps)

            # 使用 Ark chat 接口，但语义上按 completion 样式收集数据
            completion = ark_chat_with_stop(
                client=client,
                model=model,
                messages=[{"role": "assistant", "content": prompt}],
                stop_token=stop_token,
                thinking="disabled",
            )

            result = {
                "index": idx,
                "prompt": prompt,
                "completion": completion,
                "problem": problem,
                "gold_answer": item.get("solution") or item.get("answer"),
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="使用 Ark DeepSeek 在 math 数据集上跑 CoT"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="math 数据集 JSONL 文件路径（每行一个样本），或 hendrycks_math 根目录",
    )
    parser.add_argument(
        "--output",
        default="math_cot_outputs.jsonl",
        help="输出 JSONL 路径（默认：math_cot_outputs.jsonl）",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v3-2-251201",
        help="Ark 模型名称（默认：deepseek-v3-2-251201）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="最多跑多少条样本（默认：100，设为 -1 表示全量）",
    )
    parser.add_argument(
        "--test-hendrycks",
        action="store_true",
        help=(
            "从 hendrycks_math 数据集读取一条题目并跑一次 CoT 测试。"
            "若未提供 dataset，则默认使用 /data/jsg_data/hendrycks_math。"
        ),
    )

    args = parser.parse_args()

    # 默认行为：不传任何参数时，等价于
    # python cot_math.py /data/jsg_data/hendrycks_math --test-hendrycks
    if args.dataset is None and not args.test_hendrycks:
        result = run_single_cot_from_hendrycks(
            dataset_root="/data/jsg_data/hendrycks_math",
            model=args.model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)

    if args.test_hendrycks:
        base_dir = args.dataset if args.dataset is not None else "/data/jsg_data/hendrycks_math"
        result = run_single_cot_from_hendrycks(
            dataset_root=base_dir,
            model=args.model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if args.dataset is None:
            parser.error("You must provide a dataset path unless using --test-hendrycks.")

        run_cot_on_math(
            dataset_path=args.dataset,
            output_path=args.output,
            model=args.model,
            limit=None if args.limit is not None and args.limit < 0 else args.limit,
        )
