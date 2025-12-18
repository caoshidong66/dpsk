import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from cot_math import _load_one_hendrycks_sample  # type: ignore
from dataset_utils import iter_samples, normalize_sample
from llama_cot_math import run_llama_cot_on_single
from llama_tot_math import run_llama_tot_on_single

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_ID_DIR = _REPO_ROOT / "datas" / "eval_ids"
_DEFAULT_MERGED_DIR = _REPO_ROOT / "datas" / "merged_models"


def _resolve_gpu_ids(user_spec: Optional[str]) -> List[Optional[str]]:
    """
    解析 CLI / 环境 / torch 中可用的 GPU 列表。
    返回的列表中每个元素是要传给 CUDA_VISIBLE_DEVICES 的字符串，
    若只有一个 None 表示沿用当前可见 GPU、单进程评估。
    """
    if user_spec:
        return [gid.strip() for gid in user_spec.split(",") if gid.strip()]

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env:
        env_ids = [gid.strip() for gid in cuda_env.split(",") if gid.strip()]
        if env_ids:
            return env_ids

    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return [str(i) for i in range(count)]
    except Exception:
        pass

    return [None]


def _load_samples_from_hendrycks(
    dataset_root: Path,
    num_samples: int,
) -> List[Dict[str, Any]]:
    """
    复用 _load_one_hendrycks_sample 的根目录约定，从同一数据集随机采样 num_samples 条样本。

    这里为了简单，直接多次调用 _load_one_hendrycks_sample，并假设其内部有随机性
    或者数据集本身经过 shuffle；对于严格的“无放回采样”，可以后续再做改进。
    """
    samples: List[Dict[str, Any]] = []
    for _ in range(num_samples):
        sample = _load_one_hendrycks_sample(dataset_root=dataset_root)
        samples.append(sample)
    return samples


def _parse_level(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def _merge_lora_adapter(
    base_model_dir: Path,
    lora_dir: Path,
    output_dir: Path,
    *,
    reuse_if_exists: bool,
) -> Path:
    if reuse_if_exists and (output_dir / "config.json").exists():
        return output_dir

    try:
        import torch
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Merging LoRA requires `peft` and `transformers` to be installed."
        ) from exc

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), use_fast=False)
        tokenizer.save_pretrained(str(output_dir))
    except Exception:
        pass

    return output_dir


def _select_hendrycks_indices_by_level(
    dataset_root: Path,
    *,
    split: str,
    level: int,
    num_samples: int,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    indices: List[int] = []
    seen = 0
    for idx, raw in enumerate(iter_samples("hendrycks_math", dataset_root, split=split)):
        lvl = _parse_level(raw.get("level"))
        if lvl != level:
            continue
        seen += 1
        if len(indices) < num_samples:
            indices.append(idx)
            continue
        j = rng.randint(1, seen)
        if j <= num_samples:
            indices[j - 1] = idx
    if len(indices) < num_samples:
        print(
            f"[eval] Only found {len(indices)} samples for level={level} (requested {num_samples})."
        )
    return indices


def _load_hendrycks_samples_by_indices(
    dataset_root: Path,
    *,
    split: str,
    indices: List[int],
) -> List[Dict[str, Any]]:
    if not indices:
        return []
    wanted = set(indices)
    collected: Dict[int, Dict[str, Any]] = {}
    for idx, raw in enumerate(iter_samples("hendrycks_math", dataset_root, split=split)):
        if idx not in wanted:
            continue
        collected[idx] = normalize_sample("hendrycks_math", raw)
        if len(collected) == len(wanted):
            break
    return [collected[i] for i in indices if i in collected]


def evaluate_cot_and_tot_on_samples(
    samples: List[Dict[str, Any]],
    model_dir: Optional[Union[str, Path]],
    use_vllm_for_cot: bool = True,
    use_vllm_for_tot: bool = False,
    branches: int = 4,
    rollouts_per_candidate: int = 4,
    rollout_batch_size: int = 16,
    temperature: float = 0.5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    在给定的一批样本上分别跑 CoT 和 ToT，并统计正确率。
    """
    cot_correct = 0
    tot_correct = 0

    cot_results: List[Dict[str, Any]] = []
    tot_results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        print(
            f"[eval] sample {idx + 1}/{len(samples)} - running CoT "
            f"({'vLLM' if use_vllm_for_cot else 'transformers'})"
        )
        # CoT
        cot_out = run_llama_cot_on_single(
            model_dir=model_dir,
            sample=sample,
            use_vllm=use_vllm_for_cot,
        )
        cot_results.append(cot_out)
        if cot_out.get("is_correct"):
            cot_correct += 1

        print(
            f"[eval] sample {idx + 1}/{len(samples)} - running ToT "
            f"({'vLLM' if use_vllm_for_tot else 'transformers'})"
        )
        # ToT
        tot_out = run_llama_tot_on_single(
            model_dir=model_dir,
            num_step_candidates=branches,
            rollouts_per_candidate=rollouts_per_candidate,
            temperature=temperature,
            use_vllm=use_vllm_for_tot,
            rollout_batch_size=rollout_batch_size,
            sample=sample,
        )
        tot_results.append(tot_out)
        if tot_out.get("final_is_correct"):
            tot_correct += 1

        print(f"[eval] processed sample {idx + 1}/{len(samples)}")

    n = len(samples)
    cot_summary = {
        "num_samples": n,
        "num_correct": cot_correct,
        "accuracy": cot_correct / n if n > 0 else None,
    }
    tot_summary = {
        "num_samples": n,
        "num_correct": tot_correct,
        "accuracy": tot_correct / n if n > 0 else None,
    }

    return (
        {"summary": cot_summary, "details": cot_results},
        {"summary": tot_summary, "details": tot_results},
    )


def _worker_eval_one_shard(
    gpu_id: Optional[str],
    samples: List[Dict[str, Any]],
    model_dir: Optional[Union[str, Path]],
    use_vllm_for_cot: bool,
    use_vllm_for_tot: bool,
    branches: int,
    rollouts_per_candidate: int,
    rollout_batch_size: int,
    temperature: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    单个进程在指定 GPU 上跑一批样本。
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    return evaluate_cot_and_tot_on_samples(
        samples=samples,
        model_dir=model_dir,
        use_vllm_for_cot=use_vllm_for_cot,
        use_vllm_for_tot=use_vllm_for_tot,
        branches=branches,
        rollouts_per_candidate=rollouts_per_candidate,
        rollout_batch_size=rollout_batch_size,
        temperature=temperature,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="在 hendrycks_math 上随机抽取若干样本，分别评估本地 LLaMA 的 CoT 和 ToT 正确率"
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/jsg_data/hendrycks_math",
        help="hendrycks_math 数据根目录（默认：/data/jsg_data/hendrycks_math）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="随机评估多少条样本（默认：200）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="hendrycks_math 的 split（默认：train）",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="只评估指定难度 level 的样本（例如 3）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="采样随机种子（默认：42）",
    )
    parser.add_argument(
        "--id-cache",
        type=str,
        default=None,
        help="保存/读取采样样本索引的 JSON 文件路径（默认写入 datas/eval_ids/）。",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=None,
        help="LoRA adapter 权重目录；提供后会自动 merge 到 base 模型。",
    )
    parser.add_argument(
        "--merged-model-dir",
        type=str,
        default=None,
        help="merge 后的模型保存目录（默认 datas/merged_models/）。",
    )
    parser.add_argument(
        "--reuse-merged",
        action="store_true",
        help="如果 merge 输出目录已存在则直接复用（跳过重新 merge）。",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="本地模型目录，默认与 llama_tot_math.py 中的约定保持一致",
    )
    parser.add_argument(
        "--use-vllm-for-cot",
        action="store_true",
        dest="use_vllm_for_cot",
        help="在 CoT 推理中启用 vLLM（默认启用，可用 --no-vllm-for-cot 关闭）",
    )
    parser.add_argument(
        "--no-vllm-for-cot",
        action="store_false",
        dest="use_vllm_for_cot",
        help="禁用 CoT 的 vLLM，回退到 transformers (HF) 推理",
    )
    parser.add_argument(
        "--use-vllm-for-tot",
        action="store_true",
        dest="use_vllm_for_tot",
        help="在 ToT 中启用 vLLM（默认启用，可用 --no-vllm-for-tot 关闭）",
    )
    parser.add_argument(
        "--no-vllm-for-tot",
        action="store_false",
        dest="use_vllm_for_tot",
        help="禁用 ToT 的 vLLM，改用 transformers 顺序生成",
    )
    parser.add_argument(
        "--branches",
        type=int,
        default=4,
        help="ToT 中每个 step 的候选步数（默认：4）",
    )
    parser.add_argument(
        "--rollouts-per-candidate",
        type=int,
        default=4,
        help="ToT 中每个候选步的 rollout 次数（默认：4）",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=16,
        help="ToT rollout 阶段一次并行多少个 prompt（默认：16，等于 4 候选 x 4 rollout）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="采样 temperature（默认：0.5）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='0,1,2',
        help="用于并行评估的 GPU 列表，例如 '0,1,2'（默认：只用当前可见设备、单进程顺序评估）",
    )

    parser.set_defaults(use_vllm_for_cot=True, use_vllm_for_tot=True)

    args = parser.parse_args()

    if args.lora_dir:
        if not args.model_dir:
            raise ValueError("--lora-dir requires --model-dir (base model directory).")
        base_model_dir = Path(args.model_dir)
        lora_dir = Path(args.lora_dir)
        if args.merged_model_dir:
            merged_dir = Path(args.merged_model_dir)
        else:
            _DEFAULT_MERGED_DIR.mkdir(parents=True, exist_ok=True)
            merged_dir = _DEFAULT_MERGED_DIR / f"{lora_dir.name}_merged"
        args.model_dir = str(
            _merge_lora_adapter(
                base_model_dir,
                lora_dir,
                merged_dir,
                reuse_if_exists=args.reuse_merged,
            )
        )

    ds_root = Path(args.dataset_root)
    if args.level is not None:
        if args.id_cache:
            cache_path = Path(args.id_cache)
        else:
            _DEFAULT_ID_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = _DEFAULT_ID_DIR / (
                f"hendrycks_level{args.level}_{args.split}_{args.num_samples}.json"
            )

        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            indices = cached.get("indices") if isinstance(cached, dict) else None
            if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
                raise ValueError(f"Invalid id cache format: {cache_path}")
        else:
            indices = _select_hendrycks_indices_by_level(
                ds_root,
                split=args.split,
                level=int(args.level),
                num_samples=int(args.num_samples),
                seed=int(args.seed),
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset_root": str(ds_root),
                        "split": args.split,
                        "level": int(args.level),
                        "num_samples": int(args.num_samples),
                        "seed": int(args.seed),
                        "indices": indices,
                    },
                    f,
                    indent=2,
                )
            print(f"[eval] saved id cache -> {cache_path}")

        samples = _load_hendrycks_samples_by_indices(
            ds_root,
            split=args.split,
            indices=indices,
        )
    else:
        samples = _load_samples_from_hendrycks(ds_root, args.num_samples)

    # 多 GPU / 多进程并行：按 GPU 数量把样本切成若干 shard
    gpu_ids = _resolve_gpu_ids(args.gpus)
    if len(gpu_ids) > 1:
        print(f"[eval] Detected {len(gpu_ids)} GPUs -> shards: {gpu_ids}")

    if len(gpu_ids) == 1:
        only_gpu = gpu_ids[0]
        if only_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(only_gpu)
        cot_eval, tot_eval = evaluate_cot_and_tot_on_samples(
            samples=samples,
            model_dir=args.model_dir,
            use_vllm_for_cot=args.use_vllm_for_cot,
            use_vllm_for_tot=args.use_vllm_for_tot,
            branches=args.branches,
            rollouts_per_candidate=args.rollouts_per_candidate,
            rollout_batch_size=args.rollout_batch_size,
            temperature=args.temperature,
        )
    else:
        # 切 shard
        n = len(samples)
        num_shards = len(gpu_ids)
        shard_size = (n + num_shards - 1) // num_shards
        shards: List[List[Dict[str, Any]]] = []
        for i in range(num_shards):
            start = i * shard_size
            end = min(n, (i + 1) * shard_size)
            if start >= end:
                shards.append([])
            else:
                shards.append(samples[start:end])

        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        # CUDA + fork 容易死锁，这里强制使用 spawn 避免加载本地模型时卡住
        mp_ctx = get_context("spawn")

        results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        with ProcessPoolExecutor(
            max_workers=num_shards,
            mp_context=mp_ctx,
        ) as ex:
            futures = []
            for gpu_id, shard in zip(gpu_ids, shards):
                if not shard:
                    continue
                futures.append(
                    ex.submit(
                        _worker_eval_one_shard,
                        gpu_id,
                        shard,
                        args.model_dir,
                        args.use_vllm_for_cot,
                        args.use_vllm_for_tot,
                        args.branches,
                        args.rollouts_per_candidate,
                        args.rollout_batch_size,
                        args.temperature,
                    )
                )
            for fut in futures:
                results.append(fut.result())

        # 聚合结果
        total_samples = 0
        cot_correct = 0
        tot_correct = 0
        for cot_eval_part, tot_eval_part in results:
            cs = cot_eval_part["summary"]
            ts = tot_eval_part["summary"]
            total_samples += cs["num_samples"]
            cot_correct += cs["num_correct"]
            tot_correct += ts["num_correct"]

        cot_eval = {
            "summary": {
                "num_samples": total_samples,
                "num_correct": cot_correct,
                "accuracy": cot_correct / total_samples if total_samples > 0 else None,
            },
            "details": [],
        }
        tot_eval = {
            "summary": {
                "num_samples": total_samples,
                "num_correct": tot_correct,
                "accuracy": tot_correct / total_samples if total_samples > 0 else None,
            },
            "details": [],
        }

    result = {
        "cot": cot_eval["summary"],
        "tot": tot_eval["summary"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
