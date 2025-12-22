"""
Shared helpers for local LLaMA backends (mainly vLLM) so that multiple
modules can reuse the same cached engine without re-loading the model.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union


PathLike = Optional[Union[str, Path]]
_DEFAULT_MODEL_DIR = "/data/jsg_data/model/meta-llama/llama3-8b"


def resolve_model_dir(model_dir: PathLike) -> str:
    """
    Normalize model_dir into a concrete string path, falling back to the
    repository-wide default when None.
    """
    if model_dir is not None:
        return str(Path(model_dir))
    return _DEFAULT_MODEL_DIR


@lru_cache(maxsize=4)
def get_vllm_engine(model_dir_str: str, tensor_parallel_size: int = 1):
    """
    Lazily construct and cache a vLLM engine. Import vLLM only when the user
    actually needs it so that pure-transformers workflows do not depend on it.
    """
    try:
        from vllm import LLM  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "vLLM is required when --use-vllm flags are enabled. "
            "Install it via `pip install vllm`."
        ) from exc

    os.environ.setdefault("VLLM_USE_FAST_TOKENIZER", "1")
    return LLM(
        model=model_dir_str,
        dtype="bfloat16",
        tokenizer_mode="auto",
        tensor_parallel_size=max(1, int(tensor_parallel_size)),
    )
