"""
Shared helpers for local LLaMA backends (mainly vLLM) so that multiple
modules can reuse the same cached engine without re-loading the model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


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


_VLLM_ENGINES: Dict[str, Any] = {}


def get_vllm_engine(model_dir_str: str):
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
    key = model_dir_str
    engine = _VLLM_ENGINES.get(key)
    if engine is not None:
        return engine
    engine = LLM(
        model=model_dir_str,
        dtype="bfloat16",
        tokenizer_mode="auto",
    )
    _VLLM_ENGINES[key] = engine
    return engine


def shutdown_vllm_engines() -> None:
    for engine in list(_VLLM_ENGINES.values()):
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    _VLLM_ENGINES.clear()
