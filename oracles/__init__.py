from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from .onnx import _engine_forward as _onnx_engine_forward
from .llama import _engine_forward as _llama_engine_forward
from . import torch as torch_oracle
from .xformers import _engine_forward as _xformers_attention_probe


__all__ = ["try_forward"]


def try_forward(artifact: Path, inputs: Dict[str, Any]) -> str:
    """Dispatch to the appropriate engine helper and xFormers probe."""
    suffix = artifact.suffix.lower()
    if suffix == ".onnx":
        return _onnx_engine_forward(artifact, inputs)
    if suffix == ".gguf":
        return _llama_engine_forward(artifact, inputs)
    try:
        _xformers_attention_probe()
    except Exception:
        pass
    return torch_oracle._engine_forward(artifact, inputs)
