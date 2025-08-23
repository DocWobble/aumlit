"""Model header readers for various artifact formats.

This module exposes tiny utilities to introspect model files without
materialising the full weights.  Each reader returns a :class:`Meta`
object describing tensor names/shapes and any useful scalar hints
(e.g. vocab size, context length).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import hashlib


@dataclass
class TensorInfo:
    """Lightweight description of a tensor."""

    name: str
    shape: Tuple[int, ...]


@dataclass
class Meta:
    """Summary of a model's tensors and scalar hints."""

    tensors: List[TensorInfo]
    hints: Dict[str, Any] = field(default_factory=dict)


def _ensure_path(path: Path | str) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)


def read_safetensors_header(path: Path | str) -> Meta:
    """Read the header of a ``.safetensors`` file.

    Only the JSON header is parsed; tensor data is never materialised.
    Any top-level ``metadata`` section is returned verbatim as hints.
    """
    p = _ensure_path(path)
    with p.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        raw_header = f.read(header_len)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError("Malformed safetensors header") from e

    hints = header.get("__metadata__", {}) if isinstance(header, dict) else {}
    tensors = [
        TensorInfo(name=name, shape=tuple(int(d) for d in info["shape"]))
        for name, info in header.items()
        if name != "__metadata__"
    ]
    return Meta(tensors=tensors, hints=dict(hints))


def header_class_key(meta: Meta) -> str:
    """Compute a stable key representing the header class.

    The key is a SHA256 hash over tensor names/shapes and scalar hints,
    allowing cross-artifact reuse when headers match.
    """

    data = {
        "tensors": [(t.name, t.shape) for t in meta.tensors],
        "hints": meta.hints,
    }
    blob = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_onnx_header(path: Path | str) -> Meta:
    """Read tensor names and shapes from an ONNX file.

    The graph initialisers are inspected to build the tensor list.  A
    heuristic attempts to recover the vocabulary size from any
    embedding matrix.
    """
    import onnx

    p = _ensure_path(path)
    model = onnx.load(p, load_external_data=False)

    tensors = [
        TensorInfo(name=init.name, shape=tuple(int(d) for d in init.dims))
        for init in model.graph.initializer
    ]

    hints: Dict[str, Any] = {}
    for init in model.graph.initializer:
        name_l = init.name.lower()
        if len(init.dims) == 2 and any(k in name_l for k in ("embed", "vocab")):
            hints["vocab_size"] = int(init.dims[0])
            break

    return Meta(tensors=tensors, hints=hints)


def read_gguf_header(path: Path | str) -> Meta:
    """Read tensor names, shapes and common metadata from a GGUF file."""
    import gguf

    p = _ensure_path(path)
    reader = gguf.GGUFReader(p)

    tensors = [
        TensorInfo(name=t.name, shape=tuple(int(s) for s in t.shape))
        for t in reader.tensors
    ]

    hints: Dict[str, Any] = {}
    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field else None
    if arch:
        key_map = {
            "vocab_size": gguf.KEY_VOCAB_SIZE.format(arch=arch),
            "context_length": gguf.KEY_CONTEXT_LENGTH.format(arch=arch),
            "rope_dimension_count": gguf.KEY_ROPE_DIMENSION_COUNT.format(arch=arch),
            "rope_freq_base": gguf.KEY_ROPE_FREQ_BASE.format(arch=arch),
            "rope_scale": gguf.KEY_ROPE_SCALING_FACTOR.format(arch=arch),
        }
        for hint_key, field_key in key_map.items():
            field = reader.get_field(field_key)
            if field is not None:
                hints[hint_key] = field.contents()

    return Meta(tensors=tensors, hints=hints)


__all__ = [
    "TensorInfo",
    "Meta",
    "read_safetensors_header",
    "read_onnx_header",
    "read_gguf_header",
    "header_class_key",
]
