from __future__ import annotations
from pathlib import Path
from typing import Any, Dict


def _engine_forward(artifact: Path, inputs: Dict[str, Any]) -> str:
    """Run a minimal llama.cpp step via :mod:`llama_cpp`."""
    import re
    try:
        from llama_cpp import Llama
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("LLAMA_CPP_MISSING") from e

    token_id = int(inputs.get("token_id", 0))
    kv_dtype = inputs.get("kv_dtype")
    rope = inputs.get("rope")
    vocab = inputs.get("vocab")

    kwargs = {}
    if kv_dtype is not None:
        kwargs["f16_kv"] = kv_dtype == "f16"
    if rope is not None:
        kwargs["rope_freq_base"] = int(rope)
    if vocab is not None:
        kwargs["vocab_only"] = True

    try:
        llm = Llama(model_path=str(artifact), n_ctx=16, n_gpu_layers=0, **kwargs)
        llm.eval([token_id])  # one-step evaluation
    except Exception as e:  # pragma: no cover - runtime failure path
        msg = str(e).lower()
        ints = [int(x) for x in re.findall(r"\d+", msg)]
        if ints:
            if "vocab" in msg:
                raise RuntimeError(f"VOCAB: {max(ints)}") from e
            if "rope" in msg:
                raise RuntimeError(f"ROPE: {max(ints)}k") from e
            raise RuntimeError("LLAMA_FAIL: " + " ".join(map(str, ints))) from e
        raise
    return "ok"
