from __future__ import annotations
from typing import Any


def _engine_forward() -> str:
    """Attempt a tiny xFormers memory-efficient attention call."""
    try:  # pragma: no cover - optional dependency
        import torch
        import xformers.ops as xops
    except Exception:
        return "skip"

    q = torch.randn(1, 2, 1, 32, dtype=torch.float16)
    k = torch.randn(1, 2, 1, 32, dtype=torch.float16)
    v = torch.randn(1, 2, 1, 32, dtype=torch.float16)

    import re

    try:
        xops.memory_efficient_attention(q, k, v)
    except Exception as e:  # pragma: no cover - layout/type mismatches
        ints = re.findall(r"\d+", str(e))
        if ints:
            raise RuntimeError("XFORMERS_FAIL: " + " ".join(ints)) from e
        raise
    return "ok"
