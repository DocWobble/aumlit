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
        xops.memory_efficient_attention(q, k, v, mem_efficient=True)
    except Exception as e:  # pragma: no cover - layout/type mismatches
        msg = str(e)
        ints = re.findall(r"\d+", msg)
        layout = None
        m = re.search(r"layout[=:\s]+([A-Za-z0-9_\-]+)", msg)
        if m:
            layout = m.group(1)
        parts = []
        if ints:
            parts.extend(ints)
        if layout:
            parts.append(f"LAYOUT={layout}")
        if parts:
            raise RuntimeError("XFORMERS_FAIL: " + " ".join(parts)) from e
        raise
    return "ok"
