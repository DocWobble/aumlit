"""Regex-based error classifier extracting hypothesis hints."""
from __future__ import annotations

import re
from typing import Any, Dict

# Map hypothesis keys to regex patterns capturing the relevant value.
# Patterns are intentionally broad and numeric groups are prioritised.
ERROR_PATTERNS = {
    "TEXT_EMB_d": r"COND_DIM[:\s]+(?:expected\s+)?(\d+)",
    "LATENT_C": r"LATENT_C[:\s]+(?:expected\s+)?(\d+)",
    "HEAD": r"HEAD[:\s]+(?:expected\s+)?(epsilon|v|flow)",
    "LATENT_SCALE": r"LATENT_SCALE[:\s]+(?:expected\s+)?(\d+)",
    # VISION adapter failures often surface the profile name
    "VISION": r"VISION_ADAPTER[:\s]+([A-Za-z0-9\-_/]+)",
    # LLM-specific hints
    "KV_DTYPE": r"KV(?:_CACHE)?_DTYPE[:=\s]+([A-Za-z0-9_]+)",
    "VOCAB": r"VOCAB[=:\s]+(\d+)",
    "ROPE": r"ROPE[=:\s]+(\d+)(k)?",
}


def parse_reason(msg: str) -> Dict[str, Any]:
    """Extract hypothesis updates from ``msg`` using regex rules."""
    if "'" in msg:
        msg = msg.split("'", 2)[1]
    for key, pattern in ERROR_PATTERNS.items():
        m = re.search(pattern, msg, re.IGNORECASE)
        if not m:
            continue
        val: Any = m.group(1)
        if key in {"TEXT_EMB_d", "LATENT_C", "LATENT_SCALE", "VOCAB", "ROPE"}:
            val = int(val)
            if key == "ROPE" and m.group(2):
                val *= 1000
        return {key: val}
    return {}


__all__ = ["parse_reason", "ERROR_PATTERNS"]
