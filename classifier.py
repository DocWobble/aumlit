"""Regex-based error classifier extracting hypothesis hints."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from geometry import ConstraintSet, DimVar, Equality

# Map hypothesis keys to regex patterns capturing the relevant value.
# Patterns are intentionally broad and numeric groups are prioritised.
DEFAULT_ERROR_PATTERNS = {
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

ERROR_PATTERNS = DEFAULT_ERROR_PATTERNS.copy()

# Attempt to extend ERROR_PATTERNS from YAML rules file.
rules_path = Path(__file__).resolve().parent / "rules" / "error_rules.yaml"
try:
    loaded = yaml.safe_load(rules_path.read_text())
    if isinstance(loaded, dict):
        extra = {k: v for k, v in loaded.items() if isinstance(k, str) and isinstance(v, str)}
        ERROR_PATTERNS.update(extra)
except (FileNotFoundError, yaml.YAMLError, OSError):
    # Fall back to default patterns silently if file missing or malformed.
    pass


def parse_reason(msg: str) -> List[Equality]:
    """Extract constraints from ``msg`` using regex rules."""

    msg = msg.strip()
    m = re.match(r"^(['\"])(.*)\1$", msg)
    if m:
        msg = m.group(2)
    constraints: List[Equality] = []
    for key, pattern in ERROR_PATTERNS.items():
        for m in re.finditer(pattern, msg, re.IGNORECASE):
            val: Any = m.group(1)
            if key in {"TEXT_EMB_d", "LATENT_C", "LATENT_SCALE", "VOCAB", "ROPE"}:
                val = int(val)
                if key == "ROPE" and m.group(2):
                    val *= 1000
            constraints.append(Equality(DimVar(key), val))
    return constraints


def constraints_from_error(msg: str) -> ConstraintSet:
    """Parse ``msg`` into a ``ConstraintSet``."""

    cs = ConstraintSet()
    cs.add(*parse_reason(msg))
    return cs


__all__ = ["parse_reason", "constraints_from_error", "ERROR_PATTERNS"]
