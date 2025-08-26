"""Regex-based error classifier extracting hypothesis hints."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from geometry import ConstraintSet, DimVar, Equality

# Map hypothesis keys to regex patterns capturing the relevant value.
# Patterns are intentionally broad and numeric groups are prioritised.
DEFAULT_ERROR_PATTERNS: Dict[str, List[str]] = {
    "TEXT_EMB_d": [r"COND_DIM[:\s]+(?:expected\s+)?(\d+)",
                     # PyTorch matmul shape mismatch
                     r"mat1 and mat2 shapes cannot be multiplied \(\d+x\d+ and \d+x(\d+)\)",
                     r"TEXT_EMB_d[=:\s]+(\d+)",],
    "LATENT_C": [r"LATENT_C[:\s]+(?:expected\s+)?(\d+)",
                  # Conv2d channel mismatch
                  r"expected input.* to have (\d+) channels",
                  r"LATENT_C[=:\s]+(\d+)"],
    "HEAD": [r"HEAD[:\s]+(?:expected\s+)?(epsilon|v|flow)",
              r"prediction mismatch[:\s]+(?:expected\s+)?(epsilon|v|flow)"],
    "LATENT_SCALE": [r"LATENT_SCALE[:\s]+(?:expected\s+)?(\d+)",],
    # VISION adapter failures often surface the profile name
    "VISION": [r"VISION_ADAPTER[:\s]+([A-Za-z0-9\-_/]+)"],
    # LLM-specific hints
    "KV_DTYPE": [r"KV(?:_CACHE)?_DTYPE[:=\s]+([A-Za-z0-9_]+)"],
    "VOCAB": [r"VOCAB[=:\s]+(\d+)"],
    "ROPE": [r"ROPE[=:\s]+(\d+)(k)?"],
    "LAYOUT": [r"LAYOUT[=:\s]+([A-Za-z0-9_\-]+)"],
}

# Copy defaults so we can extend with YAML-provided patterns.
ERROR_PATTERNS: Dict[str, List[str]] = {
    k: v[:] for k, v in DEFAULT_ERROR_PATTERNS.items()
}

# Attempt to extend ERROR_PATTERNS from YAML rules file. Each key may map to a
# string pattern or a list of patterns.
rules_path = Path(__file__).resolve().parent / "rules" / "error_rules.yaml"
try:
    loaded = yaml.safe_load(rules_path.read_text())
    if isinstance(loaded, dict):
        for key, val in loaded.items():
            if not isinstance(key, str):
                continue
            patterns = []
            if isinstance(val, str):
                patterns = [val]
            elif isinstance(val, list):
                patterns = [v for v in val if isinstance(v, str)]
            if patterns:
                ERROR_PATTERNS.setdefault(key, [])
                ERROR_PATTERNS[key].extend(patterns)
except (FileNotFoundError, yaml.YAMLError, OSError):
    # Fall back to default patterns silently if file missing or malformed.
    pass

# Debug logging for unmatched errors.
DEBUG = os.getenv("AUM_CLASSIFIER_DEBUG") == "1"
UNMATCHED_LOG = Path(__file__).resolve().parent / "rules" / "unmatched_errors.log"


def parse_reason(msg: str) -> List[Equality]:
    """Extract constraints from ``msg`` using regex rules."""

    msg = msg.strip()
    m = re.match(r"^(['\"])(.*)\1$", msg)
    if m:
        msg = m.group(2)
    constraints: List[Equality] = []
    matched = False
    for key, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, msg, re.IGNORECASE):
                matched = True
                val: Any = m.group(1)
                if key in {"TEXT_EMB_d", "LATENT_C", "LATENT_SCALE", "VOCAB", "ROPE"}:
                    val = int(val)
                    if key == "ROPE" and m.group(2):
                        val *= 1000
                constraints.append(Equality(DimVar(key), val))
    if DEBUG and not matched:
        try:
            with UNMATCHED_LOG.open("a") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass
    return constraints


def constraints_from_error(msg: str) -> ConstraintSet:
    """Parse ``msg`` into a ``ConstraintSet``."""

    cs = ConstraintSet()
    cs.add(*parse_reason(msg))
    return cs


__all__ = ["parse_reason", "constraints_from_error", "ERROR_PATTERNS"]
