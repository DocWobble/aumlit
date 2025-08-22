from __future__ import annotations

"""Utility renderers for probe outputs.

This module collects small helpers that render the results of a probing
run into either JSON-serialisable structures or human-friendly CLI
strings.  The functions are intentionally tiny and do not attempt to be
complete – they merely provide placeholders for richer implementations.
"""

from typing import Any, Dict, Iterable, List


# Mapping from hypothesis keys to obligation templates.  Each template is
# rendered with a placeholder when the corresponding hypothesis is
# missing.
_OBLIGATION_TEMPLATES = {
    "TEXT_EMB_d": "TEXT_ENCODER(dim={})",
    "LATENT_C": "VAE(C={})",
    "LATENT_SCALE": "VAE(scale={})",
    "HEAD": "UNET(head={})",
    "VISION": "VISION_ADAPTER(profile={})",
}


def contact_trace(hyp: Dict[str, Any], validation: Dict[str, Any] | None = None, fmt: str = "json") -> Any:
    """Render a minimal contact trace.

    Parameters
    ----------
    hyp:
        Dictionary of resolved hypotheses.
    validation:
        Optional extra artefacts gathered during validation.
    fmt:
        Output format – ``"json"`` yields a dict suitable for
        ``json.dumps`` while ``"cli"`` returns a simple string for
        terminal display.
    """

    data = {"hypotheses": hyp}
    if validation:
        data["validation"] = validation

    if fmt == "json":
        return data
    if fmt == "cli":
        parts: List[str] = []
        if "TEXT_EMB_d" in hyp:
            parts.append(f"TEXT_EMB[d={hyp['TEXT_EMB_d']}]")
        if "HEAD" in hyp:
            parts.append(f"UNET(head={hyp['HEAD']})")
        if "LATENT_C" in hyp and "LATENT_SCALE" in hyp:
            parts.append(f"VAE(C={hyp['LATENT_C']},scale={hyp['LATENT_SCALE']})")
        if "VISION" in hyp:
            parts.append(f"VISION(profile={hyp['VISION']})")
        return " -> ".join(parts) or "<empty>"
    raise ValueError(f"unknown format: {fmt}")


def obligations(hyp: Dict[str, Any], fmt: str = "json") -> Any:
    """Derive outstanding component obligations from hypotheses.

    Missing hypothesis keys are rendered into user-readable obligation
    messages.
    """

    missing = [template.format("?") for key, template in _OBLIGATION_TEMPLATES.items() if key not in hyp]

    if fmt == "json":
        return missing
    if fmt == "cli":
        return "\n".join(f"* {m}" for m in missing)
    raise ValueError(f"unknown format: {fmt}")


def proof(log: Iterable[str], fmt: str = "cli") -> Any:
    """Render the proof transcript.

    ``log`` is expected to be an iterable of already-formatted lines.  By
    default a single CLI string is returned; ``fmt="json"`` yields a
    list of lines suitable for serialising.
    """

    if fmt == "json":
        return list(log)
    if fmt == "cli":
        return "\n".join(log)
    raise ValueError(f"unknown format: {fmt}")
