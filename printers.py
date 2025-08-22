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
        trace = " -> ".join(parts) or "<empty>"
        if validation:
            metrics: List[str] = []
            if "time_ms" in validation:
                metrics.append(f"{validation['time_ms']:.1f}ms")
            if "vram_mb" in validation:
                metrics.append(f"{validation['vram_mb']:.1f}MB")
            if metrics:
                trace = f"{trace} ({', '.join(metrics)})"
        return trace
    if fmt == "comfy":
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_ids: Dict[str, int] = {}

        next_id = 1

        def add_node(name: str, params: Dict[str, Any]) -> int:
            nonlocal next_id
            node = {"id": next_id, "type": name, "params": params}
            nodes.append(node)
            node_ids[name] = next_id
            next_id += 1
            return node["id"]

        if "TEXT_EMB_d" in hyp:
            add_node("TEXT_EMB", {"d": hyp["TEXT_EMB_d"]})
        if "HEAD" in hyp:
            add_node("UNET", {"head": hyp["HEAD"]})
        if "LATENT_C" in hyp or "LATENT_SCALE" in hyp:
            params: Dict[str, Any] = {}
            if "LATENT_C" in hyp:
                params["C"] = hyp["LATENT_C"]
            if "LATENT_SCALE" in hyp:
                params["scale"] = hyp["LATENT_SCALE"]
            add_node("VAE", params)
        if "VISION" in hyp:
            add_node("VISION", {"profile": hyp["VISION"]})

        # Edges expressing minimal data flow between nodes.
        if "TEXT_EMB" in node_ids and "UNET" in node_ids:
            edges.append({
                "src": node_ids["TEXT_EMB"],
                "dst": node_ids["UNET"],
                "src_port": "cond",
                "dst_port": "cond",
            })
        if "VAE" in node_ids and "UNET" in node_ids:
            edges.append({
                "src": node_ids["VAE"],
                "dst": node_ids["UNET"],
                "src_port": "latent",
                "dst_port": "latent",
            })
            edges.append({
                "src": node_ids["UNET"],
                "dst": node_ids["VAE"],
                "src_port": "latent",
                "dst_port": "latent",
            })
        if "VISION" in node_ids and "UNET" in node_ids:
            edges.append({
                "src": node_ids["VISION"],
                "dst": node_ids["UNET"],
                "src_port": "vision",
                "dst_port": "vision",
            })

        return {"nodes": nodes, "edges": edges}
    raise ValueError(f"unknown format: {fmt}")


def obligations(hyp: Dict[str, Any], fmt: str = "json") -> Any:
    """Derive outstanding component obligations from hypotheses.

    Missing hypothesis keys are rendered into user-readable obligation
    messages.
    """

    missing = [
        template.format(hyp.get(key, "?")) for key, template in _OBLIGATION_TEMPLATES.items()
    ]

    if fmt == "json":
        return missing
    if fmt == "cli":
        return "\n".join(f"* {m}" for m in missing)
    if fmt == "comfy":
        nodes: List[Dict[str, Any]] = []
        if "TEXT_EMB_d" not in hyp:
            nodes.append({"type": "TEXT_ENCODER", "params": {"dim": None}})
        vae_params: Dict[str, Any] = {}
        if "LATENT_C" not in hyp:
            vae_params["C"] = None
        if "LATENT_SCALE" not in hyp:
            vae_params["scale"] = None
        if vae_params:
            nodes.append({"type": "VAE", "params": vae_params})
        if "HEAD" not in hyp:
            nodes.append({"type": "UNET", "params": {"head": None}})
        if "VISION" not in hyp:
            nodes.append({"type": "VISION_ADAPTER", "params": {"profile": None}})
        return nodes
    raise ValueError(f"unknown format: {fmt}")


def proof(log: Iterable[str], validation: Dict[str, Any] | None = None, fmt: str = "cli") -> Any:
    """Render the proof transcript.

    ``log`` is expected to be an iterable of already-formatted lines.  By
    default a single CLI string is returned; ``fmt="json"`` yields a
    list of lines suitable for serialising.
    """

    lines = list(log)
    if validation:
        lines.append(f"validate {validation}")
    if fmt == "json":
        return lines
    if fmt in {"cli", "comfy"}:
        return "\n".join(lines)
    raise ValueError(f"unknown format: {fmt}")
