"""Command-line entry point for Reshell probing.

This module wires together the tiny planning utilities and the sandbox
to perform a single probing run.  The probing is intentionally minimal:
we iterate through candidate combinations, invoke engine helpers inside a
resource-limited subprocess, record outcomes and update our hypotheses
until one run succeeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from headers import Meta, read_gguf_header, read_onnx_header, read_safetensors_header
from planner import Planner
from puppets import latent, text_emb, vision_grid
from sandbox import spawn_sandbox


def _read_meta(artifact: Path) -> Meta:
    """Dispatch to the appropriate header reader based on suffix."""

    suffix = artifact.suffix.lower()
    if suffix == ".safetensors":
        return read_safetensors_header(artifact)
    if suffix == ".onnx":
        return read_onnx_header(artifact)
    if suffix == ".gguf":
        return read_gguf_header(artifact)
    return Meta(tensors=[], hints={})


def _dummy_engine_forward(artifact: Path, inputs: Dict[str, Any]) -> str:
    """Placeholder engine that always succeeds."""

    _ = artifact, inputs  # unused
    return "ok"


def run_pipeline(artifact: Path, out_dir: Path) -> Tuple[Path, Path]:
    """Run a tiny probing loop over candidate combinations."""

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(artifact)

    hypotheses: Dict[str, Any] = {}
    planner = Planner(seed=hypotheses)
    sandbox = spawn_sandbox({"timeout": 2.0})

    results: list[Dict[str, Any]] = []
    for combo in planner:
        puppet_inputs: Dict[str, Any] = {}
        if "TEXT_EMB_d" in combo:
            puppet_inputs["text"] = text_emb(combo["TEXT_EMB_d"])
        if "LATENT_C" in combo and "LATENT_SCALE" in combo:
            puppet_inputs["latent"] = latent(combo["LATENT_C"], combo["LATENT_SCALE"])
        if "VISION" in combo:
            puppet_inputs["vision"] = vision_grid(combo["VISION"])

        try:
            sandbox.try_forward(_dummy_engine_forward, artifact, puppet_inputs)
            hypotheses.update(combo)
            results.append({"candidate": combo, "status": "ok"})
            break
        except Exception as e:  # pragma: no cover - error path
            results.append({"candidate": combo, "status": "error", "reason": str(e)})

    contact_trace = out_dir / "contact_trace.json"
    obligations = out_dir / "obligations.json"
    contact_trace.write_text(json.dumps({"hypotheses": hypotheses, "results": results}))
    pending = {k: v for k, v in planner.candidates.items() if k not in hypotheses}
    obligations.write_text(json.dumps(pending))

    return contact_trace, obligations


def main(artifact: str, out_dir: str = "out") -> None:
    """Parse arguments and delegate to the probing pipeline.

    Parameters
    ----------
    artifact:
        Path to the model artifact to analyse.
    out_dir:
        Directory where probe results should be written.  The directory
        is created if it does not exist.
    """
    contact_trace, obligations = run_pipeline(Path(artifact), Path(out_dir))
    print(f"contact trace written to: {contact_trace}")
    print(f"obligations written to: {obligations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe a model artifact")
    parser.add_argument("--artifact", required=True, help="path to model artifact")
    parser.add_argument("--out", default="out", help="output directory for results")
    args = parser.parse_args()

    main(args.artifact, args.out)
