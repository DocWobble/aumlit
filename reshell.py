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
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from classifier import parse_reason

from headers import Meta, read_gguf_header, read_onnx_header, read_safetensors_header
from planner import Planner, probe_signature
from failure_cache import FailureCache
from printers import contact_trace, obligations, proof
from puppets import audio_mel, latent, text_emb, vision_grid
from sandbox import spawn_sandbox
from oracles import try_forward


def _parse_assignments(spec: str | None) -> Dict[str, str]:
    """Parse comma-separated ``key=value`` pairs into a dict."""

    result: Dict[str, str] = {}
    if not spec:
        return result
    for part in spec.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        result[key.strip()] = val.strip()
    return result


def _parse_guess(guess: str | None) -> Dict[str, Any]:
    """Parse ``--guess`` into hypothesis overrides."""

    mapping = {
        "d": "TEXT_EMB_d",
        "C": "LATENT_C",
        "head": "HEAD",
        "scale": "LATENT_SCALE",
        "vision": "VISION",
    }
    result: Dict[str, Any] = {}
    for key, val in _parse_assignments(guess).items():
        hyp_key = mapping.get(key, key)
        try:
            result[hyp_key] = int(val)
        except ValueError:
            result[hyp_key] = val
    return result


def _parse_limits(limits: str | None) -> Dict[str, int | float | str]:
    """Parse ``--limits`` into ``spawn_sandbox`` kwargs.

    Supported keys include ``timeout``, ``vram``, ``cpu_mem`` and
    ``cache_dir``.
    """

    result: Dict[str, int | float | str] = {}
    for key, val in _parse_assignments(limits).items():
        lk = key.strip()
        lv_raw = val.strip()
        lv = lv_raw.lower()
        if lk == "timeout":
            if lv.endswith("s"):
                lv = lv[:-1]
            try:
                result["timeout"] = float(lv)
            except ValueError:
                pass
        elif lk in {"vram", "cpu_mem"}:
            mult = 1
            if lv.endswith("gb"):
                mult = 1024 ** 3
                lv = lv[:-2]
            elif lv.endswith("mb"):
                mult = 1024 ** 2
                lv = lv[:-2]
            try:
                result[lk] = int(float(lv) * mult)
            except ValueError:
                pass
        elif lk in {"cache", "cache_dir"}:
            result["cache_dir"] = lv_raw
    return result


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


def _header_hash(meta: Meta) -> str:
    """Compute a stable hash for ``meta``."""

    blob = {
        "tensors": [(t.name, t.shape) for t in meta.tensors],
        "hints": meta.hints,
    }
    raw = json.dumps(blob, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()




def run_pipeline(
    artifact: Path,
    out_dir: Path,
    guesses: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
    fmt: str | None = None,
    failure_cache: FailureCache | None = None,
) -> Tuple[Path, Path, Path]:
    """Run a tiny probing loop over candidate combinations."""

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(artifact)
    header_id = _header_hash(meta)

    hypotheses: Dict[str, Any] = dict(meta.hints)
    if guesses:
        hypotheses.update(guesses)
    known_failures = failure_cache.get(header_id) if failure_cache else {}
    planner = Planner(seed=hypotheses, failed_probes=set(known_failures.keys()))
    limits_dict = dict(limits) if limits else {}
    cache_dir = limits_dict.pop("cache_dir", None)
    sandbox = spawn_sandbox(limits_dict or None, cache_dir=cache_dir)

    proof_log: list[str] = []
    validation: Dict[str, Any] | None = None
    for combo in planner:
        puppet_inputs: Dict[str, Any] = {}
        if "TEXT_EMB_d" in combo:
            puppet_inputs["text"] = text_emb(combo["TEXT_EMB_d"])
        if "LATENT_C" in combo and "LATENT_SCALE" in combo:
            puppet_inputs["latent"] = latent(combo["LATENT_C"], combo["LATENT_SCALE"])
        if "VISION" in combo:
            puppet_inputs["vision"] = vision_grid(combo["VISION"])
        if "AUDIO_SHAPE" in combo:
            puppet_inputs["audio"] = audio_mel(combo["AUDIO_SHAPE"])
        if "VOCAB" in combo:
            puppet_inputs["vocab"] = combo["VOCAB"]
        if "ROPE" in combo:
            puppet_inputs["rope"] = combo["ROPE"]
        if "KV_DTYPE" in combo:
            puppet_inputs["kv_dtype"] = combo["KV_DTYPE"]

        try:
            sandbox.try_forward(try_forward, artifact, puppet_inputs)
            hypotheses.update(combo)
            proof_log.append(f"candidate {combo} -> ok")
            validation = sandbox.validate_min_run(try_forward, artifact, puppet_inputs)
            break
        except Exception as e:  # pragma: no cover - error path
            reason = str(e)
            updates = parse_reason(reason)
            err_cls = next(iter(updates), "OTHER")
            if failure_cache:
                failure_cache.record(header_id, probe_signature(combo), err_cls)
            if updates:
                hypotheses.update(updates)
                planner.update(updates)
            proof_log.append(f"candidate {combo} -> {reason}")

    trace_out = contact_trace(hypotheses, validation, fmt=fmt or "json")
    oblig_out = obligations(hypotheses, fmt=fmt or "json")
    proof_out = proof(proof_log, validation, fmt=fmt or "cli")

    contact_trace_path = out_dir / "contact_trace.json"
    obligations_path = out_dir / "obligations.json"
    proof_path = out_dir / "proof.txt"
    contact_trace_path.write_text(trace_out if isinstance(trace_out, str) else json.dumps(trace_out))
    obligations_path.write_text(oblig_out if isinstance(oblig_out, str) else json.dumps(oblig_out))
    proof_path.write_text(proof_out if isinstance(proof_out, str) else json.dumps(proof_out))

    return contact_trace_path, obligations_path, proof_path


def _run(
    artifact: str,
    out_dir: str = "out",
    guess: str | None = None,
    limits: str | None = None,
    fmt: str | None = None,
    failures: str | None = None,
) -> None:
    """Delegate to the probing pipeline.

    Parameters
    ----------
    artifact:
        Path to the model artifact to analyse.
    out_dir:
        Directory where probe results should be written.  The directory
        is created if it does not exist.
    guess:
        Optional hypothesis overrides (``key=value`` pairs).
    limits:
        Optional sandbox limits and cache directory
        (``timeout=2s,vram=1GB,cache_dir=/tmp/cache``).
    fmt:
        Optional printer output format.
    """

    guesses = _parse_guess(guess)
    limits_map = _parse_limits(limits)
    cache_base = Path(limits_map.get("cache_dir", out_dir))
    failure_cache = FailureCache(cache_base / "failure_cache.json")

    if failures in {"inspect", "clear"}:
        meta = _read_meta(Path(artifact))
        header_id = _header_hash(meta)
        if failures == "inspect":
            data = failure_cache.inspect(header_id)
            print(json.dumps(data, indent=2))
        else:
            failure_cache.clear(header_id)
            print("cleared cached failures")
        return

    contact_trace, obligations, proof_file = run_pipeline(
        Path(artifact),
        Path(out_dir),
        guesses=guesses,
        limits=limits_map,
        fmt=fmt,
        failure_cache=failure_cache,
    )
    print(f"contact trace written to: {contact_trace}")
    print(f"obligations written to: {obligations}")
    print(f"proof transcript written to: {proof_file}")


def main() -> None:
    """Entry-point for the console script."""
    parser = argparse.ArgumentParser(description="Probe a model artifact")
    parser.add_argument("--artifact", required=True, help="path to model artifact")
    parser.add_argument("--out", default="out", help="output directory for results")
    parser.add_argument("--guess", help="seed hypotheses", default=None)
    parser.add_argument(
        "--limits",
        help="resource limits and cache (timeout=2s,vram=1GB,cache_dir=/tmp/cache)",
        default=None,
    )
    parser.add_argument("--format", dest="fmt", help="printer output format", default=None)
    parser.add_argument(
        "--failures",
        choices=["inspect", "clear"],
        help="inspect or clear cached probe failures",
        default=None,
    )
    args = parser.parse_args()

    _run(args.artifact, args.out, args.guess, args.limits, args.fmt, args.failures)


if __name__ == "__main__":
    main()
