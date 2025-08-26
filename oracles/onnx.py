from __future__ import annotations
from pathlib import Path
from typing import Any, Dict


def _engine_forward(artifact: Path, inputs: Dict[str, Any]) -> str:
    """Run a minimal ONNX inference step and surface failures."""
    import numpy as np
    import onnxruntime as ort
    import re

    sess = ort.InferenceSession(str(artifact))
    inputs_info = sess.get_inputs()
    input_name = inputs_info[0].name
    shape = inputs_info[0].shape
    expected = shape[-1] if shape and isinstance(shape[-1], int) else None
    hints = []
    if expected is not None:
        hints.append(f"TEXT_EMB_d={expected}")
    data = next(iter(inputs.values()))
    if hasattr(data, "numpy"):
        data = data.numpy()
    if expected is not None and data.shape[-1] != expected:
        raise RuntimeError(f"COND_DIM: expected {expected}")
    try:
        sess.run(None, {input_name: np.asarray(data)})
    except Exception as e:  # pragma: no cover - depends on runtime errors
        msg = str(e)
        ints = re.findall(r"\d+", msg)
        parts = []
        if ints:
            parts.extend(ints)
        parts.extend(hints)
        if parts:
            raise RuntimeError("ONNX_FAIL: " + " ".join(parts)) from e
        raise
    return "ok"
