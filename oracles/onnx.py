from __future__ import annotations
from pathlib import Path
from typing import Any, Dict


def _engine_forward(artifact: Path, inputs: Dict[str, Any]) -> str:
    """Run a minimal ONNX inference step and surface failures."""
    import numpy as np
    import onnxruntime as ort
    import re

    sess = ort.InferenceSession(str(artifact))
    input_name = sess.get_inputs()[0].name
    expected = sess.get_inputs()[0].shape[-1]
    data = next(iter(inputs.values()))
    if hasattr(data, "numpy"):
        data = data.numpy()
    if data.shape[-1] != expected:
        raise RuntimeError(f"COND_DIM: expected {expected}")
    try:
        sess.run(None, {input_name: np.asarray(data)})
    except Exception as e:  # pragma: no cover - depends on runtime errors
        msg = str(e)
        ints = re.findall(r"\d+", msg)
        if ints:
            raise RuntimeError("ONNX_FAIL: " + " ".join(ints)) from e
        raise
    return "ok"
