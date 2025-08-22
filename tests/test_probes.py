from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from classifier import parse_reason
from reshell import _onnx_engine_forward, _llama_engine_forward


# ---- ONNX -----------------------------------------------------------


def _save_onnx_linear(in_features: int, path: Path) -> None:
    import onnx
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, in_features])
    W = helper.make_tensor("W", TensorProto.FLOAT, [in_features, 1], [0.0] * in_features)
    B = helper.make_tensor("B", TensorProto.FLOAT, [1], [0.0])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Gemm", ["input", "W", "B"], ["output"])
    graph = helper.make_graph([node], "g", [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 10
    onnx.save(model, path)


def test_onnx_probe_success(tmp_path):
    model = tmp_path / "lin.onnx"
    _save_onnx_linear(3, model)
    out = _onnx_engine_forward(model, {"x": np.zeros((1, 3), dtype=np.float32)})
    assert out == "ok"


def test_onnx_probe_failure_classified(tmp_path):
    model = tmp_path / "lin.onnx"
    _save_onnx_linear(3, model)
    with pytest.raises(RuntimeError) as exc:
        _onnx_engine_forward(model, {"x": np.zeros((1, 4), dtype=np.float32)})
    assert parse_reason(str(exc.value)) == {"TEXT_EMB_d": 3}


# ---- llama.cpp ------------------------------------------------------


def test_llama_probe_success(monkeypatch, tmp_path):
    class Dummy:
        def __init__(self, *a, **k):
            pass

        def eval(self, tokens):
            return None

    import types
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Dummy))
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"gguf")
    out = _llama_engine_forward(artifact, {"token_id": 0})
    assert out == "ok"


def test_llama_probe_failure_classified(monkeypatch, tmp_path):
    class Dummy:
        def __init__(self, *a, **k):
            pass

        def eval(self, tokens):
            raise RuntimeError("vocab 16 expected 32000")

    import types
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Dummy))
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"gguf")
    with pytest.raises(RuntimeError) as exc:
        _llama_engine_forward(artifact, {"token_id": 0})
    assert parse_reason(str(exc.value)) == {"VOCAB": 32000}
