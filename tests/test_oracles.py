from pathlib import Path
import types
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import oracles
import oracles.onnx as onnx_oracle
import oracles.xformers as xformers_oracle
import pytest
from classifier import constraints_from_error


def test_try_forward_uses_patched_torch(monkeypatch):
    called = {}

    def fake_forward(artifact, inputs):
        called["flag"] = True
        return "patched"

    monkeypatch.setattr(oracles, "_xformers_attention_probe", lambda: "skip")
    monkeypatch.setattr(oracles.torch, "_engine_forward", fake_forward)

    result = oracles.try_forward(Path("model.pt"), {})

    assert called.get("flag") is True
    assert result == "patched"


def test_onnx_static_shape_hint(monkeypatch, tmp_path):
    class DummyInput:
        name = "x"
        shape = [1, 3]

    class DummySession:
        def __init__(self, path):
            pass

        def get_inputs(self):
            return [DummyInput()]

        def run(self, *_a, **_k):
            raise RuntimeError("runtime 7 8")

    dummy_ort = types.SimpleNamespace(InferenceSession=DummySession)
    monkeypatch.setitem(sys.modules, "onnxruntime", dummy_ort)

    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"onnx")
    with pytest.raises(RuntimeError) as exc:
        onnx_oracle._engine_forward(artifact, {"x": np.zeros((1, 3), dtype=np.float32)})
    solved = constraints_from_error(str(exc.value)).solve()
    assert solved == {"TEXT_EMB_d": 3}


def test_xformers_layout_hint(monkeypatch):
    class DummyTorch:
        float16 = "f16"

        @staticmethod
        def randn(*a, **k):
            return object()

    def fake_mea(q, k, v, mem_efficient=True):
        raise RuntimeError("layout=sbpck")

    xops = types.SimpleNamespace(memory_efficient_attention=fake_mea)
    monkeypatch.setitem(sys.modules, "torch", DummyTorch)
    monkeypatch.setitem(sys.modules, "xformers", types.SimpleNamespace(ops=xops))
    monkeypatch.setitem(sys.modules, "xformers.ops", xops)

    with pytest.raises(RuntimeError) as exc:
        xformers_oracle._engine_forward()
    solved = constraints_from_error(str(exc.value)).solve()
    assert solved == {"LAYOUT": "sbpck"}
