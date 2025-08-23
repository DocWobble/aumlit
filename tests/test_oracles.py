from pathlib import Path
import oracles


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
