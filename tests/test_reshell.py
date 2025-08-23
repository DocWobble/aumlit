from pathlib import Path
import json
import sys

import torch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import reshell
from reshell import run_pipeline
import oracles.torch as torch_oracle


def _save_linear(in_features: int, path: Path) -> None:
    """Create and save a tiny TorchScript linear module."""
    module = torch.nn.Linear(in_features, 1)
    scripted = torch.jit.script(module)
    scripted.save(str(path))


def test_run_pipeline_success(tmp_path):
    artifact = tmp_path / "lin768.pt"
    _save_linear(768, artifact)

    ct_path, oblig_path, proof_path = run_pipeline(artifact, tmp_path)

    trace = json.loads(ct_path.read_text())
    assert trace["hypotheses"]["TEXT_EMB_d"] == 768
    assert "ok" in proof_path.read_text()
    assert oblig_path.read_text()


def test_run_pipeline_failure_classification(tmp_path):
    artifact = tmp_path / "lin4.pt"
    _save_linear(4, artifact)

    ct_path, oblig_path, proof_path = run_pipeline(artifact, tmp_path)

    trace = json.loads(ct_path.read_text())
    assert trace["hypotheses"]["TEXT_EMB_d"] == 4
    proof = proof_path.read_text()
    assert "COND_DIM" in proof
    assert oblig_path.read_text()


def test_run_pipeline_cli_format(tmp_path):
    artifact = tmp_path / "lin128.pt"
    _save_linear(128, artifact)

    ct_path, oblig_path, proof_path = run_pipeline(artifact, tmp_path, fmt="cli")

    assert "TEXT_EMB" in ct_path.read_text()
    assert "TEXT_ENCODER" in oblig_path.read_text()
    assert "candidate" in proof_path.read_text()


def test_run_pipeline_multiple_hints(monkeypatch, tmp_path):
    artifact = tmp_path / "multi.pt"
    _save_linear(4, artifact)

    def fake_forward(artifact_path, inputs):
        text = inputs.get("text")
        latent = inputs.get("latent")
        msgs = []
        if text is None or text.shape[-1] != 4:
            msgs.append("COND_DIM: expected 4")
        if latent is None or latent.shape[1] != 8:
            msgs.append("LATENT_C: expected 8")
        if msgs:
            raise RuntimeError(" ".join(msgs))
        return "ok"

    monkeypatch.setattr(torch_oracle, "_engine_forward", fake_forward)

    ct_path, oblig_path, proof_path = run_pipeline(artifact, tmp_path)

    trace = json.loads(ct_path.read_text())
    assert trace["hypotheses"]["TEXT_EMB_d"] == 4
    assert trace["hypotheses"]["LATENT_C"] == 8
    assert "COND_DIM" in proof_path.read_text()


def test_run_pipeline_onnx_fixture(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    artifact = Path(__file__).resolve().parent / "fixtures" / "linear4.onnx"

    ct_path, oblig_path, proof_path = run_pipeline(artifact, tmp_path)

    trace = json.loads(ct_path.read_text())
    assert trace["hypotheses"]["TEXT_EMB_d"] == 4
    proof = proof_path.read_text()
    assert "COND_DIM" in proof
    assert oblig_path.read_text()


def test_run_pipeline_gguf_fixture(tmp_path):
    pytest.importorskip("gguf")
    pytest.importorskip("llama_cpp")
    artifact = Path(__file__).resolve().parent / "fixtures" / "tiny.gguf"

    guesses = {"VOCAB": 1, "ROPE": 1, "KV_DTYPE": "f16"}
    ct_path, oblig_path, proof_path = run_pipeline(artifact, tmp_path, guesses=guesses)

    trace = json.loads(ct_path.read_text())
    assert trace["hypotheses"]["VOCAB"] == 1
    proof = proof_path.read_text()
    assert "candidate" in proof
    assert oblig_path.read_text()


def test_run_pipeline_header_cache(monkeypatch, tmp_path):
    artifact = tmp_path / "lin256.pt"
    _save_linear(256, artifact)
    cache_dir = tmp_path / "cache"

    # first run populates cache
    run_pipeline(artifact, tmp_path / "out1", limits={"cache_dir": cache_dir})

    def fail(*_args, **_kwargs):
        raise AssertionError("try_forward should not be called on cache hit")

    monkeypatch.setattr(reshell, "try_forward", fail)

    # second run with same headers should hit cache and skip probes
    artifact2 = tmp_path / "lin256_copy.pt"
    _save_linear(256, artifact2)
    ct_path, oblig_path, proof_path = run_pipeline(
        artifact2, tmp_path / "out2", limits={"cache_dir": cache_dir}
    )

    trace = json.loads(ct_path.read_text())
    assert trace["hypotheses"]["TEXT_EMB_d"] == 256
    assert "recognized header" in proof_path.read_text()
    assert oblig_path.read_text()

