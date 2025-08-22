from pathlib import Path
import json
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from reshell import run_pipeline


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
    assert "COND_DIM" in proof and "ok" in proof
    assert oblig_path.read_text()


def test_run_pipeline_cli_format(tmp_path):
    artifact = tmp_path / "lin128.pt"
    _save_linear(128, artifact)

    ct_path, oblig_path, proof_path = run_pipeline(artifact, tmp_path, fmt="cli")

    assert "TEXT_EMB" in ct_path.read_text()
    assert oblig_path.read_text() == ""
    assert "candidate" in proof_path.read_text()
