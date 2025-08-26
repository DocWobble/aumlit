from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import printers


def test_contact_trace_comfy():
    hyp = {
        "TEXT_EMB_d": 2048,
        "HEAD": "epsilon",
        "LATENT_C": 4,
        "LATENT_SCALE": 8,
        "VISION": "ViT-H/14-grid",
        "AUDIO_SHAPE": (2, 4, 32),
    }
    trace = printers.contact_trace(hyp, fmt="comfy")
    nodes = {n["type"]: n for n in trace["nodes"]}
    assert nodes["TEXT_EMB"]["params"] == {"d": 2048}
    assert nodes["UNET"]["params"] == {"head": "epsilon"}
    assert nodes["VAE"]["params"] == {"C": 4, "scale": 8}
    assert nodes["VISION"]["params"] == {"profile": "ViT-H/14-grid"}
    assert nodes["AUDIO"]["params"] == {"shape": [2, 4, 32]}
    edges = trace["edges"]
    ids = {typ: nodes[typ]["id"] for typ in nodes}
    assert {"src": ids["TEXT_EMB"], "dst": ids["UNET"], "src_port": "cond", "dst_port": "cond"} in edges
    assert {"src": ids["VAE"], "dst": ids["UNET"], "src_port": "latent", "dst_port": "latent"} in edges
    assert {"src": ids["UNET"], "dst": ids["VAE"], "src_port": "latent", "dst_port": "latent"} in edges
    assert {"src": ids["VISION"], "dst": ids["UNET"], "src_port": "vision", "dst_port": "vision"} in edges
    assert {"src": ids["AUDIO"], "dst": ids["UNET"], "src_port": "audio", "dst_port": "audio"} in edges


def test_obligations_comfy():
    hyp = {"TEXT_EMB_d": 2048}
    oblig = printers.obligations(hyp, fmt="comfy")
    nodes = {n["type"]: n for n in oblig}
    assert "TEXT_ENCODER" not in nodes
    assert nodes["UNET"]["params"] == {"head": None}
    assert nodes["VAE"]["params"] == {"C": None, "scale": None}
    assert nodes["VISION_ADAPTER"]["params"] == {"profile": None}
    assert nodes["AUDIO_ENCODER"]["params"] == {"shape": None}


def test_obligations_cli():
    hyp = {"TEXT_EMB_d": 2048}
    oblig_cli = printers.obligations(hyp, fmt="cli")
    oblig_json = printers.obligations(hyp, fmt="json")
    assert oblig_cli == "\n".join(oblig_json)


def test_obligations_provenance():
    hyp = {"TEXT_EMB_d": 2048, "LATENT_C": 4}
    prov = {
        "TEXT_EMB_d": "candidate {'TEXT_EMB_d': 2048} -> ok",
        "LATENT_C": "candidate {'LATENT_C': 4} -> ok",
    }
    oblig_json = printers.obligations(hyp, provenance=prov, fmt="json")
    assert {
        "obligation": "TEXT_ENCODER(dim=2048)",
        "provenance": prov["TEXT_EMB_d"],
    } in oblig_json
    oblig_cli = printers.obligations(hyp, provenance=prov, fmt="cli")
    assert prov["TEXT_EMB_d"] in oblig_cli


def test_proof_ignores_comfy():
    log = ["a", "b"]
    assert printers.proof(log, fmt="comfy") == printers.proof(log, fmt="cli")


def test_contact_trace_cli_validation_metrics():
    hyp = {}
    val = {"time_ms": 1.0, "vram_mb": 2.0, "dtype_ok": False, "stable": True}
    out = printers.contact_trace(hyp, validation=val, fmt="cli")
    assert "1.0ms" in out
    assert "2.0MB" in out
    assert "dtype_mismatch" in out
    assert "stable" in out


def test_proof_validation_metrics():
    log = ["x"]
    val = {"time_ms": 3.0, "vram_mb": 0.0, "dtype_ok": True, "stable": False}
    out = printers.proof(log, validation=val)
    assert "dtype_ok=True" in out
    assert "stable=False" in out
