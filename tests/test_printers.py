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
    }
    trace = printers.contact_trace(hyp, fmt="comfy")
    nodes = {n["type"]: n for n in trace["nodes"]}
    assert nodes["TEXT_EMB"]["params"] == {"d": 2048}
    assert nodes["UNET"]["params"] == {"head": "epsilon"}
    assert nodes["VAE"]["params"] == {"C": 4, "scale": 8}
    assert nodes["VISION"]["params"] == {"profile": "ViT-H/14-grid"}
    edges = trace["edges"]
    ids = {typ: nodes[typ]["id"] for typ in nodes}
    assert {"src": ids["TEXT_EMB"], "dst": ids["UNET"], "src_port": "cond", "dst_port": "cond"} in edges
    assert {"src": ids["VAE"], "dst": ids["UNET"], "src_port": "latent", "dst_port": "latent"} in edges
    assert {"src": ids["UNET"], "dst": ids["VAE"], "src_port": "latent", "dst_port": "latent"} in edges
    assert {"src": ids["VISION"], "dst": ids["UNET"], "src_port": "vision", "dst_port": "vision"} in edges


def test_obligations_comfy():
    hyp = {"TEXT_EMB_d": 2048}
    oblig = printers.obligations(hyp, fmt="comfy")
    nodes = {n["type"]: n for n in oblig}
    assert "TEXT_ENCODER" not in nodes
    assert nodes["UNET"]["params"] == {"head": None}
    assert nodes["VAE"]["params"] == {"C": None, "scale": None}
    assert nodes["VISION_ADAPTER"]["params"] == {"profile": None}


def test_proof_ignores_comfy():
    log = ["a", "b"]
    assert printers.proof(log, fmt="comfy") == printers.proof(log, fmt="cli")
