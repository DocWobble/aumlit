from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from planner import Planner, probe_signature


def test_planner_update_prunes_and_resets():
    p = Planner(seed={})
    next(p)  # consume the first combination (TEXT_EMB_d=768)
    p.update({"TEXT_EMB_d": 4096})
    combo = next(p)
    assert combo["TEXT_EMB_d"] == 4096
    assert p.candidates["TEXT_EMB_d"] == [4096]


def test_planner_ordering():
    defaults = {
        "TEXT_EMB_d": [1, 2],
        "HEAD": ["a", "b"],
        "LATENT_C": [4],
        "LATENT_SCALE": [8],
        "VISION": ["V1", "V2"],
        "AUDIO_SHAPE": [(1, 1, 1), (2, 2, 2)],
        "VOCAB": [32000],
        "ROPE": [128000],
        "KV_DTYPE": ["f16"],
    }
    p = Planner(seed={}, defaults=defaults)
    first = next(p)
    second = next(p)
    assert list(first.keys()) == [
        "TEXT_EMB_d",
        "HEAD",
        "LATENT_C",
        "LATENT_SCALE",
        "VISION",
        "AUDIO_SHAPE",
        "VOCAB",
        "ROPE",
        "KV_DTYPE",
    ]
    assert first["AUDIO_SHAPE"] != second["AUDIO_SHAPE"]
    assert first["VISION"] == second["VISION"]
    assert first["TEXT_EMB_d"] == second["TEXT_EMB_d"]


def test_planner_skips_failed_probes():
    defaults = {"A": [1, 2], "B": [3]}
    sig = probe_signature({"A": 1, "B": 3})
    p = Planner(seed={}, defaults=defaults, order=["A", "B"], failed_probes={sig})
    combo = next(p)
    assert combo == {"A": 2, "B": 3}
