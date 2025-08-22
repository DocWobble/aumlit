from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from planner import Planner


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
    }
    p = Planner(seed={}, defaults=defaults)
    first = next(p)
    second = next(p)
    assert list(first.keys()) == ["TEXT_EMB_d", "HEAD", "LATENT_C", "LATENT_SCALE", "VISION"]
    assert first["VISION"] != second["VISION"]
    assert first["TEXT_EMB_d"] == second["TEXT_EMB_d"]
