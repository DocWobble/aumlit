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
