from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))
from classifier import parse_reason
from geometry import ConstraintSet


def test_error_corpus():
    data_path = Path(__file__).parent / "data" / "classifier_errors.json"
    cases = json.loads(data_path.read_text())
    assert len(cases) > 50
    for case in cases:
        msg = case["msg"]
        expected = case["update"]
        assert ConstraintSet(parse_reason(msg)).solve() == expected


def test_multiple_hints_collected():
    msg = "COND_DIM: expected 4 LATENT_C: expected 8"
    assert ConstraintSet(parse_reason(msg)).solve() == {"TEXT_EMB_d": 4, "LATENT_C": 8}


def test_surrounding_quotes_removed_but_inner_preserved():
    msg = "'LATENT_C: expected 4 and 'inner' noise'"
    assert ConstraintSet(parse_reason(msg)).solve() == {"LATENT_C": 4}
