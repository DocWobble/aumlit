from geometry import DimVar, Equality, Divisible, ConstraintSet
from classifier import parse_reason
import pytest


def test_equality_and_solve():
    v = DimVar("X")
    cs = ConstraintSet([Equality(v, 7)])
    assert cs.solve() == {"X": 7}


def test_divisible_constraint():
    v = DimVar("Y")
    cs = ConstraintSet([Equality(v, 8), Divisible(v, 4)])
    assert cs.solve() == {"Y": 8}
    cs_bad = ConstraintSet([Equality(v, 10), Divisible(v, 4)])
    with pytest.raises(ValueError):
        cs_bad.solve()


def test_parse_reason_integration():
    msg = "COND_DIM: expected 4 LATENT_C: expected 8"
    cs = ConstraintSet(parse_reason(msg))
    assert cs.solve() == {"TEXT_EMB_d": 4, "LATENT_C": 8}
