import pytest

from conftest import bombs_df, stream
from kata.validate.rules import MapInput
from kata.validate.score import combine, sub_score, validate

A = {"q90": 1.0, "q99": 2.0, "zero": 4.0}


@pytest.mark.parametrize("m, s", [(0.0, 1.0), (1.0, 1.0), (1.5, 0.75), (2.0, 0.5), (3.0, 0.25), (4.0, 0.0), (9.0, 0.0)])
def test_sub_score_is_piecewise_linear_between_the_anchors(m, s):
    assert sub_score(m, A) == pytest.approx(s)


def test_combine_is_a_floored_geometric_mean():
    ones = {r: 1.0 for r in ("E1", "E2", "E3", "L1", "L2", "L3")}
    assert combine(ones) == pytest.approx(100.0)
    assert combine({**ones, "E2": 0.0}) == pytest.approx(100 * 0.01 ** (1 / 6))
    assert combine({**ones, "L1": 0.5}) == pytest.approx(100 * 0.5 ** (1 / 6))


def test_validate_gates_hard_errors_but_still_scores(toy_ref):
    clean = validate(MapInput("ExpertPlus", 18.0, 0.0, 120.0, stream()), toy_ref)
    assert clean.passed and clean.score == pytest.approx(100.0) and clean.stratum == "ExpertPlus|all"
    assert {"s_E1", "score"} <= set(clean.local.columns)
    n = stream()
    t, x, y = n["time_s"][10], n["x"][10], n["y"][10]
    bad = validate(MapInput("ExpertPlus", 18.0, 0.0, 120.0, n, bombs_df([(t + 0.005, x, y)])), toy_ref)
    assert not bad.passed and bad.metrics["H1"] == 1 and bad.score == pytest.approx(100.0)
