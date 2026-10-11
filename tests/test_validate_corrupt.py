import numpy as np
import pytest

from conftest import notes_df, stream
from kata.validate.corrupt import CORRUPTIONS, Injection, corrupt, evaluate
from kata.validate.rules import MapInput, analyse
from kata.validate.score import score_analysis


@pytest.mark.parametrize("kind", sorted(CORRUPTIONS))
def test_every_corruption_is_detected_on_a_clean_stream_without_false_alarms(kind, toy_ref):
    m = MapInput("ExpertPlus", 18.0, 0.0, 120.0, stream())
    a = analyse(m)
    clean = score_analysis(a, m.difficulty, toy_ref)
    assert clean.passed and clean.issues.height == 0
    rng = np.random.default_rng(7)
    for _ in range(5):
        inj, bad = corrupt(kind, m, a, rng, toy_ref.anchors("L1", clean.stratum)["q99"])
        assert inj.rule == CORRUPTIONS[kind]
        hit, false_alarms = evaluate(inj, clean, score_analysis(analyse(bad), bad.difficulty, toy_ref))
        assert hit, inj
        assert false_alarms == 0, inj


def test_no_eligible_place_returns_none(toy_ref):
    m = MapInput("ExpertPlus", 18.0, 0.0, 120.0, stream(n_per_hand=4, stack_every=99))
    assert corrupt("C6", m, analyse(m), np.random.default_rng(0)) is None
    assert corrupt("C4", m, analyse(m), np.random.default_rng(0), l1_q99=None) is None


def _result(rows, ref, difficulty="Expert"):
    m = MapInput(difficulty, 16.0, 0.0, 120.0, notes_df(rows))
    return score_analysis(analyse(m), m.difficulty, ref)


def test_an_issue_on_the_other_hand_or_already_in_the_clean_map_is_not_a_detection(toy_ref):
    red = [(2.0 + 0.2 * i, 0, 0 if i % 2 == 0 else 2, 0, 1 if i % 2 == 0 else 0) for i in range(10)]
    blue_double_down = [(3.05, 3, 0, 1, 1), (3.2, 3, 0, 1, 1)]          # E1 on blue at 3.2
    clean = _result(red + blue_double_down, toy_ref)
    assert "E1" in clean.issues["rule"].to_list()
    assert evaluate(Injection("C1", "E1", 0, 3.0, 6.0), clean, clean) == (False, 0)


def test_a_window_already_below_half_in_the_clean_map_is_not_a_detection(toy_ref):
    slow_resets = [(2.0 + 0.5 * i, 3, 0, 1, 1) for i in range(40)]      # blue down-down: L3 far past zero
    clean = _result(slow_resets, toy_ref)
    assert (clean.local["s_L3"] < 0.5).any()
    assert evaluate(Injection("C5", "L2", 1, 10.0, 20.0, 12.0), clean, clean) == (False, 0)
