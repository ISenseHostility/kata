import numpy as np
import pytest

from conftest import stream
from kata.validate.corrupt import CORRUPTIONS, corrupt, evaluate
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
