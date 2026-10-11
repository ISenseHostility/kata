import numpy as np
import polars as pl
import pytest

from kata.baseline.patterns import place
from kata.validate.rules import LABELS, MapInput, analyse


@pytest.mark.parametrize("label", LABELS)
def test_placement_alternates_without_any_rule_issue(label):
    # 2 s bars: a strong downbeat, then 4 notes, then a breath, so both hands are free and on the
    # same parity at every downbeat (the double conditions)
    times = (2.0 + 2.0 * np.arange(40)[:, None] + np.array([0, 0.5, 0.75, 1.0, 1.25])).ravel()
    strength = np.tile([5.0, 1, 1, 1, 1], 40)
    notes = place(times, times * 2, strength, times[::5], label, 0.4, np.random.default_rng(1))
    a = analyse(MapInput(label, 16.0, 0.0, 120.0, notes))
    assert a.issues.height == 0
    assert notes.filter(pl.col("x").is_in([1, 2]) & (pl.col("y") == 1)).height == 0   # no vision blockers
    doubles = notes.group_by("time_s").len().filter(pl.col("len") > 1).height
    assert (doubles > 0) == (label in ("Hard", "Expert", "ExpertPlus"))
    if label in ("Easy", "Normal"):
        assert set(notes["cut_dir"].unique().to_list()) <= {0, 1}


def test_placement_is_deterministic_per_seed():
    times = 2.0 + np.arange(50) * 0.25
    a = place(times, times * 2, np.ones(50), np.array([]), "Expert", 0.4, np.random.default_rng(3))
    b = place(times, times * 2, np.ones(50), np.array([]), "Expert", 0.4, np.random.default_rng(3))
    assert a.equals(b)
