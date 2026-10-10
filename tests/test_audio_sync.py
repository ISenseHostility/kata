import numpy as np
import pytest

pytest.importorskip("scipy")
from kata.audio.frames import FPS  # noqa: E402
from kata.audio.sync import estimate_lag, measure_onset_bias, note_train  # noqa: E402

NOTES = np.sort(np.random.default_rng(1).choice(np.arange(2.0, 58.0, 0.25), 150, replace=False))


def onsets_for(notes, early_s):
    """Onset curve with a peak `early_s` before each note (on the frame grid), plus noise."""
    o = np.zeros(60 * FPS + 1)
    for t in notes - early_s:
        o[int(round(t * FPS))] += 1.0
    return o + np.random.default_rng(2).random(len(o)) * 0.3


@pytest.mark.parametrize("late_s", [0.0, 0.03, 0.08, -0.05])
def test_lag_of_shifted_notes(late_s):
    r = estimate_lag(onsets_for(NOTES, late_s), NOTES)
    assert r["lag_ms"] == pytest.approx(late_s * 1000, abs=2)
    assert r["peak_corr"] > 0.3


def test_detector_bias_is_added():
    r = estimate_lag(onsets_for(NOTES, 0.03), NOTES, bias_s=-0.004)
    assert r["lag_ms"] == pytest.approx(26, abs=2)


def test_unrelated_notes_have_low_correlation():
    r = estimate_lag(np.random.default_rng(3).random(60 * FPS + 1), NOTES)
    assert r["peak_corr"] < 0.15


def test_note_train_drops_duplicates_and_out_of_range():
    t = note_train([1.0, 1.0, 5.0, -1.0], 301)
    assert t.sum() == pytest.approx(1.0, abs=1e-6)
    assert int(np.argmax(t)) == 100
