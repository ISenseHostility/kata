import numpy as np
import pytest

from kata.baseline.grid import Grid, fit_grid, fit_line


def _beats(t0, period, n, jitter=0.0, drop=(), seed=0):
    b = t0 + np.arange(n) * period + np.random.default_rng(seed).uniform(-jitter, jitter, n)
    return np.delete(b, list(drop))


def test_fit_line_survives_missed_beats_and_jitter():
    t0, period, p95 = fit_line(_beats(0.23, 0.5, 400, jitter=0.01, drop=(5, 100, 101)))
    assert period == pytest.approx(0.5, abs=1e-4)
    assert t0 == pytest.approx(0.23, abs=0.003)
    assert p95 < 0.011


def test_steady_grid_puts_map_beat_one_on_the_first_audio_beat_after_half_a_period():
    g = fit_grid(_beats(0.23, 0.5, 200), duration_s=100.0)
    assert not g.piecewise and g.bpm == pytest.approx(120)
    assert g.times[0] == 0 and g.times[1] == pytest.approx(0.73)       # 0.23 < 0.25 is skipped
    assert g.bpm_events()[0] == (0.0, pytest.approx(60 / 0.73))
    assert g.bpm_events()[1] == (1.0, pytest.approx(120))
    k = np.arange(1, 50)
    np.testing.assert_allclose(g.to_beats(0.73 + 0.5 * (k - 1)), k, atol=1e-9)
    np.testing.assert_allclose(g.tempo_map().to_seconds(k), 0.73 + 0.5 * (k - 1), atol=1e-9)
    assert g.to_beats(200.0) == pytest.approx(1 + (200 - 0.73) / 0.5)   # extrapolates past the grid


def test_unsteady_song_is_fitted_per_segment():
    a = _beats(0.5, 0.5, 60)                            # 120 BPM until 30 s
    b = a[-1] + 0.6 + np.arange(80) * (60 / 140)        # then 140 BPM
    g = fit_grid(np.concatenate([a, b]), duration_s=b[-1] + 1, segments=[(0, 30.25), (30.25, b[-1] + 1)])
    assert g.piecewise
    k = g.to_beats(np.concatenate([a[5:-5], b[5:-5]]))
    np.testing.assert_allclose(k, np.round(k), atol=0.02)
    assert len(g.bpm_events()) >= 3


@pytest.mark.parametrize("bpm, per_beat", [(120, 4), (100, 8)])
def test_slots_are_quarter_beats_or_eighths_for_slow_songs(bpm, per_beat):
    period = 60 / bpm
    g = Grid(np.concatenate([[0.0], period * np.arange(1, 6)]), bpm, False)
    t, b = g.slots()
    assert b[0] == 1 and b[1] == pytest.approx(1 + 1 / per_beat) and b[-1] == 5
    np.testing.assert_allclose(np.diff(t), period / per_beat)
