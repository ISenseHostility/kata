import numpy as np
import pytest

from kata.audio.beats import beat_in_bar, f_measure, grid_agreement, tempo_stats

GRID = np.arange(0, 30, 0.5)  # 120 BPM map grid


def test_f_measure():
    assert f_measure(GRID, GRID) == 1.0
    assert f_measure(GRID, GRID + 0.05) == 1.0  # inside ±70 ms
    assert f_measure(GRID, GRID + 0.1) == 0.0
    assert f_measure([], []) == 1.0 and f_measure(GRID, []) == 0.0
    assert f_measure(GRID, GRID[::2]) == pytest.approx(2 / 3)


@pytest.mark.parametrize("beats, level", [
    (GRID + 0.01, 1.0),
    (np.arange(0, 30, 0.25), 2.0),   # audio beats twice as fast as the map grid
    (np.arange(0, 30, 0.125), 4.0),
    (np.arange(0.5, 30, 1.0), 0.5),  # half-time, odd phase
    (np.arange(0, 30, 1.0), 0.5),    # half-time, even phase
])
def test_grid_agreement_finds_the_metrical_level(beats, level):
    ga = grid_agreement(GRID, beats)
    assert ga["best_level"] == level and ga["best_f"] == pytest.approx(1.0)
    assert set(ga) == {"f_l025", "f_l05", "f_l1", "f_l2", "f_l4", "best_level", "best_f"}


def test_grid_agreement_ignores_beats_outside_the_mapped_span():
    beats = np.concatenate([np.arange(-10, 0, 0.5), GRID])
    assert grid_agreement(GRID, beats)["f_l1"] == 1.0


def test_beat_in_bar_and_tempo_stats():
    beats = np.arange(1.0, 21.0, 0.5)
    downbeats = beats[2::4]
    pos, is_db = beat_in_bar(beats, downbeats)
    assert list(pos[:8]) == [0, 0, 1, 2, 3, 4, 1, 2]
    assert is_db.sum() == len(downbeats)
    ts = tempo_stats(beats, downbeats)
    assert ts["est_bpm"] == pytest.approx(120) and ts["est_meter"] == 4 and ts["tempo_cv"] == pytest.approx(0)


def test_tempo_cv_is_robust_to_a_missed_beat():
    beats = np.delete(np.arange(0, 60, 0.5), 50)
    assert tempo_stats(beats, beats[::4])["tempo_cv"] == pytest.approx(0)
