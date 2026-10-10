import numpy as np
import pytest

from kata.audio.frames import FPS, HOP, SR, beat_grid_times, frame_index, n_frames, sample, tempo_from_rows
from kata.data.mapformat.tempo import TempoMap


def test_grid_constants():
    assert (SR, HOP, FPS) == (44100, 441, 100)


@pytest.mark.parametrize("n, want", [(0, 1), (440, 1), (441, 2), (3 * 44100, 301)])
def test_n_frames_matches_a_centred_stft(n, want):
    assert n_frames(n) == want


def test_frame_index():
    assert frame_index(1.234) == pytest.approx(123.4)


def test_sample_interpolates_and_returns_silence_outside_the_song():
    a = np.arange(10, dtype=np.float16)[:, None] * np.array([1, 2], np.float16)
    s = sample(a, [0.0, 0.015, 0.09, -0.01, 0.0901], silence=-100.0)
    assert s.dtype == np.float32
    np.testing.assert_allclose(s, [[0, 0], [1.5, 3], [9, 18], [-100, -100], [-100, -100]])


def test_sample_accepts_a_silence_row():
    s = sample(np.zeros((5, 3), np.float32), [1.0], silence=np.array([1, 2, 3], np.float32))
    np.testing.assert_array_equal(s, [[1, 2, 3]])


def test_beat_grid_times_follow_bpm_changes():
    tm = TempoMap.from_bpm_events(120, [(4, 60)])
    np.testing.assert_allclose(beat_grid_times(tm, 0, 6), [0, .5, 1, 1.5, 2, 3, 4])
    np.testing.assert_allclose(beat_grid_times(tm, 3, 5, subdiv=2), [1.5, 1.75, 2.0, 2.5, 3.0])


def test_tempo_from_rows_round_trips_tempo_maps_parquet_rows():
    tm = TempoMap.from_bpm_events(120, [(4, 60), (8, 180)])
    beats = [0, 3, 4, 7.5, 8, 12]
    rows = list(reversed(tm.rows()))  # row order must not matter
    np.testing.assert_allclose(tempo_from_rows(rows).to_seconds(beats), tm.to_seconds(beats))
