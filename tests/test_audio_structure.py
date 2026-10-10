import numpy as np
import pytest

pytest.importorskip("scipy")
from kata.audio.frames import FPS, SR, n_frames  # noqa: E402
from kata.audio.structure import boundary_density_check, segment  # noqa: E402


def abab(seed=1):
    """33 s: sections A B A B of 8 s each (16 beats at 120 BPM), distinct chroma and timbre."""
    rng = np.random.default_rng(seed)
    T = n_frames(33 * SR)
    sec = np.minimum((np.arange(T) / FPS // 8).astype(int), 3) % 2
    chroma = 0.1 + rng.random((T, 12)) * 0.1
    chroma[sec == 0, 0], chroma[sec == 0, 4] = 1.0, 0.8
    chroma[sec == 1, 7], chroma[sec == 1, 2] = 1.0, 0.8
    pa, pb = rng.normal(-30, 10, 128), rng.normal(-30, 10, 128)
    mel = np.where(sec[:, None] == 0, pa, pb) + rng.normal(0, 1, (T, 128))
    beats = np.arange(0, 32.01, 0.5)
    return chroma, mel, beats, beats[::4]


def test_abab_gives_two_repeating_labels():
    chroma, mel, beats, downbeats = abab()
    segs = segment(chroma, mel, beats, downbeats, 33.0)
    assert [s["label"] for s in segs] == [0, 1, 0, 1]
    np.testing.assert_allclose([s["start_s"] for s in segs[1:]], [8, 16, 24], atol=0.5)
    assert segs[0]["start_s"] == 0.0 and segs[-1]["end_s"] == 33.0


def test_segment_is_deterministic():
    args = abab()
    assert segment(*args, 33.0) == segment(*args, 33.0)


def test_short_song_is_one_segment():
    beats = np.arange(0, 10, 0.5)
    segs = segment(np.ones((1001, 12)), np.zeros((1001, 128)), beats, beats[::4], 10.0)
    assert segs == [{"start_s": 0.0, "end_s": 10.0, "label": 0}]


def test_boundary_density_check():
    notes = np.concatenate([np.arange(10, 20, 0.1), np.arange(0, 30, 2.0)])  # dense from 10 to 20 s
    c = boundary_density_check([10.0, 15.0, 20.0], notes, 30.0)
    assert c["hits"] == 2 and c["n"] == 3  # 15 s is mid-section: no density change there
    assert 0 < c["coverage"] < 0.7
