import numpy as np
import pytest

from kata.audio.frames import FPS
from kata.audio.store import CURVE_COLUMNS
from kata.baseline.generate import SongInput, generate


def test_generate_gives_five_passing_difficulties_with_rising_density(toy_ref):
    dur = 90.0
    curves = np.zeros((int(dur * FPS) + 1, 12), np.float32)
    curves[:, CURVE_COLUMNS.index("rms_mix")] = -20
    rng = np.random.default_rng(0)
    for k in range(int((dur - 1) / 0.125)):              # onsets on every 1/4 of a 120 BPM beat
        curves[int(round((0.5 + 0.125 * k) * FPS)), CURVE_COLUMNS.index("onset_drums")] = rng.uniform(0.3, 1)
    beats = 0.5 + 0.5 * np.arange(int(dur / 0.5) - 1)
    song = SongInput("ab" * 32, dur, curves, beats, beats[::4], [(0.0, 45.0), (45.0, dur)])
    grid, out, results = generate(song, toy_ref)
    assert grid.bpm == pytest.approx(120)
    counts = [out[label].notes.height for label in ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")]
    assert counts == sorted(counts) and counts[0] > 50
    assert all(r.passed and r.issues.height == 0 for r in results.values())
    again = generate(song, toy_ref)[1]
    assert all(out[k].notes.equals(again[k].notes) for k in out)
