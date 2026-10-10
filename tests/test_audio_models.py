"""GPU integration tests: real Demucs and Beat This weights (downloaded on first use)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("demucs")
pytest.importorskip("beat_this")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from kata.audio.frames import SR, n_frames  # noqa: E402
from kata.audio.models import BeatTracker, StemSeparator  # noqa: E402
from kata.audio.sync import click_track  # noqa: E402


def test_separator_shape_and_determinism():
    sep = StemSeparator("cuda")
    x = (np.random.default_rng(0).standard_normal((2, 10 * SR)) * 0.1).astype(np.float32)
    a, b = sep(x), sep(x)
    assert tuple(a.shape) == (4, 2, 10 * SR)
    assert torch.equal(a, b)  # shifts=0: identical across runs


def test_beat_tracker_on_a_click_track():
    times = np.arange(0.24, 29.5, 0.5)  # on Beat This's 20 ms grid
    x = click_track(times, 30.0)
    T = n_frames(len(x))
    out = BeatTracker("cuda")(x, T)
    assert out["beat_act"].shape == (T,) and out["downbeat_act"].shape == (T,)
    hit = [np.min(np.abs(out["beats"] - t)) <= 0.02 for t in times[2:-2]]
    assert np.mean(hit) >= 0.9
