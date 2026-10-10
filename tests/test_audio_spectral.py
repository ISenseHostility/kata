import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")
from kata.audio.frames import FPS, SR, n_frames  # noqa: E402
from kata.audio.spectral import (SpectralExtractor, chroma_filterbank, mel_centers,  # noqa: E402
                                 mel_filterbank)
from kata.audio.sync import click_track  # noqa: E402


@pytest.fixture(scope="module")
def sp():
    return SpectralExtractor("cpu")


def sine(freq, seconds=5.0, amp=1.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_filterbanks():
    fb = mel_filterbank(128)
    assert fb.shape == (128, 1025)
    assert (fb >= 0).all() and (fb.sum(axis=1) > 0).all()
    assert np.all(np.diff(fb.argmax(axis=1)) >= 0)
    cf = chroma_filterbank()
    assert cf.shape == (12, 2049) and set(np.unique(cf)) <= {0.0, 1.0}


def test_mix_shapes_follow_the_frame_grid(sp):
    x = sine(440, 3.3)
    m = sp.mix(x)
    T = n_frames(len(x))
    assert m["mel"].shape == (T, 128) and m["chroma"].shape == (T, 12)
    assert m["onset"].shape == (T,) and m["rms"].shape == (T,)


def test_sine_lands_in_the_right_mel_bin_and_chroma_class(sp):
    m = sp.mix(sine(440))
    want = int(np.argmin(np.abs(mel_centers(128) - 440)))
    assert abs(int(np.argmax(m["mel"][50:-50].mean(axis=0))) - want) <= 1
    assert int(np.argmax(m["chroma"][50:-50].mean(axis=0))) == 9  # A


def test_full_scale_sine_rms_is_minus_3_dbfs(sp):
    assert np.median(sp.mix(sine(440))["rms"]) == pytest.approx(-3.01, abs=0.05)


def test_silence_gives_the_floor(sp):
    m = sp.mix(np.zeros(3 * SR, np.float32))
    assert (m["mel"] == -100).all() and (m["rms"] == -100).all()
    assert (m["chroma"] == 0).all() and (m["onset"] == 0).all()


def test_onset_strength_peaks_at_clicks(sp):
    clicks = np.arange(1.0, 9.0, 0.5) + 0.0037
    flux = sp.mix(click_track(clicks, 10.0))["onset"]
    for c in clicks:
        i = int(round(c * FPS))
        assert abs(int(np.argmax(flux[i - 10:i + 11])) - 10) <= 2


def test_stem_features_are_time_major(sp):
    x = sine(220, 2.0)
    st = sp.stems(np.stack([np.stack([x, x])] * 4))
    T = n_frames(len(x))
    assert st["stem_mel"].shape == (T, 4, 64)
    assert st["onset"].shape == (4, T) and st["rms"].shape == (4, T)
    np.testing.assert_allclose(st["stem_mel"][:, 0], st["stem_mel"][:, 3])
