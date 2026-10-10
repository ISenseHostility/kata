import numpy as np
import pytest

pytest.importorskip("soxr")
import soundfile as sf  # noqa: E402

from kata.audio.decode import level_stats, load, to_canonical  # noqa: E402
from kata.audio.frames import SR  # noqa: E402

# Every source rate in the dataset (docs/phases/phase-2-audio.md §2).
SOURCE_RATES = [22050, 32000, 38000, 44099, 48000, 88200, 96000, 176400, 192000]


@pytest.mark.parametrize("sr", SOURCE_RATES)
def test_resampling_keeps_an_impulse_on_its_sample(sr):
    x = np.zeros((sr * 3, 1), np.float32)
    x[sr] = 1.0
    a = to_canonical(x, sr)
    assert a.n_samples == 3 * SR
    assert int(np.argmax(np.abs(a.mono))) == SR


@pytest.mark.parametrize("sr", [44100, 48000])
def test_vorbis_round_trip_keeps_click_timing(tmp_path, sr):
    x = np.zeros((sr * 3, 2), np.float32)
    x[sr:sr + 8] = 0.9
    path = tmp_path / "song.egg"  # Beat Saber's extension; libsndfile sniffs the content
    sf.write(path, x, sr, format="OGG", subtype="VORBIS")
    a = load(path)
    assert a.source_sr == sr and a.n_samples == 3 * SR
    assert abs(int(np.argmax(np.abs(a.mono) > 0.3)) - SR) <= 1


def test_mono_is_duplicated_and_extra_channels_dropped():
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((1000, 1)).astype(np.float32)
    a = to_canonical(x1, SR)
    assert a.stereo.shape == (2, 1000) and a.source_channels == 1
    np.testing.assert_array_equal(a.stereo[0], a.stereo[1])
    x4 = rng.standard_normal((1000, 4)).astype(np.float32)
    b = to_canonical(x4, SR)
    np.testing.assert_array_equal(b.stereo, x4[:, :2].T)
    assert b.source_channels == 4


def test_mono_is_the_channel_mean():
    x = np.stack([np.ones(10), -0.5 * np.ones(10)], axis=1).astype(np.float32)
    np.testing.assert_allclose(to_canonical(x, SR).mono, 0.25)


def test_level_stats():
    t = np.arange(SR) / SR
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None]
    s = level_stats(to_canonical(x, SR))
    assert s["peak_dbfs"] == pytest.approx(-6.02, abs=0.01)
    assert s["rms_dbfs"] == pytest.approx(-9.03, abs=0.01)
    assert s["clip_ratio"] == 0.0
