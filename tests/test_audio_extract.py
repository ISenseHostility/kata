import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("soxr")
from kata.audio.decode import to_canonical  # noqa: E402
from kata.audio.extract import Extractor  # noqa: E402
from kata.audio.frames import SR, n_frames  # noqa: E402
from kata.audio.store import CURVE_COLUMNS, STEMS  # noqa: E402


class FakeSeparator:
    """Stems = the mix scaled by 0.1 … 0.4, so stem loudness increases in STEMS order."""
    name = "fake-sep"

    def __call__(self, stereo):
        return torch.from_numpy(np.stack([stereo * g for g in (0.1, 0.2, 0.3, 0.4)]))


class FakeBeats:
    name = "fake-beats"

    def __call__(self, mono, T):
        return {"beat_act": np.full(T, 0.25, np.float32), "downbeat_act": np.full(T, 0.75, np.float32),
                "beats": np.array([0.5, 1.0, 1.5]), "downbeats": np.array([0.5])}


def extractor():
    return Extractor("cpu", separator=FakeSeparator(), beat_tracker=FakeBeats())


def test_extract_assembles_every_array_on_one_grid():
    x = (np.random.default_rng(0).standard_normal((5 * SR + 123, 2)) * 0.1).astype(np.float32)
    f = extractor()(to_canonical(x, SR))
    T = n_frames(5 * SR + 123)
    assert {k: v.shape for k, v in f.arrays.items()} == {
        "mel": (T, 128), "stem_mel": (T, 4, 64), "chroma": (T, 12), "curves": (T, 12)}
    c = f.arrays["curves"]
    assert (c[:, CURVE_COLUMNS.index("beat_act")] == 0.25).all()
    assert (c[:, CURVE_COLUMNS.index("downbeat_act")] == 0.75).all()
    rms = [float(np.median(c[:, CURVE_COLUMNS.index(f"rms_{s}")])) for s in STEMS]
    assert rms == sorted(rms)
    assert f.scalars["n_frames"] == T and f.scalars["source_sr"] == SR and f.scalars["bandlimited"] is False
    np.testing.assert_array_equal(f.beats, [0.5, 1.0, 1.5])


def test_config_names_the_models():
    cfg = extractor().config
    assert cfg["separator"] == "fake-sep" and cfg["beat_tracker"] == "fake-beats"
    assert cfg["fps"] == 100 and cfg["curve_columns"] == list(CURVE_COLUMNS)
