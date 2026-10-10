"""Phase 2 stages on a tiny synthetic dataset with a CPU stand-in for the GPU extractor."""
import json

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("soxr")
import soundfile as sf  # noqa: E402

from kata.audio import decode  # noqa: E402
from kata.audio import pipeline as ap  # noqa: E402
from kata.audio.extract import SongFeatures, feature_config  # noqa: E402
from kata.audio.frames import SR, n_frames  # noqa: E402
from kata.audio.store import FeatureStore  # noqa: E402
from kata.config import Paths  # noqa: E402


class FakeExtractor:
    """Deterministic features derived from the decoded audio; counts calls."""

    def __init__(self):
        self.calls = 0
        self.config = feature_config("fake-sep", "fake-beats")

    def __call__(self, audio):
        self.calls += 1
        T = n_frames(audio.n_samples)
        level = float(np.abs(audio.mono).mean())
        beats = np.arange(0.5, audio.n_samples / SR - 0.5, 0.5)
        curves = np.zeros((T, 12), np.float32)
        curves[:, 0] = level
        arrays = {"mel": np.full((T, 128), level, np.float32), "stem_mel": np.full((T, 4, 64), level, np.float32),
                  "chroma": np.full((T, 12), level, np.float32), "curves": curves}
        scalars = {"duration_s": audio.n_samples / SR, "n_frames": T, "source_sr": audio.source_sr,
                   "source_channels": audio.source_channels, "bandlimited": audio.source_sr < SR,
                   **decode.level_stats(audio)}
        return SongFeatures(arrays, beats, beats[::4], scalars)


@pytest.fixture
def paths(tmp_path) -> Paths:
    """Maps h1/h2 share audio 'aa' (train, h1 is the pilot map), h3 = 'bb' (val), h4 = 'cc' is corrupt."""
    p = Paths(tmp_path).ensure()
    rng = np.random.default_rng(0)
    for h, amp in (("h1", 0.1), ("h3", 0.5)):
        (p.maps / h).mkdir(parents=True)
        sf.write(p.maps / h / "song.egg", (rng.standard_normal((SR * 6, 2)) * amp).astype(np.float32), SR,
                 format="OGG", subtype="VORBIS")
    (p.maps / "h4").mkdir()
    (p.maps / "h4" / "song.egg").write_bytes(b"not audio")
    pl.DataFrame({"hash": ["h1", "h2", "h3", "h4"], "audio_sha256": ["aa", "aa", "bb", "cc"],
                  "audio_file": ["song.egg"] * 4, "audio_status": ["ok"] * 4,
                  "pilot": [True, False, False, False], "split": ["train", "train", "val", "train"]}
                 ).write_parquet(p.processed / "maps.parquet")
    diff = {"characteristic": ["Standard"] * 2, "difficulty": ["Expert"] * 2}
    pl.DataFrame({"hash": ["h1", "h3"], **diff, "tier": ["A", "B"], "parse_status": ["ok"] * 2,
                  "difficulty_rank": [7, 7]}).write_parquet(p.processed / "difficulties.parquet")
    beats = np.arange(2.0, 10.0, 1.0)
    pl.DataFrame({"hash": ["h1"] * 8 + ["h3"] * 8, "characteristic": ["Standard"] * 16,
                  "difficulty": ["Expert"] * 16, "beat": np.tile(beats, 2), "time_s": np.tile(beats / 2, 2)}
                 ).write_parquet(p.processed / "notes.parquet")
    pl.DataFrame({"hash": ["h1", "h3"], **diff, "seg_idx": [0, 0], "start_beat": [0.0, 0.0],
                  "start_s": [0.0, 0.0], "bpm": [120.0, 120.0]}).write_parquet(p.processed / "tempo_maps.parquet")
    return p


def test_features_once_per_unique_audio_and_failures_recorded(paths):
    fx = FakeExtractor()
    s = ap.features(paths, extractor=fx)
    assert fx.calls == 2 and (s["songs"], s["ok"], s["error"]) == (3, 2, 1)
    af = pl.read_parquet(paths.processed / "audio_features.parquet").sort("audio_sha256")
    assert af["status"].to_list() == ["ok", "ok", "error"]
    assert FeatureStore(paths.features).has("aa")
    beats = pl.read_parquet(paths.processed / "audio_beats.parquet")
    assert set(beats["audio_sha256"]) == {"aa", "bb"} and beats["beat_in_bar"].max() == 4


def test_features_is_resumable(paths):
    fx = FakeExtractor()
    ap.features(paths, extractor=fx)
    s = ap.features(paths, extractor=fx)
    assert fx.calls == 2 and s["processed"] == 1  # only the failed song is retried


def test_features_pilot_subset(paths):
    fx = FakeExtractor()
    s = ap.features(paths, pilot=True, extractor=fx)
    assert s["songs"] == 1 and fx.calls == 1


def test_features_refuses_a_changed_config(paths):
    ap.features(paths, extractor=FakeExtractor())
    other = FakeExtractor()
    other.config = {**other.config, "fps": 50}
    with pytest.raises(ValueError, match="bump FEATURE_VERSION"):
        ap.features(paths, extractor=other)


def test_structure_sync_stats(paths):
    ap.features(paths, extractor=FakeExtractor())
    assert ap.structure(paths)["status"] == {"ok": 2}
    assert pl.read_parquet(paths.processed / "audio_segments.parquet")["audio_sha256"].n_unique() == 2

    assert ap.sync(paths)["difficulties"] == 2
    sync = pl.read_parquet(paths.processed / "map_audio_sync.parquet")
    assert sync.height == 4 and set(sync["src"]) == {"mix", "drums"}
    ba = pl.read_parquet(paths.processed / "beat_agreement.parquet").sort("hash")
    assert ba["best_level"].to_list() == [1.0, 1.0] and ba["best_f"].to_list() == [1.0, 1.0]

    ap.stats(paths)
    stats = json.loads((FeatureStore(paths.features).stats_path).read_text())
    assert stats["split"] == "train" and stats["songs"] == ["aa"]  # bb is val, cc failed
    stored = np.asarray(FeatureStore(paths.features).load("aa", "mel"), np.float64)
    assert stats["arrays"]["mel"]["mean"][0] == pytest.approx(stored[:, 0].mean())
    assert stats["arrays"]["stem_mel"]["shape"] == [4, 64]


def test_report(paths):
    ap.features(paths, extractor=FakeExtractor())
    for stage in (ap.structure, ap.sync, ap.stats):
        stage(paths)
    path = ap.report(paths, name="test")["path"]
    text = open(path, encoding="utf-8").read()
    for heading in ("## 2. Features", "## 4. Map grid vs. audio beats", "## 5. Map ↔ audio sync",
                    "## 6. Structure", "## 7. Normalisation stats"):
        assert heading in text
