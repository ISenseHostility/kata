# Phase 2 — Audio Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `kata.audio`: a resumable pipeline that turns each of the 5,347 unique Phase 1 songs into cached fp16 feature arrays on a 100 fps grid, plus beats, structure, map↔audio sync, and normalisation stats, and runs it on the pilot and then the full dataset.

**Architecture:** Small pure modules (frame grid, decoding, spectral maths, beat statistics, structure, sync, feature store) are composed by `extract.py` (one song → all outputs) and driven by `pipeline.py` stages that mirror Phase 1's conventions: `_with_state` runs, a per-song resumable `features` stage, and table-rebuilding `structure` / `sync` / `stats` / `report` stages. The GPU models (Demucs, Beat This!) are isolated behind two small wrappers, so everything else is tested on CPU with fakes.

**Tech Stack:** Python 3.13 (uv), torch 2.11.0+cu130, demucs 4.1.0 (`htdemucs`), beat-this 1.1.0, soxr, scipy, numpy, polars, duckdb, soundfile, typer, pytest.

**Spec:** `docs/phases/phase-2-audio.md` (approved 2026-10-09; prototype corrections in §17). Read §3 (time conventions), §4 (feature spec), §9 (storage) before starting.

## Global Constraints

- Install and run with the GPU extra: `uv sync --extra audio`, and **always `uv run --extra audio …`** so uv never drops torch from the environment.
- `kata/cli.py` must not import `kata.audio` at module level: `kata data …` must keep working without the extra. Import inside each `audio` command.
- Test modules that need torch / scipy / soxr start with `pytest.importorskip(...)`. GPU tests also skip without CUDA. No live-network tests (model weights are cached after Step 0).
- Frame grid: `SR = 44100`, `HOP = 441`, `FPS = 100`; frame *i* is centred at `i / 100` s from the first decoded sample (spec §3).
- Arrays are fp16 and time-major: `mel (T,128)`, `stem_mel (T,4,64)`, `chroma (T,12)`, `curves (T,12)`.
- `curves` columns, in this order: `onset_mix, onset_drums, onset_bass, onset_other, onset_vocals, rms_mix, rms_drums, rms_bass, rms_other, rms_vocals, beat_act, downbeat_act`.
- Stem order: `drums, bass, other, vocals`. Demucs `htdemucs` with `shifts=0, overlap=0.25, split=True`. Beat This! checkpoint `final0`, minimal post-processing.
- Feature store root `$KATA_DATA/features/v1/`; a changed feature config must bump `FEATURE_VERSION`, never overwrite.
- No real audio, maps or features are committed (Phase 1 D6). Test fixtures are synthetic.
- Every stage stops when the data drive has < 50 GB free (`kata.data.pipeline._check_disk`).
- Commit after every task. End each commit message with the attribution lines from the session's system reminder.

## Review Focus

1. **Notes or windows beyond the end of the audio.** Expected: feature sampling returns silence rows (−100 dB / 0), never repeated edge values; sync ignores out-of-range notes. Pinned by `test_sample_interpolates_and_returns_silence_outside_the_song`, `test_window_pads_with_silence_rows` and `test_note_train_drops_duplicates_and_out_of_range` (Tasks 1, 3, 5).
2. **A song that fails mid-run (corrupt file, CUDA error).** Expected: the stage records the error and continues; the next run retries only failed songs. Pinned by `test_features_once_per_unique_audio_and_failures_recorded` and `test_features_is_resumable` (Task 10).
3. **Changing a feature parameter after features exist.** Expected: refusal with "bump FEATURE_VERSION", never a mix of versions in one directory. Pinned by `test_config_is_pinned_per_version` (Task 3) and `test_features_refuses_a_changed_config` (Task 10).
4. **Maps with doubled/halved BPM or BPM changes.** Expected: grid agreement reports the metrical level instead of a low score; beat grids follow BPM changes. Pinned by `test_grid_agreement_finds_the_metrical_level` (Task 4) and `test_beat_grid_times_follow_bpm_changes` (Task 1).
5. **Normalisation stats leaking val/test songs.** Expected: `stats.json` lists and uses train-split songs only. Pinned by `test_structure_sync_stats` (Task 11).

## File Structure

| File | Responsibility |
|------|----------------|
| `src/kata/config.py` (modify) | `Paths.features` |
| `src/kata/audio/__init__.py` | package docstring |
| `src/kata/audio/frames.py` | frame grid constants, interpolation, beat-grid times, tempo map from Parquet rows |
| `src/kata/audio/decode.py` | soundfile decode → soxr VHQ resample → stereo/mono canonical audio; level stats |
| `src/kata/audio/store.py` | feature version dir, config pinning, atomic `.npy` writes, mmap reads, windows |
| `src/kata/audio/beats.py` | F-measure, map-grid agreement at metrical levels, bar positions, tempo stats |
| `src/kata/audio/sync.py` | note impulse train, lag estimation, click track, onset-bias calibration |
| `src/kata/audio/spectral.py` | torch STFT, mel/chroma filterbanks, log-mel, spectral flux, RMS |
| `src/kata/audio/structure.py` | beat-sync features, recurrence + spectral clustering, boundary/density check |
| `src/kata/audio/models.py` | Demucs and Beat This! wrappers (the only GPU-model code) |
| `src/kata/audio/extract.py` | one canonical song → `SongFeatures`; the feature config dict |
| `src/kata/audio/pipeline.py` | stages `features`, `structure`, `sync`, `stats`, `report`; `PILOT_STAGES` |
| `src/kata/audio/report.py` | Phase 2 Markdown report |
| `src/kata/cli.py` (modify) | `kata audio <stage>` commands |
| `tests/test_audio_*.py` | one test module per source module |

Task order follows the dependencies: 1 → 2 → 3 → 4 → 5 → 6 (spectral uses `sync.click_track`) → 7 → 8 → 9 → 10 → 11 → 12 → 13 (pilot) → 14 (full run).

---

### Task 1: Frame grid and `Paths.features`

**Files:**
- Create: `src/kata/audio/__init__.py`, `src/kata/audio/frames.py`
- Modify: `src/kata/config.py`
- Test: `tests/test_audio_frames.py`

**Interfaces:**
- Consumes: `kata.data.mapformat.tempo.TempoMap`, `Segment` (Phase 1).
- Produces: `SR`, `HOP`, `FPS`; `n_frames(n_samples: int) -> int`; `frame_index(t_s) -> np.ndarray`; `sample(arr, times_s, silence) -> np.ndarray[float32]`; `beat_grid_times(tempo: TempoMap, start_beat, end_beat, subdiv=1) -> np.ndarray`; `tempo_from_rows(rows: list[dict]) -> TempoMap`; `Paths.features -> Path` (`root / "features"`, created by `Paths.ensure()`).

- [ ] **Step 1: Write the failing test** — `tests/test_audio_frames.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_frames.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'kata.audio'`.

- [ ] **Step 3: Implement**

`src/kata/audio/__init__.py`:

```python
"""Phase 2 audio features (docs/phases/phase-2-audio.md)."""
```

`src/kata/audio/frames.py`:

```python
"""Feature time grid (docs/phases/phase-2-audio.md §3, §7.2).

Frame i of every feature array is centred at t = i / FPS seconds, counted from the first
decoded audio sample: the same clock as map `time_s` and replay `frame.time`.
"""

from __future__ import annotations

import math

import numpy as np

from kata.data.mapformat.tempo import Segment, TempoMap

SR = 44100
HOP = 441
FPS = SR // HOP  # 100


def n_frames(n_samples: int) -> int:
    """Frames of a centred STFT with hop HOP over n_samples samples."""
    return n_samples // HOP + 1


def frame_index(t_s) -> np.ndarray:
    """Fractional frame index of time(s) in seconds."""
    return np.asarray(t_s, dtype=np.float64) * FPS


def sample(arr: np.ndarray, times_s, silence) -> np.ndarray:
    """Linearly interpolate `arr` (time on axis 0) at `times_s` -> float32 (len(times_s), ...).

    Times outside the song return `silence` (a scalar, or a row shaped like arr.shape[1:]),
    never a repeated edge value.
    """
    a = np.asarray(arr)
    f = np.atleast_1d(frame_index(times_s))
    last = a.shape[0] - 1
    i0 = np.clip(np.floor(f).astype(np.int64), 0, last)
    i1 = np.minimum(i0 + 1, last)
    w = (f - i0).astype(np.float32).reshape((-1,) + (1,) * (a.ndim - 1))
    out = a[i0].astype(np.float32) * (1 - w) + a[i1].astype(np.float32) * w
    out[(f < 0) | (f > last)] = silence
    return out


def beat_grid_times(tempo: TempoMap, start_beat: float, end_beat: float, subdiv: int = 1) -> np.ndarray:
    """Seconds of every 1/subdiv beat in [start_beat, end_beat]."""
    first = math.ceil(start_beat * subdiv - 1e-9)
    last = math.floor(end_beat * subdiv + 1e-9)
    return tempo.to_seconds(np.arange(first, last + 1) / subdiv)


def tempo_from_rows(rows: list[dict]) -> TempoMap:
    """Rebuild a TempoMap from `tempo_maps.parquet` rows of one difficulty (offset already applied)."""
    rows = sorted(rows, key=lambda r: r["seg_idx"])
    return TempoMap([Segment(r["start_beat"], r["start_s"], 60.0 / r["bpm"]) for r in rows])
```

In `src/kata/config.py`, add this property to `Paths` directly above `state_db`:

```python
    @property
    def features(self) -> Path:
        """Phase 2 feature store root (docs/phases/phase-2-audio.md §9)."""
        return self.root / "features"
```

and add `self.features` to the tuple in `Paths.ensure()`:

```python
        for p in (self.catalog_pages, self.maps, self.bl_leaderboards, self.bl_scores,
                  self.replays, self.processed, self.replay_frames, self.features,
                  self.state_db.parent, self.reports):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_frames.py -q` → `10 passed`.
Run: `uv run --extra audio pytest -q` → all Phase 1 tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/__init__.py src/kata/audio/frames.py src/kata/config.py tests/test_audio_frames.py
git commit -m "Phase 2: frame grid, sampling and beat-grid helpers"
```

---

### Task 2: Canonical audio decoding

**Files:**
- Create: `src/kata/audio/decode.py`
- Test: `tests/test_audio_decode.py`

**Interfaces:**
- Consumes: `frames.SR`.
- Produces: `CanonicalAudio(stereo: (2,N) float32, source_sr: int, source_channels: int)` with `.mono -> (N,)` and `.n_samples`; `to_canonical(x: (N,C) float32, sr) -> CanonicalAudio`; `load(path) -> CanonicalAudio`; `level_stats(audio) -> {"peak_dbfs", "rms_dbfs", "clip_ratio"}` (plain floats).

- [ ] **Step 1: Write the failing test** — `tests/test_audio_decode.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_decode.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.audio.decode'`.

- [ ] **Step 3: Implement** — `src/kata/audio/decode.py`:

```python
"""Canonical audio (§3): soundfile decode -> soxr VHQ resample to 44.1 kHz -> stereo float32."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from kata.audio.frames import SR


@dataclass
class CanonicalAudio:
    stereo: np.ndarray  # (2, N) float32 at SR
    source_sr: int
    source_channels: int

    @property
    def mono(self) -> np.ndarray:
        return self.stereo.mean(axis=0)

    @property
    def n_samples(self) -> int:
        return self.stereo.shape[1]


def to_canonical(x: np.ndarray, sr: int) -> CanonicalAudio:
    """x: (N, C) float32 as soundfile returns it. Mono is duplicated; channels beyond 2 dropped."""
    channels = x.shape[1]
    x = np.repeat(x, 2, axis=1) if channels == 1 else x[:, :2]
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="VHQ")
    return CanonicalAudio(np.ascontiguousarray(x.T, dtype=np.float32), sr, channels)


def load(path: Path | str) -> CanonicalAudio:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return to_canonical(x, sr)


def level_stats(audio: CanonicalAudio) -> dict:
    peak = float(np.abs(audio.stereo).max())
    rms = float(np.sqrt(np.mean(audio.mono.astype(np.float64) ** 2)))
    return {"peak_dbfs": float(20 * np.log10(max(peak, 1e-10))),
            "rms_dbfs": float(20 * np.log10(max(rms, 1e-10))),
            "clip_ratio": float(np.mean(np.abs(audio.stereo) >= 0.999))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_decode.py -q` → `14 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/decode.py tests/test_audio_decode.py
git commit -m "Phase 2: canonical audio decode and resampling"
```

---

### Task 3: Feature store

**Files:**
- Create: `src/kata/audio/store.py`
- Test: `tests/test_audio_store.py`

**Interfaces:**
- Consumes: `frames.FPS`.
- Produces: `FEATURE_VERSION = "v1"`, `STEMS`, `CURVE_COLUMNS`, `ARRAYS = ("mel","stem_mel","chroma","curves")`, `EVENTS`, `SILENCE` (per-array silence value/row); `FeatureStore(root)` with `.dir`, `.config_path`, `.stats_path`, `open_for_write(config: dict)`, `song_dir(sha)`, `has(sha) -> bool`, `write(sha, arrays, beats, downbeats)`, `load(sha, name, mmap=True)`, `events(sha) -> (beats, downbeats)`, `window(sha, start_s, end_s) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write the failing test** — `tests/test_audio_store.py`:

```python
import numpy as np
import pytest

from kata.audio.store import ARRAYS, CURVE_COLUMNS, FEATURE_VERSION, SILENCE, FeatureStore


def arrays(T, v=1.0):
    return {"mel": np.full((T, 128), v), "stem_mel": np.full((T, 4, 64), v),
            "chroma": np.full((T, 12), v), "curves": np.full((T, 12), v)}


def test_write_load_round_trip(tmp_path):
    st = FeatureStore(tmp_path)
    st.open_for_write({"a": 1})
    st.write("abcdef", arrays(300, 0.5), [0.5, 1.0], [0.5])
    assert st.has("abcdef") and not st.has("other")
    assert st.song_dir("abcdef") == tmp_path / FEATURE_VERSION / "ab" / "abcdef"
    mel = st.load("abcdef", "mel")
    assert isinstance(mel, np.memmap) and mel.dtype == np.float16 and mel.shape == (300, 128)
    b, d = st.events("abcdef")
    np.testing.assert_array_equal(b, [0.5, 1.0])
    np.testing.assert_array_equal(d, [0.5])
    assert not list(st.song_dir("abcdef").glob("*.tmp"))


def test_frame_count_mismatch_is_rejected(tmp_path):
    a = arrays(300)
    a["chroma"] = np.zeros((299, 12))
    with pytest.raises(ValueError, match="frame count"):
        FeatureStore(tmp_path).write("abcdef", a, [], [])


def test_config_is_pinned_per_version(tmp_path):
    FeatureStore(tmp_path).open_for_write({"fps": 100})
    FeatureStore(tmp_path).open_for_write({"fps": 100})  # same config: fine
    with pytest.raises(ValueError, match="bump FEATURE_VERSION"):
        FeatureStore(tmp_path).open_for_write({"fps": 50})


def test_window_pads_with_silence_rows(tmp_path):
    st = FeatureStore(tmp_path)
    st.write("abcdef", arrays(300, 1.0), [], [])  # 3 s
    w = st.window("abcdef", 2.5, 3.5)  # 100 frames, 50 inside the song
    assert {k: v.shape[0] for k, v in w.items()} == {k: 100 for k in ARRAYS}
    assert (w["mel"][:50] == 1).all() and (w["mel"][50:] == -100).all()
    np.testing.assert_array_equal(w["curves"][-1], SILENCE["curves"])
    assert (st.window("abcdef", -0.5, 0.5)["chroma"][:50] == 0).all()


def test_curve_silence_row_matches_the_columns():
    assert len(SILENCE["curves"]) == len(CURVE_COLUMNS)
    assert all((v == -100) == c.startswith("rms_") for v, c in zip(SILENCE["curves"], CURVE_COLUMNS))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_store.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.audio.store'`.

- [ ] **Step 3: Implement** — `src/kata/audio/store.py`:

```python
"""Versioned per-song feature files (§9): one fp16 .npy per array, time on axis 0."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from kata.audio.frames import FPS

FEATURE_VERSION = "v1"
STEMS = ("drums", "bass", "other", "vocals")
CURVE_COLUMNS = ("onset_mix", "onset_drums", "onset_bass", "onset_other", "onset_vocals",
                 "rms_mix", "rms_drums", "rms_bass", "rms_other", "rms_vocals",
                 "beat_act", "downbeat_act")
ARRAYS = ("mel", "stem_mel", "chroma", "curves")
EVENTS = ("beats", "downbeats")
SILENCE = {
    "mel": np.float32(-100.0),
    "stem_mel": np.float32(-100.0),
    "chroma": np.float32(0.0),
    "curves": np.array([0, 0, 0, 0, 0, -100, -100, -100, -100, -100, 0, 0], dtype=np.float32),
}


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


class FeatureStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / FEATURE_VERSION

    @property
    def config_path(self) -> Path:
        return self.dir / "config.json"

    @property
    def stats_path(self) -> Path:
        return self.dir / "stats.json"

    def open_for_write(self, config: dict) -> None:
        """Pin the feature config of this version dir; a different config must bump FEATURE_VERSION."""
        self.dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(config, sort_keys=True, indent=1)
        if self.config_path.exists():
            if json.loads(self.config_path.read_text()) != json.loads(text):
                raise ValueError(f"feature config differs from {self.config_path}; bump FEATURE_VERSION")
            return
        self.config_path.write_text(text)

    def song_dir(self, sha: str) -> Path:
        return self.dir / sha[:2] / sha

    def has(self, sha: str) -> bool:
        d = self.song_dir(sha)
        return all((d / f"{n}.npy").exists() for n in ARRAYS + EVENTS)

    def write(self, sha: str, arrays: dict[str, np.ndarray], beats, downbeats) -> None:
        lengths = {n: arrays[n].shape[0] for n in ARRAYS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"arrays disagree on frame count: {lengths}")
        d = self.song_dir(sha)
        d.mkdir(parents=True, exist_ok=True)
        for n in ARRAYS:
            _atomic_save(d / f"{n}.npy", np.asarray(arrays[n], dtype=np.float16))
        # events last: has() is true only once everything is on disk
        _atomic_save(d / "beats.npy", np.asarray(beats, dtype=np.float64))
        _atomic_save(d / "downbeats.npy", np.asarray(downbeats, dtype=np.float64))

    def load(self, sha: str, name: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.song_dir(sha) / f"{name}.npy", mmap_mode="r" if mmap else None)

    def events(self, sha: str) -> tuple[np.ndarray, np.ndarray]:
        return self.load(sha, "beats", mmap=False), self.load(sha, "downbeats", mmap=False)

    def window(self, sha: str, start_s: float, end_s: float) -> dict[str, np.ndarray]:
        """Frames [round(start_s*FPS), round(end_s*FPS)) of every array; outside the song = silence."""
        i0, i1 = int(round(start_s * FPS)), int(round(end_s * FPS))
        out = {}
        for n in ARRAYS:
            a = self.load(sha, n)
            w = np.empty((i1 - i0,) + a.shape[1:], dtype=np.float16)
            w[...] = SILENCE[n]
            lo, hi = max(i0, 0), min(i1, a.shape[0])
            if hi > lo:
                w[lo - i0:hi - i0] = a[lo:hi]
            out[n] = w
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_store.py -q` → `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/store.py tests/test_audio_store.py
git commit -m "Phase 2: versioned fp16 feature store with windows"
```

---

### Task 4: Beat statistics and map-grid agreement

**Files:**
- Create: `src/kata/audio/beats.py`
- Test: `tests/test_audio_beats.py`

**Interfaces:**
- Produces: `TOLERANCE_S = 0.07`, `LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)`, `LEVEL_COLUMNS`; `f_measure(ref, est, tol) -> float`; `grid_variants(grid, level) -> list[np.ndarray]`; `grid_agreement(grid, beats) -> {"f_l025","f_l05","f_l1","f_l2","f_l4","best_level","best_f"}`; `beat_in_bar(beats, downbeats) -> (int16 positions, bool is_downbeat)`; `tempo_stats(beats, downbeats) -> {"est_bpm","est_meter","tempo_cv"}`.

- [ ] **Step 1: Write the failing test** — `tests/test_audio_beats.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_beats.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.audio.beats'`.

- [ ] **Step 3: Implement** — `src/kata/audio/beats.py`:

```python
"""Beat statistics and map-grid agreement (§5). Pure numpy; no model code here."""

from __future__ import annotations

import numpy as np

TOLERANCE_S = 0.07
LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)
LEVEL_COLUMNS = {0.25: "f_l025", 0.5: "f_l05", 1.0: "f_l1", 2.0: "f_l2", 4.0: "f_l4"}


def f_measure(ref, est, tol: float = TOLERANCE_S) -> float:
    """Beat F-measure: one-to-one matching of est to ref within ±tol seconds."""
    ref = np.sort(np.asarray(ref, dtype=np.float64))
    est = np.sort(np.asarray(est, dtype=np.float64))
    if len(ref) == 0 and len(est) == 0:
        return 1.0
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    used = np.zeros(len(est), dtype=bool)
    hits = 0
    for r in ref:
        lo = np.searchsorted(est, r - tol, side="left")
        hi = np.searchsorted(est, r + tol, side="right")
        free = [j for j in range(lo, hi) if not used[j]]
        if free:
            used[min(free, key=lambda j: abs(est[j] - r))] = True
            hits += 1
    if hits == 0:
        return 0.0
    p, r = hits / len(est), hits / len(ref)
    return 2 * p * r / (p + r)


def grid_variants(grid, level: float) -> list[np.ndarray]:
    """The map grid at a metrical level: L > 1 subdivides, L < 1 keeps every 1/L-th beat (each phase)."""
    g = np.asarray(grid, dtype=np.float64)
    if level >= 1:
        k = int(round(level))
        parts = [g] + [g[:-1] + (g[1:] - g[:-1]) * i / k for i in range(1, k)]
        return [np.sort(np.concatenate(parts))]
    m = int(round(1 / level))
    return [g[p::m] for p in range(m)]


def grid_agreement(grid, beats, tol: float = TOLERANCE_S) -> dict:
    """F-measure of audio beats vs. the map grid at each level in LEVELS (best phase)."""
    g = np.asarray(grid, dtype=np.float64)
    b = np.asarray(beats, dtype=np.float64)
    if len(g):
        b = b[(b >= g[0] - tol) & (b <= g[-1] + tol)]
    out = {LEVEL_COLUMNS[L]: max(f_measure(v, b, tol) for v in grid_variants(g, L)) for L in LEVELS}
    best = max(LEVELS, key=lambda L: (out[LEVEL_COLUMNS[L]], L == 1.0))
    out["best_level"] = best
    out["best_f"] = out[LEVEL_COLUMNS[best]]
    return out


def beat_in_bar(beats, downbeats, tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """-> (beat_in_bar, is_downbeat). 1 = downbeat; 0 = before the first downbeat."""
    beats = np.asarray(beats, dtype=np.float64)
    db = np.sort(np.asarray(downbeats, dtype=np.float64))
    if len(db):
        j = np.clip(np.searchsorted(db, beats), 1, len(db)) - 1
        k = np.clip(j + 1, 0, len(db) - 1)
        is_db = (np.abs(db[j] - beats) <= tol) | (np.abs(db[k] - beats) <= tol)
    else:
        is_db = np.zeros(len(beats), dtype=bool)
    pos = np.zeros(len(beats), dtype=np.int16)
    count = 0
    for i, d in enumerate(is_db):
        count = 1 if d else (count + 1 if count else 0)
        pos[i] = count
    return pos, is_db


def tempo_stats(beats, downbeats) -> dict:
    beats = np.asarray(beats, dtype=np.float64)
    ibi = np.diff(beats)
    if len(ibi) == 0:
        return {"est_bpm": float("nan"), "est_meter": 0, "tempo_cv": float("nan")}
    pos, is_db = beat_in_bar(beats, downbeats)
    bar_lengths = [int(pos[i - 1]) for i in np.nonzero(is_db)[0] if i > 0 and pos[i - 1] > 0]
    meter = int(np.bincount(bar_lengths).argmax()) if bar_lengths else 0
    med = float(np.median(ibi))
    # robust spread (MAD / median): a few missed or extra beats must not dominate
    return {"est_bpm": 60.0 / med, "est_meter": meter,
            "tempo_cv": float(np.median(np.abs(ibi - med)) / med)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_beats.py -q` → `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/beats.py tests/test_audio_beats.py
git commit -m "Phase 2: beat F-measure, grid agreement, tempo stats"
```

---

### Task 5: Map ↔ audio sync

**Files:**
- Create: `src/kata/audio/sync.py`
- Test: `tests/test_audio_sync.py`

**Interfaces:**
- Consumes: `frames.FPS`, `frames.SR`, `frames.frame_index`.
- Produces: `note_train(note_times, n) -> np.ndarray`; `estimate_lag(onset, note_times, bias_s=0.0) -> {"lag_ms","peak_corr","peak_ratio"}` (lag > 0 = notes after onsets); `click_track(times_s, dur_s, noise_db=-40, seed=0) -> float32 mono`; `measure_onset_bias(spectral) -> float` seconds (needs a `SpectralExtractor`, Task 6).

- [ ] **Step 1: Write the failing test** — `tests/test_audio_sync.py`:

```python
import numpy as np
import pytest

pytest.importorskip("scipy")
from kata.audio.frames import FPS  # noqa: E402
from kata.audio.sync import estimate_lag, measure_onset_bias, note_train  # noqa: E402

NOTES = np.sort(np.random.default_rng(1).choice(np.arange(2.0, 58.0, 0.25), 150, replace=False))


def onsets_for(notes, early_s):
    """Onset curve with a peak `early_s` before each note (on the frame grid), plus noise."""
    o = np.zeros(60 * FPS + 1)
    for t in notes - early_s:
        o[int(round(t * FPS))] += 1.0
    return o + np.random.default_rng(2).random(len(o)) * 0.3


@pytest.mark.parametrize("late_s", [0.0, 0.03, 0.08, -0.05])
def test_lag_of_shifted_notes(late_s):
    r = estimate_lag(onsets_for(NOTES, late_s), NOTES)
    assert r["lag_ms"] == pytest.approx(late_s * 1000, abs=2)
    assert r["peak_corr"] > 0.3


def test_detector_bias_is_added():
    r = estimate_lag(onsets_for(NOTES, 0.03), NOTES, bias_s=-0.004)
    assert r["lag_ms"] == pytest.approx(26, abs=2)


def test_unrelated_notes_have_low_correlation():
    r = estimate_lag(np.random.default_rng(3).random(60 * FPS + 1), NOTES)
    assert r["peak_corr"] < 0.15


def test_note_train_drops_duplicates_and_out_of_range():
    t = note_train([1.0, 1.0, 5.0, -1.0], 301)
    assert t.sum() == pytest.approx(1.0, abs=1e-6)
    assert int(np.argmax(t)) == 100
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_sync.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.audio.sync'`.

- [ ] **Step 3: Implement** — `src/kata/audio/sync.py`:

```python
"""Map <-> audio sync: how far a difficulty's notes sit from the audio onsets (§7.3)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from kata.audio.frames import FPS, SR, frame_index

MAX_LAG_S = 0.2
SMOOTH_S = 0.02
CONTEXT_S = 1.0


def note_train(note_times, n: int) -> np.ndarray:
    """Unit impulses at the note times (split linearly between frames), Gaussian-smoothed."""
    f = frame_index(np.unique(np.asarray(note_times, dtype=np.float64)))
    f = f[(f >= 0) & (f <= n - 1)]
    out = np.zeros(n, dtype=np.float64)
    i0 = np.floor(f).astype(np.int64)
    w = f - i0
    np.add.at(out, i0, 1 - w)
    np.add.at(out, np.minimum(i0 + 1, n - 1), w)
    return gaussian_filter1d(out, SMOOTH_S * FPS)


def _xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """c[l + max_lag] = mean_t a[t] * b[t - l] for l in [-max_lag, max_lag]."""
    out = np.empty(2 * max_lag + 1)
    for i, l in enumerate(range(-max_lag, max_lag + 1)):
        x, y = (a[l:], b[:len(b) - l]) if l >= 0 else (a[:l], b[-l:])
        out[i] = np.dot(x, y) / len(x)
    return out


def estimate_lag(onset, note_times, bias_s: float = 0.0) -> dict:
    """Lag > 0 means the notes come *after* the audio onsets.

    bias_s is the onset detector's own offset (peak time - true onset), from measure_onset_bias().
    """
    onset = np.asarray(onset, dtype=np.float64)
    times = np.asarray(note_times, dtype=np.float64)
    lo = max(0, int((times.min() - CONTEXT_S) * FPS))
    hi = min(len(onset), int((times.max() + CONTEXT_S) * FPS) + 1)
    o = onset[lo:hi]
    t = note_train(times - lo / FPS, hi - lo)
    if o.std() == 0 or t.std() == 0:
        return {"lag_ms": float("nan"), "peak_corr": float("nan"), "peak_ratio": float("nan")}
    o = (o - o.mean()) / o.std()
    t = (t - t.mean()) / t.std()
    L = int(round(MAX_LAG_S * FPS))
    c = _xcorr(t, o, L)
    k = int(np.argmax(c))
    frac = 0.0
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        frac = 0.5 * (c[k - 1] - c[k + 1]) / den if den != 0 else 0.0
    lag_s = (k - L + frac) / FPS + bias_s
    return {"lag_ms": lag_s * 1000.0, "peak_corr": float(c[k]),
            "peak_ratio": float(c[k] / max(np.median(np.abs(c)), 1e-9))}


def click_track(times_s, dur_s: float, noise_db: float = -40.0, seed: int = 0) -> np.ndarray:
    """Decaying 1 kHz clicks at times_s over white noise; mono float32 at SR."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(int(dur_s * SR)) * 10 ** (noise_db / 20)).astype(np.float32)
    k = np.arange(int(0.03 * SR))
    tone = (np.sin(2 * np.pi * 1000 * k / SR) * np.exp(-k / (0.005 * SR))).astype(np.float32)
    for t in times_s:
        s = int(round(t * SR))
        x[s:s + len(tone)] += tone[: len(x) - s]
    return x


def measure_onset_bias(spectral) -> float:
    """Median (flux peak - click time) in seconds on a synthetic click track, refined to sub-frame."""
    times = np.arange(1.0, 29.0, 0.4615) + 0.0037  # off-grid on purpose
    flux = spectral.mix(click_track(times, 30.0))["onset"].astype(np.float64)
    errs = []
    for t in times:
        c = int(round(t * FPS))
        j = c - 5 + int(np.argmax(flux[c - 5:c + 6]))
        a, b, d = flux[j - 1], flux[j], flux[j + 1]
        den = a - 2 * b + d
        errs.append((j + (0.5 * (a - d) / den if den != 0 else 0.0)) / FPS - t)
    return float(np.median(errs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_sync.py -q` → `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/sync.py tests/test_audio_sync.py
git commit -m "Phase 2: map-audio lag estimation and click-track helpers"
```

---

### Task 6: Spectral features (GPU-capable, tested on CPU)

**Files:**
- Create: `src/kata/audio/spectral.py`
- Test: `tests/test_audio_spectral.py`; append to `tests/test_audio_sync.py`

**Interfaces:**
- Consumes: `frames.HOP`, `frames.SR`; tests use `sync.click_track`.
- Produces: constants `N_FFT, CHROMA_N_FFT, MEL_FMIN, MEL_FMAX, CHROMA_FMIN, CHROMA_FMAX, DB_FLOOR`; `hz_to_mel`, `mel_to_hz`, `mel_centers(n_mels)`, `mel_filterbank(n_mels) -> (n_mels, 1025)`, `chroma_filterbank() -> (12, 2049)`, `spectral_flux(log_mel: Tensor (T,B)) -> Tensor (T,)`; `SpectralExtractor(device)` with `.mix(mono) -> {"mel": (T,128), "chroma": (T,12), "onset": (T,), "rms": (T,)}` and `.stems(stems (4,2,N)) -> {"stem_mel": (T,4,64), "onset": (4,T), "rms": (4,T)}`, all float32 numpy.

- [ ] **Step 1: Write the failing tests** — `tests/test_audio_spectral.py`:

```python
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
```

and append to `tests/test_audio_sync.py`:

```python
def test_measure_onset_bias_is_small_and_stable():
    pytest.importorskip("torch")
    from kata.audio.spectral import SpectralExtractor
    sp = SpectralExtractor("cpu")
    b = measure_onset_bias(sp)
    assert -0.03 < b < 0.0
    assert b == measure_onset_bias(sp)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra audio pytest tests/test_audio_spectral.py tests/test_audio_sync.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.audio.spectral'`.

- [ ] **Step 3: Implement** — `src/kata/audio/spectral.py`:

```python
"""Spectral features on the frame grid, in PyTorch so they run on the GPU (§4.1, §4.2)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from kata.audio.frames import HOP, SR

N_FFT = 2048
CHROMA_N_FFT = 4096
MEL_FMIN = 30.0
MEL_FMAX = 16000.0
CHROMA_FMIN = 55.0
CHROMA_FMAX = 5000.0
DB_FLOOR = -100.0

_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(f) -> np.ndarray:
    """Slaney mel scale (linear below 1 kHz, log above), as librosa's default."""
    f = np.asarray(f, dtype=np.float64)
    return np.where(f >= _MIN_LOG_HZ,
                    _MIN_LOG_MEL + np.log(np.maximum(f, 1e-10) / _MIN_LOG_HZ) / _LOGSTEP,
                    f / _F_SP)


def mel_to_hz(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.where(m >= _MIN_LOG_MEL, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), _F_SP * m)


def mel_centers(n_mels: int, fmin: float = MEL_FMIN, fmax: float = MEL_FMAX) -> np.ndarray:
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))[1:-1]


def mel_filterbank(n_mels: int, n_fft: int = N_FFT, sr: int = SR,
                   fmin: float = MEL_FMIN, fmax: float = MEL_FMAX) -> np.ndarray:
    """(n_mels, n_fft//2 + 1) triangular filters with Slaney area normalisation."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(pts)
    ramps = pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0.0, np.minimum(lower, upper))
    w *= (2.0 / (pts[2:] - pts[:-2]))[:, None]
    return w.astype(np.float32)


def chroma_filterbank(n_fft: int = CHROMA_N_FFT, sr: int = SR,
                      fmin: float = CHROMA_FMIN, fmax: float = CHROMA_FMAX) -> np.ndarray:
    """(12, n_fft//2 + 1): each FFT bin in [fmin, fmax] goes to its nearest pitch class (0 = C)."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((12, len(freqs)), np.float32)
    idx = np.nonzero((freqs >= fmin) & (freqs <= fmax))[0]
    midi = 69 + 12 * np.log2(freqs[idx] / 440.0)
    fb[np.round(midi).astype(np.int64) % 12, idx] = 1.0
    return fb


def spectral_flux(log_mel: torch.Tensor) -> torch.Tensor:
    """(T, B) dB -> (T,) SuperFlux-style onset strength: positive rise vs. the 3-bin max of the previous frame."""
    ref = F.max_pool1d(log_mel[None], kernel_size=3, stride=1, padding=1)[0]  # max over bins
    flux = (log_mel[1:] - ref[:-1]).clamp_min(0).sum(dim=1)
    return torch.cat([flux.new_zeros(1), flux])


class SpectralExtractor:
    """Builds windows and filterbanks once; every method returns float32 numpy, time on axis 0."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.window = torch.hann_window(N_FFT, device=self.device)
        self.chroma_window = torch.hann_window(CHROMA_N_FFT, device=self.device)
        self.mel_fb = {n: torch.from_numpy(mel_filterbank(n)).to(self.device) for n in (64, 128)}
        self.chroma_fb = torch.from_numpy(chroma_filterbank()).to(self.device)

    def _tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32).to(self.device)

    def _power(self, x: torch.Tensor, n_fft: int, window: torch.Tensor) -> torch.Tensor:
        """(..., N) -> (..., n_fft//2 + 1, T) power spectrogram, centred frames."""
        s = torch.stft(x, n_fft, hop_length=HOP, window=window, center=True,
                       pad_mode="reflect", return_complex=True)
        return s.real.square() + s.imag.square()

    def _log_mel(self, x: torch.Tensor, n_mels: int) -> torch.Tensor:
        m = self.mel_fb[n_mels] @ self._power(x, N_FFT, self.window)
        return (10.0 * torch.log10(m.clamp_min(1e-10))).clamp_min(DB_FLOOR).transpose(-1, -2)

    def _rms_db(self, x: torch.Tensor) -> torch.Tensor:
        """(..., N) -> (..., T) frame RMS in dBFS over N_FFT-sample centred windows."""
        pad = N_FFT // 2
        xp = F.pad(x.reshape(-1, 1, x.shape[-1]), (pad, pad), mode="reflect")
        frames = xp[:, 0].unfold(-1, N_FFT, HOP)
        rms = frames.square().mean(dim=-1).sqrt().reshape(*x.shape[:-1], -1)
        return (20.0 * torch.log10(rms.clamp_min(1e-5))).clamp_min(DB_FLOOR)

    def _chroma(self, x: torch.Tensor) -> torch.Tensor:
        c = self.chroma_fb @ self._power(x, CHROMA_N_FFT, self.chroma_window)
        return (c / c.amax(dim=0, keepdim=True).clamp_min(1e-10)).T

    @torch.inference_mode()
    def mix(self, mono) -> dict[str, np.ndarray]:
        """mono (N,) -> mel (T,128), chroma (T,12), onset (T,), rms (T,)."""
        x = self._tensor(mono)
        mel = self._log_mel(x, 128)
        out = {"mel": mel, "chroma": self._chroma(x), "onset": spectral_flux(mel), "rms": self._rms_db(x)}
        return {k: v.float().cpu().numpy() for k, v in out.items()}

    @torch.inference_mode()
    def stems(self, stems) -> dict[str, np.ndarray]:
        """stems (4, 2, N) -> stem_mel (T,4,64), onset (4,T), rms (4,T)."""
        x = self._tensor(stems).mean(dim=1)  # (4, N) mono per stem
        mel = self._log_mel(x, 64)  # (4, T, 64)
        onset = torch.stack([spectral_flux(m) for m in mel])
        return {"stem_mel": mel.permute(1, 0, 2).float().cpu().numpy(),
                "onset": onset.float().cpu().numpy(),
                "rms": self._rms_db(x).float().cpu().numpy()}
```

Note: the 3-bin max filter in `spectral_flux` must pool over **bins** (`log_mel[None]` is `(1, T, B)`, pooled along `B`). Pooling over time makes the flux identically zero; `test_onset_strength_peaks_at_clicks` catches that.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_spectral.py tests/test_audio_sync.py -q` → `15 passed`.
`measure_onset_bias` should be about −0.011 s (the flux peak sits one frame before the click).

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/spectral.py tests/test_audio_spectral.py tests/test_audio_sync.py
git commit -m "Phase 2: torch spectral features (log-mel, flux, RMS, chroma)"
```

---

### Task 7: Structure segmentation

**Files:**
- Create: `src/kata/audio/structure.py`
- Test: `tests/test_audio_structure.py`

**Interfaces:**
- Consumes: `frames.FPS`.
- Produces: `MIN_SEG_BEATS = 8`, `K_MAX = 8`; `beat_sync(feat, beats) -> (n_beats-1, D)`; `segment(chroma, mel, beats, downbeats, duration_s) -> list[{"start_s","end_s","label"}]` (labels renumbered by first appearance, first segment starts at 0, last ends at `duration_s`); `boundary_density_check(boundaries, note_times, duration_s) -> {"hits","n","coverage"}`.

- [ ] **Step 1: Write the failing test** — `tests/test_audio_structure.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_structure.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.audio.structure'`.

- [ ] **Step 3: Implement** — `src/kata/audio/structure.py`:

```python
"""Unsupervised structure: segment boundaries + repetition labels (§6).

Follows McFee & Ellis (2014): a beat-synchronous recurrence graph (chroma) plus a local
path graph (MFCC), spectral clustering of the combined graph, k chosen by the eigengap.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.fft import dct
from scipy.linalg import eigh
from scipy.ndimage import median_filter

from kata.audio.frames import FPS

MIN_SEG_BEATS = 8
K_MAX = 8
N_MFCC = 20


def beat_sync(feat: np.ndarray, beats) -> np.ndarray:
    """(T, D) frame features -> (len(beats) - 1, D) means over each inter-beat interval."""
    b = np.clip(np.round(np.asarray(beats) * FPS).astype(np.int64), 0, len(feat) - 1)
    out = np.empty((len(b) - 1, feat.shape[1]), dtype=np.float64)
    for i in range(len(b) - 1):
        out[i] = feat[b[i]:max(b[i + 1], b[i] + 1)].mean(axis=0)
    return out


def _recurrence(x: np.ndarray) -> np.ndarray:
    """Mutual k-NN recurrence on time-delay-stacked features, path-enhanced along diagonals."""
    n = len(x)
    # centred delay embedding (previous, current, next beat; edges repeated, no wrap-around):
    # a past-only embedding reacts late and pushes boundaries about a bar too far
    prev = np.concatenate([x[:1], x[:-1]])
    nxt = np.concatenate([x[1:], x[-1:]])
    stacked = np.concatenate([prev, x, nxt], axis=1)
    stacked /= np.linalg.norm(stacked, axis=1, keepdims=True).clip(1e-9)
    d = 1.0 - stacked @ stacked.T
    for off in (-2, -1, 0, 1, 2):  # ignore trivial self-similarity near the diagonal
        np.fill_diagonal(d[max(0, off):, max(0, -off):], np.inf)
    k = min(n - 3, int(np.ceil(2 * np.sqrt(n))))
    nn = np.argsort(d, axis=1)[:, :k]
    knn = np.zeros((n, n), dtype=bool)
    knn[np.repeat(np.arange(n), k), nn.ravel()] = True
    rec = (knn & knn.T).astype(np.float64)
    return median_filter(rec, footprint=np.eye(7, dtype=bool), mode="constant")


def _path(m: np.ndarray) -> np.ndarray:
    d2 = np.sum(np.diff(m, axis=0) ** 2, axis=1)
    sigma = np.median(d2) if np.median(d2) > 0 else 1.0
    w = np.exp(-d2 / sigma)
    return np.diag(w, 1) + np.diag(w, -1)


def _embedding(chroma_b: np.ndarray, mfcc_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rec, path = _recurrence(chroma_b), _path(mfcc_b)
    deg_r, deg_p = rec.sum(axis=1), path.sum(axis=1)
    mu = deg_p.dot(deg_p + deg_r) / max(np.sum((deg_p + deg_r) ** 2), 1e-9)
    a = mu * rec + (1 - mu) * path
    deg = a.sum(axis=1).clip(1e-9)
    lap = np.eye(len(a)) - a / np.sqrt(np.outer(deg, deg))
    evals, evecs = eigh(lap)
    return evals, median_filter(evecs, size=(9, 1))


def _cluster(evecs: np.ndarray, k: int) -> np.ndarray:
    x = evecs[:, :k] / np.linalg.norm(evecs[:, :k], axis=1, keepdims=True).clip(1e-9)
    _, lab = kmeans2(x, k, seed=np.random.default_rng(0), minit="++")
    return lab


def _runs(lab: np.ndarray) -> list[list[int]]:
    """[[start, end, label], ...] over beat-interval indices, end exclusive."""
    runs, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            runs.append([s, i, int(lab[s])])
            s = i
    return runs


def _merge_short(runs: list[list[int]], min_len: int) -> list[list[int]]:
    while len(runs) > 1:
        lens = [e - s for s, e, _ in runs]
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            break
        j = i - 1 if i > 0 else 1  # absorb into the previous run (or the next, for the first)
        lo, hi = min(i, j), max(i, j)
        keep = runs[j][2]
        runs[lo:hi + 1] = [[runs[lo][0], runs[hi][1], keep]]
        # neighbours that now share a label become one run
        merged = [runs[0]]
        for r in runs[1:]:
            if r[2] == merged[-1][2]:
                merged[-1] = [merged[-1][0], r[1], r[2]]
            else:
                merged.append(r)
        runs = merged
    return runs


def segment(chroma: np.ndarray, mel: np.ndarray, beats, downbeats, duration_s: float,
            min_beats: int = MIN_SEG_BEATS, k_max: int = K_MAX) -> list[dict]:
    """-> [{"start_s", "end_s", "label"}]; labels are renumbered by first appearance."""
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 4 * min_beats:
        return [{"start_s": 0.0, "end_s": float(duration_s), "label": 0}]
    mfcc = dct(np.asarray(mel, dtype=np.float64), type=2, norm="ortho", axis=1)[:, :N_MFCC]
    evals, evecs = _embedding(beat_sync(np.asarray(chroma, dtype=np.float64), beats), beat_sync(mfcc, beats))
    # k by eigengap (largest λ_k - λ_{k-1}), among the k whose merged segmentation still has ≥ 2 labels
    k_max = min(k_max, len(evals) - 1)
    best = None
    for k in sorted(range(2, k_max + 1), key=lambda k: evals[k - 1] - evals[k]):
        runs = _merge_short(_runs(_cluster(evecs, k)), min_beats)
        if len({r[2] for r in runs}) >= 2:
            best = runs
            break
    runs = best or [[0, len(beats) - 1, 0]]
    db = np.asarray(downbeats, dtype=np.float64)
    ibi = float(np.median(np.diff(beats)))
    bounds = []
    for s, _, _ in runs[1:]:
        t = beats[s]
        if len(db):
            near = db[np.argmin(np.abs(db - t))]
            t = near if abs(near - t) <= 2 * ibi else t
        bounds.append(float(t))
    edges = [0.0] + bounds + [float(duration_s)]
    order: dict[int, int] = {}
    return [{"start_s": edges[i], "end_s": edges[i + 1], "label": order.setdefault(r[2], len(order))}
            for i, r in enumerate(runs)]


def boundary_density_check(boundaries, note_times, duration_s: float, win_s: float = 4.0,
                           tol_s: float = 2.0, rel: float = 0.5, step: float = 0.1) -> dict:
    """How many boundaries lie within tol_s of a large note-density change (a report diagnostic).

    Density change at t = |notes in (t, t + win/2] - notes in (t - win/2, t]|; "large" means at
    least `rel` × the mean count per half-window. `coverage` is the fraction of the song within
    tol_s of a large change, i.e. the hit rate of randomly placed boundaries.
    """
    t = np.arange(0.0, duration_s, step)
    counts = np.histogram(np.asarray(note_times, dtype=np.float64), bins=np.append(t, duration_s))[0]
    cs = np.concatenate([[0], np.cumsum(counts)])
    w = int(round(win_s / 2 / step))
    i = np.arange(len(t))
    before = cs[i] - cs[np.maximum(i - w, 0)]
    after = cs[np.minimum(i + w, len(t))] - cs[i]
    large = np.abs(after - before) >= rel * max(counts.mean() * w, 1e-9)
    k = int(round(tol_s / step))
    near = np.convolve(large.astype(np.int64), np.ones(2 * k + 1, dtype=np.int64), mode="same") > 0
    idx = np.clip((np.asarray(boundaries, dtype=np.float64) / step).astype(np.int64), 0, len(t) - 1)
    return {"hits": int(near[idx].sum()), "n": int(len(idx)), "coverage": float(near.mean())}
```

Two details matter (spec §6, §17): the delay embedding is **centred** (previous, current, next beat; a past-only embedding puts boundaries a bar late and fails the A-B-A-B test on some seeds), and *k* is tried in eigengap order until a merged segmentation keeps ≥ 2 labels.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_structure.py -q` → `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/structure.py tests/test_audio_structure.py
git commit -m "Phase 2: unsupervised structure segmentation"
```

---

### Task 8: GPU model wrappers

**Files:**
- Create: `src/kata/audio/models.py`
- Test: `tests/test_audio_models.py`

**Interfaces:**
- Consumes: `frames.SR`, `store.STEMS`; test uses `sync.click_track`, `frames.n_frames`.
- Produces: `StemSeparator(device="cuda")` with `.name = "htdemucs"` and `__call__(stereo (2,N) float32) -> Tensor (4,2,N)` in `STEMS` order on the device; `BeatTracker(device="cuda")` with `.name = "beat_this:final0"` and `__call__(mono, n_frames) -> {"beat_act": (T,), "downbeat_act": (T,), "beats": s, "downbeats": s}`.

- [ ] **Step 1: Write the failing test** — `tests/test_audio_models.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_models.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.audio.models'` (or 2 skipped on a machine without CUDA; on this machine it must run).

- [ ] **Step 3: Implement** — `src/kata/audio/models.py`:

```python
"""GPU model wrappers: Demucs stems (§4.2) and Beat This! beats (§4.3, §5.1)."""

from __future__ import annotations

import numpy as np
import torch

from kata.audio.frames import SR
from kata.audio.store import STEMS

DEMUCS_MODEL = "htdemucs"
BEAT_CHECKPOINT = "final0"
BEAT_FPS = 50


class StemSeparator:
    name = DEMUCS_MODEL

    def __init__(self, device: str = "cuda"):
        from demucs.api import Separator
        self.device = device
        self.sep = Separator(model=DEMUCS_MODEL, device=device, shifts=0, overlap=0.25,
                             split=True, progress=False)
        if tuple(self.sep.model.sources) != STEMS:
            raise RuntimeError(f"unexpected stem order {self.sep.model.sources}")

    def __call__(self, stereo: np.ndarray) -> torch.Tensor:
        """(2, N) float32 at SR -> (4, 2, N) on the model device, in STEMS order."""
        wav = torch.from_numpy(stereo).to(self.device)
        _, stems = self.sep.separate_tensor(wav, SR)
        return torch.stack([stems[s] for s in STEMS])


class BeatTracker:
    name = f"beat_this:{BEAT_CHECKPOINT}"

    def __init__(self, device: str = "cuda"):
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor
        self.a2f = Audio2Frames(checkpoint_path=BEAT_CHECKPOINT, device=device)
        self.post = Postprocessor(type="minimal", fps=BEAT_FPS)

    def __call__(self, mono: np.ndarray, n_frames: int) -> dict[str, np.ndarray]:
        """-> beat_act, downbeat_act (n_frames,) at 100 fps; beats, downbeats in seconds."""
        beat_logits, down_logits = self.a2f(mono, SR)
        beats, downbeats = self.post(beat_logits, down_logits)
        src = np.arange(len(beat_logits), dtype=np.float64)  # 50 fps frame j is centred at j / 50 s
        dst = np.arange(n_frames, dtype=np.float64) / 2.0
        act = lambda l: np.interp(dst, src, torch.sigmoid(l).cpu().numpy(), right=0.0)  # noqa: E731
        return {"beat_act": act(beat_logits).astype(np.float32),
                "downbeat_act": act(down_logits).astype(np.float32),
                "beats": np.asarray(beats, dtype=np.float64),
                "downbeats": np.asarray(downbeats, dtype=np.float64)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_models.py -q` → `2 passed` (not skipped).

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/models.py tests/test_audio_models.py
git commit -m "Phase 2: Demucs and Beat This wrappers"
```

---

### Task 9: Per-song extraction

**Files:**
- Create: `src/kata/audio/extract.py`
- Test: `tests/test_audio_extract.py`

**Interfaces:**
- Consumes: `decode.CanonicalAudio`, `decode.level_stats`, `spectral.SpectralExtractor` and constants, `frames.*`, `store.CURVE_COLUMNS`, `store.STEMS`, `models.StemSeparator`, `models.BeatTracker` (only when no fakes are injected).
- Produces: `SongFeatures(arrays, beats, downbeats, scalars)`; `feature_config(separator_name, beat_name) -> dict`; `Extractor(device="cuda", separator=None, beat_tracker=None)` with `.config` and `__call__(audio: CanonicalAudio) -> SongFeatures`. Scalars: `source_sr, source_channels, duration_s, n_frames, bandlimited, peak_dbfs, rms_dbfs, clip_ratio`.

- [ ] **Step 1: Write the failing test** — `tests/test_audio_extract.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_extract.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.audio.extract'`.

- [ ] **Step 3: Implement** — `src/kata/audio/extract.py`:

```python
"""One song in, every Phase 2 per-song output out (§4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kata.audio import decode
from kata.audio import spectral
from kata.audio.frames import FPS, HOP, SR, n_frames
from kata.audio.store import CURVE_COLUMNS, STEMS


@dataclass
class SongFeatures:
    arrays: dict[str, np.ndarray]
    beats: np.ndarray
    downbeats: np.ndarray
    scalars: dict = field(default_factory=dict)


def feature_config(separator_name: str, beat_name: str) -> dict:
    return {"sr": SR, "hop": HOP, "fps": FPS, "n_fft": spectral.N_FFT,
            "chroma_n_fft": spectral.CHROMA_N_FFT, "mel_fmin": spectral.MEL_FMIN,
            "mel_fmax": spectral.MEL_FMAX, "chroma_fmin": spectral.CHROMA_FMIN,
            "chroma_fmax": spectral.CHROMA_FMAX, "db_floor": spectral.DB_FLOOR,
            "n_mels_mix": 128, "n_mels_stem": 64, "resampler": "soxr-VHQ",
            "stems": list(STEMS), "curve_columns": list(CURVE_COLUMNS),
            "separator": separator_name, "separator_args": {"shifts": 0, "overlap": 0.25, "split": True},
            "beat_tracker": beat_name}


class Extractor:
    def __init__(self, device: str = "cuda", separator=None, beat_tracker=None):
        from kata.audio.models import BeatTracker, StemSeparator
        self.spectral = spectral.SpectralExtractor(device)
        self.separator = separator if separator is not None else StemSeparator(device)
        self.beat_tracker = beat_tracker if beat_tracker is not None else BeatTracker(device)

    @property
    def config(self) -> dict:
        return feature_config(self.separator.name, self.beat_tracker.name)

    def __call__(self, audio: decode.CanonicalAudio) -> SongFeatures:
        T = n_frames(audio.n_samples)
        mono = audio.mono
        mix = self.spectral.mix(mono)
        st = self.spectral.stems(self.separator(audio.stereo))
        bt = self.beat_tracker(mono, T)
        curves = np.stack([mix["onset"], *st["onset"], mix["rms"], *st["rms"],
                           bt["beat_act"], bt["downbeat_act"]], axis=1)
        arrays = {"mel": mix["mel"], "stem_mel": st["stem_mel"], "chroma": mix["chroma"], "curves": curves}
        bad = {k: v.shape for k, v in arrays.items() if v.shape[0] != T}
        if bad:
            raise RuntimeError(f"frame count mismatch, expected {T}: {bad}")
        scalars = {"source_sr": audio.source_sr, "source_channels": audio.source_channels,
                   "duration_s": audio.n_samples / SR, "n_frames": T,
                   "bandlimited": audio.source_sr < SR, **decode.level_stats(audio)}
        return SongFeatures(arrays, bt["beats"], bt["downbeats"], scalars)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_extract.py -q` → `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/extract.py tests/test_audio_extract.py
git commit -m "Phase 2: per-song feature extraction"
```

---

### Task 10: `features` stage

**Files:**
- Create: `src/kata/audio/pipeline.py`
- Test: `tests/test_audio_pipeline.py`

**Interfaces:**
- Consumes: `kata.data.pipeline._with_state`, `_check_disk`; `kata.data.state.State`; `store.FeatureStore`; `beats.beat_in_bar`, `beats.tempo_stats`; `decode.load`; `extract.Extractor`.
- Produces: `FEATURES_STAGE = "audio-features"`; `songs(paths, pilot=False) -> DataFrame[audio_sha256, hash, audio_file]` (one row per unique audio, smallest map hash); `ok_songs(paths) -> DataFrame[audio_sha256, duration_s]`; `write_feature_tables(paths, state, store) -> dict`; `features(paths, pilot=False, limit=0, device="cuda", extractor=None) -> summary dict` writing `audio_features.parquet`, `audio_beats.parquet`, `audio_tempo.parquet`.

The module header below already imports what Tasks 11–12 add, so later tasks only append code.

- [ ] **Step 1: Write the failing test** — `tests/test_audio_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_pipeline.py -q`
Expected: `ImportError: cannot import name 'pipeline' from 'kata.audio'`.

- [ ] **Step 3: Implement** — `src/kata/audio/pipeline.py`:

```python
"""Phase 2 stages (docs/phases/phase-2-audio.md §11). Same conventions as kata.data.pipeline:
every stage runs inside `_with_state`; `features` is resumable per audio_sha256, the others
rebuild their tables from the stored features each run."""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from kata.audio.beats import beat_in_bar, grid_agreement, tempo_stats
from kata.audio.frames import beat_grid_times, tempo_from_rows
from kata.audio.store import ARRAYS, CURVE_COLUMNS, FeatureStore
from kata.audio.structure import segment
from kata.audio.sync import estimate_lag
from kata.config import Paths
from kata.data.pipeline import _check_disk, _with_state

FEATURES_STAGE = "audio-features"
STRUCTURE_STAGE = "audio-structure"
STATS_MAX_SONGS = 500
DIFF_KEY = ["hash", "characteristic", "difficulty"]


def songs(paths: Paths, pilot: bool = False) -> pl.DataFrame:
    """One row per unique audio file: audio_sha256, hash (smallest map hash using it), audio_file."""
    maps = pl.read_parquet(paths.processed / "maps.parquet").filter(pl.col("audio_status") == "ok")
    if pilot:
        maps = maps.filter(pl.col("pilot"))
    return (maps.sort("hash").group_by("audio_sha256", maintain_order=True).first()
            .select("audio_sha256", "hash", "audio_file").sort("audio_sha256"))


def ok_songs(paths: Paths) -> pl.DataFrame:
    f = paths.processed / "audio_features.parquet"
    if not f.exists():
        return pl.DataFrame({"audio_sha256": [], "duration_s": []},
                            schema={"audio_sha256": pl.String, "duration_s": pl.Float64})
    return pl.read_parquet(f).filter(pl.col("status") == "ok").select("audio_sha256", "duration_s")


# --------------------------------------------------------------------------- features

def write_feature_tables(paths: Paths, state, store: FeatureStore) -> dict:
    """audio_features / audio_beats / audio_tempo, rebuilt from the state DB and the stored events."""
    rows = state.con.execute("SELECT key, status, error, info FROM items WHERE stage = ? ORDER BY key",
                             [FEATURES_STAGE]).fetchall()
    feats, beat_parts, tempo = [], [], []
    for sha, status, error, info in rows:
        feats.append({"audio_sha256": sha, "status": status, "error": error, **json.loads(info or "{}")})
        if status != "ok":
            continue
        b, d = store.events(sha)
        pos, is_db = beat_in_bar(b, d)
        beat_parts.append(pl.DataFrame({"audio_sha256": [sha] * len(b), "idx": np.arange(len(b)),
                                        "time_s": b, "beat_in_bar": pos, "is_downbeat": is_db}))
        tempo.append({"audio_sha256": sha, "n_beats": len(b), "n_downbeats": len(d), **tempo_stats(b, d)})
    P = paths.processed
    pl.DataFrame(feats, infer_schema_length=None).write_parquet(P / "audio_features.parquet")
    (pl.concat(beat_parts) if beat_parts else pl.DataFrame()).write_parquet(P / "audio_beats.parquet")
    pl.DataFrame(tempo, infer_schema_length=None).write_parquet(P / "audio_tempo.parquet")
    return {"songs_ok": len(tempo), "songs_error": len(feats) - len(tempo)}


def _features(paths, state, summary, pilot=False, limit=0, device="cuda", extractor=None):
    from kata.audio import decode
    from kata.audio.extract import Extractor

    all_songs = songs(paths, pilot)
    store = FeatureStore(paths.features)
    ex = extractor if extractor is not None else Extractor(device)
    store.open_for_write(ex.config)
    done = state.done(FEATURES_STAGE)
    todo = [r for r in all_songs.iter_rows(named=True)
            if not (r["audio_sha256"] in done and store.has(r["audio_sha256"]))]
    if limit:
        todo = todo[:limit]
    counts = {"ok": 0, "error": 0}
    # Decode + resample the next songs on CPU threads while the GPU works on the current one.
    with ThreadPoolExecutor(max_workers=2) as pool:
        queue: deque = deque()
        it = iter(todo)

        def submit():
            r = next(it, None)
            if r is not None:
                queue.append((r, pool.submit(decode.load, paths.maps / r["hash"] / r["audio_file"])))

        for _ in range(3):
            submit()
        n = 0
        while queue:
            r, fut = queue.popleft()
            submit()
            sha = r["audio_sha256"]
            try:
                f = ex(fut.result())
                store.write(sha, f.arrays, f.beats, f.downbeats)
                state.mark(FEATURES_STAGE, sha, "ok", info=f.scalars)
                counts["ok"] += 1
            except Exception as e:  # recorded per song; the stage keeps going
                state.mark(FEATURES_STAGE, sha, "error", error=f"{type(e).__name__}: {e}"[:500])
                counts["error"] += 1
            n += 1
            if n % 50 == 0 or n == len(todo):
                _check_disk(paths)
                print(f"  [{FEATURES_STAGE}] {n}/{len(todo)}", flush=True)
    summary.update(songs=all_songs.height, processed=len(todo), **counts,
                   status=state.counts(FEATURES_STAGE), **write_feature_tables(paths, state, store))

features = _with_state(FEATURES_STAGE, _features)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_pipeline.py -q` → `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/pipeline.py tests/test_audio_pipeline.py
git commit -m "Phase 2: resumable features stage"
```

---

### Task 11: `structure`, `sync` and `stats` stages

**Files:**
- Modify: `src/kata/audio/pipeline.py` (append)
- Test: `tests/test_audio_pipeline.py` (append)

**Interfaces:**
- Consumes: Task 10's `songs`, `ok_songs`, `FeatureStore`; `structure.segment`; `sync.estimate_lag`, `sync.measure_onset_bias`; `spectral.SpectralExtractor`; `beats.grid_agreement`; `frames.beat_grid_times`, `frames.tempo_from_rows`; Phase 1 tables `maps`, `difficulties`, `notes`, `tempo_maps`.
- Produces: `structure(paths, pilot=False)` → `audio_segments.parquet` (`audio_sha256, seg_idx, start_s, end_s, label, n_labels`); `sync(paths, pilot=False)` → `map_audio_sync.parquet` (`hash, characteristic, difficulty, tier, audio_sha256, src ∈ {mix, drums}, n_notes, lag_ms, peak_corr, peak_ratio`) and `beat_agreement.parquet` (`…, map_bpm, n_grid, f_l025 … f_l4, best_level, best_f`), summary `onset_bias_ms`; `stats(paths, pilot=False)` → `features/v1/stats.json` (`split, n_songs, songs, arrays{name: shape, mean, std, n_frames}`), ≤ 500 train songs.

- [ ] **Step 1: Write the failing test** — append to `tests/test_audio_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_pipeline.py -q`
Expected: `AttributeError: module 'kata.audio.pipeline' has no attribute 'structure'`.

- [ ] **Step 3: Implement** — append to `src/kata/audio/pipeline.py`:

```python
# --------------------------------------------------------------------------- structure

def _structure_one(features_root: str, sha: str, duration_s: float) -> list[dict]:
    store = FeatureStore(Path(features_root))
    b, d = store.events(sha)
    segs = segment(np.asarray(store.load(sha, "chroma"), np.float32),
                   np.asarray(store.load(sha, "mel"), np.float32), b, d, duration_s)
    n_labels = len({s["label"] for s in segs})
    return [{"audio_sha256": sha, "seg_idx": i, **s, "n_labels": n_labels} for i, s in enumerate(segs)]


def _structure(paths, state, summary, pilot=False):
    todo = ok_songs(paths)
    if pilot:
        todo = todo.join(songs(paths, pilot=True).select("audio_sha256"), on="audio_sha256", how="semi")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = {r["audio_sha256"]: pool.submit(_structure_one, str(paths.features), r["audio_sha256"],
                                                  r["duration_s"]) for r in todo.iter_rows(named=True)}
        for sha, fut in futures.items():
            try:
                rows += fut.result()
                state.mark(STRUCTURE_STAGE, sha, "ok")
            except Exception as e:
                state.mark(STRUCTURE_STAGE, sha, "error", error=f"{type(e).__name__}: {e}"[:500])
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(paths.processed / "audio_segments.parquet")
    summary.update(songs=todo.height, segments=len(rows), status=state.counts(STRUCTURE_STAGE))

structure = _with_state(STRUCTURE_STAGE, _structure)


# --------------------------------------------------------------------------- sync

def _sync(paths, state, summary, pilot=False):
    from kata.audio.spectral import SpectralExtractor
    from kata.audio.sync import measure_onset_bias

    P = paths.processed
    maps = pl.read_parquet(P / "maps.parquet", columns=["hash", "audio_sha256", "pilot"])
    if pilot:
        maps = maps.filter(pl.col("pilot"))
    maps = maps.join(ok_songs(paths), on="audio_sha256", how="semi").select("hash", "audio_sha256")
    diffs = (pl.read_parquet(P / "difficulties.parquet", columns=DIFF_KEY + ["tier", "parse_status"])
             .filter(pl.col("parse_status") == "ok").join(maps, on="hash"))
    notes = (pl.read_parquet(P / "notes.parquet", columns=DIFF_KEY + ["beat", "time_s"])
             .join(maps.select("hash"), on="hash", how="semi"))
    tempo = pl.read_parquet(P / "tempo_maps.parquet").join(maps.select("hash"), on="hash", how="semi")
    notes_by = notes.partition_by(DIFF_KEY, as_dict=True)
    tempo_by = tempo.partition_by(DIFF_KEY, as_dict=True)
    bias = measure_onset_bias(SpectralExtractor("cpu"))
    store = FeatureStore(paths.features)
    sync_rows, grid_rows = [], []
    for (sha,), group in diffs.group_by("audio_sha256"):
        curves = np.asarray(store.load(sha, "curves"), np.float32)
        beats, _ = store.events(sha)
        for d in group.iter_rows(named=True):
            key = tuple(d[k] for k in DIFF_KEY)
            n = notes_by.get(key)
            if n is None or n.height == 0:
                continue
            base = {**dict(zip(DIFF_KEY, key)), "tier": d["tier"], "audio_sha256": sha}
            times = n["time_s"].to_numpy()
            for src, col in (("mix", "onset_mix"), ("drums", "onset_drums")):
                lag = estimate_lag(curves[:, CURVE_COLUMNS.index(col)], times, bias)
                sync_rows.append({**base, "src": src, "n_notes": int(len(np.unique(times))), **lag})
            rows = tempo_by[key].to_dicts()
            beat = n["beat"].to_numpy()
            grid = beat_grid_times(tempo_from_rows(rows), math.ceil(beat.min()), math.floor(beat.max()))
            map_bpm = min(rows, key=lambda r: r["seg_idx"])["bpm"]
            grid_rows.append({**base, "map_bpm": map_bpm, "n_grid": len(grid), **grid_agreement(grid, beats)})
    pl.DataFrame(sync_rows, infer_schema_length=None).write_parquet(P / "map_audio_sync.parquet")
    pl.DataFrame(grid_rows, infer_schema_length=None).write_parquet(P / "beat_agreement.parquet")
    summary.update(onset_bias_ms=bias * 1000, difficulties=len(grid_rows))

sync = _with_state("audio-sync", _sync)


# --------------------------------------------------------------------------- stats

def _stats(paths, state, summary, pilot=False):
    maps = pl.read_parquet(paths.processed / "maps.parquet", columns=["audio_sha256", "split", "pilot"])
    if pilot:
        maps = maps.filter(pl.col("pilot"))
    train = maps.filter(pl.col("split") == "train").join(ok_songs(paths), on="audio_sha256", how="semi")
    shas = sorted(set(train["audio_sha256"]))
    if len(shas) > STATS_MAX_SONGS:
        shas = sorted(random.Random(0).sample(shas, STATS_MAX_SONGS))
    store = FeatureStore(paths.features)
    acc: dict[str, list] = {}
    for sha in shas:
        for name in ARRAYS:
            a = np.asarray(store.load(sha, name), dtype=np.float64)
            flat = a.reshape(a.shape[0], -1)
            s = acc.setdefault(name, [np.zeros(flat.shape[1]), np.zeros(flat.shape[1]), 0, a.shape[1:]])
            s[0] += flat.sum(axis=0)
            s[1] += np.square(flat).sum(axis=0)
            s[2] += flat.shape[0]
    arrays = {}
    for name, (s1, s2, n, shape) in acc.items():
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
        arrays[name] = {"shape": list(shape), "mean": mean.reshape(shape).tolist(),
                        "std": std.reshape(shape).tolist(), "n_frames": int(n)}
    out = {"split": "train", "n_songs": len(shas), "songs": shas, "arrays": arrays}
    store.stats_path.write_text(json.dumps(out))
    summary.update(n_songs=len(shas), path=str(store.stats_path))

stats = _with_state("audio-stats", _stats)
```

`_structure_one` must stay a module-level function: it runs in a `ProcessPoolExecutor`, which on Windows pickles it by name.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest tests/test_audio_pipeline.py -q` → `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/pipeline.py tests/test_audio_pipeline.py
git commit -m "Phase 2: structure, sync and stats stages"
```

---

### Task 12: Report, pilot sequence and CLI

**Files:**
- Create: `src/kata/audio/report.py`
- Modify: `src/kata/audio/pipeline.py` (append), `src/kata/cli.py`
- Test: `tests/test_audio_pipeline.py` (append)

**Interfaces:**
- Consumes: `kata.data.report.md_table`, `dir_size`, `gb`; `store.FeatureStore`; `structure.boundary_density_check`; the Phase 2 tables.
- Produces: `report.write_report(paths, state, name) -> Path` (`reports/{name}-YYYY-MM-DD.md`); `pipeline.report(paths, name="phase2")`; `pipeline.PILOT_STAGES`; CLI `kata audio features|structure|sync|stats|report|pilot`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_audio_pipeline.py`:

```python
def test_report(paths):
    ap.features(paths, extractor=FakeExtractor())
    for stage in (ap.structure, ap.sync, ap.stats):
        stage(paths)
    path = ap.report(paths, name="test")["path"]
    text = open(path, encoding="utf-8").read()
    for heading in ("## 2. Features", "## 4. Map grid vs. audio beats", "## 5. Map ↔ audio sync",
                    "## 6. Structure", "## 7. Normalisation stats"):
        assert heading in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra audio pytest tests/test_audio_pipeline.py -q`
Expected: `AttributeError: module 'kata.audio.pipeline' has no attribute 'report'`.

- [ ] **Step 3: Implement**

`src/kata/audio/report.py`:

```python
"""Phase 2 Markdown report (docs/phases/phase-2-audio.md §12, criterion 10)."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from kata.audio.store import FeatureStore
from kata.audio.structure import boundary_density_check
from kata.config import Paths
from kata.data.report import dir_size, gb, md_table

TABLES = ("audio_features", "audio_beats", "audio_tempo", "audio_segments", "map_audio_sync",
          "beat_agreement", "maps", "difficulties", "notes")


def window_benchmark(store: FeatureStore, songs: pl.DataFrame, n: int = 200, seconds: float = 12.0) -> dict:
    """Median / p95 time of a random `seconds`-long window read (criterion 6)."""
    rng = np.random.default_rng(0)
    rows = songs.to_dicts()
    times = []
    for _ in range(n if rows else 0):
        r = rows[rng.integers(len(rows))]
        start = rng.uniform(0, max(r["duration_s"] - seconds, 0))
        t0 = time.perf_counter()
        store.window(r["audio_sha256"], start, start + seconds)
        times.append((time.perf_counter() - t0) * 1000)
    return {"n": len(times), "median_ms": float(np.median(times)) if times else None,
            "p95_ms": float(np.percentile(times, 95)) if times else None}


def structure_vs_density(con) -> pl.DataFrame:
    """Per tier: segment boundaries near a large note-density change, vs. chance (coverage)."""
    hardest = con.execute("""
        SELECT d.hash, d.characteristic, d.difficulty, d.tier, m.audio_sha256
        FROM difficulties d JOIN maps m USING (hash)
        WHERE d.parse_status = 'ok' AND m.audio_sha256 IN (SELECT audio_sha256 FROM audio_segments)
        QUALIFY row_number() OVER (PARTITION BY d.hash ORDER BY d.difficulty_rank DESC) = 1""").pl()
    out = []
    for r in hardest.iter_rows(named=True):
        times = con.execute("SELECT time_s FROM notes WHERE hash = ? AND characteristic = ? AND difficulty = ?",
                            [r["hash"], r["characteristic"], r["difficulty"]]).pl()["time_s"].to_numpy()
        segs = con.execute("SELECT start_s, end_s FROM audio_segments WHERE audio_sha256 = ? ORDER BY seg_idx",
                           [r["audio_sha256"]]).pl()
        if segs.height < 2 or len(times) == 0:
            continue
        c = boundary_density_check(segs["start_s"].to_numpy()[1:], times, float(segs["end_s"].max()))
        out.append({"tier": r["tier"], **c})
    if not out:
        return pl.DataFrame()
    return (pl.DataFrame(out).group_by("tier").agg(
        pl.len().alias("maps"), (pl.col("hits").sum() / pl.col("n").sum()).alias("boundary_hit_rate"),
        pl.col("coverage").mean().alias("chance_rate")).sort("tier"))


def write_report(paths: Paths, state, name: str) -> Path:
    P = paths.processed
    con = duckdb.connect()
    for t in TABLES:
        if (P / f"{t}.parquet").exists():
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{(P / f'{t}.parquet').as_posix()}')")
    has = lambda t: (P / f"{t}.parquet").exists()  # noqa: E731
    q = lambda sql: con.execute(sql).pl()  # noqa: E731
    store = FeatureStore(paths.features)
    out = [f"# Phase 2 audio report — {name}", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data root `{paths.root}`.", ""]

    runs = state.con.execute("""
        SELECT stage, args, round(seconds, 1) AS seconds, summary FROM runs WHERE stage LIKE 'audio-%'
        QUALIFY row_number() OVER (PARTITION BY stage, args ORDER BY started_at DESC) = 1
        ORDER BY started_at""").pl()
    last = {r["stage"]: json.loads(r["summary"] or "{}") for r in runs.iter_rows(named=True)}
    out += ["## 1. Stage timings (latest run of each)", "", md_table(runs.select("stage", "args", "seconds")), ""]

    if has("audio_features"):
        feats = q("SELECT * FROM audio_features")
        ok = feats.filter(pl.col("status") == "ok")
        size = dir_size(store.dir)
        frames = int(ok["n_frames"].sum()) if ok.height else 0
        out += ["## 2. Features", "",
                f"- Songs: {feats.height:,} · ok {ok.height:,} · errors {feats.height - ok.height:,}",
                f"- Audio: {ok['duration_s'].sum() / 3600 if ok.height else 0:.1f} h · {frames:,} frames",
                f"- Disk: {gb(size)} under `{store.dir}` ({size / max(frames, 1):.0f} B/frame)",
                f"- Window read (12 s, warm): `{window_benchmark(store, ok)}`", ""]
        if feats.height > ok.height:
            out += ["Errors:", "", md_table(q("""SELECT error, count(*) n FROM audio_features
                WHERE status <> 'ok' GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")), ""]
        if ok.height:
            out += [md_table(q("""SELECT source_sr, count(*) songs, sum(CAST(bandlimited AS INT)) bandlimited,
                    round(median(rms_dbfs), 1) median_rms_dbfs, round(avg(clip_ratio) * 100, 3) clip_pct
                    FROM audio_features WHERE status = 'ok' GROUP BY 1 ORDER BY 2 DESC""")), ""]

    if has("audio_tempo"):
        out += ["## 3. Tempo", "", md_table(q("""SELECT est_meter, count(*) songs,
                round(quantile_cont(est_bpm, 0.5), 1) median_bpm, round(quantile_cont(tempo_cv, 0.5), 4) median_cv,
                round(quantile_cont(tempo_cv, 0.95), 4) p95_cv FROM audio_tempo GROUP BY 1 ORDER BY 2 DESC""")), ""]

    if has("beat_agreement"):
        out += ["## 4. Map grid vs. audio beats", "",
                md_table(q("""SELECT tier, best_level, count(*) diffs, round(avg(best_f), 3) mean_f
                    FROM beat_agreement GROUP BY ALL ORDER BY tier, best_level""")),
                md_table(q("""SELECT tier, count(*) diffs,
                    round(avg(CAST(best_f >= 0.8 AS INT)), 3) share_best_f_ge_08,
                    round(quantile_cont(f_l1, 0.5), 3) median_f_level1
                    FROM beat_agreement GROUP BY 1 ORDER BY 1""")), ""]

    if has("map_audio_sync"):
        out += ["## 5. Map ↔ audio sync", "",
                f"Onset detector bias (subtracted): {last.get('audio-sync', {}).get('onset_bias_ms', float('nan')):+.2f} ms", "",
                md_table(q("""SELECT tier, src, count(*) diffs,
                    round(median(lag_ms), 2) median_lag_ms,
                    round(quantile_cont(lag_ms, 0.25), 2) p25, round(quantile_cont(lag_ms, 0.75), 2) p75,
                    round(avg(CAST(abs(lag_ms) <= 30 AS INT)), 3) share_within_30ms,
                    count(*) FILTER (WHERE abs(lag_ms) > 50) n_over_50ms,
                    round(median(peak_ratio), 2) median_peak_ratio
                    FROM map_audio_sync WHERE lag_ms IS NOT NULL AND NOT isnan(lag_ms)
                    GROUP BY ALL ORDER BY tier, src""")),
                "Tier A, mix onsets — lag histogram (5 ms buckets, clipped to ±50 ms):", "",
                md_table(q("""SELECT greatest(-50, least(50, floor(lag_ms / 5) * 5)) bucket_ms, count(*) diffs
                    FROM map_audio_sync WHERE tier = 'A' AND src = 'mix' AND NOT isnan(lag_ms)
                    GROUP BY 1 ORDER BY 1""")),
                "Per-map consistency (std of lag across a map's difficulties, mix):", "",
                md_table(q("""SELECT round(quantile_cont(s, 0.5), 2) median_std_ms, round(quantile_cont(s, 0.95), 2) p95_std_ms
                    FROM (SELECT hash, stddev_samp(lag_ms) s FROM map_audio_sync WHERE src = 'mix'
                          GROUP BY hash HAVING count(*) > 1)""")), ""]

    if has("audio_segments"):
        out += ["## 6. Structure", "",
                md_table(q("""SELECT count(DISTINCT audio_sha256) songs,
                    round(avg(n) , 1) mean_segments, round(quantile_cont(n, 0.5), 1) median_segments,
                    round(quantile_cont(len, 0.5), 1) median_seg_s, round(avg(nl), 2) mean_labels FROM (
                      SELECT audio_sha256, count(*) OVER w n, end_s - start_s len, n_labels nl
                      FROM audio_segments WINDOW w AS (PARTITION BY audio_sha256))""")),
                "Boundaries within 2 s of a large note-density change (hardest diff per map) vs. chance:", "",
                md_table(structure_vs_density(con)) if has("notes") else "_(no notes table)_" + chr(10), ""]

    if store.stats_path.exists():
        st = json.loads(store.stats_path.read_text())
        out += ["## 7. Normalisation stats", "",
                f"- From {st['n_songs']} {st['split']}-split songs; arrays {sorted(st['arrays'])}", ""]

    path = paths.reports / f"{name}-{datetime.now():%Y-%m-%d}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    return path
```

Append to `src/kata/audio/pipeline.py`:

```python
# --------------------------------------------------------------------------- report

def _report(paths, state, summary, name="phase2"):
    from kata.audio.report import write_report
    summary["path"] = str(write_report(paths, state, name))

report = _with_state("audio-report", _report)

PILOT_STAGES = [
    ("audio-features", features, {"pilot": True}),
    ("audio-structure", structure, {"pilot": True}),
    ("audio-sync", sync, {"pilot": True}),
    ("audio-stats", stats, {"pilot": True}),
    ("audio-report", report, {"name": "phase2-pilot"}),
]
```

In `src/kata/cli.py`, register the group directly below `app.add_typer(data, name="data")`:

```python
audio = typer.Typer(no_args_is_help=True, help="Phase 2 audio feature stages (needs `uv sync --extra audio`).")
app.add_typer(audio, name="audio")
```

and append at the end of the file:

```python
# --------------------------------------------------------------------------- Phase 2
# kata.audio needs the optional `audio` extra, so it is imported inside each command.

@audio.command("features")
def audio_features(pilot: bool = PILOT, limit: int = typer.Option(0, help="Max songs (0 = all)."),
                   device: str = typer.Option("cuda", help="Torch device.")):
    """Stems, spectral features, beats per unique audio file (GPU, resumable)."""
    from kata.audio import pipeline as ap
    _run("audio-features", ap.features, pilot=pilot, limit=limit, device=device)


@audio.command("structure")
def audio_structure(pilot: bool = PILOT):
    """Segment boundaries + repetition labels from the stored features."""
    from kata.audio import pipeline as ap
    _run("audio-structure", ap.structure, pilot=pilot)


@audio.command("sync")
def audio_sync(pilot: bool = PILOT):
    """Map <-> audio lag and map-grid vs. audio-beat agreement per difficulty."""
    from kata.audio import pipeline as ap
    _run("audio-sync", ap.sync, pilot=pilot)


@audio.command("stats")
def audio_stats(pilot: bool = PILOT):
    """Train-split normalisation statistics -> features/v1/stats.json."""
    from kata.audio import pipeline as ap
    _run("audio-stats", ap.stats, pilot=pilot)


@audio.command("report")
def audio_report(name: str = typer.Option("phase2", help="Report file name prefix.")):
    from kata.audio import pipeline as ap
    _run("audio-report", ap.report, name=name)


@audio.command("pilot")
def audio_pilot():
    """Run every Phase 2 stage on the pilot subset, then write the pilot report."""
    from kata.audio import pipeline as ap
    for stage, fn, kwargs in ap.PILOT_STAGES:
        _run(stage, fn, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra audio pytest -q` → every test passes (Phase 1's 33 + Phase 2's 67 = 100).
Run: `uv run --extra audio kata audio --help` → lists `features, structure, sync, stats, report, pilot`.
Run: `uv run kata data --help` → still works (the `audio` group imports nothing at startup).

- [ ] **Step 5: Commit**

```bash
git add src/kata/audio/report.py src/kata/audio/pipeline.py src/kata/cli.py tests/test_audio_pipeline.py
git commit -m "Phase 2: report, pilot sequence and kata audio CLI"
```

---

### Task 13: Pilot run and findings (checkpoint with the user)

Operational task: no new code unless the pilot finds a bug (then: write a failing test reproducing it, fix, commit, re-run).

**Files:**
- Modify: `docs/phases/phase-2-audio.md` (new §18 Pilot findings, decision log, criterion thresholds)

- [ ] **Step 1: Run the pilot**

Run: `uv run --extra audio kata audio pilot`
Expected: ~199 songs in `audio-features` (`"error": 0` or a handful with reasons), then `audio-structure`, `audio-sync`, `audio-stats`, `audio-report` summaries; report at `E:\kata-data\reports\phase2-pilot-YYYY-MM-DD.md`. Takes ~10 minutes.

- [ ] **Step 2: Peak VRAM on the longest song (spec §10 open item)**

Run:

```bash
uv run --extra audio python -c "
import duckdb, torch
from kata.audio import decode
from kata.audio.extract import Extractor
h, f, d = duckdb.sql(\"select hash, audio_file, audio_duration_s from 'E:/kata-data/processed/maps.parquet' order by audio_duration_s desc limit 1\").fetchone()
ex = Extractor('cuda'); torch.cuda.reset_peak_memory_stats()
ex(decode.load(f'E:/kata-data/raw/maps/{h}/{f}'))
print(f'{d:.0f} s song: peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')"
```

Expected: a 710 s song, peak well under 16 GiB (1.14 GiB was measured at 204 s).

- [ ] **Step 3: Check the Beat This! checkpoint licence (spec §13 open item)**

Read the licence section of `https://github.com/CPJKU/beat_this` (README and LICENSE) and record what it says about the `final0` checkpoint in spec §13.

- [ ] **Step 4: Evaluate the pilot report against spec §12**

Record in a new spec section **§18 Pilot findings**: features ok/errors, B/frame, window read p95, the sync table (mix vs. drums, per tier), the beat-agreement table, the structure numbers and boundary↔density rates, stats song count, VRAM, licence.
Then propose (do not decide alone):
- criterion 4: which onset source (`mix` or `drums`) and which median / ±30 ms thresholds, from the pilot's Tier A distribution (spec §17 shows mix sits ~6 ms earlier than drums);
- criterion 5: the `best_f ≥ 0.8` share threshold;
- whether structure quality (boundary hit rate vs. chance) is acceptable for v1.

- [ ] **Step 5: Checkpoint**

Show the user the pilot report and the proposals. **The full run starts only after the user approves.** Record the decisions in the spec's decision log and commit:

```bash
git add docs/phases/phase-2-audio.md
git commit -m "Phase 2: pilot findings and tuned acceptance thresholds"
```

---

### Task 14: Full run and acceptance

**Files:**
- Modify: `docs/phases/phase-2-audio.md` (new §19 v1 build results; status line), `docs/PLAN.md` (Phase 2 ✅)

- [ ] **Step 1: Run `features` on the full set (overnight)**

Run in the background: `uv run --extra audio kata audio features`
Expected: ~5,347 songs at ~2.5 s each ≈ 4 h. Resumable: if interrupted, run the same command again.

- [ ] **Step 2: Run the remaining stages and rebuild views**

```bash
uv run --extra audio kata audio structure
uv run --extra audio kata audio sync
uv run --extra audio kata audio stats
uv run --extra audio kata audio report
uv run kata data build
```

- [ ] **Step 3: Verify every acceptance criterion (spec §12)**

Run these and record each result:

```bash
uv run --extra audio python -c "
import duckdb, json
c = duckdb.connect('E:/kata-data/kata.duckdb', read_only=True)
print('1 features ok / total:', c.sql(\"select count(*) filter (where status='ok'), count(*) from audio_features\").fetchone())
print('1 unique audio in maps:', c.sql('select count(distinct audio_sha256) from maps').fetchone())
print('2 n_frames == n_samples//441+1 (via duration):', c.sql(\"select count(*) from audio_features where status='ok' and n_frames <> floor(round(duration_s*44100)/441)+1\").fetchone(), 'mismatches')
print('4 sync (tier A):', c.sql(\"select src, median(lag_ms), avg(cast(abs(lag_ms)<=30 as int)) from map_audio_sync where tier='A' group by 1\").fetchall())
print('5 beats (tier A, best_f>=0.8):', c.sql(\"select avg(cast(best_f>=0.8 as int)) from beat_agreement where tier='A'\").fetchone())
s = json.load(open('E:/kata-data/features/v1/stats.json'))
print('7 stats songs all train:', c.sql(\"select count(*) from maps where audio_sha256 in (select unnest(?)) and split<>'train'\", params=[s['songs']]).fetchone(), 'non-train')
"
uv run --extra audio pytest -q
uv run --extra audio kata audio features
```

Expected: criterion 1 ≥ 99.5 % ok; criterion 2: 0 mismatches; criteria 4 and 5 meet the thresholds fixed in Task 13; criterion 7: 0 non-train; all tests pass (criterion 3); the final `features` re-run reports `"processed": 0` (criterion 8). Criteria 6 (window read p95 < 5 ms), 9 (disk ≤ 100 GB) and 10 (report contents) are read from the report.

- [ ] **Step 4: Record and commit**

Write **§19 v1 build results** in the spec (same shape as Phase 1 §13.1: coverage, timings, disk, every criterion with its number), set the status line to done, mark Phase 2 ✅ in `docs/PLAN.md`, then:

```bash
git add docs/phases/phase-2-audio.md docs/PLAN.md
git commit -m "Phase 2: v1 audio feature build results"
```
