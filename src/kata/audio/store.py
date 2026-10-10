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
