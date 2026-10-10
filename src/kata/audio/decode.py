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
