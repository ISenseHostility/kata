"""One song in, every Phase 2 per-song output out (§4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version

import numpy as np

from kata.audio import decode
from kata.audio import models, spectral
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
            "flux_max_bins": spectral.FLUX_MAX_BINS, "rms_floor": spectral.RMS_FLOOR, "chroma_norm": "max",
            "dtype": "float16",
            "separator": separator_name, "separator_args": models.DEMUCS_ARGS,
            "beat_tracker": beat_name, "beat_postprocess": models.BEAT_POSTPROCESS, "beat_fps": models.BEAT_FPS,
            "packages": {p: version(p) for p in ("demucs", "beat-this", "torch", "soxr")}}


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
