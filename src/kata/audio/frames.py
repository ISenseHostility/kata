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
