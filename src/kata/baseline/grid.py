"""Tempo grid from audio beats (spec §8.1): map beat k+1 is fitted audio beat k.

The grid is a list of times, one per map beat, starting with map beat 0 at t = 0. Beat 0 -> 1 is a
lead-in whose length puts map beat 1 on the first fitted audio beat at least half a period in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kata.data.mapformat.tempo import TempoMap

STEADY_P95_S = 0.030   # residual p95 above this -> fit per structure segment (measured: Beat This
                       # beats on 20 ms frames give 15-30 ms on steady songs)
SLOW_BPM = 110.0       # below this the slots are 1/8 of a fitted beat, so slow songs reach Expert+ density
MIN_SEGMENT_BEATS = 4


@dataclass
class Grid:
    times: np.ndarray        # time of map beat k, k = 0 .. len-1; times[0] = 0
    bpm: float               # Info BPM: the main fitted tempo
    piecewise: bool

    def to_beats(self, t) -> np.ndarray:
        """Map beat of time(s) t; linear beyond the last grid time at the last tempo."""
        t = np.asarray(t, dtype=np.float64)
        k = np.arange(len(self.times), dtype=np.float64)
        last = (t - self.times[-1]) / (self.times[-1] - self.times[-2]) + k[-1]
        return np.where(t > self.times[-1], last, np.interp(t, self.times, k))

    def bpm_events(self) -> list[tuple[float, float]]:
        """(beat, bpm) at every tempo change, including the lead-in at beat 0."""
        bpm = 60.0 / np.diff(self.times)
        out = [(0.0, float(bpm[0]))]
        for k in range(1, len(bpm)):
            if abs(bpm[k] - out[-1][1]) > 1e-6 * bpm[k]:
                out.append((float(k), float(bpm[k])))
        return out

    def tempo_map(self) -> TempoMap:
        return TempoMap.from_bpm_events(self.bpm, self.bpm_events())

    def slots(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, beats) of every 1/4 beat (1/8 below SLOW_BPM) from map beat 1 to the last grid time."""
        per = 8 if self.bpm < SLOW_BPM else 4
        t, k = self.times[1:], np.arange(1, len(self.times), dtype=np.float64)
        q = np.arange(per) / per
        times = (t[:-1, None] + np.diff(t)[:, None] * q).ravel()
        beats = (k[:-1, None] + q).ravel()
        return np.append(times, t[-1]), np.append(beats, k[-1])


def fit_line(beats: np.ndarray) -> tuple[float, float, float]:
    """(t0, period, residual p95) of t = t0 + k * period. Indices step by each interval divided by
    the median interval, rounded, so a missed beat leaves a gap instead of shifting later indices
    (rounding (b - b0) / median instead drifts by whole beats over a long song)."""
    b = np.sort(np.asarray(beats, dtype=np.float64))
    steps = np.maximum(np.round(np.diff(b) / np.median(np.diff(b))), 1)
    k = np.concatenate([[0.0], np.cumsum(steps)])
    period, t0 = np.polyfit(k, b, 1)
    resid = np.abs(b - (t0 + k * period))
    return float(t0), float(period), float(np.percentile(resid, 95))


def _line_times(t0: float, period: float, start: float, end: float) -> np.ndarray:
    j0 = int(np.ceil((start - t0) / period - 1e-9))
    j1 = int(np.floor((end - t0) / period + 1e-9))
    return t0 + np.arange(j0, j1 + 1) * period


def fit_grid(beats, duration_s: float, segments: list[tuple[float, float]] | None = None) -> Grid:
    """Grid covering [0, duration_s] from Beat This beats; piecewise per segment if unsteady."""
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    if len(beats) < MIN_SEGMENT_BEATS:
        raise ValueError(f"only {len(beats)} beats")
    t0, period, p95 = fit_line(beats)
    piecewise = p95 > STEADY_P95_S and bool(segments)
    if not piecewise:
        audio = _line_times(t0, period, 0.0, duration_s)
    else:
        parts, start = [], 0.0
        for k, (_, end) in enumerate(segments):
            last = k == len(segments) - 1
            seg_end = duration_s if last else end
            inside = beats[(beats >= start) & (beats < seg_end)]
            if len(inside) < MIN_SEGMENT_BEATS and not last:
                continue   # too short to fit: merged into the next segment
            if len(inside) < MIN_SEGMENT_BEATS:
                inside = beats[beats >= start - period * MIN_SEGMENT_BEATS]
            a, p, _ = fit_line(inside)
            parts.append(_line_times(a, p, start, seg_end - 1e-9))
            start = seg_end
        audio = np.concatenate(parts)
        keep = np.concatenate([[True], np.diff(audio) > 0.5 * period])
        audio = audio[keep]
    audio = audio[audio >= 0.5 * period]
    times = np.concatenate([[0.0], audio])
    return Grid(times, 60.0 / period, piecewise)
