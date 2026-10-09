"""Piecewise-linear beat -> seconds mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start_beat: float
    start_s: float
    sec_per_beat: float

    @property
    def bpm(self) -> float:
        return 60.0 / self.sec_per_beat


class TempoMap:
    def __init__(self, segments: list[Segment], offset_s: float = 0.0):
        if not segments:
            raise ValueError("tempo map needs at least one segment")
        self.segments = segments
        self.offset_s = offset_s
        self._beats = np.array([s.start_beat for s in segments])
        self._secs = np.array([s.start_s for s in segments])
        self._spb = np.array([s.sec_per_beat for s in segments])

    @classmethod
    def from_bpm_events(cls, base_bpm: float, events: list[tuple[float, float]],
                        offset_s: float = 0.0) -> "TempoMap":
        """v2 (`_events` type 100) and v3 (`bpmEvents`): BPM holds from its beat onwards."""
        if base_bpm <= 0:
            raise ValueError(f"invalid base BPM {base_bpm}")
        segments = [Segment(0.0, 0.0, 60.0 / base_bpm)]
        for beat, bpm in sorted(e for e in events if e[1] > 0):
            last = segments[-1]
            if beat <= last.start_beat:
                # An event at (or before) the current segment start replaces its tempo.
                segments[-1] = Segment(last.start_beat, last.start_s, 60.0 / bpm)
                continue
            start_s = last.start_s + (beat - last.start_beat) * last.sec_per_beat
            segments.append(Segment(beat, start_s, 60.0 / bpm))
        return cls(segments, offset_s)

    @classmethod
    def from_audio_regions(cls, base_bpm: float, frequency: int,
                           regions: list[dict]) -> "TempoMap":
        """v4 AudioData `bpmData`: regions map sample indices [si, ei] to beats [sb, eb]."""
        usable = sorted((r for r in regions if r.get("eb", 0) > r.get("sb", 0)),
                        key=lambda r: r["sb"])
        if not usable or frequency <= 0:
            return cls.from_bpm_events(base_bpm, [])
        segments = []
        for r in usable:
            spb = (r["ei"] - r["si"]) / frequency / (r["eb"] - r["sb"])
            segments.append(Segment(float(r["sb"]), r["si"] / frequency, spb))
        if segments[0].start_beat > 0:
            first = segments[0]
            segments.insert(0, Segment(0.0, first.start_s - first.start_beat * first.sec_per_beat,
                                       first.sec_per_beat))
        return cls(segments)

    def to_seconds(self, beats) -> np.ndarray:
        b = np.asarray(beats, dtype=np.float64)
        i = np.clip(np.searchsorted(self._beats, b, side="right") - 1, 0, len(self._beats) - 1)
        return self._secs[i] + (b - self._beats[i]) * self._spb[i] + self.offset_s

    def rows(self) -> list[dict]:
        return [{"seg_idx": i, "start_beat": s.start_beat, "start_s": s.start_s + self.offset_s,
                 "bpm": s.bpm} for i, s in enumerate(self.segments)]
