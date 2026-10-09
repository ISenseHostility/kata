"""Replay <-> map alignment check (docs/phases/phase-1-data.md §6.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kata.data.bsor import EVENT_BOMB, decode_note_id

# BeatLeader encodes scoringType + 2; 7 = BurstSliderElement (chain links have no map note).
SCORING_CHAIN_ELEMENT = 7
MATCH_WINDOW_S = 0.050


@dataclass
class AlignResult:
    matched_idx: np.ndarray  # per event, -1 when unmatched or skipped
    residual_ms: np.ndarray  # per event, NaN when unmatched or skipped
    n_considered: int
    n_matched: int

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_considered if self.n_considered else 0.0

    def residual_quantile(self, q: float) -> float:
        r = np.abs(self.residual_ms[~np.isnan(self.residual_ms)])
        return float(np.quantile(r, q)) if len(r) else float("nan")


def align(notes_x, notes_y, notes_color, notes_cut, notes_time_s, events: list[dict]) -> AlignResult:
    """Match each replay note event to the nearest map note with the same lane/layer/color/cut."""
    by_key: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    keys = np.stack([np.asarray(a, dtype=np.int64) for a in (notes_x, notes_y, notes_color, notes_cut)], 1)
    times = np.asarray(notes_time_s, dtype=np.float64)
    for key in {tuple(k) for k in keys.tolist()}:
        idx = np.flatnonzero((keys == key).all(axis=1))
        order = np.argsort(times[idx])
        by_key[key] = (times[idx][order], idx[order])

    matched = np.full(len(events), -1, dtype=np.int64)
    residual = np.full(len(events), np.nan)
    considered = 0
    for i, e in enumerate(events):
        if e["event_type"] == EVENT_BOMB:
            continue
        scoring, x, y, color, cut = decode_note_id(e["note_id"])
        if scoring == SCORING_CHAIN_ELEMENT:
            continue
        considered += 1
        entry = by_key.get((x, y, color, cut))
        if entry is None:
            continue
        t, idx = entry
        j = int(np.searchsorted(t, e["spawn_time"]))
        best = min((k for k in (j - 1, j) if 0 <= k < len(t)), key=lambda k: abs(t[k] - e["spawn_time"]))
        delta = e["spawn_time"] - t[best]
        if abs(delta) <= MATCH_WINDOW_S:
            matched[i] = idx[best]
            residual[i] = delta * 1000.0
    return AlignResult(matched, residual, considered, int((matched >= 0).sum()))
