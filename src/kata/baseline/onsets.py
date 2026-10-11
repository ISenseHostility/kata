"""Note times from onset curves (spec §8.2)."""

from __future__ import annotations

import numpy as np

from kata.audio.frames import FPS
from kata.audio.store import CURVE_COLUMNS

ONSET_WEIGHTS = {"onset_drums": 1.0, "onset_mix": 0.5, "onset_bass": 0.3, "onset_other": 0.3,
                 "onset_vocals": 0.3}
POOL_S = 0.03            # slot strength = max over ±30 ms
LEAD_IN_S, LEAD_OUT_S = 1.5, 2.0
ACTIVE_FRACTION = 0.2    # the mapped span runs from the first to the last slot this strong (× max)


def onset_strength(curves: np.ndarray) -> np.ndarray:
    """Weighted sum of the onset curves, each divided by its own 99th percentile."""
    out = np.zeros(curves.shape[0], np.float64)
    for name, w in ONSET_WEIGHTS.items():
        c = np.asarray(curves[:, CURVE_COLUMNS.index(name)], dtype=np.float64)
        p99 = np.percentile(c, 99)
        out += w * c / (p99 if p99 > 0 else 1.0)
    return out


def loudness_boost(curves: np.ndarray, segments: list[tuple[float, float]]) -> np.ndarray:
    """Per frame: clip(1 + 0.5 z, 0.5, 1.5), z = the segment's mean rms_mix z-scored within the song."""
    rms = np.asarray(curves[:, CURVE_COLUMNS.index("rms_mix")], dtype=np.float64)
    mu, sd = rms.mean(), rms.std() or 1.0
    boost = np.ones(len(rms))
    for start, end in segments:
        a, b = int(round(start * FPS)), min(int(round(end * FPS)), len(rms))
        if b > a:
            boost[a:b] = np.clip(1 + 0.5 * (rms[a:b].mean() - mu) / sd, 0.5, 1.5)
    return boost


def slot_strengths(frame_strength: np.ndarray, slot_times: np.ndarray) -> np.ndarray:
    r = int(round(POOL_S * FPS))
    centre = np.round(np.asarray(slot_times) * FPS).astype(np.int64)
    idx = np.clip(centre[:, None] + np.arange(-r, r + 1), 0, len(frame_strength) - 1)
    return frame_strength[idx].max(axis=1)


def mapped_span(slot_times: np.ndarray, strengths: np.ndarray, duration_s: float) -> np.ndarray:
    """Mask of slots inside [first, last] strong slot, at least LEAD_IN_S / LEAD_OUT_S from the ends."""
    ok = (slot_times >= LEAD_IN_S) & (slot_times <= duration_s - LEAD_OUT_S)
    strong = np.flatnonzero(ok & (strengths >= ACTIVE_FRACTION * strengths[ok].max())) if ok.any() else []
    if len(strong) == 0:
        return np.zeros(len(slot_times), bool)
    return ok & (np.arange(len(slot_times)) >= strong[0]) & (np.arange(len(slot_times)) <= strong[-1])


def select(slot_times: np.ndarray, strengths: np.ndarray, n_target: int, min_gap_s: float) -> np.ndarray:
    """Greedy by strength: indices (time-ordered) of up to n_target slots at least min_gap_s apart."""
    chosen: list[float] = []
    picked: list[int] = []
    for i in np.argsort(-strengths, kind="stable"):
        t = slot_times[i]
        k = np.searchsorted(chosen, t)
        if (k > 0 and t - chosen[k - 1] < min_gap_s - 1e-9) or (k < len(chosen) and chosen[k] - t < min_gap_s - 1e-9):
            continue
        chosen.insert(k, t)
        picked.insert(k, int(i))
        if len(picked) >= n_target:
            break
    return np.asarray(picked, dtype=np.int64)
