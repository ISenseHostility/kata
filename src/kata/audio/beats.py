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
