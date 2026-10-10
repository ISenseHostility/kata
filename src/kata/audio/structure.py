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
