"""Map <-> audio sync: how far a difficulty's notes sit from the audio onsets (§7.3)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from kata.audio.frames import FPS, SR, frame_index

MAX_LAG_S = 0.2
SMOOTH_S = 0.02
CONTEXT_S = 1.0


def note_train(note_times, n: int) -> np.ndarray:
    """Unit impulses at the note times (split linearly between frames), Gaussian-smoothed."""
    f = frame_index(np.unique(np.asarray(note_times, dtype=np.float64)))
    f = f[(f >= 0) & (f <= n - 1)]
    out = np.zeros(n, dtype=np.float64)
    i0 = np.floor(f).astype(np.int64)
    w = f - i0
    np.add.at(out, i0, 1 - w)
    np.add.at(out, np.minimum(i0 + 1, n - 1), w)
    return gaussian_filter1d(out, SMOOTH_S * FPS)


def _xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """c[l + max_lag] = mean_t a[t] * b[t - l] for l in [-max_lag, max_lag]."""
    out = np.empty(2 * max_lag + 1)
    for i, l in enumerate(range(-max_lag, max_lag + 1)):
        x, y = (a[l:], b[:len(b) - l]) if l >= 0 else (a[:l], b[-l:])
        out[i] = np.dot(x, y) / len(x)
    return out


def estimate_lag(onset, note_times, bias_s: float = 0.0) -> dict:
    """Lag > 0 means the notes come *after* the audio onsets.

    bias_s is the onset detector's own offset (peak time - true onset), from measure_onset_bias().
    """
    onset = np.asarray(onset, dtype=np.float64)
    times = np.asarray(note_times, dtype=np.float64)
    lo = max(0, int((times.min() - CONTEXT_S) * FPS))
    hi = min(len(onset), int((times.max() + CONTEXT_S) * FPS) + 1)
    o = onset[lo:hi]
    t = note_train(times - lo / FPS, hi - lo)
    if o.std() == 0 or t.std() == 0:
        return {"lag_ms": float("nan"), "peak_corr": float("nan"), "peak_ratio": float("nan")}
    o = (o - o.mean()) / o.std()
    t = (t - t.mean()) / t.std()
    L = int(round(MAX_LAG_S * FPS))
    c = _xcorr(t, o, L)
    k = int(np.argmax(c))
    frac = 0.0
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        frac = 0.5 * (c[k - 1] - c[k + 1]) / den if den != 0 else 0.0
    lag_s = (k - L + frac) / FPS + bias_s
    return {"lag_ms": lag_s * 1000.0, "peak_corr": float(c[k]),
            "peak_ratio": float(c[k] / max(np.median(np.abs(c)), 1e-9))}


def click_track(times_s, dur_s: float, noise_db: float = -40.0, seed: int = 0) -> np.ndarray:
    """Decaying 1 kHz clicks at times_s over white noise; mono float32 at SR."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(int(dur_s * SR)) * 10 ** (noise_db / 20)).astype(np.float32)
    k = np.arange(int(0.03 * SR))
    tone = (np.sin(2 * np.pi * 1000 * k / SR) * np.exp(-k / (0.005 * SR))).astype(np.float32)
    for t in times_s:
        s = int(round(t * SR))
        x[s:s + len(tone)] += tone[: len(x) - s]
    return x


def measure_onset_bias(spectral) -> float:
    """Median (flux peak - click time) in seconds on a synthetic click track, refined to sub-frame."""
    times = np.arange(1.0, 29.0, 0.4615) + 0.0037  # off-grid on purpose
    flux = spectral.mix(click_track(times, 30.0))["onset"].astype(np.float64)
    errs = []
    for t in times:
        c = int(round(t * FPS))
        j = c - 5 + int(np.argmax(flux[c - 5:c + 6]))
        a, b, d = flux[j - 1], flux[j], flux[j + 1]
        den = a - 2 * b + d
        errs.append((j + (0.5 * (a - d) / den if den != 0 else 0.0)) / FPS - t)
    return float(np.median(errs))
