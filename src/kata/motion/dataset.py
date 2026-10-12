"""Training windows (docs/phases/phase-4-motion.md §3.6, §4, §6.2). numpy only.

A batch is a pure function of (seed, step), so a resumed run sees exactly the batches it would
have seen without the interruption, whatever the number of loader workers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from kata.audio.store import ARRAYS, FeatureStore
from kata.motion.prepare import FPS, HIT, MOTION_DIM, OFF, TRACK, MotionStore
from kata.validate.rules import LABELS

WINDOW = 640                 # 12.8 s (spec §5.1)
CTX = 100                    # 2 s of audio past each end (§4.2)
AUDIO_FRAME_DIM = 408        # mel 128 + stem_mel 256 + chroma 12 + curves 12 per 10 ms
AUDIO_DIM = 2 * AUDIO_FRAME_DIM + 4   # two 10 ms frames per 20 ms + beat/bar phase sin/cos
NONE_LABEL = len(LABELS)     # "no label" for classifier-free guidance
HIT_SIGMA = 1.0              # frames (§5.5)
STEM_COLS = slice(128, 384)  # stem_mel inside one 408-value audio frame
START_TRIES = 20


@dataclass(frozen=True)
class TrackRef:
    score_id: str
    sha: str
    frame0: int
    n_frames: int
    label: int               # index into LABELS
    nps: float
    key: tuple               # (hash, characteristic, difficulty)


def load_tracks(tracks: pl.DataFrame, split: str, songs_with_features, window: int = WINDOW) -> list[TrackRef]:
    """Accepted tracks of one split whose song has features and that are at least one window long."""
    t = (tracks.filter((pl.col("status") == "ok") & (pl.col("split") == split) & (pl.col("n_frames") >= window)
                       & pl.col("audio_sha256").is_in(list(songs_with_features)))
         .sort("score_id"))
    return [TrackRef(r["score_id"], r["audio_sha256"], r["frame0"], r["n_frames"], LABELS.index(r["label"]),
                     float(r["nps"]), (r["hash"], r["characteristic"], r["difficulty"]))
            for r in t.iter_rows(named=True)]


class Normaliser:
    """Train-split statistics: motion/{v}/stats.json (27 channels) and features/{v}/stats.json (Phase 2)."""

    def __init__(self, motion_stats: dict, audio_stats: dict):
        self.m_mean = np.asarray(motion_stats["mean"], np.float32)
        self.m_std = np.asarray(motion_stats["std"], np.float32)
        arrays = audio_stats["arrays"]
        self.a_mean = np.concatenate([np.asarray(arrays[n]["mean"], np.float32).ravel() for n in ARRAYS])
        self.a_std = np.maximum(np.concatenate([np.asarray(arrays[n]["std"], np.float32).ravel() for n in ARRAYS]),
                                1e-3)

    def motion(self, raw: np.ndarray) -> np.ndarray:
        """(..., 31) raw track rows -> model space: motion standardised, hit flags -> pulse, offsets x2."""
        out = np.empty(raw.shape, np.float32)
        out[..., :MOTION_DIM] = (raw[..., :MOTION_DIM] - self.m_mean) / self.m_std
        for h in (0, 1):
            out[..., HIT[h]] = hit_pulse(raw[..., HIT[h]])
            out[..., OFF[h]] = 2.0 * raw[..., OFF[h]] * (raw[..., HIT[h]] > 0)
        return out

    def denorm_motion(self, x: np.ndarray) -> np.ndarray:
        """Model space -> metres and 6D rotations for the 27 motion channels."""
        return x[..., :MOTION_DIM] * self.m_std + self.m_mean

    def audio(self, frames: np.ndarray) -> np.ndarray:
        return ((np.asarray(frames, np.float32) - self.a_mean) / self.a_std).astype(np.float32)


def hit_pulse(flags: np.ndarray) -> np.ndarray:
    """(T,) hit flags -> Gaussian bumps (σ = HIT_SIGMA frames, peak 1) mapped to [-1, 1]."""
    flags = np.asarray(flags)
    pulse = np.zeros(flags.shape, np.float32)
    for f in np.flatnonzero(flags > 0):
        lo, hi = max(f - 4, 0), min(f + 5, len(flags))
        k = np.arange(lo, hi)
        pulse[lo:hi] = np.maximum(pulse[lo:hi], np.exp(-0.5 * ((k - f) / HIT_SIGMA) ** 2))
    return 2.0 * pulse - 1.0


def decode_hits(pulse: np.ndarray, offset: np.ndarray, frame0: int = 0) -> np.ndarray:
    """Model-space pulse and offset channels of one hand -> hit times (s): local maxima above 0."""
    p = np.asarray(pulse, np.float64)
    left = np.concatenate([[-np.inf], p[:-1]])
    right = np.concatenate([p[1:], [-np.inf]])
    peaks = np.flatnonzero((p > 0) & (p >= left) & (p > right))
    off = np.clip(np.asarray(offset, np.float64)[peaks] / 2.0, -0.5, 0.5)
    return (frame0 + peaks + off) / FPS


def phase(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Fractional position in [0, 1) between consecutive events; outside them, the median period
    continues the grid. Fewer than two events -> zeros."""
    times = np.asarray(times, np.float64)
    ev = np.asarray(events, np.float64)
    if len(ev) < 2:
        return np.zeros(len(times))
    period = float(np.median(np.diff(ev)))
    j = np.searchsorted(ev, times, side="right") - 1
    jj = np.clip(j, 0, len(ev) - 2)
    out = (times - ev[jj]) / (ev[jj + 1] - ev[jj])
    out = np.where(j < 0, ((times - ev[0]) / period) % 1.0, out)
    return np.where(j >= len(ev) - 1, ((times - ev[-1]) / period) % 1.0, out)


def mirror_track(raw: np.ndarray) -> np.ndarray:
    """Mirror augmentation (§3.5) on raw rows: x -> -x, rotations M R M, hands (and their hits) swapped."""
    out = np.array(raw, dtype=np.float32, copy=True)
    for part in ("head", "left", "right"):
        s = TRACK[part].start
        out[..., s] *= -1                                   # position x
        out[..., s + 4] *= -1                               # col0 y
        out[..., s + 5] *= -1                               # col0 z
        out[..., s + 6] *= -1                               # col1 x
    left, right = TRACK["left"], TRACK["right"]
    out[..., left], out[..., right] = out[..., right].copy(), out[..., left].copy()
    for a, b in (HIT, OFF):
        out[..., [a, b]] = out[..., [b, a]]
    return out


class WindowSampler:
    """Song-first window sampling (§6.2) and the audio context for a window (§4)."""

    def __init__(self, tracks: list[TrackRef], motion: MotionStore, features: FeatureStore, norm: Normaliser,
                 window: int = WINDOW, ctx: int = CTX, seed: int = 0, batch: int = 64,
                 mirror_aug: bool = False, use_stems: bool = True):
        self.motion, self.features, self.norm = motion, features, norm
        self.window, self.ctx, self.seed, self.batch_size = window, ctx, seed, batch
        self.mirror_aug, self.use_stems = mirror_aug, use_stems
        songs: dict[str, dict[tuple, list[TrackRef]]] = {}
        for t in tracks:
            songs.setdefault(t.sha, {}).setdefault(t.key, []).append(t)
        self.songs = [(sha, [diffs[k] for k in sorted(diffs)]) for sha, diffs in sorted(songs.items())]
        self._events: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def events(self, sha: str) -> tuple[np.ndarray, np.ndarray]:
        if sha not in self._events:
            self._events[sha] = self.features.events(sha)
        return self._events[sha]

    def audio(self, sha: str, start: int, n: int) -> np.ndarray:
        """(n, AUDIO_DIM) for 50 fps grid frames start .. start + n - 1 (absolute grid indices)."""
        w = self.features.window(sha, start / FPS, (start + n) / FPS)
        flat = np.concatenate([w["mel"], w["stem_mel"].reshape(len(w["mel"]), -1), w["chroma"], w["curves"]], 1)
        flat = self.norm.audio(flat)
        if not self.use_stems:
            flat[:, STEM_COLS] = 0.0
        pairs = flat.reshape(n, 2 * AUDIO_FRAME_DIM)
        t = (start + np.arange(n)) / FPS
        beats, downbeats = self.events(sha)
        ph = 2 * np.pi * np.stack([phase(t, beats), phase(t, downbeats)], 1)
        return np.concatenate([pairs, np.sin(ph), np.cos(ph)], 1).astype(np.float32)

    def example(self, track: TrackRef, start: int, mirror: bool = False) -> dict:
        raw = np.asarray(self.motion.load(track.score_id)[start:start + self.window], np.float32)
        if mirror:
            raw = mirror_track(raw)
        g0 = track.frame0 + start
        hits = np.stack([raw[:, HIT[0]] > 0, raw[:, HIT[1]] > 0], 1)
        return {"x": self.norm.motion(raw), "hit_mask": hits.astype(np.float32),
                "audio": self.audio(track.sha, g0 - self.ctx, self.window + 2 * self.ctx),
                "label": track.label, "nps": track.nps}

    def _start(self, rng: np.random.Generator, track: TrackRef) -> int | None:
        valid = self.motion.valid(track.score_id)
        for _ in range(START_TRIES):
            s = int(rng.integers(0, track.n_frames - self.window + 1))
            if valid[s:s + self.window].all():
                return s
        return None

    def batch(self, step: int) -> dict:
        rng = np.random.default_rng([self.seed, step])
        out = []
        while len(out) < self.batch_size:
            _, diffs = self.songs[int(rng.integers(len(self.songs)))]
            replays = diffs[int(rng.integers(len(diffs)))]
            track = replays[int(rng.integers(len(replays)))]
            start = self._start(rng, track)
            if start is None:
                continue
            mirror = bool(self.mirror_aug and rng.random() < 0.5)
            out.append(self.example(track, start, mirror))
        return {k: np.stack([o[k] for o in out]).astype(np.float32 if k != "label" else np.int64) for k in out[0]}
