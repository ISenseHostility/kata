"""Evaluation metrics (docs/phases/phase-4-motion.md §7.2) on de-normalised motion, plus the quick
windowed validation that picks checkpoints during training (§6.4).

Motion here is (T, 27): head, left, right as position + 6D rotation, on frames frame0 + i at FPS.
"""

from __future__ import annotations

import numpy as np

from kata.audio.beats import f_measure
from kata.motion.geometry import blade, rot6_to_matrix
from kata.motion.prepare import FPS, HIT, OFF, TRACK

TIMING_TOL_S = 0.03          # Phase 3's timing agreement tolerance (baseline TIMING_TOL_S)
AGREE_TOL_FRAMES = 2         # ±40 ms (§7.2.3)
MIN_HIT_SPEED = 1.0          # m/s
QUANTITIES = ("tip_speed_at_hits", "peak_speed_per_swing", "roll_range", "hand_head_dist", "head_disp_1s")


def timing_f(hit_times, map_times) -> float:
    """Phase 3's measure: F at ±30 ms between unique map note times and unique hit times."""
    return f_measure(np.unique(np.asarray(map_times, np.float64)), np.unique(np.asarray(hit_times, np.float64)),
                     TIMING_TOL_S)


def tips(motion: np.ndarray) -> np.ndarray:
    """(T, 27) -> saber tips (T, 2, 3)."""
    out = []
    for part in ("left", "right"):
        s = TRACK[part].start
        out.append(blade(motion[:, s:s + 3], rot6_to_matrix(motion[:, s + 3:s + 9]))[1])
    return np.stack(out, 1)


def tip_velocity(motion: np.ndarray) -> np.ndarray:
    """(T, 2, 3) in m/s, central differences."""
    return np.gradient(tips(motion), axis=0) * FPS


def _hit_frames(hits, frame0: int, n: int) -> np.ndarray:
    f = np.rint(np.asarray(hits, np.float64) * FPS - frame0).astype(np.int64)
    return f[(f >= 0) & (f < n)]


def quantities(motion: np.ndarray, hits: tuple, frame0: int = 0, valid: np.ndarray | None = None) -> dict:
    """The five plausibility distributions of §7.2.2 for one track. hits: (left times, right times)."""
    valid = np.ones(len(motion), bool) if valid is None else np.asarray(valid, bool)
    speed = np.linalg.norm(tip_velocity(motion), axis=-1)
    speed[~valid] = np.nan
    at_hits, peaks, rolls = [], [], []
    for h, part in enumerate(("left", "right")):
        f = np.sort(_hit_frames(hits[h], frame0, len(motion)))
        at_hits.append(speed[f, h])
        segs = [speed[a:b + 1, h] for a, b in zip(f[:-1], f[1:]) if b > a]
        peaks += [np.nanmax(g) for g in segs if np.isfinite(g).any()]
        s = TRACK[part].start
        rot = rot6_to_matrix(motion[valid, s + 3:s + 9])
        roll = np.degrees(np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0])))
        rolls.append(np.percentile(roll, 95) - np.percentile(roll, 5) if len(roll) else np.nan)
    head = motion[:, 0:3]
    reach = np.concatenate([np.linalg.norm(motion[valid, s:s + 3] - head[valid], axis=-1)[::5]
                            for s in (TRACK["left"].start, TRACK["right"].start)])
    ok = valid[:-FPS] & valid[FPS:] if len(motion) > FPS else np.zeros(0, bool)
    disp = np.linalg.norm(head[FPS:] - head[:-FPS], axis=-1)[ok][::25] if len(motion) > FPS else np.zeros(0)
    out = {"tip_speed_at_hits": np.concatenate(at_hits), "peak_speed_per_swing": np.asarray(peaks),
           "roll_range": np.asarray(rolls), "hand_head_dist": reach, "head_disp_1s": disp}
    return {k: v[np.isfinite(v)] for k, v in out.items()}


def agreement(motion: np.ndarray, hits: tuple, frame0: int = 0) -> tuple[int, int]:
    """(hits within ±40 ms of a local tip-speed maximum and moving >= 1 m/s, all hits) (§7.2.3)."""
    speed = np.linalg.norm(tip_velocity(motion), axis=-1)
    good = total = 0
    for h in (0, 1):
        s = speed[:, h]
        is_max = np.zeros(len(s), bool)
        is_max[1:-1] = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:])
        for f in _hit_frames(hits[h], frame0, len(motion)):
            lo, hi = max(f - AGREE_TOL_FRAMES, 0), min(f + AGREE_TOL_FRAMES + 1, len(s))
            good += bool(is_max[lo:hi].any() and s[f] >= MIN_HIT_SPEED)
            total += 1
    return good, total


def tip_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean saber-tip distance (m) between two tracks on the same frames (§7.2.5); no alignment:
    real players on one map are already aligned by its notes."""
    n = min(len(a), len(b))
    return float(np.linalg.norm(tips(a[:n]) - tips(b[:n]), axis=-1).mean())


def velocity_jumps(motion: np.ndarray, frames=None) -> np.ndarray:
    """|Δ tip velocity| (m/s, worse hand) between frame f - 1 and f, for every frame or the given ones."""
    v = tip_velocity(motion)
    jump = np.linalg.norm(np.diff(v, axis=0), axis=-1).max(-1)       # jump[f - 1] is between f - 1 and f
    if frames is None:
        return jump
    f = np.asarray(frames, np.int64)
    return jump[f[(f >= 1) & (f <= len(jump))] - 1]


def real_hits(raw: np.ndarray, frame0: int) -> tuple[np.ndarray, np.ndarray]:
    """Hit times per hand from a raw track's flag and offset channels."""
    out = []
    for h in (0, 1):
        rows = np.flatnonzero(raw[:, HIT[h]] > 0)
        out.append((frame0 + rows + raw[rows, OFF[h]]) / FPS)
    return out[0], out[1]


# --------------------------------------------------------------------------- quick validation (training)

def quick_windows(sampler, n: int) -> list:
    """Up to n val difficulties, evenly spread: their first replay's middle window (if fully valid)."""
    firsts = [diff[0] for _, diffs in sampler.songs for diff in diffs]
    if not firsts:
        return []
    picks = [firsts[i] for i in sorted(set(np.linspace(0, len(firsts) - 1, min(n, len(firsts))).round().astype(int)))]
    out = []
    for t in picks:
        start = (t.n_frames - sampler.window) // 2
        if sampler.motion.valid(t.score_id)[start:start + sampler.window].all():
            out.append((t, start))
    return out


def quick_eval(model, diffusion, sampler, windows: list, guidance: float, steps: int, device: str,
               chunk: int = 16) -> dict:
    """Generate each window once (seeded by its index) and score hits against the replay's own hits:
    median timing F (±30 ms) and median generated/real hit-count ratio."""
    import torch
    from kata.motion.dataset import decode_hits
    fs, ratios = [], []
    for c0 in range(0, len(windows), chunk):
        part = windows[c0:c0 + chunk]
        ex = [sampler.example(t, s) for t, s in part]
        gens = [torch.Generator().manual_seed(c0 + i) for i in range(len(part))]
        x_shape = ex[0]["x"].shape
        with torch.inference_mode():
            tokens = model.encode_audio(torch.from_numpy(np.stack([e["audio"] for e in ex])).to(device))
            x = diffusion.ddim(model, tokens, torch.tensor([e["label"] for e in ex], device=device),
                               torch.tensor([e["nps"] for e in ex], dtype=torch.float32, device=device),
                               lambda: torch.stack([torch.randn(x_shape, generator=g) for g in gens]).to(device),
                               steps, guidance)
        x = x.float().cpu().numpy()
        for (t, s), xi in zip(part, x):
            g0 = t.frame0 + s
            gen = np.concatenate([decode_hits(xi[:, HIT[h]], xi[:, OFF[h]], g0) for h in (0, 1)])
            raw = np.asarray(sampler.motion.load(t.score_id)[s:s + sampler.window], np.float32)
            real = np.concatenate(real_hits(raw, g0))
            if len(real):
                fs.append(timing_f(gen, real))
                ratios.append(len(np.unique(np.round(gen, 3))) / len(np.unique(real)))
    return {"val_timing_f": float(np.median(fs)) if fs else None,
            "val_rate_ratio": float(np.median(ratios)) if ratios else None, "n": len(fs)}
