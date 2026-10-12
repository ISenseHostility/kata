"""Replays -> 50 fps motion tracks (docs/phases/phase-4-motion.md §3). Pure numpy.

A track is a (T, 31) float32 array on the grid t = (frame0 + i) / FPS, plus a validity mask.
Channels: head, left saber, right saber as position (3) + 6D rotation (6) each, then per hand a
hit flag and the hit's sub-frame offset (§3.3, §3.4).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kata.data.align import align
from kata.data.bsor import EVENT_BAD, EVENT_BOMB, EVENT_GOOD, Replay, decode_note_id
from kata.data.pipeline import ALIGN_MAX_P95_MS, ALIGN_MIN_MATCH_RATE
from kata.motion.geometry import (MIRROR_CUT, blade, interpolate, matrix_to_rot6, monotonic,
                                  quat_to_matrix, reflect_x, resample, swept_plane_distance)

MOTION_VERSION = "v2"
FPS = 50
MAX_GAP_S = 0.1
BLADE_MAX_M = 0.02          # median cut point to swept blade plane; pilot: all replays <= 0.012 (spec §3.5)
SWEEP_S = 1 / 90            # the swept plane runs through the tip this long before the cut
MIN_CUTS = 10
HEAD_Y_REF_M = 1.60         # heights are scaled to this median head height (pilot median 1.61 m; spec §3.2)
PARTS = ("head", "left", "right")
N_CHANNELS = 31
MOTION_DIM = 27
TRACK = {"head": slice(0, 9), "left": slice(9, 18), "right": slice(18, 27)}
HIT = (27, 28)              # left, right
OFF = (29, 30)
CHANNELS = tuple(f"{p}_{c}" for p in PARTS for c in ("px", "py", "pz", "r0", "r1", "r2", "r3", "r4", "r5")) + (
    "left_hit", "right_hit", "left_off", "right_off")


def motion_config() -> dict:
    """Every parameter of §3; pinned in motion/{MOTION_VERSION}/config.json."""
    from kata.motion import geometry
    return {"version": MOTION_VERSION, "fps": FPS, "max_gap_s": MAX_GAP_S, "blade_max_m": BLADE_MAX_M,
            "blade_length_m": geometry.BLADE_LENGTH_M, "blade_axis": geometry.BLADE_AXIS,
            "blade_check": "median cut point to swept blade plane", "sweep_s": SWEEP_S,
            "min_cuts": MIN_CUTS, "channels": list(CHANNELS), "centre": "median head x,z",
            "height": "y scaled by head_y_ref_m / median head y", "head_y_ref_m": HEAD_Y_REF_M,
            "hit_time": "map time of the matched note", "align_min_match_rate": ALIGN_MIN_MATCH_RATE,
            "align_max_p95_ms": ALIGN_MAX_P95_MS}


class Rejected(Exception):
    """A replay that cannot become a track; the message is the reject reason."""


@dataclass
class Prepared:
    frame0: int
    data: np.ndarray            # (T, 31) float32
    valid: np.ndarray           # (T,) bool
    hit_times: np.ndarray       # unique map times of all hits, both hands (s)
    info: dict = field(default_factory=dict)


def mirror_note_id(note_id: int) -> int:
    """The note id the un-mirrored map would give: lane 3 - x, colours swapped, cut mirrored."""
    scoring, x, y, color, cut = decode_note_id(note_id)
    return scoring * 10000 + (3 - x) * 1000 + y * 100 + (1 - color) * 10 + int(MIRROR_CUT[cut])


def _poses(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = frames["t"].astype(np.float64)
    pos = np.stack([np.stack([frames[f"{p}_p{c}"] for c in "xyz"], -1) for p in PARTS], 1).astype(np.float64)
    quat = np.stack([np.stack([frames[f"{p}_q{c}"] for c in "xyzw"], -1) for p in PARTS], 1).astype(np.float64)
    return t, pos, quat


def prepare_replay(replay: Replay, notes: dict[str, np.ndarray], left_handed: bool) -> Prepared:
    """One replay -> a track. notes: the difficulty's x, y, color, cut_dir, time_s arrays.

    Raises Rejected(reason) when alignment, the frame check or the blade check fails.
    """
    events = replay.notes
    if left_handed:
        events = [e if e["event_type"] == EVENT_BOMB else {**e, "note_id": mirror_note_id(e["note_id"])}
                  for e in events]
    res = align(notes["x"], notes["y"], notes["color"], notes["cut_dir"], notes["time_s"], events)
    if res.match_rate < ALIGN_MIN_MATCH_RATE:
        raise Rejected("align_match_rate")
    if not res.residual_quantile(0.95) <= ALIGN_MAX_P95_MS:
        raise Rejected("align_residual")

    frames = replay.frames[monotonic(replay.frames["t"])]
    if len(frames) < 2:
        raise Rejected("too_few_frames")
    t, pos, quat = _poses(frames)

    # Frame and blade checks (§3.5) on every matched cut inside the recording.
    cuts = [i for i, e in enumerate(events) if e["event_type"] in (EVENT_GOOD, EVENT_BAD)
            and res.matched_idx[i] >= 0 and t[0] <= e["event_time"] <= t[-1]]
    if len(cuts) < MIN_CUTS:
        raise Rejected("too_few_cuts")
    ct = np.array([events[i]["event_time"] for i in cuts], dtype=np.float64)
    cp = np.array([[events[i][f"cut_point_{c}"] for c in "xyz"] for i in cuts], dtype=np.float64)
    idx = res.matched_idx[cuts]
    p_at, q_at, _ = interpolate(t, pos, quat, ct)
    p_prev, q_prev, _ = interpolate(t, pos, quat, ct - SWEEP_S)
    rot, rot_prev = quat_to_matrix(q_at), quat_to_matrix(q_prev)
    dist = np.stack([swept_plane_distance(cp, *blade(p_at[:, k], rot[:, k]), blade(p_prev[:, k], rot_prev[:, k])[1])
                     for k in (1, 2)], 1)
    hand = dist.argmin(1)                                   # 0 = left track, 1 = right track
    color = np.asarray(notes["color"])[idx]
    lane = np.asarray(notes["x"], dtype=np.float64)[idx] - 1.5
    hands_swapped = bool((hand != color).mean() > 0.5)
    world_mirrored = bool(lane.std() > 0 and cp[:, 0].std() > 0 and np.corrcoef(cp[:, 0], lane)[0, 1] < 0)
    blade_median = float(np.median(dist.min(1)))
    layer = np.asarray(notes["y"])[idx]
    hand_y = {f"hand_y_l{k}": (float(p_at[layer == k, 1 + hand[layer == k], 1].mean()) if (layer == k).any()
                               else None) for k in range(3)}
    if not left_handed and (hands_swapped or world_mirrored):
        raise Rejected("frame_check")
    if blade_median > BLADE_MAX_M:
        raise Rejected("blade_check")
    if world_mirrored:
        pos, quat = reflect_x(pos, quat)
    if hands_swapped:
        pos, quat, hand = pos[:, [0, 2, 1]], quat[:, [0, 2, 1]], 1 - hand

    frame0, p, q, valid = resample(t, pos, quat, FPS, MAX_GAP_S)
    if not valid.any():
        raise Rejected("no_valid_frames")
    p[..., 0] -= np.median(p[valid, 0, 0])
    p[..., 2] -= np.median(p[valid, 0, 2])
    head_y = float(np.median(p[valid, 0, 1]))
    p[..., 1] *= HEAD_Y_REF_M / head_y
    data = np.zeros((len(p), N_CHANNELS), dtype=np.float32)
    for k, part in enumerate(PARTS):
        s = TRACK[part].start
        data[:, s:s + 3] = p[:, k]
        data[:, s + 3:s + 9] = matrix_to_rot6(quat_to_matrix(q[:, k]))

    hit_t = np.asarray(notes["time_s"], dtype=np.float64)[idx]
    f = hit_t * FPS - frame0
    fi = np.rint(f).astype(np.int64)
    inside = (fi >= 0) & (fi < len(data))
    for h in (0, 1):
        sel = inside & (hand == h)
        frames_h, inv = np.unique(fi[sel], return_inverse=True)
        off = np.bincount(inv, weights=(f - fi)[sel], minlength=len(frames_h)) / np.bincount(inv)
        data[frames_h, HIT[h]] = 1.0
        data[frames_h, OFF[h]] = off
    info = {"n_frames": len(data), "n_valid": int(valid.sum()),
            "n_hits_left": int((data[:, HIT[0]] > 0).sum()), "n_hits_right": int((data[:, HIT[1]] > 0).sum()),
            "mirrored": left_handed, "world_mirrored": world_mirrored, "hands_swapped": hands_swapped,
            "blade_median_m": blade_median, "head_y_median_m": head_y, **hand_y}
    return Prepared(frame0, data, valid, np.unique(hit_t[inside]), info)


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


class MotionStore:
    """motion/{MOTION_VERSION}/: one fp16 (T, 31) array and one bool mask per replay (§3.7)."""

    def __init__(self, root: Path):
        self.version = MOTION_VERSION
        self.dir = Path(root) / self.version

    @property
    def config_path(self) -> Path:
        return self.dir / "config.json"

    @property
    def stats_path(self) -> Path:
        return self.dir / "stats.json"

    def open_for_write(self, config: dict) -> None:
        """Pin this version dir's config; a different config must bump MOTION_VERSION."""
        self.dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(config, sort_keys=True, indent=1)
        if self.config_path.exists():
            if json.loads(self.config_path.read_text()) != json.loads(text):
                raise ValueError(f"motion config differs from {self.config_path}; bump MOTION_VERSION")
            return
        self.config_path.write_text(text)

    def write(self, score_id: str, data: np.ndarray, valid: np.ndarray) -> None:
        _atomic_save(self.dir / f"{score_id}.valid.npy", np.asarray(valid, dtype=bool))
        _atomic_save(self.dir / f"{score_id}.npy", np.asarray(data, dtype=np.float16))   # last: marks complete

    def has(self, score_id: str) -> bool:
        return (self.dir / f"{score_id}.npy").exists()

    def load(self, score_id: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.dir / f"{score_id}.npy", mmap_mode="r" if mmap else None)

    def valid(self, score_id: str) -> np.ndarray:
        return np.load(self.dir / f"{score_id}.valid.npy")


def channel_stats(store: MotionStore, score_ids) -> dict:
    """Mean and std of the 27 motion channels over the valid frames of the given tracks."""
    s1 = np.zeros(MOTION_DIM)
    s2 = np.zeros(MOTION_DIM)
    n = 0
    for sid in score_ids:
        a = np.asarray(store.load(sid), dtype=np.float64)[store.valid(sid), :MOTION_DIM]
        s1 += a.sum(0)
        s2 += np.square(a).sum(0)
        n += len(a)
    mean = s1 / max(n, 1)
    std = np.sqrt(np.maximum(s2 / max(n, 1) - mean ** 2, 0.0))
    return {"split": "train", "n_tracks": len(score_ids), "n_frames": int(n), "channels": list(CHANNELS[:MOTION_DIM]),
            "mean": mean.tolist(), "std": np.maximum(std, 1e-3).tolist()}
