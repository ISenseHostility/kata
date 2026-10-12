"""Synthetic maps and replays for the Phase 4 tests: alternating down-cuts, red in lane 1 and blue in
lane 2, with sabers whose blades pass exactly through each recorded cut point."""
import numpy as np

from kata.data.bsor import EVENT_GOOD, FRAME_DTYPE, Replay
from kata.motion.prepare import mirror_note_id

HAND_X = (-0.3, 0.3)            # left, right saber origin x (m)


def synth_notes(n: int = 40, t0: float = 2.0, step: float = 0.507) -> dict[str, np.ndarray]:
    k = np.arange(n)
    color = k % 2
    return {"x": np.where(color == 0, 1, 2), "y": (k // 2) % 3, "color": color,
            "cut_dir": np.ones(n, dtype=np.int64), "time_s": t0 + step * k}


def hand_y(t):
    return 1.0 + 0.2 * np.sin(2 * np.pi * np.asarray(t, dtype=np.float64))


def synth_replay(notes: dict, mode: str = "right", fps: float = 90.0, seed: int = 0,
                 head_xz: tuple[float, float] = (0.35, 0.3), gap: tuple[float, float] | None = None,
                 spawn_shift_s: float = 0.0, saber_dy: float = 0.0) -> Replay:
    """mode: 'right' (a normal replay); 'mirror_all' (left-handed: ids, world and hands mirrored);
    'mirror_ids' (left-handed: only the note ids mirrored); 'right_mirrored_world' (a broken
    right-handed replay whose frames are mirrored)."""
    rng = np.random.default_rng(seed)
    t = np.sort(np.arange(0.0, notes["time_s"].max() + 1.5, 1 / fps) + rng.uniform(-0.002, 0.002))
    if gap is not None:
        t = t[(t < gap[0]) | (t > gap[1])]
    t = t[t >= 0]
    f = np.zeros(len(t), dtype=FRAME_DTYPE)
    f["t"] = t
    f["fps"] = int(fps)
    f["head_px"], f["head_py"], f["head_pz"] = head_xz[0], 1.6, head_xz[1]
    for name, x in zip(("left", "right"), HAND_X):
        f[f"{name}_px"], f[f"{name}_py"], f[f"{name}_pz"] = x, hand_y(t) + saber_dy, 0.0
    for name in ("head", "left", "right"):
        f[f"{name}_qw"] = 1.0
    mirror_world = mode in ("mirror_all", "right_mirrored_world")
    if mirror_world:
        for name in ("head", "left", "right"):
            f[f"{name}_px"] *= -1
        left = {c: f[f"left_{c}"].copy() for c in ("px", "py", "pz", "qx", "qy", "qz", "qw")}
        for c in left:
            f[f"left_{c}"] = f[f"right_{c}"]
            f[f"right_{c}"] = left[c]
    events = []
    for x, y, c, d, ts in zip(notes["x"], notes["y"], notes["color"], notes["cut_dir"], notes["time_s"]):
        nid = 3 * 10000 + int(x) * 1000 + int(y) * 100 + int(c) * 10 + int(d)
        if mode in ("mirror_all", "mirror_ids"):
            nid = mirror_note_id(nid)
        et = float(ts) + 0.001
        cx = HAND_X[int(c)] * (-1 if mirror_world else 1)
        events.append({"note_id": nid, "event_time": et, "spawn_time": float(ts) + spawn_shift_s,
                       "event_type": EVENT_GOOD, "saber_type": int(c), "cut_point_x": cx,
                       "cut_point_y": float(hand_y(et)), "cut_point_z": 0.5})
    info = {"leftHanded": mode in ("mirror_all", "mirror_ids"), "height": 1.75}
    return Replay(info, f, events, [], [(1.75, 0.0)], [])
