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
                 spawn_shift_s: float = 0.0, saber_dx: float = 0.0, cut_z: float = 0.5,
                 scale_y: float = 1.0) -> Replay:
    """mode: 'right' (a normal replay); 'mirror_all' (left-handed: ids, world and hands mirrored);
    'mirror_ids' (left-handed: only the note ids mirrored); 'right_mirrored_world' (a broken
    right-handed replay whose frames are mirrored). saber_dx moves both sabers outwards, off the
    cut points' plane; cut_z puts the cut points along the swept plane (beyond the 1 m tip when
    > 1); scale_y scales every height, as for a taller player."""
    rng = np.random.default_rng(seed)
    t = np.sort(np.arange(0.0, notes["time_s"].max() + 1.5, 1 / fps) + rng.uniform(-0.002, 0.002))
    if gap is not None:
        t = t[(t < gap[0]) | (t > gap[1])]
    t = t[t >= 0]
    f = np.zeros(len(t), dtype=FRAME_DTYPE)
    f["t"] = t
    f["fps"] = int(fps)
    f["head_px"], f["head_py"], f["head_pz"] = head_xz[0], 1.6 * scale_y, head_xz[1]
    for name, x in zip(("left", "right"), HAND_X):
        f[f"{name}_px"], f[f"{name}_py"], f[f"{name}_pz"] = x + np.sign(x) * saber_dx, hand_y(t) * scale_y, 0.0
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
                       "cut_point_y": float(hand_y(et)) * scale_y, "cut_point_z": cut_z})
    info = {"leftHanded": mode in ("mirror_all", "mirror_ids"), "height": 1.75}
    return Replay(info, f, events, [], [(1.75, 0.0)], [])


SONGS = {"h1": ("a" * 64, "train"), "h2": ("b" * 64, "val")}


def write_features(p, sha: str, duration_s: float) -> None:
    """Phase 2 features for one song: mel bin 0 carries the 100 fps frame index (for alignment
    tests), stems are all ones, and beats fall every 0.5 s with a downbeat every fourth."""
    from kata.audio.store import FeatureStore
    T = int(duration_s * 100) + 1
    mel = np.zeros((T, 128), np.float32)
    mel[:, 0] = np.arange(T) % 2000
    curves = np.zeros((T, 12), np.float32)
    curves[:, 5:10] = -20.0
    beats = np.arange(0.5, duration_s - 0.5, 0.5)
    FeatureStore(p.features).write(sha, {"mel": mel, "stem_mel": np.ones((T, 4, 64), np.float32),
                                         "chroma": np.zeros((T, 12)), "curves": curves}, beats, beats[::4])


def write_audio_stats(p) -> None:
    """Identity normalisation (mean 0, std 1) in Phase 2's stats.json format."""
    import json
    from kata.audio.store import FeatureStore
    shapes = {"mel": [128], "stem_mel": [4, 64], "chroma": [12], "curves": [12]}
    arrays = {n: {"shape": s, "mean": np.zeros(s).tolist(), "std": np.ones(s).tolist(), "n_frames": 1}
              for n, s in shapes.items()}
    store = FeatureStore(p.features)
    store.dir.mkdir(parents=True, exist_ok=True)
    store.stats_path.write_text(json.dumps({"split": "train", "n_songs": 1, "songs": [], "arrays": arrays}))


def write_root(p, n_notes: int = 40) -> dict:
    """Two Tier A maps (train h1, val h2) with Expert notes; four replays on h1 (one normal, one
    left-handed, one hash mismatch, one with a broken saber pose) and one on h2. Features for both
    songs, and the tables prepare, the sampler and evaluation read."""
    import polars as pl
    from kata.data import bsor
    notes = synth_notes(n_notes)
    duration = float(notes["time_s"].max() + 3.0)
    pl.concat([pl.DataFrame({"hash": h, "characteristic": "Standard", "difficulty": "Expert", **notes})
               for h in SONGS]).write_parquet(p.processed / "notes.parquet")
    pl.DataFrame({"hash": list(SONGS), "split": [s for _, s in SONGS.values()],
                  "audio_sha256": [a for a, _ in SONGS.values()], "audio_file": ["song.egg"] * 2,
                  "pilot": [True, True]}).write_parquet(
        p.processed / "maps.parquet")
    pl.DataFrame({"hash": ["h1", "h2", "h3"], "characteristic": ["Standard"] * 3, "difficulty": ["Expert"] * 3,
                  "tier": ["A", "A", "B"], "parse_status": ["ok"] * 3}).write_parquet(
        p.processed / "difficulties.parquet")
    pl.DataFrame({"hash": ["h1"] * 4 + ["h2"] * 2, "src": ["drums"] * 3 + ["mix", "drums", "mix"],
                  "difficulty": ["Expert", "Hard", "Normal", "Expert", "Expert", "Expert"],
                  "lag_ms": [10.0, 12.0, 190.0, 30.0, 20.0, 0.0]}).write_parquet(p.processed / "map_audio_sync.parquet")
    pl.DataFrame({"audio_sha256": [a for a, _ in SONGS.values()], "status": ["ok", "ok"],
                  "duration_s": [duration, duration]}).write_parquet(p.processed / "audio_features.parquet")
    for sha, _ in SONGS.values():
        write_features(p, sha, duration)
    write_audio_stats(p)
    replays = [  # score_id, hash, mode, status, reject_reason, saber_dx
        ("1", "h1", "right", "ok", None, 0.0),
        ("2", "h1", "mirror_all", "rejected", "left_handed,align_match_rate", 0.0),
        ("3", "h1", "right", "rejected", "hash_mismatch", 0.0),
        ("4", "h1", "right", "ok", None, 0.05),
        ("5", "h2", "right", "ok", None, 0.0),
    ]
    for sid, h, mode, _status, _reason, dx in replays:
        (p.replays / h).mkdir(parents=True, exist_ok=True)
        r = synth_replay(notes, mode=mode, saber_dx=dx, seed=int(sid))
        (p.replays / h / f"{sid}.bsor").write_bytes(bsor.write(r))
    pl.DataFrame({"score_id": [r[0] for r in replays], "hash": [r[1] for r in replays],
                  "characteristic": ["Standard"] * 5, "difficulty": ["Expert"] * 5,
                  "status": [r[3] for r in replays], "reject_reason": [r[4] for r in replays],
                  "left_handed": [r[2] != "right" for r in replays], "height_median": [1.75] * 5}
                 ).write_parquet(p.processed / "replays.parquet")
    return notes
