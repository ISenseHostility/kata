# Phase 4 — Motion Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `kata.motion`: replays → 50 fps motion tracks with per-hand hits, an audio-conditioned diffusion transformer that generates whole-song head + saber motion and hits, the training loop with exact overnight resume, evaluation against real replays and the Phase 3 baseline, and a local 3D viewer. Then pilot it on the replays already on disk, fetch Tier A replays when the user says go, train, and run acceptance.

**Architecture:** `geometry` (rotations, resampling, mirroring) → `prepare` (one replay → one track, checks, store) → `dataset` (song-first windows, audio context, normalisation; a batch is a pure function of seed and step) → `model` (audio encoder + adaLN denoiser with rotary attention) → `diffusion` (cosine schedule, x0 losses, DDIM with guidance, overlap inpainting) → `train` (runs, checkpoints, exact resume, time limit) → `sample` (`generate()`, split sampling) → `evaluate` (§7.2 metrics, criteria) → `viewer` / `report`. Stages follow the Phase 1–3 conventions (`_with_state`, deterministic table rebuilds, config pinned per version directory).

**Tech Stack:** Python 3.13 (uv), numpy, polars, duckdb, typer, pytest; torch 2.11 from the existing `audio` extra. No new dependencies. The viewer page loads three.js 0.160.0 from jsDelivr.

**Spec:** `docs/phases/phase-4-motion.md` (approved 2026-10-12; prototype corrections in §17). Read §3 (tracks), §4 (audio), §5 (model), §6 (training), §7 (evaluation) and §8 (acceptance) before starting.

**Provenance:** every code block in Tasks 1–10 ran in a throwaway prototype on 2026-10-12 (a scratch git worktree, CPU only): 232 tests passing in total (45 of them Phase 4), `ruff check --select F` clean, the viewer page rendered in a browser, and one full-size (46.4M-parameter) training step run on CPU. No real replays were used: the GPU machine and `E:\kata-data` were not available. Tasks 11–14 run on the training machine. Copy the code as written; if something fails, it is an environment difference worth reporting, not a placeholder.

## Global Constraints

- Install with `uv sync --extra audio` (torch). `kata motion prepare` is numpy-only; every other motion stage imports torch inside the stage, like Phase 2. `kata/cli.py` imports Phase 4 modules inside each command.
- Track parameters are pinned per version: `MOTION_VERSION = "v1"`; `motion/v1/config.json` is written once and a different config raises `ValueError("... bump MOTION_VERSION")`.
- Grid: `FPS = 50`, frame `i` of a track is `t = (frame0 + i) / 50` s on the song clock. Gaps over `MAX_GAP_S = 0.1` s mark frames invalid; windows never contain invalid frames.
- Axes as the game stores them: +y up, +z towards the notes, +x right, metres, quaternions (x, y, z, w). Mirror: positions (−x, y, z), quaternions (qx, −qy, −qz, qw), hands swapped. Blade: 1 m along saber-local +z.
- Track channels (31): head, left, right × (position 3 + 6D rotation 6), then `left_hit, right_hit, left_off, right_off`. Hits sit at the matched note's **map** time.
- Window 640 frames (12.8 s), audio context 100 frames each side, audio 820 values per frame (2 × 408 Phase 2 values + beat/bar phase sin/cos). Model: d = 448, 8 heads, 4 encoder + 8 decoder layers, dropout 0.1 → 46.4M parameters.
- Diffusion: x0 prediction, cosine schedule, 1,000 steps, DDIM 50 steps, label and NPS each dropped with p = 0.1; guidance from {1, 1.5, 2, 3, 4} tuned on val.
- Training: batch 64, AdamW lr 2e-4 (β 0.9/0.99, wd 0.01), 2k warmup, cosine to 10 %, grad clip 1.0, EMA 0.999, checkpoint every 30 min and at exit. **Every run fits one night** (`--hours 12`); a longer run resumes the next night.
- Splits come from Phase 1 unchanged. Normalisation stats from **train only**. Checkpoints and guidance chosen on **val**. **Test is evaluated once** per run; `evaluate --split test` refuses a second time without `--force` and the user's approval.
- Replays are not fetched until the pilot is done **and the user says go** (Task 12). Tier A only (`crawl-leaderboards --tier A`).
- No real replays, maps, audio or features are committed (Phase 1 D6). Test fixtures are synthetic (`tests/motion_synth.py`); real-data checks read `KATA_DATA` and are run by hand. Generated motion and viewer pages stay local; the viewer never embeds audio.
- Every stage that downloads checks the 50 GB free-space floor (`kata.data.pipeline._check_disk`).
- Commit after every task. End each commit message with the attribution line from the session's system reminder (shown in each commit step).

## Review Focus

1. **A training run interrupted overnight and resumed the next night.** Expected: bit-identical weights to an uninterrupted run. The prototype diverged one step after resume because creating a `DataLoader` iterator draws from the global RNG; the iterator is now created before the RNG is restored. Pinned by `test_resumed_training_is_identical_to_straight_training` (Task 8).
2. **Left-handed replays, whichever way the game mirrors them.** Expected: the recovered track equals the right-handed one; a right-handed replay with mirrored frames is rejected, not silently flipped. Pinned by `test_left_handed_replays_are_recovered_whichever_way_the_game_mirrors` and `test_broken_replays_are_rejected_with_a_reason` (Task 2).
3. **Blind mode giving away the real clip.** Expected: nothing on the page distinguishes the clips before Reveal. The prototype showed map notes only on the real clip; both clips get them now. Pinned by the notes assertion in `test_viewer_pages_and_report` (Task 10).
4. **Re-running a stage after a crash.** Expected: `prepare` rebuilds an identical table, `sample` skips finished files, `evaluate` replaces its rows instead of duplicating them, and test is never evaluated twice by accident. Pinned by `test_prepare_stage_tracks_rejects_stats_and_idempotency` (Task 3) and `test_sampling_resumes_evaluation_replaces_and_generate_works` (Task 9).
5. **Short songs, short tracks and tracking gaps.** Expected: songs shorter than a window generate in one window; tracks shorter than a window or without features never reach training; tracking gaps are flagged, never interpolated into windows. Pinned by `test_long_songs_are_stitched_on_the_frame_grid` (Task 6), `test_tracks_shorter_than_a_window_or_without_features_are_left_out` (Task 4) and `test_tracking_gaps_are_flagged_not_dropped` (Task 2).

## File Structure

| File | Responsibility |
|------|----------------|
| `src/kata/motion/__init__.py` | package docstring |
| `src/kata/motion/geometry.py` | quaternions, 6D rotations, slerp, resampling, mirroring, blade geometry (spec §3) |
| `src/kata/motion/prepare.py` | one replay → track: alignment, frame/blade checks, left-handed recovery, hits; `MotionStore`; train stats (§3) |
| `src/kata/motion/dataset.py` | `TrackRef`, `Normaliser`, hit pulses and decoding, beat phase, mirror augmentation, `WindowSampler` (§3.6, §4, §6.2) |
| `src/kata/motion/model.py` | `ModelConfig`, rotary attention, audio encoder, adaLN denoiser (§5.2–5.3) |
| `src/kata/motion/diffusion.py` | schedule, losses, DDIM + guidance + inpainting, `sample_long` (§5.4–5.6) |
| `src/kata/motion/evaluate.py` | §7.2 metrics, quick val check, split evaluation, acceptance criteria |
| `src/kata/motion/train.py` | `RunConfig`, `Run`, training loop, checkpoints, exact resume (§6.3–6.4) |
| `src/kata/motion/sample.py` | `MotionTrack`, `Generator`, `generate()`, split sampling (§5.7, §7.1) |
| `src/kata/motion/viewer.py` | self-contained three.js pages, blind mode (§7.3) |
| `src/kata/motion/report.py` | Phase 4 Markdown report |
| `src/kata/motion/pipeline.py` | stages: prepare, train, tune, sample, evaluate, final, viewer, report |
| `src/kata/config.py` (modify) | `Paths.motion`, `Paths.motion_runs` |
| `src/kata/data/pipeline.py` (modify) | `leaderboard_diffs(..., tier)`; `crawl-leaderboards --tier` (D6) |
| `src/kata/cli.py` (modify) | `kata data crawl-leaderboards --tier`, `kata motion …` |
| `tests/motion_synth.py` | synthetic maps, replays (normal, left-handed, broken), features and a whole data root |
| `tests/test_motion_*.py` | one test module per source module, plus `test_motion_pipeline.py` and `test_motion_cli.py` |

Task order follows the dependencies: 1 → 2 → 3 (tracks) → 4 (windows) → 5 → 6 (model, diffusion) → 7 (metrics, needed by training) → 8 (training) → 9 (sampling, evaluation, stages) → 10 (viewer, report, CLI) → 11 (pilot, checkpoint with the user) → 12 (replay fetch, on the user's go) → 13 (training nights) → 14 (acceptance).

---

### Task 1: Geometry

**Files:**
- Create: `src/kata/motion/__init__.py`, `src/kata/motion/geometry.py`
- Test: `tests/test_motion_geometry.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `MIRROR_CUT` (9-element array), `BLADE_LENGTH_M = 1.0`, `BLADE_AXIS = 2`; `quat_to_matrix(q) -> (...,3,3)`, `matrix_to_quat(m) -> (...,4)`, `matrix_to_rot6(m) -> (...,6)`, `rot6_to_matrix(r6) -> (...,3,3)`, `slerp(q0, q1, u)`, `monotonic(t) -> bool mask`, `interpolate(t, pos (N,K,3), quat (N,K,4), t_new) -> (pos, quat, gap)`, `resample(t, pos, quat, fps, max_gap_s) -> (frame0, pos, quat, valid)`, `reflect_x(pos, quat)`, `blade(pos, rot) -> (origin, tip)`, `point_segment_distance(p, a, b)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_motion_geometry.py`:

```python
"""Rotation conversions, resampling and mirroring (spec §3)."""
import numpy as np

from kata.motion import geometry as g


def _random_quats(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_quat_matrix_round_trip_and_rotation_properties():
    q = _random_quats(200)
    m = g.quat_to_matrix(q)
    assert np.allclose(m @ np.swapaxes(m, -1, -2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(m), 1.0)
    back = g.matrix_to_quat(m)
    assert np.allclose(np.abs((back * q).sum(-1)), 1.0)          # same rotation up to sign
    # 90 degrees about +y turns +z into +x
    q90 = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    assert np.allclose(g.quat_to_matrix(q90) @ [0, 0, 1], [1, 0, 0])


def test_rot6_round_trip_and_gram_schmidt_repairs_noise():
    m = g.quat_to_matrix(_random_quats(50))
    assert np.allclose(g.rot6_to_matrix(g.matrix_to_rot6(m)), m)
    noisy = g.matrix_to_rot6(m) + np.random.default_rng(1).normal(scale=0.05, size=(50, 6))
    r = g.rot6_to_matrix(noisy)
    assert np.allclose(r @ np.swapaxes(r, -1, -2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(r), 1.0)


def test_slerp_takes_the_short_way_and_hits_the_midpoint():
    a = np.array([0.0, 0.0, 0.0, 1.0])
    b = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])     # 90° about y
    mid = g.slerp(a, -b, 0.5)                                          # -b is the same rotation
    assert np.allclose(g.quat_to_matrix(mid) @ [0, 0, 1], [np.sin(np.pi / 4), 0, np.cos(np.pi / 4)])
    assert np.allclose(g.slerp(a, a, 0.3), a)


def test_resample_grid_gaps_and_monotonic_cleanup():
    t = np.array([0.011, 0.02, 0.02, 0.015, 0.031, 0.042, 0.30, 0.311])   # repeat, jump back, 258 ms gap
    keep = g.monotonic(t)
    assert keep.tolist() == [True, True, False, False, True, True, True, True]
    t = t[keep]
    pos = np.zeros((len(t), 1, 3))
    pos[:, 0, 0] = t                                                     # x = t: linear in time
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1, 1))
    frame0, p, q, valid = g.resample(t, pos, quat, fps=50, max_gap_s=0.1)
    assert frame0 == 1                                                  # first grid point 0.02 s
    assert np.allclose(p[:, 0, 0], (frame0 + np.arange(len(p))) / 50)
    times = (frame0 + np.arange(len(p))) / 50
    assert valid.tolist() == [bool(not (0.042 <= x < 0.30)) for x in times]
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)


def test_reflection_is_the_mirror_of_the_rotation_matrix():
    q = _random_quats(20, seed=3)
    pos = np.random.default_rng(4).normal(size=(20, 3))
    p2, q2 = g.reflect_x(pos, q)
    mirror = np.diag([-1.0, 1.0, 1.0])
    assert np.allclose(g.quat_to_matrix(q2), mirror @ g.quat_to_matrix(q) @ mirror)
    assert np.allclose(p2, pos @ mirror)
    assert g.MIRROR_CUT[g.MIRROR_CUT].tolist() == list(range(9))      # an involution


def test_blade_and_point_segment_distance():
    origin, tip = g.blade(np.zeros(3), np.eye(3))
    assert np.allclose(tip, [0, 0, g.BLADE_LENGTH_M])
    p = np.array([[0.1, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.2, -0.3]])
    d = g.point_segment_distance(p, np.broadcast_to(origin, p.shape), np.broadcast_to(tip, p.shape))
    assert np.allclose(d, [0.1, 0.5, np.hypot(0.2, 0.3)])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_geometry.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'kata.motion'`.

- [ ] **Step 3: Implement** — `src/kata/motion/__init__.py`:

```python
"""Phase 4 motion model (docs/phases/phase-4-motion.md)."""
```

`src/kata/motion/geometry.py`:

```python
"""Rotations, resampling and mirroring for replay motion (docs/phases/phase-4-motion.md §3). Pure numpy.

Axes are the game's: +y up, +z towards the notes, +x right. Quaternions are (x, y, z, w).
"""

from __future__ import annotations

import numpy as np

# Mirror image of each cut direction under x -> -x: left <-> right; up, down and dot unchanged.
MIRROR_CUT = np.array([0, 1, 3, 2, 5, 4, 7, 6, 8])
BLADE_LENGTH_M = 1.0   # spec §3.5 [verify in pilot]
BLADE_AXIS = 2         # saber-local +z [verify in pilot]


def quat_to_matrix(q) -> np.ndarray:
    """(..., 4) xyzw -> (..., 3, 3); normalises first."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def matrix_to_quat(m) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) xyzw with w >= 0 (Shepperd's method, vectorised)."""
    m = np.asarray(m, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cand = np.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], 1)
    k = cand.argmax(1)
    q = np.empty((len(m), 4))
    for case in range(4):
        s = k == case
        if not s.any():
            continue
        a = m[s]
        if case == 0:
            r = np.sqrt(1 + tr[s]) * 2
            q[s] = np.stack([(a[:, 2, 1] - a[:, 1, 2]) / r, (a[:, 0, 2] - a[:, 2, 0]) / r,
                             (a[:, 1, 0] - a[:, 0, 1]) / r, r / 4], 1)
        elif case == 1:
            r = np.sqrt(1 + a[:, 0, 0] - a[:, 1, 1] - a[:, 2, 2]) * 2
            q[s] = np.stack([r / 4, (a[:, 0, 1] + a[:, 1, 0]) / r, (a[:, 0, 2] + a[:, 2, 0]) / r,
                             (a[:, 2, 1] - a[:, 1, 2]) / r], 1)
        elif case == 2:
            r = np.sqrt(1 + a[:, 1, 1] - a[:, 0, 0] - a[:, 2, 2]) * 2
            q[s] = np.stack([(a[:, 0, 1] + a[:, 1, 0]) / r, r / 4, (a[:, 1, 2] + a[:, 2, 1]) / r,
                             (a[:, 0, 2] - a[:, 2, 0]) / r], 1)
        else:
            r = np.sqrt(1 + a[:, 2, 2] - a[:, 0, 0] - a[:, 1, 1]) * 2
            q[s] = np.stack([(a[:, 0, 2] + a[:, 2, 0]) / r, (a[:, 1, 2] + a[:, 2, 1]) / r, r / 4,
                             (a[:, 1, 0] - a[:, 0, 1]) / r], 1)
    q *= np.where(q[:, 3:4] < 0, -1.0, 1.0)
    return q.reshape(shape + (4,))


def matrix_to_rot6(m) -> np.ndarray:
    """First two columns of the rotation matrix (Zhou et al. 2019): (..., 3, 3) -> (..., 6)."""
    m = np.asarray(m)
    return np.concatenate([m[..., :, 0], m[..., :, 1]], -1)


def rot6_to_matrix(r6) -> np.ndarray:
    """Gram-Schmidt back to a rotation matrix: (..., 6) -> (..., 3, 3)."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -1)


def slerp(q0, q1, u) -> np.ndarray:
    """Spherical interpolation between unit quaternions (..., 4) at fraction u (...)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)[..., None]
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)          # shortest path
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    small = sin < 1e-6
    w0 = np.where(small, 1 - u, np.sin((1 - u) * theta) / np.where(small, 1, sin))
    w1 = np.where(small, u, np.sin(u * theta) / np.where(small, 1, sin))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def monotonic(t) -> np.ndarray:
    """Mask keeping only frames later than every earlier frame (drops repeats and jumps back)."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return np.zeros(0, dtype=bool)
    return np.concatenate([[True], t[1:] > np.maximum.accumulate(t)[:-1]])


def interpolate(t, pos, quat, t_new) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear positions and slerped rotations at t_new.

    t: (N,) strictly increasing; pos: (N, K, 3); quat: (N, K, 4); t_new inside [t[0], t[-1]].
    Returns (pos (M, K, 3), quat (M, K, 4), gap (M,)) where gap is the raw-frame spacing used.
    """
    t = np.asarray(t, dtype=np.float64)
    t_new = np.asarray(t_new, dtype=np.float64)
    j = np.clip(np.searchsorted(t, t_new, side="right") - 1, 0, len(t) - 2)
    gap = t[j + 1] - t[j]
    u = ((t_new - t[j]) / gap).clip(0.0, 1.0)
    p = pos[j] + (pos[j + 1] - pos[j]) * u[:, None, None]
    q = slerp(quat[j], quat[j + 1], np.repeat(u[:, None], quat.shape[1], 1))
    return p, q, gap


def resample(t, pos, quat, fps: int, max_gap_s: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Resample onto the grid t = i / fps between the first and last raw frame.

    Returns (frame0, pos (T, K, 3), quat (T, K, 4), valid (T,)); valid is False where the raw
    frames around a grid point are more than max_gap_s apart.
    """
    t = np.asarray(t, dtype=np.float64)
    i0 = int(np.ceil(t[0] * fps - 1e-9))
    i1 = int(np.floor(t[-1] * fps + 1e-9))
    grid = np.arange(i0, i1 + 1) / fps
    p, q, gap = interpolate(t, pos, quat, grid)
    return i0, p, q, gap <= max_gap_s


def reflect_x(pos, quat) -> tuple[np.ndarray, np.ndarray]:
    """The mirror x -> -x: positions (-x, y, z); quaternions (qx, -qy, -qz, qw)."""
    return pos * np.array([-1.0, 1.0, 1.0]), quat * np.array([1.0, -1.0, -1.0, 1.0])


def blade(pos, rot) -> tuple[np.ndarray, np.ndarray]:
    """Saber origin and tip: tip = origin + BLADE_LENGTH_M along the saber-local BLADE_AXIS."""
    pos = np.asarray(pos, dtype=np.float64)
    return pos, pos + BLADE_LENGTH_M * np.asarray(rot)[..., :, BLADE_AXIS]


def point_segment_distance(p, a, b) -> np.ndarray:
    """Distance from points p to segments a-b, all (..., 3)."""
    ab = b - a
    u = ((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)
    closest = a + u.clip(0.0, 1.0)[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_geometry.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/motion/__init__.py src/kata/motion/geometry.py tests/test_motion_geometry.py
git commit -m "Phase 4: motion geometry (rotations, resampling, mirroring)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: One replay → one track

**Files:**
- Create: `src/kata/motion/prepare.py`, `tests/motion_synth.py`
- Test: `tests/test_motion_prepare.py`

**Interfaces:**
- Consumes: Task 1 geometry; `kata.data.align.align`, `kata.data.bsor` (`Replay`, `EVENT_*`, `decode_note_id`), `kata.data.pipeline.ALIGN_MIN_MATCH_RATE`, `ALIGN_MAX_P95_MS`.
- Produces: `MOTION_VERSION = "v1"`, `FPS = 50`, `MAX_GAP_S`, `BLADE_MAX_M`, `N_CHANNELS = 31`, `MOTION_DIM = 27`, `TRACK = {"head": slice(0, 9), "left": slice(9, 18), "right": slice(18, 27)}`, `HIT = (27, 28)`, `OFF = (29, 30)`, `CHANNELS`; `motion_config() -> dict`; `Rejected(Exception)`; `Prepared(frame0, data (T,31) float32, valid (T,), hit_times, info)`; `mirror_note_id(int) -> int`; `prepare_replay(replay, notes: dict[str, ndarray], left_handed: bool) -> Prepared`; `MotionStore(root)` with `.dir`, `.config_path`, `.stats_path`, `.open_for_write(config)`, `.write(score_id, data, valid)`, `.has(score_id)`, `.load(score_id, mmap=True)`, `.valid(score_id)`; `channel_stats(store, score_ids) -> dict`. Test helpers: `motion_synth.synth_notes(n, t0, step)`, `synth_replay(notes, mode, ...)`, `HAND_X`.

- [ ] **Step 1: Write the test helpers** — `tests/motion_synth.py` (Task 3 appends to it):

```python
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
```

- [ ] **Step 2: Write the failing tests** — `tests/test_motion_prepare.py`:

```python
"""Replays -> motion tracks (spec §3): grid, centring, hits, checks, left-handed recovery, store."""
import numpy as np
import pytest

from kata.motion.prepare import (FPS, HIT, OFF, TRACK, MotionStore, Rejected, channel_stats,
                                 mirror_note_id, motion_config, prepare_replay)
from motion_synth import HAND_X, synth_notes, synth_replay


def _hit_times(p, h):
    rows = np.flatnonzero(p.data[:, HIT[h]] > 0)
    return (p.frame0 + rows + p.data[rows, OFF[h]]) / FPS


def test_mirror_note_id():
    assert mirror_note_id(32018) == 31008          # lane 2 -> 1, blue -> red, dot stays a dot
    assert mirror_note_id(30102) == 33113          # lane 0 -> 3, layer 1, red -> blue, left -> right
    assert mirror_note_id(mirror_note_id(31114)) == 31114


def test_right_handed_track_grid_centring_and_hits():
    notes = synth_notes()
    p = prepare_replay(synth_replay(notes), notes, left_handed=False)
    assert p.data.shape[1] == 31 and p.data.dtype == np.float32
    assert p.valid.all()
    assert abs(np.median(p.data[:, TRACK["head"].start])) < 1e-6        # head x centred
    assert abs(np.median(p.data[:, TRACK["head"].start + 2])) < 1e-6    # head z centred
    assert np.allclose(p.data[:, TRACK["left"].start] - p.data[:, TRACK["head"].start], HAND_X[0] - 0.35)
    for h in (0, 1):                                                    # map times, both hands
        assert np.allclose(_hit_times(p, h), notes["time_s"][notes["color"] == h], atol=1e-5)
    assert np.abs(p.data[:, OFF[0]]).max() <= 0.5
    assert p.info["blade_median_m"] < 1e-3
    assert p.info["n_hits_left"] == 20 and p.info["n_hits_right"] == 20
    assert not p.info["world_mirrored"] and not p.info["hands_swapped"]
    assert np.allclose(p.hit_times, notes["time_s"])


@pytest.mark.parametrize("mode,flags", [("mirror_all", (True, True)), ("mirror_ids", (False, False))])
def test_left_handed_replays_are_recovered_whichever_way_the_game_mirrors(mode, flags):
    notes = synth_notes()
    ref = prepare_replay(synth_replay(notes), notes, left_handed=False)
    p = prepare_replay(synth_replay(notes, mode=mode), notes, left_handed=True)
    assert (p.info["world_mirrored"], p.info["hands_swapped"]) == flags
    assert p.info["mirrored"]
    assert p.frame0 == ref.frame0
    assert np.allclose(p.data, ref.data, atol=1e-5)


def test_broken_replays_are_rejected_with_a_reason():
    notes = synth_notes()
    cases = {"frame_check": dict(mode="right_mirrored_world"), "blade_check": dict(saber_dy=0.5),
             "align_residual": dict(spawn_shift_s=0.01)}
    for reason, kw in cases.items():
        with pytest.raises(Rejected, match=reason):
            prepare_replay(synth_replay(notes, **kw), notes, left_handed=False)
    with pytest.raises(Rejected, match="align_match_rate"):           # left-handed flag on a normal replay
        prepare_replay(synth_replay(notes), notes, left_handed=True)


def test_tracking_gaps_are_flagged_not_dropped():
    notes = synth_notes()
    p = prepare_replay(synth_replay(notes, gap=(5.0, 5.3)), notes, left_handed=False)
    t = (p.frame0 + np.arange(len(p.data))) / FPS
    assert not p.valid[(t > 5.01) & (t < 5.29)].any()
    assert p.valid[(t < 4.9) | (t > 5.4)].all()


def test_store_pins_config_and_round_trips(tmp_path):
    store = MotionStore(tmp_path)
    store.open_for_write(motion_config())
    store.open_for_write(motion_config())                                # same config: fine
    notes = synth_notes()
    p = prepare_replay(synth_replay(notes), notes, left_handed=False)
    store.write("123", p.data, p.valid)
    assert store.has("123") and store.load("123").dtype == np.float16
    assert np.allclose(store.load("123"), p.data, atol=2e-3)
    s = channel_stats(store, ["123"])
    assert len(s["mean"]) == 27 and min(s["std"]) >= 1e-3
    store.config_path.write_text('{"version": "v1", "fps": 30}')
    with pytest.raises(ValueError, match="bump MOTION_VERSION"):
        store.open_for_write(motion_config())
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_prepare.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'kata.motion.prepare'`.

- [ ] **Step 4: Implement** — `src/kata/motion/prepare.py`:

```python
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
                                  point_segment_distance, quat_to_matrix, reflect_x, resample)

MOTION_VERSION = "v1"
FPS = 50
MAX_GAP_S = 0.1
BLADE_MAX_M = 0.30          # median cut-point-to-blade distance; [verify in pilot] (spec §3.5)
MIN_CUTS = 10
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
            "min_cuts": MIN_CUTS, "channels": list(CHANNELS), "centre": "median head x,z",
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
    rot = quat_to_matrix(q_at)
    dist = np.stack([point_segment_distance(cp, *blade(p_at[:, k], rot[:, k])) for k in (1, 2)], 1)
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
            "blade_median_m": blade_median, **hand_y}
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_prepare.py -q`
Expected: `7 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/kata/motion/prepare.py tests/motion_synth.py tests/test_motion_prepare.py
git commit -m "Phase 4: replay to motion track, frame and blade checks, left-handed recovery" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Prepare stage, data paths and the Tier A crawl filter

**Files:**
- Create: `src/kata/motion/pipeline.py` (first part; Tasks 9 and 10 append)
- Modify: `src/kata/config.py`, `src/kata/data/pipeline.py`, `src/kata/cli.py`, `tests/motion_synth.py` (append)
- Test: `tests/test_motion_pipeline.py` (first part; Tasks 9 and 10 append)

**Interfaces:**
- Consumes: Task 2 (`prepare_replay`, `MotionStore`, `channel_stats`, `motion_config`, `Rejected`); `kata.validate.rules.nps`; `kata.data.pipeline._with_state`.
- Produces: `Paths.motion` (`root/motion`), `Paths.motion_runs` (`root/runs/motion`); `kata.data.pipeline.leaderboard_diffs(paths, pilot=False, tier=None) -> DataFrame`; `crawl_leaderboards(..., tier=None)`; `kata.motion.pipeline.prepare_stage(paths, pilot=False) -> summary` writing `processed/motion_tracks{_pilot}.parquet` (columns: `score_id, hash, characteristic, difficulty, label, split, audio_sha256, tier, nps, height_median, motion_version, status, reject_reason, frame0, hps, n_frames, n_valid, n_hits_left, n_hits_right, mirrored, world_mirrored, hands_swapped, blade_median_m, hand_y_l0, hand_y_l1, hand_y_l2`) and `motion/v1/stats.json`. Test helpers: `motion_synth.write_features`, `write_audio_stats`, `write_root(p) -> notes`, `SONGS`.

- [ ] **Step 1: Append the data-root helpers** to `tests/motion_synth.py`:

```python
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
                  "audio_sha256": [a for a, _ in SONGS.values()], "pilot": [True, True]}).write_parquet(
        p.processed / "maps.parquet")
    pl.DataFrame({"hash": ["h1", "h2", "h3"], "characteristic": ["Standard"] * 3, "difficulty": ["Expert"] * 3,
                  "tier": ["A", "A", "B"], "parse_status": ["ok"] * 3}).write_parquet(
        p.processed / "difficulties.parquet")
    pl.DataFrame({"audio_sha256": [a for a, _ in SONGS.values()], "status": ["ok", "ok"],
                  "duration_s": [duration, duration]}).write_parquet(p.processed / "audio_features.parquet")
    for sha, _ in SONGS.values():
        write_features(p, sha, duration)
    write_audio_stats(p)
    replays = [  # score_id, hash, mode, status, reject_reason, saber_dy
        ("1", "h1", "right", "ok", None, 0.0),
        ("2", "h1", "mirror_all", "rejected", "left_handed,align_match_rate", 0.0),
        ("3", "h1", "right", "rejected", "hash_mismatch", 0.0),
        ("4", "h1", "right", "ok", None, 0.5),
        ("5", "h2", "right", "ok", None, 0.0),
    ]
    for sid, h, mode, _status, _reason, dy in replays:
        (p.replays / h).mkdir(parents=True, exist_ok=True)
        r = synth_replay(notes, mode=mode, saber_dy=dy, seed=int(sid))
        (p.replays / h / f"{sid}.bsor").write_bytes(bsor.write(r))
    pl.DataFrame({"score_id": [r[0] for r in replays], "hash": [r[1] for r in replays],
                  "characteristic": ["Standard"] * 5, "difficulty": ["Expert"] * 5,
                  "status": [r[3] for r in replays], "reject_reason": [r[4] for r in replays],
                  "left_handed": [r[2] != "right" for r in replays], "height_median": [1.75] * 5}
                 ).write_parquet(p.processed / "replays.parquet")
    return notes
```

- [ ] **Step 2: Write the failing tests** — `tests/test_motion_pipeline.py`:

```python
"""Phase 4 stages on a tiny synthetic data root."""
import json

import numpy as np
import polars as pl
import pytest

from kata.config import Paths
from kata.data.pipeline import leaderboard_diffs
from kata.motion import pipeline as mp
from kata.motion.prepare import MotionStore
from motion_synth import write_root

KEY = ["hash", "characteristic", "difficulty"]


@pytest.fixture
def paths(tmp_path) -> Paths:
    p = Paths(tmp_path).ensure()
    write_root(p)
    return p


def test_prepare_stage_tracks_rejects_stats_and_idempotency(paths):
    s = mp.prepare_stage(paths)
    t = pl.read_parquet(paths.processed / "motion_tracks.parquet")
    assert t["score_id"].to_list() == ["1", "2", "4", "5"]              # hash mismatch never considered
    assert s["tracks"] == {"ok": 3, "rejected": 1}
    assert s["reject_reasons"] == {"blade_check": 1}
    assert s["left_handed_recovered"] == 1
    ok = t.filter(pl.col("status") == "ok")
    assert set(ok["split"]) == {"train", "val"}
    assert ok["nps"].n_unique() == 1 and ok["hps"][0] == pytest.approx(ok["nps"][0])   # no stacks
    store = MotionStore(paths.motion)
    assert all(store.has(sid) for sid in ok["score_id"])
    stats = json.loads(store.stats_path.read_text())
    assert stats["n_tracks"] == 2                                     # train only: 1 and 2
    first = t
    mp.prepare_stage(paths)
    assert pl.read_parquet(paths.processed / "motion_tracks.parquet").equals(first)


def test_leaderboard_crawl_can_be_limited_to_one_tier(paths):
    assert sorted(leaderboard_diffs(paths)["hash"]) == ["h1", "h2", "h3"]
    assert sorted(leaderboard_diffs(paths, tier="A")["hash"]) == ["h1", "h2"]
    assert np.array_equal(leaderboard_diffs(paths, tier="B")["hash"].to_numpy(), ["h3"])
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_pipeline.py -q`
Expected: collection error, `ImportError: cannot import name 'leaderboard_diffs' from 'kata.data.pipeline'`.

- [ ] **Step 4: Add the paths** — in `src/kata/config.py`, insert before the `state_db` property:

```python
    @property
    def motion(self) -> Path:
        """Phase 4 motion tracks (docs/phases/phase-4-motion.md §9)."""
        return self.root / "motion"

    @property
    def motion_runs(self) -> Path:
        """Phase 4 training runs (docs/phases/phase-4-motion.md §6.4)."""
        return self.root / "runs" / "motion"
```

and in `Paths.ensure()` replace the tuple's last line

```python
                  self.lowrated, self.validator, self.generated, self.state_db.parent, self.reports):
```

with

```python
                  self.lowrated, self.validator, self.generated, self.motion, self.motion_runs,
                  self.state_db.parent, self.reports):
```

- [ ] **Step 5: Add the tier filter** — in `src/kata/data/pipeline.py`, replace the first three lines of `_crawl_leaderboards`

```python
async def _crawl_leaderboards(paths, state, summary, pilot=False):
    diffs = pl.read_parquet(paths.processed / "difficulties.parquet").filter(pl.col("parse_status") == "ok")
    diffs = _subset(diffs, paths, pilot)
```

with

```python
def leaderboard_diffs(paths: Paths, pilot: bool = False, tier: str | None = None) -> pl.DataFrame:
    """Parsed difficulties whose leaderboards are crawled; `tier` restricts to one tier (Phase 4 D6)."""
    diffs = pl.read_parquet(paths.processed / "difficulties.parquet").filter(pl.col("parse_status") == "ok")
    if tier is not None:
        diffs = diffs.filter(pl.col("tier") == tier)
    return _subset(diffs, paths, pilot)


async def _crawl_leaderboards(paths, state, summary, pilot=False, tier=None):
    diffs = leaderboard_diffs(paths, pilot, tier)
```

and in `src/kata/cli.py` replace the `crawl-leaderboards` command body

```python
def crawl_leaderboards(pilot: bool = PILOT):
    _run("crawl-leaderboards", pipeline.crawl_leaderboards, pilot=pilot)
```

with

```python
def crawl_leaderboards(pilot: bool = PILOT,
                       tier: str = typer.Option(None, help="Only this tier's difficulties (Phase 4: A).")):
    _run("crawl-leaderboards", pipeline.crawl_leaderboards, pilot=pilot, tier=tier)
```

- [ ] **Step 6: Implement the stage** — `src/kata/motion/pipeline.py`:

```python
"""Phase 4 stages (docs/phases/phase-4-motion.md §10-§11). Same conventions as Phases 1-3: every
stage runs inside `_with_state`; tables are rebuilt deterministically from their inputs."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from kata.config import Paths
from kata.data.pipeline import _with_state
from kata.motion.prepare import MotionStore, Rejected, channel_stats, motion_config, prepare_replay
from kata.validate.rules import nps

KEY = ["hash", "characteristic", "difficulty"]
# Phase 1 rejects a left-handed replay as "left_handed" and, because its note ids are mirrored,
# usually also on alignment. Those are the ones prepare recovers (spec §3.5, D7).
RECOVERABLE = {"left_handed", "align_match_rate", "align_residual"}


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _recoverable(reason: str | None) -> bool:
    parts = set((reason or "").split(","))
    return "left_handed" in parts and parts <= RECOVERABLE


def _prepare_one(root: str, score_id: str, bsor_path: str, notes: dict, left_handed: bool) -> dict:
    from kata.data import bsor
    store = MotionStore(Paths(Path(root)).motion)
    try:
        p = prepare_replay(bsor.parse(Path(bsor_path).read_bytes()), notes, left_handed)
    except Rejected as e:
        return {"status": "rejected", "reject_reason": str(e)}
    except Exception as e:  # recorded per replay; the stage keeps going
        return {"status": "error", "reject_reason": f"{type(e).__name__}: {e}"[:300]}
    store.write(score_id, p.data, p.valid)
    return {"status": "ok", "reject_reason": None, "frame0": p.frame0, "hps": nps(p.hit_times), **p.info}


def _prepare(paths, state, summary, pilot=False):
    suffix = "_pilot" if pilot else ""
    P = paths.processed
    replays = pl.read_parquet(P / f"replays{suffix}.parquet")
    replays = replays.filter((pl.col("status") == "ok")
                             | pl.col("reject_reason").map_elements(_recoverable, return_dtype=pl.Boolean))
    maps = pl.read_parquet(P / "maps.parquet").select("hash", "split", "audio_sha256")
    diffs = (pl.read_parquet(P / "difficulties.parquet").filter(pl.col("parse_status") == "ok")
             .select(KEY + ["tier"]))
    replays = replays.join(maps, on="hash").join(diffs, on=KEY).sort("score_id")
    notes = pl.read_parquet(P / "notes.parquet").join(replays.select(KEY).unique(), on=KEY, how="semi")
    by_diff = {k: {c: g[c].to_numpy() for c in ("x", "y", "color", "cut_dir", "time_s")}
               for k, g in notes.partition_by(KEY, as_dict=True).items()}
    store = MotionStore(paths.motion)
    store.open_for_write(motion_config())

    rows = []
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        futures = []
        for r in replays.iter_rows(named=True):
            sid = str(r["score_id"])
            futures.append((r, pool.submit(_prepare_one, str(paths.root), sid,
                                           str(paths.replays / r["hash"] / f"{sid}.bsor"),
                                           by_diff[(r["hash"], r["characteristic"], r["difficulty"])],
                                           bool(r["left_handed"]))))
        for i, (r, f) in enumerate(futures):
            key = (r["hash"], r["characteristic"], r["difficulty"])
            rows.append({"score_id": str(r["score_id"]), **{k: r[k] for k in KEY}, "label": r["difficulty"],
                         "split": r["split"], "audio_sha256": r["audio_sha256"], "tier": r["tier"],
                         "nps": nps(by_diff[key]["time_s"]), "height_median": r["height_median"],
                         "motion_version": store.version, **f.result()})
            if (i + 1) % 1000 == 0:
                print(f"  [motion-prepare] {i + 1}/{len(futures)}", flush=True)
    tracks = pl.DataFrame(rows, infer_schema_length=None).sort("score_id")
    tracks.write_parquet(P / f"motion_tracks{suffix}.parquet")

    train = tracks.filter((pl.col("status") == "ok") & (pl.col("split") == "train"))["score_id"].to_list()
    store.stats_path.write_text(json.dumps(channel_stats(store, train)))
    summary["tracks"] = dict(tracks.group_by("status").len().sort("status").iter_rows())
    summary["reject_reasons"] = dict(tracks.filter(pl.col("status") != "ok").group_by("reject_reason").len()
                                     .sort("reject_reason").iter_rows())
    ok = tracks.filter(pl.col("status") == "ok")
    summary["left_handed_recovered"] = ok.filter(pl.col("mirrored")).height
    summary["stats_train_tracks"] = len(train)

prepare_stage = _with_state("motion-prepare", _prepare)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_pipeline.py -q`
Expected: `2 passed`. Then the whole suite: `uv run pytest -q` → no failures (the Phase 1–3 tests are untouched by the `tier=None` default).

- [ ] **Step 8: Commit**

```bash
git add src/kata/motion/pipeline.py src/kata/config.py src/kata/data/pipeline.py src/kata/cli.py tests/motion_synth.py tests/test_motion_pipeline.py
git commit -m "Phase 4: motion prepare stage and Tier A leaderboard crawl" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Training windows

**Files:**
- Create: `src/kata/motion/dataset.py`
- Test: `tests/test_motion_dataset.py`

**Interfaces:**
- Consumes: Task 2 (`FPS`, `HIT`, `OFF`, `TRACK`, `MOTION_DIM`, `MotionStore`), Task 3 (`prepare_stage`, `write_root` for tests); `kata.audio.store.FeatureStore`, `ARRAYS`; `kata.validate.rules.LABELS`.
- Produces: `WINDOW = 640`, `CTX = 100`, `AUDIO_FRAME_DIM = 408`, `AUDIO_DIM = 820`, `NONE_LABEL = 5`, `HIT_SIGMA`, `STEM_COLS`; `TrackRef(score_id, sha, frame0, n_frames, label, nps, key)`; `load_tracks(tracks_df, split, songs_with_features, window=WINDOW) -> list[TrackRef]`; `Normaliser(motion_stats, audio_stats)` with `.m_mean`, `.m_std`, `.motion(raw) -> model space`, `.denorm_motion(x) -> (…, 27)`, `.audio(frames)`; `hit_pulse(flags)`; `decode_hits(pulse, offset, frame0=0) -> times`; `phase(times, events)`; `mirror_track(raw)`; `WindowSampler(tracks, motion, features, norm, window, ctx, seed, batch, mirror_aug, use_stems)` with `.songs`, `.window`, `.motion`, `.audio(sha, start, n) -> (n, 820)`, `.example(track, start, mirror=False) -> dict(x, hit_mask, audio, label, nps)`, `.batch(step) -> dict of stacked arrays`.

- [ ] **Step 1: Write the failing tests** — `tests/test_motion_dataset.py`:

```python
"""Training windows (spec §4, §6.2): pulses, phases, mirroring, sampling, audio alignment."""
import json

import numpy as np
import pytest

from kata.audio.store import FeatureStore
from kata.config import Paths
from kata.motion import geometry as g
from kata.motion import pipeline as mp
from kata.motion.dataset import (AUDIO_DIM, AUDIO_FRAME_DIM, CTX, STEM_COLS, Normaliser, WindowSampler,
                                 decode_hits, hit_pulse, load_tracks, mirror_track, phase)
from kata.motion.prepare import HIT, OFF, TRACK, MotionStore
from motion_synth import write_root


def test_hit_pulse_and_decoding_round_trip():
    flags = np.zeros(60)
    flags[[10, 13, 40]] = 1
    pulse = hit_pulse(flags)
    assert pulse.min() >= -1 and pulse[10] == pytest.approx(1.0) and pulse[0] == pytest.approx(-1.0)
    offset = np.zeros(60)
    offset[[10, 13, 40]] = [0.4, -0.6, 0.0]                      # model space: 2 × frames
    t = decode_hits(pulse, offset, frame0=100)
    assert np.allclose(t, (100 + np.array([10.2, 12.7, 40.0])) / 50)


def test_phase_inside_and_outside_the_events():
    beats = np.arange(1.0, 5.0, 0.5)
    assert np.allclose(phase(np.array([1.25, 3.0, 0.75, 4.75, 0.0]), beats), [0.5, 0.0, 0.5, 0.5, 0.0])
    assert np.allclose(phase(np.array([1.0, 2.0]), np.array([1.0])), 0.0)


def test_mirror_track_matches_the_geometric_reflection():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 3, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    pos = rng.normal(size=(5, 3, 3))
    raw = np.zeros((5, 31), np.float32)
    for k, part in enumerate(("head", "left", "right")):
        s = TRACK[part].start
        raw[:, s:s + 3] = pos[:, k]
        raw[:, s + 3:s + 9] = g.matrix_to_rot6(g.quat_to_matrix(q[:, k]))
    raw[2, HIT[0]], raw[2, OFF[0]] = 1, 0.3
    m = mirror_track(raw)
    p2, q2 = g.reflect_x(pos, q)
    for k, part in enumerate(("head", "right", "left")):             # hands swap
        s = TRACK[("head", "left", "right")[k]].start
        assert np.allclose(m[:, s:s + 3], p2[:, ("head", "left", "right").index(part)], atol=1e-6)
        assert np.allclose(m[:, s + 3:s + 9], g.matrix_to_rot6(g.quat_to_matrix(
            q2[:, ("head", "left", "right").index(part)])), atol=1e-6)
    assert m[2, HIT[1]] == 1 and m[2, OFF[1]] == pytest.approx(0.3) and m[2, HIT[0]] == 0
    assert np.allclose(mirror_track(m), raw)


@pytest.fixture
def sampler(tmp_path):
    p = Paths(tmp_path).ensure()
    write_root(p)
    mp.prepare_stage(p)
    import polars as pl
    tracks = load_tracks(pl.read_parquet(p.processed / "motion_tracks.parquet"), "train", {"a" * 64, "b" * 64})
    motion = MotionStore(p.motion)
    feats = FeatureStore(p.features)
    norm = Normaliser(json.loads(motion.stats_path.read_text()), json.loads(feats.stats_path.read_text()))
    return WindowSampler(tracks, motion, feats, norm, seed=7, batch=3), tracks


def test_batches_are_a_pure_function_of_seed_and_step(sampler):
    s, tracks = sampler
    assert [t.score_id for t in tracks] == ["1", "2"]
    b = s.batch(5)
    assert b["x"].shape == (3, 640, 31) and b["audio"].shape == (3, 640 + 2 * CTX, AUDIO_DIM)
    assert b["hit_mask"].shape == (3, 640, 2) and b["label"].dtype == np.int64
    assert all(np.array_equal(b[k], s.batch(5)[k]) for k in b)
    assert not np.array_equal(b["x"], s.batch(6)["x"])


def test_window_audio_lines_up_with_the_motion_grid(sampler):
    s, tracks = sampler
    t = tracks[0]
    ex = s.example(t, start=37)
    assert np.allclose(ex["audio"][:60, 0], -100.0)    # before t = 0: Phase 2 silence
    g0 = t.frame0 + 37                              # absolute 50 fps index of motion row 0
    # audio row k covers 100 fps frames 2(g0 - CTX + k) and that + 1; mel bin 0 holds the frame index
    k = np.arange(70, 80)                           # inside the song (rows < 62 are before t = 0)
    assert np.allclose(ex["audio"][k, 0], 2 * (g0 - CTX + k))
    assert np.allclose(ex["audio"][k, AUDIO_FRAME_DIM], 2 * (g0 - CTX + k) + 1)
    raw = np.asarray(s.motion.load(t.score_id)[37:37 + 640], np.float32)
    assert np.allclose(s.norm.denorm_motion(ex["x"]), raw[:, :27], atol=1e-4)
    assert ex["hit_mask"].sum() == (raw[:, [HIT[0], HIT[1]]] > 0).sum()


def test_stems_can_be_switched_off(sampler):
    s, tracks = sampler
    s.use_stems = False
    a = s.example(tracks[0], 0)["audio"]
    assert not a[:, STEM_COLS].any() and not a[:, AUDIO_FRAME_DIM + 128:AUDIO_FRAME_DIM + 384].any()


def test_tracks_shorter_than_a_window_or_without_features_are_left_out(tmp_path):
    import polars as pl
    p = Paths(tmp_path).ensure()
    write_root(p)
    mp.prepare_stage(p)
    t = pl.read_parquet(p.processed / "motion_tracks.parquet")
    shas = {"a" * 64, "b" * 64}
    assert len(load_tracks(t, "train", shas)) == 2
    assert load_tracks(t, "train", shas, window=5000) == []
    assert load_tracks(t, "train", {"b" * 64}) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_dataset.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'kata.motion.dataset'`.

- [ ] **Step 3: Implement** — `src/kata/motion/dataset.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_dataset.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/motion/dataset.py tests/test_motion_dataset.py
git commit -m "Phase 4: training windows, audio context and normalisation" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Model

**Files:**
- Create: `src/kata/motion/model.py`
- Test: `tests/test_motion_model.py`

**Interfaces:**
- Consumes: torch only.
- Produces: `ModelConfig(d=448, heads=8, enc_layers=4, dec_layers=8, ff_mult=4, dropout=0.1, n_in=31, audio_dim=820, n_labels=6)`; `rope(x, pos)`; `MotionDenoiser(config)` with `.config`, `.encode_audio(audio (B, W+2c, 820)) -> tokens`, `.forward(x_t (B,W,31), t (B,), tokens, label (B,), nps (B,), nps_drop (B,) bool) -> x0 (B,W,31)`; `n_params(model)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_motion_model.py`:

```python
"""Audio encoder + denoiser (spec §5.2-5.3)."""
import pytest

torch = pytest.importorskip("torch")

from kata.motion.model import ModelConfig, MotionDenoiser, n_params, rope  # noqa: E402

TINY = ModelConfig(d=32, heads=4, enc_layers=1, dec_layers=2, dropout=0.0)


def _inputs(B=2, W=16, ctx=4, c=TINY):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(B, W, c.n_in, generator=g), torch.tensor([3, 900]),
            torch.randn(B, W + 2 * ctx, c.audio_dim, generator=g), torch.tensor([1, 5]),
            torch.tensor([4.0, 7.5]), torch.tensor([False, True]))


def test_full_size_model_is_about_45m_parameters():
    n = n_params(MotionDenoiser(ModelConfig()))
    assert 40e6 < n < 50e6


def test_shapes_and_zero_initialised_output():
    m = MotionDenoiser(TINY)
    x, t, audio, label, nps, drop = _inputs()
    out = m(x, t, m.encode_audio(audio), label, nps, drop)
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0                     # adaLN-Zero: the untrained model predicts 0


def test_rope_is_relative():
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    a = (rope(q, torch.tensor([3.0])) * rope(k, torch.tensor([1.0]))).sum()
    b = (rope(q, torch.tensor([12.0])) * rope(k, torch.tensor([10.0]))).sum()
    assert torch.allclose(a, b, atol=1e-5)


def test_conditioning_and_audio_change_the_output_after_a_step():
    torch.manual_seed(0)
    m = MotionDenoiser(TINY)
    x, t, audio, label, nps, drop = _inputs()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    target = torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    for _ in range(3):                                       # move the zero-initialised gates
        loss = (m(x, t, m.encode_audio(audio), label, nps, drop) - target).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    base = m(x, t, m.encode_audio(audio), label, nps, drop)
    assert not torch.allclose(base, m(x, t, m.encode_audio(audio), torch.tensor([2, 5]), nps, drop))
    assert not torch.allclose(base, m(x, t, m.encode_audio(audio + 1), label, nps, drop))
    assert not torch.allclose(base, m(x, t, m.encode_audio(audio), label, nps + 3, drop))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_model.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'kata.motion.model'`.

- [ ] **Step 3: Implement** — `src/kata/motion/model.py`:

```python
"""Audio encoder + denoiser transformer (docs/phases/phase-4-motion.md §5.2-5.3).

Self- and cross-attention use rotary positions. Motion frame k sits at audio position k + ctx, so
cross-attention sees the true time offset between a motion frame and every audio frame.
Conditioning (diffusion step, label, NPS) enters every denoiser layer as FiLM shift/scale/gate
(adaLN-Zero: gates start at zero).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d: int = 448                 # ~46M parameters with the layer counts below (spec §5.3)
    heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    n_in: int = 31
    audio_dim: int = 820
    n_labels: int = 6            # 5 labels + "none"


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary embedding on (B, H, L, Dh) at float positions (L,)."""
    half = x.shape[-1] // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.float()[:, None] * freqs[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, ctx, q_pos, k_pos):
        B, L, D = x.shape
        h = self.heads
        q = self.q(x).view(B, L, h, D // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, h, D // h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, q_pos), rope(k, k_pos)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, L, D))


def feed_forward(d: int, mult: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * d, d))


class EncoderBlock(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.attn = Attention(c.d, c.heads, c.dropout)
        self.ff = feed_forward(c.d, c.ff_mult, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, a, pos):
        h = self.n1(a)
        a = a + self.drop(self.attn(h, h, pos, pos))
        return a + self.drop(self.ff(self.n2(a)))


class DecoderBlock(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1 = nn.LayerNorm(c.d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(c.d)
        self.n3 = nn.LayerNorm(c.d, elementwise_affine=False)
        self.self_attn = Attention(c.d, c.heads, c.dropout)
        self.cross_attn = Attention(c.d, c.heads, c.dropout)
        self.ff = feed_forward(c.d, c.ff_mult, c.dropout)
        self.drop = nn.Dropout(c.dropout)
        self.mod = nn.Linear(c.d, 6 * c.d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond, audio, x_pos, a_pos):
        s1, sc1, g1, s2, sc2, g2 = self.mod(F.silu(cond))[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + sc1) + s1
        x = x + g1 * self.drop(self.self_attn(h, h, x_pos, x_pos))
        x = x + self.drop(self.cross_attn(self.n2(x), audio, x_pos, a_pos))
        h = self.n3(x) * (1 + sc2) + s2
        return x + g2 * self.drop(self.ff(h))


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], -1)


class MotionDenoiser(nn.Module):
    """x_t (B, W, n_in), t (B,), audio (B, W + 2·ctx, audio_dim) -> predicted clean x0 (B, W, n_in)."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.config = c
        self.audio_in = nn.Linear(c.audio_dim, c.d)
        self.encoder = nn.ModuleList(EncoderBlock(c) for _ in range(c.enc_layers))
        self.audio_norm = nn.LayerNorm(c.d)
        self.x_in = nn.Linear(c.n_in, c.d)
        self.t_mlp = nn.Sequential(nn.Linear(256, c.d), nn.SiLU(), nn.Linear(c.d, c.d))
        self.label = nn.Embedding(c.n_labels, c.d)
        self.nps_mlp = nn.Sequential(nn.Linear(1, c.d), nn.SiLU(), nn.Linear(c.d, c.d))
        self.nps_none = nn.Parameter(torch.zeros(c.d))
        self.blocks = nn.ModuleList(DecoderBlock(c) for _ in range(c.dec_layers))
        self.out_norm = nn.LayerNorm(c.d, elementwise_affine=False)
        self.out_mod = nn.Linear(c.d, 2 * c.d)
        self.out = nn.Linear(c.d, c.n_in)
        for m in (self.out_mod, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        a = self.audio_in(audio)
        pos = torch.arange(audio.shape[1], device=audio.device)
        for blk in self.encoder:
            a = blk(a, pos)
        return self.audio_norm(a)

    def cond(self, t, label, nps, nps_drop) -> torch.Tensor:
        n = self.nps_mlp(torch.log(nps.clamp(min=0.1)).float()[:, None])
        n = torch.where(nps_drop[:, None], self.nps_none.expand_as(n), n)
        return self.t_mlp(timestep_embedding(t)) + self.label(label) + n

    def forward(self, x_t, t, audio_tokens, label, nps, nps_drop) -> torch.Tensor:
        W = x_t.shape[1]
        ctx = (audio_tokens.shape[1] - W) // 2
        x_pos = torch.arange(W, device=x_t.device) + ctx
        a_pos = torch.arange(audio_tokens.shape[1], device=x_t.device)
        c = self.cond(t, label, nps, nps_drop)
        x = self.x_in(x_t)
        for blk in self.blocks:
            x = blk(x, c, audio_tokens, x_pos, a_pos)
        shift, scale = self.out_mod(F.silu(c))[:, None].chunk(2, -1)
        return self.out(self.out_norm(x) * (1 + scale) + shift)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_model.py -q`
Expected: `4 passed` (the full-size model measures 46,376,991 parameters).

- [ ] **Step 5: Commit**

```bash
git add src/kata/motion/model.py tests/test_motion_model.py
git commit -m "Phase 4: audio encoder and adaLN denoiser transformer" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Diffusion

**Files:**
- Create: `src/kata/motion/diffusion.py`
- Test: `tests/test_motion_diffusion.py`

**Interfaces:**
- Consumes: Task 4 (`NONE_LABEL`), Task 2 (`HIT`, `OFF`, `MOTION_DIM`, `TRACK`), Task 1 (`BLADE_AXIS`, `BLADE_LENGTH_M`), Task 5 (`MotionDenoiser` in tests).
- Produces: `T_STEPS = 1000`, `P_UNCOND = 0.1`; `cosine_alpha_bar(T)`; `rot6_to_matrix(tensor)`; `saber_tips(motion (…,T,27)) -> (…,T,2,3)`; `LossWeights(tip, tip_vel)`; `Diffusion(motion_mean, motion_std)` with `.q_sample`, `.denorm`, `.loss(model, batch, weights, p_uncond) -> (total, parts dict)`, `.ddim(model, tokens, label, nps (NaN = dropped), noise_fn, steps, guidance, known=None, known_mask=None)`; `Job(sha, label|None, nps|None, seed, n_frames)`; `n_windows(n_frames, window)`; `sample_long(model, diffusion, jobs, audio_fn, window, ctx, steps, guidance, device) -> list[(model-space track (n_frames, 31), joins)]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_motion_diffusion.py`:

```python
"""Schedule, losses, DDIM with guidance and overlap inpainting (spec §5.4-5.6)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kata.motion import geometry as g  # noqa: E402
from kata.motion.diffusion import (Diffusion, Job, cosine_alpha_bar, n_windows, rot6_to_matrix,  # noqa: E402
                                   sample_long, saber_tips)
from kata.motion.model import ModelConfig, MotionDenoiser  # noqa: E402

TINY = ModelConfig(d=32, heads=4, enc_layers=1, dec_layers=1, dropout=0.0)


def _diffusion():
    return Diffusion(np.zeros(27), np.ones(27))


def test_cosine_schedule():
    ab = cosine_alpha_bar()
    assert ab.shape == (1000,) and (ab.diff() < 0).all()
    assert ab[0] > 0.999 and ab[-1] < 1e-3


def test_rot6_and_tips_match_numpy():
    r6 = np.random.default_rng(0).normal(size=(5, 6))
    assert np.allclose(rot6_to_matrix(torch.tensor(r6)).numpy(), g.rot6_to_matrix(r6), atol=1e-6)
    m = torch.zeros(1, 3, 27)
    m[..., 9:12] = torch.tensor([0.1, 1.0, 0.2])                       # left saber position
    m[..., 12:18] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])             # identity rotation
    m[..., 21:27] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
    tips = saber_tips(m)
    assert torch.allclose(tips[0, 0, 0], torch.tensor([0.1, 1.0, 1.2]))


def test_loss_is_finite_and_offsets_count_only_at_hits():
    torch.manual_seed(0)
    model = MotionDenoiser(TINY)
    B, W, C = 2, 16, 4
    batch = {"x": torch.randn(B, W, 31), "hit_mask": torch.zeros(B, W, 2), "audio": torch.randn(B, W + 2 * C, 820),
             "label": torch.tensor([0, 4]), "nps": torch.tensor([3.0, 8.0])}
    total, parts = _diffusion().loss(model, batch)
    assert torch.isfinite(total) and parts["l_off"] == 0.0
    batch["hit_mask"][0, 3, 1] = 1
    _, parts = _diffusion().loss(model, batch)
    assert parts["l_off"] > 0


class Oracle(torch.nn.Module):
    """Predicts x0 = the absolute frame index (audio column 0) in channel 0, zeros elsewhere."""

    def __init__(self):
        super().__init__()
        self.config = ModelConfig(n_in=31)

    def encode_audio(self, audio):
        return audio

    def forward(self, x_t, t, audio_tokens, label, nps, nps_drop):
        W = x_t.shape[1]
        ctx = (audio_tokens.shape[1] - W) // 2
        out = torch.zeros_like(x_t)
        out[..., 0] = audio_tokens[:, ctx:ctx + W, 0]
        return out


def _frame_audio(sha, start, n):
    a = np.zeros((n, 820), np.float32)
    a[:, 0] = start + np.arange(n)
    return a


def test_ddim_returns_the_predicted_x0_and_holds_known_frames():
    d = _diffusion()
    model = Oracle()
    audio = torch.from_numpy(_frame_audio("s", -4, 24))[None]
    gen = torch.Generator().manual_seed(0)
    x = d.ddim(model, audio, torch.tensor([1]), torch.tensor([5.0]), lambda: torch.randn(1, 16, 31, generator=gen),
               steps=10, guidance=2.5)
    assert torch.allclose(x[0, :, 0], torch.arange(16.0), atol=1e-4) and x[0, :, 1:].abs().max() < 1e-4
    known = torch.full((1, 16, 31), 7.0)
    mask = torch.zeros_like(known, dtype=torch.bool)
    mask[:, :5] = True
    x = d.ddim(model, audio, torch.tensor([1]), torch.tensor([float("nan")]),
               lambda: torch.randn(1, 16, 31, generator=gen), steps=10, known=known, known_mask=mask)
    assert torch.equal(x[0, :5], known[0, :5])


def test_long_songs_are_stitched_on_the_frame_grid():
    assert [n_windows(n, 16) for n in (10, 16, 17, 24, 25)] == [1, 1, 2, 2, 3]
    jobs = [Job("s", 1, 4.0, 0, 50), Job("t", None, None, 1, 12)]
    res = sample_long(Oracle(), _diffusion(), jobs, _frame_audio, window=16, ctx=4, steps=5)
    (a, joins_a), (b, joins_b) = res
    assert a.shape == (50, 31) and b.shape == (12, 31)
    assert np.allclose(a[:, 0], np.arange(50), atol=1e-3)
    assert joins_a == [16, 24, 32, 40, 48] and joins_b == []


def test_a_jobs_output_does_not_depend_on_its_batch_mates():
    torch.manual_seed(0)
    model = MotionDenoiser(TINY)
    for p in model.out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    d = _diffusion()

    def audio_fn(sha, start, n):
        return np.random.default_rng([ord(sha), start + 1000]).normal(size=(n, 820)).astype(np.float32)

    alone = sample_long(model, d, [Job("s", 2, 5.0, 3, 30)], audio_fn, window=16, ctx=4, steps=4, guidance=2.0)
    pair = sample_long(model, d, [Job("s", 2, 5.0, 3, 30), Job("t", 0, 2.0, 9, 40)], audio_fn, window=16, ctx=4,
                       steps=4, guidance=2.0)
    assert np.allclose(alone[0][0], pair[0][0], atol=1e-5)
    again = sample_long(model, d, [Job("s", 2, 5.0, 3, 30)], audio_fn, window=16, ctx=4, steps=4, guidance=2.0)
    assert np.array_equal(alone[0][0], again[0][0])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_diffusion.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'kata.motion.diffusion'`.

- [ ] **Step 3: Implement** — `src/kata/motion/diffusion.py`:

```python
"""Diffusion: cosine schedule, x0 losses, DDIM with classifier-free guidance, overlap inpainting
(docs/phases/phase-4-motion.md §5.4-5.6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from kata.motion.dataset import NONE_LABEL
from kata.motion.prepare import HIT, MOTION_DIM, OFF, TRACK

T_STEPS = 1000
P_UNCOND = 0.1


def cosine_alpha_bar(T: int = T_STEPS, s: float = 0.008) -> torch.Tensor:
    """ᾱ for steps 0 .. T-1 (Nichol & Dhariwal 2021), betas clipped at 0.999."""
    f = lambda x: math.cos((x / T + s) / (1 + s) * math.pi / 2) ** 2  # noqa: E731
    betas = torch.tensor([min(1 - f(i + 1) / f(i), 0.999) for i in range(T)], dtype=torch.float64)
    return torch.cumprod(1 - betas, 0)


def rot6_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3), Gram-Schmidt; the torch twin of geometry.rot6_to_matrix."""
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


def saber_tips(motion: torch.Tensor) -> torch.Tensor:
    """De-normalised (..., T, 27) motion -> saber tips (..., T, 2, 3): origin + 1 m along local +z."""
    from kata.motion.geometry import BLADE_AXIS, BLADE_LENGTH_M
    tips = []
    for part in ("left", "right"):
        s = TRACK[part].start
        rot = rot6_to_matrix(motion[..., s + 3:s + 9])
        tips.append(motion[..., s:s + 3] + BLADE_LENGTH_M * rot[..., :, BLADE_AXIS])
    return torch.stack(tips, -2)


@dataclass
class LossWeights:
    tip: float = 1.0
    tip_vel: float = 1.0


class Diffusion:
    def __init__(self, motion_mean, motion_std, T: int = T_STEPS):
        self.T = T
        self.alpha_bar = cosine_alpha_bar(T)
        self.m_mean = torch.as_tensor(np.asarray(motion_mean), dtype=torch.float32)
        self.m_std = torch.as_tensor(np.asarray(motion_std), dtype=torch.float32)

    def _ab(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(like.device)[t.to(like.device)].to(like.dtype).view(-1, *([1] * (like.dim() - 1)))

    def q_sample(self, x0, t, noise):
        ab = self._ab(t, x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :MOTION_DIM] * self.m_std.to(x.device) + self.m_mean.to(x.device)

    def loss(self, model, batch: dict, weights: LossWeights = LossWeights(),
             p_uncond: float = P_UNCOND) -> tuple[torch.Tensor, dict]:
        """batch: x (B,W,31), hit_mask (B,W,2), audio (B,W+2c,A), label (B,), nps (B,), all tensors."""
        x0 = batch["x"]
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        label = torch.where(torch.rand(B, device=x0.device) < p_uncond,
                            torch.full_like(batch["label"], NONE_LABEL), batch["label"])
        nps_drop = torch.rand(B, device=x0.device) < p_uncond
        pred = model(x_t, t, model.encode_audio(batch["audio"]), label, batch["nps"], nps_drop).float()
        dense = list(range(MOTION_DIM)) + list(HIT)
        l_simple = (pred[..., dense] - x0[..., dense]).square().mean()
        mask = batch["hit_mask"]
        l_off = ((pred[..., list(OFF)] - x0[..., list(OFF)]).square() * mask).sum() / mask.sum().clamp(min=1)
        tip_p, tip_x = saber_tips(self.denorm(pred)), saber_tips(self.denorm(x0))
        l_tip = (tip_p - tip_x).square().mean()
        l_vel = (tip_p.diff(dim=1) - tip_x.diff(dim=1)).square().mean()
        total = l_simple + l_off + weights.tip * l_tip + weights.tip_vel * l_vel
        return total, {"l_simple": l_simple.item(), "l_off": l_off.item(), "l_tip": l_tip.item(),
                       "l_vel": l_vel.item()}

    def _x0(self, model, x, t, audio_tokens, label, nps, guidance: float):
        """Guided x0: x0_u + w (x0_c - x0_u); w = 1 skips the unconditional pass."""
        B = x.shape[0]
        tt = torch.full((B,), int(t), device=x.device, dtype=torch.long)
        no_drop = torch.zeros(B, dtype=torch.bool, device=x.device)
        nps_drop = torch.isnan(nps)
        nps_val = torch.nan_to_num(nps, nan=1.0)
        if guidance == 1.0:
            return model(x, tt, audio_tokens, label, nps_val, nps_drop).float()
        both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([audio_tokens, audio_tokens]),
                     torch.cat([label, torch.full_like(label, NONE_LABEL)]), torch.cat([nps_val, nps_val]),
                     torch.cat([nps_drop, ~no_drop])).float()
        cond, uncond = both[:B], both[B:]
        return uncond + guidance * (cond - uncond)

    @torch.no_grad()
    def ddim(self, model, audio_tokens, label, nps, noise_fn: Callable[[], torch.Tensor], steps: int = 50,
             guidance: float = 1.0, known: torch.Tensor | None = None, known_mask: torch.Tensor | None = None):
        """DDIM (η = 0) from pure noise. nps NaN = dropped. known/known_mask (B, W, 31): values held
        fixed (noised to each step's level) at every step, the overlap-inpainting rule (§5.6).
        noise_fn() returns a fresh (B, W, 31) noise tensor (per-job generators, see sample_long)."""
        ts = torch.linspace(self.T - 1, 0, steps).round().long().unique_consecutive().tolist()
        x = noise_fn()
        if known is not None:
            x = torch.where(known_mask, self.q_sample(known, torch.full((x.shape[0],), ts[0]), noise_fn()), x)
        for i, t in enumerate(ts):
            x0 = self._x0(model, x, t, audio_tokens, label, nps, guidance)
            ab = self.alpha_bar[t].item()
            eps = (x - math.sqrt(ab) * x0) / math.sqrt(1 - ab)
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            ab_prev = self.alpha_bar[t_prev].item() if t_prev >= 0 else 1.0
            x = math.sqrt(ab_prev) * x0 + math.sqrt(1 - ab_prev) * eps
            if known is not None:
                held = (self.q_sample(known, torch.full((x.shape[0],), t_prev), noise_fn()) if t_prev >= 0
                        else known)
                x = torch.where(known_mask, held, x)
        return x


@dataclass
class Job:
    """One song at one difficulty: label index or None, NPS or None, a seed, and n_frames at 50 fps."""
    sha: str
    label: int | None
    nps: float | None
    seed: int
    n_frames: int


def n_windows(n_frames: int, window: int) -> int:
    half = window // 2
    return 1 if n_frames <= window else math.ceil((n_frames - window) / half) + 1


def sample_long(model, diffusion: Diffusion, jobs: list[Job], audio_fn: Callable[[str, int, int], np.ndarray],
                window: int, ctx: int, steps: int = 50, guidance: float = 1.0,
                device: str = "cpu") -> list[tuple[np.ndarray, list[int]]]:
    """Whole songs by overlapping windows (§5.6), all jobs in lockstep.

    audio_fn(sha, start, n) -> (n, AUDIO_DIM) model-space audio for 50 fps frames start .. start+n-1.
    Returns per job (model-space track (n_frames, 31), join frames). Each job draws its noise from its
    own generator, so a job's output does not depend on which other jobs share the batch.
    """
    half = window // 2
    gens = [torch.Generator().manual_seed(j.seed) for j in jobs]
    out = [np.zeros((n_windows(j.n_frames, window) * half + half, model.config.n_in), np.float32) for j in jobs]
    joins: list[list[int]] = [[] for _ in jobs]
    model.eval()
    for w in range(max(n_windows(j.n_frames, window) for j in jobs)):
        act = [k for k, j in enumerate(jobs) if w < n_windows(j.n_frames, window)]
        start = w * half
        audio = torch.from_numpy(np.stack([audio_fn(jobs[k].sha, start - ctx, window + 2 * ctx) for k in act]))
        label = torch.tensor([NONE_LABEL if jobs[k].label is None else jobs[k].label for k in act])
        nps = torch.tensor([float("nan") if jobs[k].nps is None else jobs[k].nps for k in act])
        known = mask = None
        if w > 0:
            known = torch.zeros(len(act), window, model.config.n_in)
            known[:, :half] = torch.from_numpy(np.stack([out[k][start:start + half] for k in act]))
            mask = torch.zeros_like(known, dtype=torch.bool)
            mask[:, :half] = True
            for k in act:
                joins[k].append(start + half)

        def noise_fn():
            return torch.stack([torch.randn(window, model.config.n_in, generator=gens[k]) for k in act]).to(device)

        with torch.inference_mode():
            tokens = model.encode_audio(audio.to(device))
            x = diffusion.ddim(model, tokens, label.to(device), nps.to(device), noise_fn, steps, guidance,
                               None if known is None else known.to(device), None if mask is None else mask.to(device))
        x = x.float().cpu().numpy()
        for i, k in enumerate(act):
            if w == 0:
                out[k][:window] = x[i]
            else:
                out[k][start + half:start + window] = x[i, half:]
    return [(o[:j.n_frames], [f for f in jn if f < j.n_frames]) for o, j, jn in zip(out, jobs, joins)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_diffusion.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/motion/diffusion.py tests/test_motion_diffusion.py
git commit -m "Phase 4: diffusion schedule, losses, guided DDIM and overlap inpainting" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Evaluation metrics and the quick val check

**Files:**
- Create: `src/kata/motion/evaluate.py` (first part; Task 9 appends)
- Test: `tests/test_motion_evaluate.py`

**Interfaces:**
- Consumes: Task 1 (`blade`, `rot6_to_matrix`), Task 2 (`FPS`, `HIT`, `OFF`, `TRACK`), Task 4 (`decode_hits`, `WindowSampler` in `quick_eval`), Task 6 (`Diffusion.ddim` in `quick_eval`); `kata.audio.beats.f_measure`.
- Produces: `TIMING_TOL_S = 0.03`, `QUANTITIES`; `timing_f(hit_times, map_times)`; `tips(motion)`; `tip_velocity(motion)`; `quantities(motion, hits, frame0=0, valid=None) -> dict[str, ndarray]`; `agreement(motion, hits, frame0=0) -> (good, total)`; `tip_distance(a, b)`; `velocity_jumps(motion, frames=None)`; `real_hits(raw, frame0) -> (left, right)`; `quick_windows(sampler, n) -> list[(TrackRef, start)]`; `quick_eval(model, diffusion, sampler, windows, guidance, steps, device) -> {"val_timing_f", "val_rate_ratio", "n"}`.

- [ ] **Step 1: Write the failing tests** — `tests/test_motion_evaluate.py`:

```python
"""Evaluation metrics (spec §7.2) on hand-built motion."""
import numpy as np
import pytest

from kata.motion import evaluate as ev
from kata.motion.prepare import FPS, TRACK, prepare_replay
from motion_synth import synth_notes, synth_replay

IDENTITY6 = [1.0, 0, 0, 0, 1.0, 0]


def swinging(n=500, period_frames=50, phase=0.0) -> np.ndarray:
    """Head still at 1.6 m; both sabers point +z and swing up/down sinusoidally (0.3 m amplitude)."""
    m = np.zeros((n, 27))
    m[:, 1] = 1.6
    m[:, 3:9] = IDENTITY6
    k = np.arange(n)
    for part, x in (("left", -0.3), ("right", 0.3)):
        s = TRACK[part].start
        m[:, s] = x
        m[:, s + 1] = 1.0 + 0.3 * np.sin(2 * np.pi * (k + phase) / period_frames)
        m[:, s + 3:s + 9] = IDENTITY6
    return m


def test_timing_f_matches_phase_3():
    t = np.arange(2.0, 10.0, 0.5)
    assert ev.timing_f(t, t) == 1.0
    assert ev.timing_f(t + 0.02, t) == 1.0 and ev.timing_f(t + 0.04, t) == 0.0
    assert ev.timing_f(np.concatenate([t, t]), t) == 1.0              # duplicates collapse


def test_agreement_finds_hits_at_speed_peaks():
    m = swinging()
    peaks = (np.arange(0, 500, 25) / FPS)[1:-1]                      # speed peaks at every zero crossing
    good, total = ev.agreement(m, (peaks, peaks))
    assert total == 2 * len(peaks) and good == total
    good, total = ev.agreement(m, (peaks + 12 / FPS, peaks + 12 / FPS))  # at the turning points instead
    assert good == 0


def test_quantities_on_a_known_swing():
    m = swinging()
    hits = (np.arange(0, 500, 25)[1:-1] / FPS,) * 2
    q = ev.quantities(m, hits)
    v_max = 0.3 * 2 * np.pi / 50 * FPS
    assert np.allclose(q["tip_speed_at_hits"], v_max, rtol=0.01)
    assert np.allclose(q["peak_speed_per_swing"], v_max, rtol=0.01)
    assert np.allclose(q["roll_range"], 0.0) and np.allclose(q["head_disp_1s"], 0.0)
    assert set(q) == set(ev.QUANTITIES)
    valid = np.ones(500, bool)
    valid[100:200] = False
    assert len(ev.quantities(m, hits, valid=valid)["tip_speed_at_hits"]) < len(q["tip_speed_at_hits"])


def test_distance_and_velocity_jumps():
    a = swinging()
    b = a.copy()
    b[:, TRACK["left"].start] += 0.1
    b[:, TRACK["right"].start] += 0.1
    assert ev.tip_distance(a, a) == 0.0 and ev.tip_distance(a, b) == pytest.approx(0.1)
    c = swinging()
    c[250:] = swinging(phase=25)[250:]                                 # a seam that reverses the swing
    jumps = ev.velocity_jumps(c, [100, 250])
    assert jumps[0] < 0.3 and jumps[1] > 1.0
    assert len(ev.velocity_jumps(c)) == 499


def test_real_hits_from_a_prepared_track():
    notes = synth_notes()
    p = prepare_replay(synth_replay(notes), notes, left_handed=False)
    left, right = ev.real_hits(p.data, p.frame0)
    assert np.allclose(np.sort(np.concatenate([left, right])), notes["time_s"], atol=1e-5)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_evaluate.py -q`
Expected: collection error, `ImportError: cannot import name 'evaluate' from 'kata.motion'`.

- [ ] **Step 3: Implement** — `src/kata/motion/evaluate.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_evaluate.py -q`
Expected: `5 passed`. (`quick_eval` is exercised by the training tests in Task 8.)

- [ ] **Step 5: Commit**

```bash
git add src/kata/motion/evaluate.py tests/test_motion_evaluate.py
git commit -m "Phase 4: evaluation metrics and the quick val check" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Training, runs and exact resume

**Files:**
- Create: `src/kata/motion/train.py`
- Test: `tests/test_motion_train.py`

**Interfaces:**
- Consumes: Tasks 2–7: `MotionStore`, `load_tracks`, `Normaliser`, `WindowSampler`, `CTX`, `WINDOW`, `Diffusion`, `LossWeights`, `ModelConfig`, `MotionDenoiser`, `quick_eval`, `quick_windows`; `kata.audio.store.FeatureStore`.
- Produces: `RunConfig` (all §6 hyperparameters, plus `pilot`, `workers`, `only_songs`, `.to_dict()`, `.from_dict()`); `tracks_table(paths, pilot)`; `songs_with_features(paths)`; `Run(paths, run_id)` with `.dir`, `.config_path`, `.log_path`, `.ckpt_dir`, `.motion_stats_path`, `.audio_stats_path`, `.guidance_path`, `Run.create(paths, run_id, config)`, `.config()`, `.normaliser()`, `.diffusion()`, `.sampler(split, batch=None, seed=None)`, `.load_model(which="best", device="cpu")` (EMA weights), `.guidance()`, `.log(row)`; `lr_factor(config)`; `StepBatches`; `train(paths, run_id, config=None, resume=False, hours=None, device="cuda", max_steps=None) -> {"run", "step", "stopped", "best_val_timing_f", "seconds"}`.

- [ ] **Step 1: Write the failing tests** — `tests/test_motion_train.py`:

```python
"""Runs, checkpoints, exact resume and the time limit (spec §6.3-6.4), on CPU with a tiny model."""
import json

import pytest

torch = pytest.importorskip("torch")

from kata.config import Paths  # noqa: E402
from kata.motion import pipeline as mp  # noqa: E402
from kata.motion.model import ModelConfig  # noqa: E402
from kata.motion.train import Run, RunConfig, lr_factor, train  # noqa: E402
from motion_synth import write_root  # noqa: E402

TINY = RunConfig(model=ModelConfig(d=32, heads=4, enc_layers=1, dec_layers=1, dropout=0.1), batch=2, warmup=2,
                 total_steps=6, eval_every=3, eval_diffs=2, ddim_steps=2, workers=0, log_every=1)


@pytest.fixture
def paths(tmp_path) -> Paths:
    p = Paths(tmp_path).ensure()
    write_root(p)
    mp.prepare_stage(p)
    return p


def _state(paths, run_id):
    return torch.load(Run(paths, run_id).ckpt_dir / "last.pt", weights_only=True)


def test_resumed_training_is_identical_to_straight_training(paths):
    a = train(paths, "straight", TINY, device="cpu", max_steps=4)
    assert a["step"] == 4 and a["stopped"] == "max_steps"
    train(paths, "resumed", TINY, device="cpu", max_steps=2)
    b = train(paths, "resumed", resume=True, device="cpu", max_steps=4)
    assert b["step"] == 4
    sa, sb = _state(paths, "straight"), _state(paths, "resumed")
    for part in ("model", "ema"):
        assert all(torch.equal(sa[part][k], sb[part][k]) for k in sa[part]), part
    assert torch.equal(sa["rng"]["cpu"], sb["rng"]["cpu"])


def test_run_guards_and_logs(paths):
    train(paths, "r", TINY, device="cpu", max_steps=3)
    with pytest.raises(ValueError, match="resume"):
        train(paths, "r", TINY, device="cpu")
    with pytest.raises(FileNotFoundError):
        train(paths, "missing", resume=True, device="cpu")
    other = RunConfig(**{**TINY.__dict__, "lr": 1e-3})
    with pytest.raises(ValueError, match="different config"):
        Run.create(paths, "r", other)
    run = Run(paths, "r")
    rows = [json.loads(line) for line in run.log_path.read_text().splitlines()]
    assert [r["step"] for r in rows if "loss" in r] == [1, 2, 3]
    ev = [r["eval"] for r in rows if "eval" in r]
    assert len(ev) == 1 and ev[0]["n"] == 1 and 0.0 <= ev[0]["val_timing_f"] <= 1.0
    assert (run.ckpt_dir / "best.pt").exists()
    assert run.load_model("best").config == TINY.model
    assert run.motion_stats_path.exists() and run.audio_stats_path.exists()


def test_time_limit_stops_cleanly_with_a_checkpoint(paths):
    s = train(paths, "t", TINY, device="cpu", hours=0.0)
    assert s["stopped"] == "time_limit" and s["step"] == 1
    assert _state(paths, "t")["step"] == 1


def test_lr_schedule():
    f = lr_factor(RunConfig(warmup=10, total_steps=110, min_lr_frac=0.1))
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(60) == pytest.approx(0.55) and f(110) == pytest.approx(0.1)


def test_overfit_mode_trains_and_checks_on_the_chosen_song_only(paths):
    cfg = RunConfig(**{**TINY.__dict__, "only_songs": ("a" * 64,)})
    assert train(paths, "o", cfg, device="cpu", max_steps=2)["step"] == 2
    run = Run(paths, "o")
    assert run.config().only_songs == ("a" * 64,)                 # survives the JSON round trip
    assert [sha for sha, _ in run.sampler("train").songs] == ["a" * 64]
    assert run.sampler("val").songs == []
    ev = [json.loads(x)["eval"] for x in run.log_path.read_text().splitlines() if '"eval"' in x]
    assert ev[-1]["n"] == 1                                       # the quick check ran on the train song
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_train.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'kata.motion.train'`.

- [ ] **Step 3: Implement** — `src/kata/motion/train.py`:

```python
"""Training loop, runs, checkpoints and exact resume (docs/phases/phase-4-motion.md §6.3-6.4)."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch

from kata.audio.store import FeatureStore
from kata.config import Paths
from kata.motion.dataset import CTX, WINDOW, Normaliser, WindowSampler, load_tracks
from kata.motion.diffusion import Diffusion, LossWeights
from kata.motion.model import ModelConfig, MotionDenoiser
from kata.motion.prepare import MotionStore


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    window: int = WINDOW
    ctx: int = CTX
    batch: int = 64
    lr: float = 2e-4
    warmup: int = 2000
    total_steps: int = 80_000        # sets the lr schedule; a run may span nights (§6.4)
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    ema: float = 0.999
    seed: int = 0
    p_uncond: float = 0.1
    w_tip: float = 1.0
    w_tip_vel: float = 1.0
    ckpt_minutes: float = 30.0
    eval_every: int = 5000
    log_every: int = 100
    eval_diffs: int = 50
    eval_guidance: float = 2.0
    ddim_steps: int = 50
    mirror_aug: bool = False
    use_stems: bool = True
    pilot: bool = False
    workers: int = 4
    only_songs: tuple[str, ...] = ()  # overfit check (§6.5): train and quick-check on these songs only

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        d["model"] = ModelConfig(**d["model"])
        d["only_songs"] = tuple(d.get("only_songs", ()))
        return cls(**d)


def tracks_table(paths: Paths, pilot: bool) -> pl.DataFrame:
    return pl.read_parquet(paths.processed / f"motion_tracks{'_pilot' if pilot else ''}.parquet")


def songs_with_features(paths: Paths) -> set[str]:
    f = pl.read_parquet(paths.processed / "audio_features.parquet")
    return set(f.filter(pl.col("status") == "ok")["audio_sha256"])


class Run:
    """runs/motion/{run_id}/: config.json, the stats it was trained with, log.jsonl, ckpt/{last,best}.pt."""

    def __init__(self, paths: Paths, run_id: str):
        self.paths, self.id = paths, run_id
        self.dir = paths.motion_runs / run_id
        self.config_path = self.dir / "config.json"
        self.log_path = self.dir / "log.jsonl"
        self.ckpt_dir = self.dir / "ckpt"
        self.motion_stats_path = self.dir / "motion_stats.json"
        self.audio_stats_path = self.dir / "audio_stats.json"
        self.guidance_path = self.dir / "guidance.json"

    @classmethod
    def create(cls, paths: Paths, run_id: str, config: RunConfig) -> "Run":
        run = cls(paths, run_id)
        if run.config_path.exists():
            if json.loads(run.config_path.read_text()) != config.to_dict():
                raise ValueError(f"run {run_id} exists with a different config; use a new run id")
            return run
        run.ckpt_dir.mkdir(parents=True, exist_ok=True)
        run.config_path.write_text(json.dumps(config.to_dict(), indent=1, sort_keys=True))
        run.motion_stats_path.write_bytes(MotionStore(paths.motion).stats_path.read_bytes())
        run.audio_stats_path.write_bytes(FeatureStore(paths.features).stats_path.read_bytes())
        return run

    def config(self) -> RunConfig:
        return RunConfig.from_dict(json.loads(self.config_path.read_text()))

    def normaliser(self) -> Normaliser:
        return Normaliser(json.loads(self.motion_stats_path.read_text()), json.loads(self.audio_stats_path.read_text()))

    def diffusion(self) -> Diffusion:
        n = self.normaliser()
        return Diffusion(n.m_mean, n.m_std)

    def sampler(self, split: str, batch: int | None = None, seed: int | None = None) -> WindowSampler:
        c = self.config()
        songs = set(c.only_songs) if c.only_songs else songs_with_features(self.paths)
        tracks = load_tracks(tracks_table(self.paths, c.pilot), split, songs, c.window)
        return WindowSampler(tracks, MotionStore(self.paths.motion), FeatureStore(self.paths.features),
                             self.normaliser(), c.window, c.ctx, c.seed if seed is None else seed,
                             c.batch if batch is None else batch, c.mirror_aug and split == "train", c.use_stems)

    def load_model(self, which: str = "best", device: str = "cpu") -> MotionDenoiser:
        """The EMA weights of ckpt/{which}.pt, in eval mode."""
        state = torch.load(self.ckpt_dir / f"{which}.pt", map_location="cpu", weights_only=True)
        model = MotionDenoiser(self.config().model)
        model.load_state_dict(state["ema"])
        return model.to(device).eval()

    def guidance(self) -> float:
        if self.guidance_path.exists():
            return float(json.loads(self.guidance_path.read_text())["guidance"])
        return self.config().eval_guidance

    def log(self, row: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def lr_factor(c: RunConfig):
    def f(step: int) -> float:
        if step < c.warmup:
            return (step + 1) / c.warmup
        progress = min(1.0, (step - c.warmup) / max(1, c.total_steps - c.warmup))
        return c.min_lr_frac + (1 - c.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


class StepBatches(torch.utils.data.Dataset):
    """Item i is the whole batch of global step first + i (WindowSampler.batch is a function of step)."""

    def __init__(self, sampler: WindowSampler, first: int, last: int):
        self.sampler, self.first, self.last = sampler, first, last

    def __len__(self) -> int:
        return max(0, self.last - self.first)

    def __getitem__(self, i: int) -> dict:
        return {k: torch.from_numpy(v) for k, v in self.sampler.batch(self.first + i).items()}


def _save(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _rng_state(device: str) -> dict:
    s = {"cpu": torch.get_rng_state()}
    if device.startswith("cuda"):
        s["cuda"] = torch.cuda.get_rng_state(torch.device(device))
    return s


def _set_rng_state(s: dict, device: str) -> None:
    torch.set_rng_state(s["cpu"])
    if "cuda" in s and device.startswith("cuda"):
        torch.cuda.set_rng_state(s["cuda"], torch.device(device))


def train(paths: Paths, run_id: str, config: RunConfig | None = None, resume: bool = False,
          hours: float | None = None, device: str = "cuda", max_steps: int | None = None, log=print) -> dict:
    """Train run_id. Fresh (config given, no checkpoint yet) or --resume from ckpt/last.pt.

    Stops at total_steps, at max_steps, or after `hours` of wall time; always saves last.pt on exit.
    """
    from kata.motion.evaluate import quick_eval, quick_windows
    run = Run(paths, run_id)
    if resume:
        if not (run.ckpt_dir / "last.pt").exists():
            raise FileNotFoundError(f"no checkpoint to resume in {run.ckpt_dir}")
        c = run.config()
    else:
        if config is None:
            raise ValueError("a fresh run needs a config")
        if (run.ckpt_dir / "last.pt").exists():
            raise ValueError(f"run {run_id} already has a checkpoint; pass resume=True")
        run = Run.create(paths, run_id, config)
        c = config
    torch.manual_seed(c.seed)
    model = MotionDenoiser(c.model).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, betas=(c.beta1, c.beta2), weight_decay=c.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(c))
    step, best = 0, -math.inf
    if resume:
        state = torch.load(run.ckpt_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, best = state["step"], state["best"]
    diffusion = run.diffusion()
    weights = LossWeights(c.w_tip, c.w_tip_vel)
    sampler = run.sampler("train")
    val_sampler = run.sampler("train" if c.only_songs else "val", seed=0)    # overfit: check the song itself
    val_windows = quick_windows(val_sampler, c.eval_diffs)
    stop_at = min(c.total_steps, max_steps or c.total_steps)
    loader = torch.utils.data.DataLoader(StepBatches(sampler, step, stop_at), batch_size=None,
                                         num_workers=c.workers, persistent_workers=c.workers > 0,
                                         prefetch_factor=2 if c.workers > 0 else None)
    # The iterator draws its base seed from the global RNG, so create it before restoring the RNG:
    # a resumed run then continues from exactly the state a straight run would have.
    batches = iter(loader)
    if resume:
        _set_rng_state(state["rng"], device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast(
        "cpu", enabled=False)

    def checkpoint(name: str) -> None:
        _save(run.ckpt_dir / f"{name}.pt", {"model": model.state_dict(), "ema": ema.state_dict(),
                                             "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                                             "best": best, "rng": _rng_state(device)})

    t_start = t_ckpt = time.monotonic()
    first_step, reason = step, "total_steps"
    model.train()
    for batch in batches:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with amp:
            loss, parts = diffusion.loss(model, batch, weights, c.p_uncond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 1 - c.ema)
        step += 1
        now = time.monotonic()
        if step % c.log_every == 0:
            row = {"step": step, "loss": loss.item(), **parts, "lr": sched.get_last_lr()[0],
                   "steps_per_s": (step - first_step) / max(now - t_start, 1e-9)}
            run.log(row)
            log(f"  [motion-train] {json.dumps(row)}")
        if step % c.eval_every == 0 or step == stop_at:
            ev = quick_eval(ema, diffusion, val_sampler, val_windows, c.eval_guidance, c.ddim_steps, device)
            run.log({"step": step, "eval": ev})
            if ev["val_timing_f"] is not None and ev["val_timing_f"] > best:
                best = ev["val_timing_f"]
                checkpoint("best")
        if now - t_ckpt >= c.ckpt_minutes * 60:
            checkpoint("last")
            t_ckpt = now
        if hours is not None and now - t_start >= hours * 3600:
            reason = "time_limit"
            break
        if max_steps is not None and step >= max_steps:
            reason = "max_steps"
            break
    checkpoint("last")
    return {"run": run_id, "step": step, "stopped": reason, "best_val_timing_f": best,
            "seconds": time.monotonic() - t_start}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_train.py -q`
Expected: `5 passed` (about 15 s on CPU). If `test_resumed_training_is_identical_to_straight_training` fails, check that `iter(loader)` comes **before** `_set_rng_state` (Review Focus 1).

- [ ] **Step 5: Commit**

```bash
git add src/kata/motion/train.py tests/test_motion_train.py
git commit -m "Phase 4: training loop, runs, checkpoints and exact resume" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Sampling, split evaluation and the training stages

**Files:**
- Create: `src/kata/motion/sample.py`
- Modify: `src/kata/motion/evaluate.py` (append), `src/kata/motion/pipeline.py` (append)
- Test: `tests/test_motion_pipeline.py` (append)

**Interfaces:**
- Consumes: Task 8 (`Run`, `tracks_table`, `train`), Task 6 (`Job`, `sample_long`), Task 4 (`WindowSampler`, `decode_hits`), Task 7 metrics.
- Produces: `sample.MotionTrack(t, head_pos, head_rot, left_pos, left_rot, right_pos, right_rot, hits_left, hits_right)`; `to_track`; `final_run(paths)`; `durations(paths)`; `Generator(paths, run_id, which="best", device=None)` with `.run`, `.config`, `.run_jobs(jobs, guidance) -> list[dict(motion, hits_left, hits_right, joins)]`; **`generate(audio_sha256, label, nps, seed=0, guidance=None, run=None, paths=None, device=None) -> MotionTrack`** (Phase 5's entry point, spec §5.7); `eval_targets(paths, run, split)`; `spread(df, n)`; `sample_path(out_dir, key, mode, seed)`; `sample_targets(gen, targets, out_dir, guidance, seeds=(0, 1), label_only=True, batch_jobs=16)`; `load_sample(path)`. `evaluate.spearman`, `evaluate_samples(paths, run_id, split, targets, sample_dir) -> (rows, pooled)`, `criteria(rows, pooled, baseline=None) -> list[dict(criterion, name, value, pass)]`. Stages: `train_stage`, `tune_stage`, `sample_stage`, `evaluate_stage` (refuses a second test evaluation without `force=True`), `final_stage`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_motion_pipeline.py`:

```python
TINY_RUN = dict(batch=2, warmup=1, total_steps=2, eval_every=2, eval_diffs=1, ddim_steps=2, workers=0)


@pytest.fixture
def trained(paths) -> Paths:
    """A tiny run 'r1' trained for two steps on CPU, its guidance tuned and the val split sampled."""
    pytest.importorskip("torch")
    from kata.motion.model import ModelConfig
    from kata.motion.train import RunConfig
    mp.prepare_stage(paths)
    s = mp.train_stage(paths, run="r1", config=RunConfig(model=ModelConfig(d=32, heads=4, enc_layers=1,
                                                                             dec_layers=1), **TINY_RUN),
                       device="cpu")
    assert s["step"] == 2
    t = mp.tune_stage(paths, run="r1", n=1, values=(1.0, 2.0), device="cpu")
    assert t["guidance"] in (1.0, 2.0) and len(t["table"]) == 2
    assert json.loads((paths.motion_runs / "r1" / "guidance.json").read_text())["guidance"] == t["guidance"]
    s = mp.sample_stage(paths, run="r1", split="val", device="cpu")
    assert s["jobs"] == 3 and s["targets"] == 1                     # full seeds 0 and 1, label-only seed 0
    return paths


def test_sampling_resumes_evaluation_replaces_and_generate_works(trained):
    from kata.motion.sample import generate
    paths = trained
    assert mp.sample_stage(paths, run="r1", split="val", device="cpu")["jobs"] == 0   # resumable
    e = mp.evaluate_stage(paths, run="r1", split="val")
    assert [c["criterion"] for c in e["criteria"]] == [2, 3, 4, 5, 6]   # no baseline_eval: criterion 1 skipped
    ev = pl.read_parquet(paths.processed / "motion_eval.parquet")
    assert set(ev["source"]) == {"generated", "real"} and ev.filter(pl.col("source") == "generated").height == 3
    assert ev.filter(pl.col("source") == "real").height == 1
    mp.evaluate_stage(paths, run="r1", split="val")                  # replaces, never duplicates
    assert pl.read_parquet(paths.processed / "motion_eval.parquet").height == ev.height
    assert (paths.motion_runs / "r1" / "eval_val.json").exists()
    (paths.motion_runs / "r1" / "eval_test.json").write_text("{}")
    with pytest.raises(ValueError, match="already evaluated on test"):     # test is evaluated once
        mp.evaluate_stage(paths, run="r1", split="test")
    with pytest.raises(FileNotFoundError):
        mp.final_stage(paths, run="nope")
    mp.final_stage(paths, run="r1")
    track = generate("b" * 64, "Expert", 2.0, paths=paths, device="cpu")
    assert track.left_rot.shape == (len(track.t), 3, 3) and track.t[1] == pytest.approx(0.02)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_pipeline.py -q`
Expected: `2 passed, 1 error`; the error (in the `trained` fixture) is `AttributeError: module 'kata.motion.pipeline' has no attribute 'train_stage'`.

- [ ] **Step 3: Implement sampling** — `src/kata/motion/sample.py`:

```python
"""Whole-song generation from a trained run (docs/phases/phase-4-motion.md §5.6-5.7, §7.1)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from kata.audio.store import FeatureStore
from kata.config import Paths
from kata.motion.dataset import WindowSampler, decode_hits
from kata.motion.diffusion import Job, sample_long
from kata.motion.geometry import rot6_to_matrix
from kata.motion.prepare import FPS, HIT, OFF, TRACK, MotionStore
from kata.motion.train import Run
from kata.validate.rules import LABELS

KEY = ["hash", "characteristic", "difficulty"]


@dataclass
class MotionTrack:
    """Generated motion on t = i / FPS from 0 s, in the play-space frame (§3.2)."""
    t: np.ndarray                       # (T,)
    head_pos: np.ndarray                # (T, 3) m
    head_rot: np.ndarray                # (T, 3, 3)
    left_pos: np.ndarray
    left_rot: np.ndarray
    right_pos: np.ndarray
    right_rot: np.ndarray
    hits_left: np.ndarray               # sorted hit times (s)
    hits_right: np.ndarray


def to_track(motion: np.ndarray, hits_left: np.ndarray, hits_right: np.ndarray) -> MotionTrack:
    parts = {}
    for part in ("head", "left", "right"):
        s = TRACK[part].start
        parts[f"{part}_pos"] = motion[:, s:s + 3].astype(np.float64)
        parts[f"{part}_rot"] = rot6_to_matrix(motion[:, s + 3:s + 9])
    return MotionTrack(np.arange(len(motion)) / FPS, **parts, hits_left=np.sort(hits_left),
                       hits_right=np.sort(hits_right))


def final_run(paths: Paths) -> str:
    return (paths.motion_runs / "final").read_text().strip()


def durations(paths: Paths) -> dict[str, float]:
    f = pl.read_parquet(paths.processed / "audio_features.parquet").filter(pl.col("status") == "ok")
    return dict(zip(f["audio_sha256"], f["duration_s"]))


class Generator:
    """A run's model, ready to sample whole songs in batches of jobs."""

    def __init__(self, paths: Paths, run_id: str, which: str = "best", device: str | None = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.run = Run(paths, run_id)
        self.config = self.run.config()
        self.model = self.run.load_model(which, self.device)
        self.diffusion = self.run.diffusion()
        self.norm = self.run.normaliser()
        self.audio = WindowSampler([], MotionStore(paths.motion), FeatureStore(paths.features), self.norm,
                                   self.config.window, self.config.ctx, use_stems=self.config.use_stems).audio

    def run_jobs(self, jobs: list[Job], guidance: float) -> list[dict]:
        """-> per job: motion (T, 27) de-normalised, hits_left, hits_right, joins."""
        import torch
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda")
               else torch.autocast("cpu", enabled=False))
        with amp:
            res = sample_long(self.model, self.diffusion, jobs, self.audio, self.config.window, self.config.ctx,
                              self.config.ddim_steps, guidance, self.device)
        return [{"motion": self.norm.denorm_motion(x).astype(np.float32),
                 "hits_left": decode_hits(x[:, HIT[0]], x[:, OFF[0]]),
                 "hits_right": decode_hits(x[:, HIT[1]], x[:, OFF[1]]), "joins": np.asarray(joins, np.int64)}
                for x, joins in res]


def generate(audio_sha256: str, label: str | None, nps: float | None, seed: int = 0, guidance: float | None = None,
             run: str | None = None, paths: Paths | None = None, device: str | None = None) -> MotionTrack:
    """Phase 5's entry point (§5.7): one whole song at one difficulty."""
    paths = paths or Paths()
    gen = Generator(paths, run or final_run(paths), device=device)
    n = math.ceil(durations(paths)[audio_sha256] * FPS)
    job = Job(audio_sha256, None if label is None else LABELS.index(label), nps, seed, n)
    out = gen.run_jobs([job], gen.run.guidance() if guidance is None else guidance)[0]
    return to_track(out["motion"], out["hits_left"], out["hits_right"])


# --------------------------------------------------------------------------- split sampling (§7.1)

def eval_targets(paths: Paths, run: Run, split: str) -> pl.DataFrame:
    """One row per difficulty of the split with at least one accepted track: key, song, label, nps."""
    from kata.motion.train import tracks_table
    t = tracks_table(paths, run.config().pilot).filter((pl.col("status") == "ok") & (pl.col("split") == split))
    return (t.group_by(KEY).agg(pl.col("audio_sha256").first(), pl.col("label").first(), pl.col("nps").first(),
                                pl.col("hps").mean().alias("real_hps"), pl.col("score_id").sort().alias("score_ids"))
            .join(pl.DataFrame({"audio_sha256": list(durations(paths))}), on="audio_sha256", how="semi")
            .sort(KEY))


def spread(df: pl.DataFrame, n: int | None) -> pl.DataFrame:
    """n rows evenly spread over df (all when n is None or larger)."""
    if n is None or n >= df.height:
        return df
    return df[sorted(set(np.linspace(0, df.height - 1, n).round().astype(int).tolist()))]


def sample_path(out_dir: Path, key: tuple, mode: str, seed: int) -> Path:
    return out_dir / f"{key[0]}_{key[1]}_{key[2]}_{mode}_{seed}.npz"


def _save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def sample_targets(gen: Generator, targets: pl.DataFrame, out_dir: Path, guidance: float, seeds=(0, 1),
                   label_only: bool = True, batch_jobs: int = 16, log=print) -> dict:
    """Generate every target (mode 'full' per seed, plus 'label_only' seed 0) into out_dir; existing
    files are kept, so an interrupted run resumes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = durations(gen.run.paths)
    todo = []
    for r in targets.iter_rows(named=True):
        key = (r["hash"], r["characteristic"], r["difficulty"])
        n = math.ceil(dur[r["audio_sha256"]] * FPS)
        label = LABELS.index(r["label"])
        modes = [("full", s, r["nps"]) for s in seeds] + ([("label_only", 0, None)] if label_only else [])
        for mode, seed, nps in modes:
            path = sample_path(out_dir, key, mode, seed)
            if not path.exists():
                todo.append((path, Job(r["audio_sha256"], label, nps, seed, n),
                             {"key": list(key), "audio_sha256": r["audio_sha256"], "label": r["label"],
                              "nps": nps, "seed": seed, "mode": mode, "guidance": guidance}))
    import time
    t0 = time.monotonic()
    for i in range(0, len(todo), batch_jobs):
        part = todo[i:i + batch_jobs]
        for (path, _, meta), out in zip(part, gen.run_jobs([j for _, j, _ in part], guidance)):
            _save_npz(path, meta=np.array(json.dumps(meta)), **out)
        log(f"  [motion-sample] {min(i + batch_jobs, len(todo))}/{len(todo)}")
    return {"jobs": len(todo), "seconds": time.monotonic() - t0}


def load_sample(path: Path) -> dict:
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out
```

- [ ] **Step 4: Append split evaluation** to `src/kata/motion/evaluate.py`:

```python
# --------------------------------------------------------------------------- split evaluation (§7.1, §8)

def spearman(a, b) -> float:
    import polars as pl
    ra = pl.Series(np.asarray(a, np.float64)).rank("average").to_numpy()
    rb = pl.Series(np.asarray(b, np.float64)).rank("average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 else float("nan")


def _overlap(f0a: int, na: int, f0b: int, nb: int) -> tuple[slice, slice]:
    """Rows of two tracks that cover the same absolute frames."""
    lo, hi = max(f0a, f0b), max(min(f0a + na, f0b + nb), max(f0a, f0b))
    return slice(lo - f0a, hi - f0a), slice(lo - f0b, hi - f0b)


def evaluate_samples(paths, run_id: str, split: str, targets, sample_dir) -> tuple:
    """Rows for every generated sample in sample_dir and every real replay of the targets, plus the
    pooled plausibility percentiles and the real frame-to-frame velocity-jump p99.

    targets: sample.eval_targets() rows. Returns (rows DataFrame, pooled dict)."""
    import polars as pl
    from kata.motion.prepare import MotionStore
    from kata.motion.sample import load_sample, sample_path
    from kata.motion.train import Run, tracks_table
    from kata.validate.rules import nps as nps_of
    keys = ["hash", "characteristic", "difficulty"]
    store = MotionStore(paths.motion)
    meta = tracks_table(paths, Run(paths, run_id).config().pilot).select("score_id", "frame0")
    frame0 = dict(zip(meta["score_id"], meta["frame0"]))
    notes = (pl.scan_parquet(paths.processed / "notes.parquet").join(targets.select(keys).lazy(), on=keys, how="semi")
             .select(keys + ["time_s"]).collect())
    note_times = {k: g["time_s"].to_numpy() for k, g in notes.partition_by(keys, as_dict=True).items()}
    pooled = {"generated": {q: [] for q in QUANTITIES}, "real": {q: [] for q in QUANTITIES}}
    rows, real_jumps = [], []
    for r in targets.iter_rows(named=True):
        key = (r["hash"], r["characteristic"], r["difficulty"])
        base = {"run": run_id, "split": split, **dict(zip(keys, key)), "audio_sha256": r["audio_sha256"],
                "label": r["label"], "nps": r["nps"], "real_hps": r["real_hps"]}
        reals = []
        for sid in r["score_ids"]:
            raw = np.asarray(store.load(sid), np.float32)
            valid = store.valid(sid)
            hits = real_hits(raw, frame0[sid])
            for q, v in quantities(raw[:, :27], hits, frame0[sid], valid).items():
                pooled["real"][q].append(v)
            good, total = agreement(raw[:, :27], hits, frame0[sid])
            real_jumps.append(velocity_jumps(raw[:, :27])[valid[1:] & valid[:-1]])
            reals.append((raw[:, :27], frame0[sid], valid))
            rows.append({**base, "source": "real", "mode": "replay", "seed": None, "score_id": sid,
                         "timing_f": timing_f(np.concatenate(hits), note_times[key]),
                         "hit_rate": nps_of(np.unique(np.concatenate(hits))), "agree_good": good,
                         "agree_total": total})
        pair = []
        for i in range(len(reals)):
            for k in range(i + 1, len(reals)):
                (a, fa, va), (b, fb, vb) = reals[i], reals[k]
                sa, sb = _overlap(fa, len(a), fb, len(b))
                ok = va[sa] & vb[sb]
                if ok.sum() > FPS:
                    pair.append(tip_distance(a[sa][ok], b[sb][ok]))
        if pair:
            rows[-len(reals)]["diversity"] = float(np.mean(pair))
        gen = {}
        for mode, seed in (("full", 0), ("full", 1), ("label_only", 0)):
            path = sample_path(sample_dir, key, mode, seed)
            if not path.exists():
                continue
            s = load_sample(path)
            gen[(mode, seed)] = len(rows)
            hits = (s["hits_left"], s["hits_right"])
            if mode == "full":
                for q, v in quantities(s["motion"], hits).items():
                    pooled["generated"][q].append(v)
            good, total = agreement(s["motion"], hits)
            rows.append({**base, "source": "generated", "mode": mode, "seed": seed, "score_id": None,
                         "timing_f": timing_f(np.concatenate(hits), note_times[key]),
                         "hit_rate": nps_of(np.unique(np.round(np.concatenate(hits), 4))),
                         "agree_good": good, "agree_total": total,
                         "seam_jumps": velocity_jumps(s["motion"], s["joins"]).tolist(), "_motion": s["motion"]})
        if ("full", 0) in gen and ("full", 1) in gen:
            rows[gen[("full", 0)]]["diversity"] = tip_distance(rows[gen[("full", 0)]]["_motion"],
                                                               rows[gen[("full", 1)]]["_motion"])
    for row in rows:
        row.pop("_motion", None)
    jumps = np.concatenate(real_jumps) if real_jumps else np.zeros(0)
    out = {}
    for src, d in pooled.items():
        out[src] = {}
        for q, parts in d.items():
            v = np.concatenate(parts) if parts else np.zeros(0)
            out[src][q] = [float(np.percentile(v, 50)), float(np.percentile(v, 95))] if len(v) else None
    out["real_jump_p99"] = float(np.percentile(jumps, 99)) if len(jumps) else None
    df = pl.DataFrame(rows, infer_schema_length=None)
    for col, dtype in (("diversity", pl.Float64), ("seam_jumps", pl.List(pl.Float64))):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return df, out


def criteria(rows, pooled: dict, baseline=None) -> list[dict]:
    """Acceptance criteria 1-6 (§8) from evaluate_samples' output. baseline: baseline_eval rows
    (audio_sha256, difficulty, timing_f) for criterion 1, or None to skip it."""
    import polars as pl
    from kata.validate.rules import LABELS
    gen = rows.filter(pl.col("source") == "generated")
    full = gen.filter(pl.col("mode") == "full")
    full0 = full.filter(pl.col("seed") == 0)
    real = rows.filter(pl.col("source") == "real")
    out = []
    if baseline is not None:                                    # 1. same songs and labels as the baseline
        ours = full0.sort("hash").group_by("audio_sha256", "difficulty").agg(pl.col("timing_f").first())
        both = ours.join(baseline.select("audio_sha256", "difficulty", pl.col("timing_f").alias("base_f")),
                         on=["audio_sha256", "difficulty"]).drop_nulls()
        per_label = both.group_by("difficulty").agg(pl.col("timing_f").median(), pl.col("base_f").median())
        wins = int((per_label["timing_f"] > per_label["base_f"]).sum())
        ok = both.height > 0 and both["timing_f"].median() > both["base_f"].median() \
            and wins >= min(4, per_label.height)
        value = (f"median F {both['timing_f'].median():.3f} vs. baseline {both['base_f'].median():.3f}; "
                 f"ahead in {wins}/{per_label.height} labels") if both.height else "no common songs"
        out.append({"criterion": 1, "name": "hit timing vs. baseline", "value": value, "pass": bool(ok)})
    worst, ok2 = 0.0, True                                      # 2. p50, p95 within ±15 % of real
    for q in QUANTITIES:
        g, r = pooled["generated"].get(q), pooled["real"].get(q)
        if not g or not r:
            ok2 = False
            continue
        for gi, ri in zip(g, r):
            dev = abs(gi / ri - 1) if ri else float("inf")
            worst, ok2 = max(worst, dev), ok2 and dev <= 0.15
    out.append({"criterion": 2, "name": "physical plausibility", "value": f"worst deviation {worst:.1%}",
                "pass": bool(ok2)})
    g_rate = full["agree_good"].sum() / max(full["agree_total"].sum(), 1)    # 3. hits agree with motion
    r_rate = real["agree_good"].sum() / max(real["agree_total"].sum(), 1)
    out.append({"criterion": 3, "name": "hits agree with motion", "value": f"{g_rate:.1%} vs. real {r_rate:.1%}",
                "pass": bool(g_rate >= r_rate - 0.05)})
    rho = spearman(full0["nps"], full0["hit_rate"])                          # 4. control
    ratio = float((full0["hit_rate"] / full0["real_hps"] - 1).abs().median() or 0.0) if full0.height else float("nan")
    by_label = dict(gen.filter(pl.col("mode") == "label_only").group_by("label").agg(pl.col("hit_rate").median())
                    .iter_rows())
    order = [by_label[lab] for lab in LABELS if lab in by_label]
    increasing = all(b > a for a, b in zip(order, order[1:]))
    out.append({"criterion": 4, "name": "control",
                "value": f"Spearman {rho:.2f}; median |rate/real - 1| {ratio:.2f}; label-only increasing: {increasing}",
                "pass": bool(rho >= 0.9 and ratio <= 0.15 and increasing)})
    g_div, r_div = full0["diversity"].drop_nulls().mean(), real["diversity"].drop_nulls().mean()  # 5. diversity
    out.append({"criterion": 5, "name": "diversity",
                "value": f"{g_div if g_div is not None else float('nan'):.3f} m vs. real "
                         f"{r_div if r_div is not None else float('nan'):.3f} m",
                "pass": bool(g_div is not None and r_div and g_div >= 0.5 * r_div)})
    lists = [np.asarray(j, np.float64) for j in gen["seam_jumps"].drop_nulls().to_list()]   # 6. seams
    jumps = np.concatenate(lists) if lists else np.zeros(0)
    p99 = pooled.get("real_jump_p99")
    share = float((jumps <= p99).mean()) if len(jumps) and p99 is not None else 1.0
    out.append({"criterion": 6, "name": "seams", "value": f"{share:.1%} of {len(jumps)} joins within real p99",
                "pass": bool(share >= 0.99)})
    return out
```

- [ ] **Step 5: Append the stages** to `src/kata/motion/pipeline.py`:

```python
# --------------------------------------------------------------------------- training and sampling
# These stages import torch (the `audio` extra) inside the function, like the Phase 2 stages.

GUIDANCE_VALUES = (1.0, 1.5, 2.0, 3.0, 4.0)
TUNE_DIFFS = 50
SPEED_SONGS = 3


def _train(paths, state, summary, run, config=None, resume=False, hours=None, device="cuda", max_steps=None):
    from kata.motion.train import train
    summary.update(train(paths, run, config, resume, hours, device, max_steps))

train_stage = _with_state("motion-train", _train)


def _sample_dir(paths: Paths, run: str, split: str) -> Path:
    return paths.generated / "motion" / run / split


def _tune(paths, state, summary, run, n=TUNE_DIFFS, values=GUIDANCE_VALUES, device=None):
    """Guidance weight on val (§7.1): best median timing F among the values passing criteria 2 and 4."""
    from kata.motion.evaluate import criteria, evaluate_samples
    from kata.motion.sample import Generator, eval_targets, sample_targets, spread
    gen = Generator(paths, run, device=device)
    targets = spread(eval_targets(paths, gen.run, "val"), n)
    table = []
    for w in values:
        d = _sample_dir(paths, run, "tune") / f"w{w}"
        sample_targets(gen, targets, d, w, seeds=(0,), label_only=False)
        rows, pooled = evaluate_samples(paths, run, "val", targets, d)
        crit = {c["criterion"]: c for c in criteria(rows, pooled)}
        f = rows.filter(pl.col("source") == "generated")["timing_f"].median()
        table.append({"guidance": w, "median_f": f, "plausible": crit[2]["pass"], "control": crit[4]["pass"]})
    passing = [r for r in table if r["plausible"] and r["control"]]
    best = max(passing or table, key=lambda r: r["median_f"] or 0.0)
    note = None if passing else "no value passes criteria 2 and 4; best timing F chosen"
    gen.run.guidance_path.write_text(json.dumps({"guidance": best["guidance"], "table": table, "note": note},
                                                indent=1))
    summary.update(guidance=best["guidance"], table=table, note=note, n_targets=targets.height)

tune_stage = _with_state("motion-tune", _tune)


def _sample(paths, state, summary, run, split="val", limit=None, device=None, batch_jobs=16):
    from kata.motion.sample import Generator, eval_targets, sample_targets, spread
    gen = Generator(paths, run, device=device)
    targets = spread(eval_targets(paths, gen.run, split), limit)
    summary.update(sample_targets(gen, targets, _sample_dir(paths, run, split), gen.run.guidance(),
                                  batch_jobs=batch_jobs), targets=targets.height, guidance=gen.run.guidance())

sample_stage = _with_state("motion-sample", _sample)


def _evaluate(paths, state, summary, run, split="val", limit=None, device=None, force=False):
    """Criteria for one split. Test is evaluated once per run (criterion 9); force only with approval."""
    import time

    from kata.motion.evaluate import criteria, evaluate_samples
    from kata.motion.sample import durations, eval_targets, generate, spread
    from kata.motion.train import Run
    if split == "test" and (paths.motion_runs / run / "eval_test.json").exists() and not force:
        raise ValueError(f"run {run} was already evaluated on test; re-running needs force=True and your approval")
    targets = spread(eval_targets(paths, Run(paths, run), split), limit)
    rows, pooled = evaluate_samples(paths, run, split, targets, _sample_dir(paths, run, split))
    base_path = paths.processed / "baseline_eval.parquet"
    baseline = pl.read_parquet(base_path).filter(pl.col("split") == split) if base_path.exists() else None
    crit = criteria(rows, pooled, baseline)
    speed = []
    if split == "test":                                      # criterion 11, on the first few test songs
        dur = durations(paths)
        for r in targets.head(SPEED_SONGS).iter_rows(named=True):
            t0 = time.monotonic()
            generate(r["audio_sha256"], r["label"], r["nps"], run=run, paths=paths, device=device)
            speed.append((time.monotonic() - t0) * 180.0 / dur[r["audio_sha256"]])
        crit.append({"criterion": 11, "name": "speed", "value": f"max {max(speed):.1f} s per 3-min song",
                     "pass": max(speed) <= 60.0})
    f = paths.processed / "motion_eval.parquet"
    if f.exists():
        old = pl.read_parquet(f).filter(~((pl.col("run") == run) & (pl.col("split") == split)))
        rows = pl.concat([old, rows], how="diagonal_relaxed")
    rows.sort("run", "split", "hash", "characteristic", "difficulty", "source", "mode", "seed", "score_id",
              nulls_last=True).write_parquet(f)
    out = {"criteria": crit, "pooled": pooled, "speed": speed, "targets": targets.height}
    (paths.motion_runs / run / f"eval_{split}.json").write_text(json.dumps(out, indent=1))
    summary.update(criteria=[{k: c[k] for k in ("criterion", "value", "pass")} for c in crit])

evaluate_stage = _with_state("motion-evaluate", _evaluate)


def _final(paths, state, summary, run):
    if not (paths.motion_runs / run / "ckpt" / "best.pt").exists():
        raise FileNotFoundError(f"run {run} has no best checkpoint")
    (paths.motion_runs / "final").write_text(run)
    summary["final"] = run

final_stage = _with_state("motion-final", _final)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_pipeline.py -q`
Expected: `3 passed` (about 10 s on CPU: two training steps of a tiny model, guidance tuning over two values, sampling and evaluation).

- [ ] **Step 7: Commit**

```bash
git add src/kata/motion/sample.py src/kata/motion/evaluate.py src/kata/motion/pipeline.py tests/test_motion_pipeline.py
git commit -m "Phase 4: generate(), split sampling, evaluation and criteria stages" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Viewer, report and CLI

**Files:**
- Create: `src/kata/motion/viewer.py`, `src/kata/motion/report.py`
- Modify: `src/kata/motion/pipeline.py` (append), `src/kata/cli.py`
- Test: `tests/test_motion_pipeline.py` (append), `tests/test_motion_cli.py`

**Interfaces:**
- Consumes: Task 9 (`eval_targets`, `load_sample`, `sample_path`, `spread`, stages), Task 7 (`real_hits`, `QUANTITIES`), Task 1 (`matrix_to_quat`, `rot6_to_matrix`); `kata.data.report.md_table`.
- Produces: `viewer.THREE_URL`, `clip(motion, hits, frame0=0, notes=None) -> dict`, `page(title, a, b, labels, blind, seed=0) -> str`, `write_page(path, ...)`; `report.write_report(paths, state, name, run=None, pilot=False) -> Path`; stages `viewer_stage`, `report_stage`; CLI `kata motion prepare | train | tune | sample | evaluate | final | viewer | report`.

- [ ] **Step 1: Write the failing tests** — add `import re` under `import json` at the top of `tests/test_motion_pipeline.py`, then append:

```python
def test_viewer_pages_and_report(trained):
    paths = trained
    mp.evaluate_stage(paths, run="r1", split="val")
    mp.final_stage(paths, run="r1")
    v = mp.viewer_stage(paths, run="r1", split="val", n=1)
    page = open(v["pages"][0], encoding="utf-8").read()
    assert "three@0.160.0" in page and '"blind":true' in page
    data = json.loads(re.search(r"const DATA = (\{.*?\});$", page, re.M).group(1))
    assert data["clips"][0]["notes"] == data["clips"][1]["notes"] != []     # notes never reveal the real clip
    r = mp.report_stage(paths, name="t")
    text = open(r["path"], encoding="utf-8").read()
    assert "Acceptance criteria" in text and "Blade check" in text and "| r1 |" in text
```

and create `tests/test_motion_cli.py`:

```python
"""`kata motion …` commands are wired to their stages."""
from typer.testing import CliRunner

from kata.cli import app
from kata.config import Paths
from motion_synth import write_root


def test_motion_commands_are_registered():
    out = CliRunner().invoke(app, ["motion", "--help"]).output
    for cmd in ("prepare", "train", "tune", "sample", "evaluate", "final", "viewer", "report"):
        assert cmd in out
    assert "--tier" in CliRunner().invoke(app, ["data", "crawl-leaderboards", "--help"]).output


def test_motion_prepare_runs_from_the_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("KATA_DATA", str(tmp_path))
    write_root(Paths(tmp_path).ensure())
    res = CliRunner().invoke(app, ["motion", "prepare"])
    assert res.exit_code == 0, res.output
    assert '"ok": 3' in res.output and (tmp_path / "processed" / "motion_tracks.parquet").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_motion_pipeline.py tests/test_motion_cli.py -q`
Expected: failures with `AttributeError: module 'kata.motion.pipeline' has no attribute 'viewer_stage'` and, in the CLI test, `motion` missing from the help output.

- [ ] **Step 3: Implement the viewer** — `src/kata/motion/viewer.py`:

```python
"""Local 3D viewer pages (docs/phases/phase-4-motion.md §7.3): two clips of one song side by side,
synced to an audio file the viewer picks. Pages stay on this machine; audio is never embedded.

The game's axes are left-handed (+z towards the notes); three.js is right-handed. Clips are
converted by flipping z: positions (x, y, -z), quaternions (-qx, -qy, qz, qw). The blade, along
saber-local +z in the game, is then along local -z.
"""

from __future__ import annotations

import html
import json
import random
from pathlib import Path

import numpy as np

from kata.motion.geometry import matrix_to_quat, rot6_to_matrix
from kata.motion.prepare import FPS, TRACK

THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"


def clip(motion: np.ndarray, hits: tuple, frame0: int = 0, notes: list | None = None) -> dict:
    """(T, 27) motion -> viewer JSON: per part flat [x, y, z, qx, qy, qz, qw] per frame (z-flipped)."""
    out = {"fps": FPS, "frame0": int(frame0), "hits": [np.round(np.asarray(h, float), 4).tolist() for h in hits],
           "notes": notes or []}
    for part in ("head", "left", "right"):
        s = TRACK[part].start
        pos = motion[:, s:s + 3].astype(np.float64) * [1, 1, -1]
        q = matrix_to_quat(rot6_to_matrix(motion[:, s + 3:s + 9])) * [-1, -1, 1, 1]
        out[part] = np.round(np.concatenate([pos, q], 1), 4).ravel().tolist()
    return out


def page(title: str, a: dict, b: dict, labels: tuple[str, str], blind: bool, seed: int = 0) -> str:
    """One HTML page; with blind=True the clips are shown as A/B in random order until revealed."""
    clips = [(a, labels[0]), (b, labels[1])]
    if blind:
        random.Random(seed).shuffle(clips)
    data = {"clips": [c for c, _ in clips], "labels": [lab for _, lab in clips], "blind": blind}
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__THREE__", THREE_URL).replace(
        "__DATA__", json.dumps(data, separators=(",", ":")))


def write_page(path: Path, *args, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page(*args, **kwargs), encoding="utf-8")
    return path


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg: #f4f4f2; --fg: #1d1d1b; --muted: #6b6b66; --panel: #ffffff; --line: #d8d8d2; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecece8; --muted: #9a9a94; --panel: #20201e; --line: #34342f; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; }
header { padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
h1 { font-size: 16px; margin: 0 12px 0 0; }
.views { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; padding: 0 16px 16px; }
.view { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.view canvas { width: 100%; height: 420px; display: block; }
.view .tag { padding: 8px 12px; color: var(--muted); }
input[type=range] { flex: 1 1 240px; }
button { font: inherit; padding: 4px 12px; }
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <label>Song audio <input id="file" type="file" accept="audio/*,.egg,.ogg"></label>
  <button id="play">Play</button>
  <input id="seek" type="range" min="0" max="1" step="0.001" value="0">
  <span id="time">0.0 s</span>
  <button id="reveal" hidden>Reveal</button>
</header>
<div class="views"><div class="view"><canvas id="c0"></canvas><div class="tag" id="t0"></div></div>
<div class="view"><canvas id="c1"></canvas><div class="tag" id="t1"></div></div></div>
<audio id="audio"></audio>
<script type="module">
import * as THREE from "__THREE__";
const DATA = __DATA__;
const audio = document.getElementById("audio");
let playing = false, clock0 = 0, offset = 0;
const duration = Math.max(...DATA.clips.map(c => (c.frame0 + c.head.length / 7) / c.fps));
const now = () => audio.src ? audio.currentTime : (playing ? offset + (performance.now() - clock0) / 1000 : offset);
function makeView(canvas, c) {
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue("--panel").trim());
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));
  const cam = new THREE.PerspectiveCamera(55, 1, 0.05, 100);
  cam.position.set(0, 1.7, 2.6); cam.lookAt(0, 1.2, -2);
  const grid = new THREE.GridHelper(4, 8, 0x888888, 0x555555); scene.add(grid);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 16), new THREE.MeshStandardMaterial({color: 0xbbbbbb}));
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.04, 0.12, 12), new THREE.MeshStandardMaterial({color: 0x777777}));
  nose.rotation.x = -Math.PI / 2; nose.position.z = -0.14; head.add(nose); scene.add(head);
  const sabers = [0xd8332a, 0x2a6fd8].map(col => {
    const g = new THREE.BoxGeometry(0.03, 0.03, 1.0); g.translate(0, 0, -0.5);
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({color: col, emissive: 0x000000}));
    scene.add(m); return m;
  });
  const notes = c.notes.map(n => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.4),
      new THREE.MeshStandardMaterial({color: n[3] === 0 ? 0xd8332a : 0x2a6fd8, transparent: true, opacity: 0.8}));
    m.position.set((n[1] - 1.5) * 0.6, 0.25 + n[2] * 0.55, 0); scene.add(m); return [n[0], m];
  });
  const parts = [["head", head], ["left", sabers[0]], ["right", sabers[1]]];
  function pose(arr, i, obj) {
    const k = 7 * i; obj.position.set(arr[k], arr[k + 1], arr[k + 2]);
    obj.quaternion.set(arr[k + 3], arr[k + 4], arr[k + 5], arr[k + 6]);
  }
  function draw(t) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) { renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
    const n = c.head.length / 7;
    const i = Math.min(n - 1, Math.max(0, Math.round(t * c.fps) - c.frame0));
    for (const [name, obj] of parts) pose(c[name], i, obj);
    c.hits.forEach((hs, s) => {
      const lit = hs.some(x => t >= x && t - x < 0.1);
      sabers[s].material.emissive.setHex(lit ? 0x666666 : 0x000000);
    });
    for (const [tn, m] of notes) { const z = -(tn - t) * 10; m.visible = z < 0.5 && z > -30; m.position.z = z; }
    renderer.render(scene, cam);
  }
  return draw;
}
const views = DATA.clips.map((c, k) => makeView(document.getElementById("c" + k), c));
const tags = [0, 1].map(k => document.getElementById("t" + k));
function label(reveal) { tags.forEach((el, k) => el.textContent = DATA.blind && !reveal ? ["A", "B"][k] : DATA.labels[k]); }
label(false);
if (DATA.blind) { const r = document.getElementById("reveal"); r.hidden = false; r.onclick = () => label(true); }
document.getElementById("file").onchange = e => { audio.src = URL.createObjectURL(e.target.files[0]); };
document.getElementById("play").onclick = () => {
  if (audio.src) { audio.paused ? audio.play() : audio.pause(); playing = !audio.paused; }
  else { if (playing) { offset = now(); playing = false; } else { clock0 = performance.now(); playing = true; } }
};
const seek = document.getElementById("seek");
seek.oninput = () => { const t = seek.value * duration; if (audio.src) audio.currentTime = t; offset = t; clock0 = performance.now(); };
(function loop() {
  const t = now();
  views.forEach(v => v(t));
  document.getElementById("time").textContent = t.toFixed(1) + " s";
  if (document.activeElement !== seek) seek.value = t / duration;
  requestAnimationFrame(loop);
})();
</script></body></html>
"""
```

- [ ] **Step 4: Implement the report** — `src/kata/motion/report.py`:

```python
"""Phase 4 Markdown report (docs/phases/phase-4-motion.md §8, criterion 12)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from kata.config import Paths
from kata.data.report import md_table
from kata.motion.evaluate import QUANTITIES


def _tracks_section(paths: Paths, suffix: str) -> list[str]:
    f = paths.processed / f"motion_tracks{suffix}.parquet"
    if not f.exists():
        return ["_(no motion tracks yet)_", ""]
    t = pl.read_parquet(f)
    out = [md_table(t.group_by("status", "reject_reason").len().sort("status", "reject_reason")), ""]
    ok = t.filter(pl.col("status") == "ok")
    out += [f"Accepted: {ok.height} (left-handed recovered: {ok.filter(pl.col('mirrored')).height}; "
            f"world mirrored: {ok.filter(pl.col('world_mirrored')).height}; "
            f"hands swapped: {ok.filter(pl.col('hands_swapped')).height}).", ""]
    if ok.height:
        b = ok["blade_median_m"].to_numpy()
        out += ["**Blade check** (median cut point to blade, m): " +
                ", ".join(f"p{q} {np.percentile(b, q):.3f}" for q in (5, 50, 95, 99)), ""]
        rows = []
        for k in range(3):
            sub = ok.drop_nulls(["height_median", f"hand_y_l{k}"])
            if sub.height >= 3 and sub["height_median"].n_unique() >= 2:
                slope, icpt = np.polyfit(sub["height_median"].to_numpy(), sub[f"hand_y_l{k}"].to_numpy(), 1)
                rows.append({"layer": k, "n": sub.height, "slope_m_per_m": float(slope), "intercept_m": float(icpt)})
        out += ["**Hand height at hits vs. player height** (spec §3.2):", "",
                md_table(pl.DataFrame(rows)) if rows else "_(too few tracks)_", ""]
    return out


def _runs_section(paths: Paths) -> list[str]:
    rows = []
    for d in sorted(p for p in paths.motion_runs.glob("*") if p.is_dir()):
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        log = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()] if (d / "log.jsonl").exists() else []
        evals = [r for r in log if "eval" in r]
        speed = [r["steps_per_s"] for r in log if "steps_per_s" in r]
        g = json.loads((d / "guidance.json").read_text())["guidance"] if (d / "guidance.json").exists() else None
        rows.append({"run": d.name, "pilot": cfg.get("pilot"), "window": cfg.get("window"),
                     "mirror_aug": cfg.get("mirror_aug"), "use_stems": cfg.get("use_stems"),
                     "steps": log[-1]["step"] if log else 0, "steps_per_s": speed[-1] if speed else None,
                     "last_val_f": evals[-1]["eval"]["val_timing_f"] if evals else None,
                     "best_val_f": max((e["eval"]["val_timing_f"] or 0) for e in evals) if evals else None,
                     "last_rate_ratio": evals[-1]["eval"]["val_rate_ratio"] if evals else None, "guidance": g})
    return [md_table(pl.DataFrame(rows)) if rows else "_(no runs yet)_", ""]


def _eval_section(paths: Paths, run: str) -> list[str]:
    out = []
    d = paths.motion_runs / run
    ev_path = paths.processed / "motion_eval.parquet"
    for split in ("val", "test"):
        f = d / f"eval_{split}.json"
        if not f.exists():
            continue
        e = json.loads(f.read_text())
        out += [f"### {split}", "", "**Acceptance criteria**", "",
                md_table(pl.DataFrame(e["criteria"]).select("criterion", "name", "value", "pass")), ""]
        q = [{"quantity": k, "gen_p50": (e["pooled"]["generated"][k] or [None, None])[0],
              "real_p50": (e["pooled"]["real"][k] or [None, None])[0],
              "gen_p95": (e["pooled"]["generated"][k] or [None, None])[1],
              "real_p95": (e["pooled"]["real"][k] or [None, None])[1]} for k in QUANTITIES]
        out += ["**Plausibility percentiles**", "", md_table(pl.DataFrame(q)), ""]
        if ev_path.exists():
            r = pl.read_parquet(ev_path).filter((pl.col("run") == run) & (pl.col("split") == split)
                                                 & pl.col("mode").is_in(["full", "replay"]))
            per = (r.group_by("label", "source").agg(pl.len().alias("n"), pl.col("timing_f").median(),
                                                     pl.col("hit_rate").median(), pl.col("nps").median())
                   .sort("label", "source"))
            out += ["**Per label**", "", md_table(per), ""]
        if e.get("speed"):
            out += [f"**Speed** (generate(), one song at one difficulty, scaled to 3 min): "
                    f"max {max(e['speed']):.1f} s over {len(e['speed'])} songs.", ""]
    return out or ["_(not evaluated yet)_", ""]


def write_report(paths: Paths, state, name: str, run: str | None = None, pilot: bool = False) -> Path:
    out = [f"# Phase 4 motion report — {name}", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data root `{paths.root}`.", ""]
    runs = state.con.execute("""
        SELECT stage, args, round(seconds, 1) AS seconds, summary FROM runs WHERE stage LIKE 'motion-%'
        QUALIFY row_number() OVER (PARTITION BY stage, args ORDER BY started_at DESC) = 1
        ORDER BY started_at""").pl()
    out += ["## 1. Stage timings (latest run of each)", "", md_table(runs.select("stage", "args", "seconds")), ""]
    out += ["## 2. Motion tracks (spec §3)", ""] + _tracks_section(paths, "_pilot" if pilot else "")
    out += ["## 3. Training runs and ablations (spec §6)", ""] + _runs_section(paths)
    if run is None and (paths.motion_runs / "final").exists():
        run = (paths.motion_runs / "final").read_text().strip()
    out += [f"## 4. Evaluation of run `{run}` (spec §7-§8)", ""] + (_eval_section(paths, run) if run else
                                                                   ["_(no run selected)_", ""])
    path = paths.reports / f"{name}-{datetime.now():%Y-%m-%d}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
```

- [ ] **Step 5: Append the stages** to `src/kata/motion/pipeline.py`:

```python
def _viewer(paths, state, summary, run, split="test", n=6, blind=True):
    """Generated (full, seed 0) next to the first real replay for n songs spread over tempo (§7.3).
    Both clips show the same map notes, so the notes never give the real one away in blind mode."""
    import numpy as np

    from kata.motion.evaluate import real_hits
    from kata.motion.sample import eval_targets, load_sample, sample_path, spread
    from kata.motion.train import Run, tracks_table
    from kata.motion.viewer import clip, write_page
    targets = eval_targets(paths, Run(paths, run), split)
    tempo = paths.processed / "audio_tempo.parquet"
    if tempo.exists():
        targets = targets.join(pl.read_parquet(tempo).select("audio_sha256", "est_bpm"), on="audio_sha256",
                               how="left").sort("est_bpm", nulls_last=True)
    targets = targets.unique("audio_sha256", keep="first", maintain_order=True)
    store = MotionStore(paths.motion)
    meta = tracks_table(paths, Run(paths, run).config().pilot).select("score_id", "frame0")
    frame0 = dict(zip(meta["score_id"], meta["frame0"]))
    notes = pl.read_parquet(paths.processed / "notes.parquet")
    written = []
    for i, r in enumerate(spread(targets, n).iter_rows(named=True)):
        key = (r["hash"], r["characteristic"], r["difficulty"])
        path = sample_path(_sample_dir(paths, run, split), key, "full", 0)
        if not path.exists():
            continue
        s = load_sample(path)
        sid = r["score_ids"][0]
        raw = np.asarray(store.load(sid), np.float32)
        kn = notes.filter((pl.col("hash") == key[0]) & (pl.col("characteristic") == key[1])
                          & (pl.col("difficulty") == key[2]))
        note_list = [[float(t), int(x), int(y), int(c), int(d)] for t, x, y, c, d in
                     kn.select("time_s", "x", "y", "color", "cut_dir").iter_rows()]
        real = clip(raw[:, :27], real_hits(raw, frame0[sid]), frame0[sid], note_list)
        generated = clip(s["motion"], (s["hits_left"], s["hits_right"]), notes=note_list)   # both: blind-fair
        out = paths.reports / "motion-viewer" / f"{run}_{split}_{i:02d}_{key[0]}_{key[2]}.html"
        write_page(out, f"{key[0]} {key[2]} ({run})", generated, real, ("generated", "real"), blind, seed=i)
        written.append(str(out))
    summary.update(pages=written)

viewer_stage = _with_state("motion-viewer", _viewer)


def _report(paths, state, summary, name="phase4", run=None, pilot=False):
    from kata.motion.report import write_report
    summary["path"] = str(write_report(paths, state, name, run, pilot))

report_stage = _with_state("motion-report", _report)
```

- [ ] **Step 6: Wire the CLI** — in `src/kata/cli.py`, after `app.add_typer(baseline, name="baseline")` add

```python
motion = typer.Typer(no_args_is_help=True, help="Phase 4 motion model (all but `prepare` need `uv sync --extra audio`).")
app.add_typer(motion, name="motion")
```

and append at the end of the file:

```python
# --------------------------------------------------------------------------- Phase 4
# kata.motion.prepare is numpy-only; training, sampling and evaluation import torch inside each stage.

RUN = typer.Argument(..., help="Run id: a folder under runs/motion/.")
DEVICE = typer.Option(None, help="Torch device (default: cuda when available).")


@motion.command("prepare")
def motion_prepare(pilot: bool = PILOT):
    """Replays -> 50 fps motion tracks, motion_tracks.parquet and train stats (spec §3)."""
    from kata.motion import pipeline as mp
    _run("motion-prepare", mp.prepare_stage, pilot=pilot)


@motion.command("train")
def motion_train(run: str = RUN, resume: bool = typer.Option(False, help="Continue from ckpt/last.pt."),
                 hours: float = typer.Option(None, help="Stop cleanly after this many hours."),
                 device: str = typer.Option("cuda", help="Torch device."),
                 pilot: bool = typer.Option(False, help="Train on the pilot tracks (new runs only)."),
                 window: int = typer.Option(640, help="Window in 50 fps frames (new runs only)."),
                 mirror_aug: bool = typer.Option(False, help="Mirror augmentation (new runs only)."),
                 stems: bool = typer.Option(True, help="Use stem features (new runs only)."),
                 total_steps: int = typer.Option(80_000, help="Steps the lr schedule spans (new runs only)."),
                 batch: int = typer.Option(64, help="Windows per step (new runs only)."),
                 workers: int = typer.Option(4, help="DataLoader worker processes (new runs only)."),
                 only_song: list[str] = typer.Option([], help="Overfit check: train on this audio sha only.")):
    """Train a run overnight; resume it the next night with --resume (spec §6.4)."""
    from kata.motion import pipeline as mp
    from kata.motion.train import RunConfig
    config = None if resume else RunConfig(window=window, mirror_aug=mirror_aug, use_stems=stems, pilot=pilot,
                                           total_steps=total_steps, batch=batch, workers=workers,
                                           only_songs=tuple(only_song))
    _run("motion-train", mp.train_stage, run=run, config=config, resume=resume, hours=hours, device=device)


@motion.command("tune")
def motion_tune(run: str = RUN, device: str = DEVICE):
    """Pick the guidance weight on 50 val difficulties -> runs/motion/{run}/guidance.json (spec §7.1)."""
    from kata.motion import pipeline as mp
    _run("motion-tune", mp.tune_stage, run=run, device=device)


@motion.command("sample")
def motion_sample(run: str = RUN, split: str = typer.Option("val", help="val or test."),
                  limit: int = typer.Option(None, help="Only this many difficulties, evenly spread."),
                  device: str = DEVICE):
    """Whole songs for every difficulty of the split -> generated/motion/{run}/{split}/ (resumable)."""
    from kata.motion import pipeline as mp
    _run("motion-sample", mp.sample_stage, run=run, split=split, limit=limit, device=device)


@motion.command("evaluate")
def motion_evaluate(run: str = RUN, split: str = typer.Option("val", help="val or test (test: once, §8)."),
                    device: str = DEVICE,
                    force: bool = typer.Option(False, help="Re-evaluate test (only with approval).")):
    """Metrics and acceptance criteria -> motion_eval.parquet, runs/motion/{run}/eval_{split}.json."""
    from kata.motion import pipeline as mp
    _run("motion-evaluate", mp.evaluate_stage, run=run, split=split, device=device, force=force)


@motion.command("final")
def motion_final(run: str = RUN):
    """Mark the run generate() uses by default."""
    from kata.motion import pipeline as mp
    _run("motion-final", mp.final_stage, run=run)


@motion.command("viewer")
def motion_viewer(run: str = RUN, split: str = typer.Option("test", help="val or test."),
                  n: int = typer.Option(6, help="Songs, spread over tempo."),
                  blind: bool = typer.Option(True, help="Show clips as A/B until revealed.")):
    """Generated vs. real 3D viewer pages -> reports/motion-viewer/ (spec §7.3)."""
    from kata.motion import pipeline as mp
    _run("motion-viewer", mp.viewer_stage, run=run, split=split, n=n, blind=blind)


@motion.command("report")
def motion_report(name: str = typer.Option("phase4", help="Report file name prefix."),
                  run: str = typer.Option(None, help="Run to report (default: the final run)."),
                  pilot: bool = PILOT):
    """Phase 4 Markdown report under reports/."""
    from kata.motion import pipeline as mp
    _run("motion-report", mp.report_stage, name=name, run=run, pilot=pilot)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion_pipeline.py tests/test_motion_cli.py -q`
Expected: `6 passed`. Then `uv run pytest -q` → `232 passed, 3 skipped` (the skips are the existing real-data tests), and `uvx ruff check --select F src tests` → `All checks passed!`.

- [ ] **Step 8: Look at a page.** Run `uv run python -m http.server 8765 --bind 127.0.0.1` in the folder of a page written by the test (or any `reports/motion-viewer/*.html` later), open it, drag the time slider: head, red and blue sabers and the grid draw, notes fly in on **both** clips, the clips are labelled A/B, and Reveal names them. The theme follows the system light/dark setting.

- [ ] **Step 9: Commit**

```bash
git add src/kata/motion/viewer.py src/kata/motion/report.py src/kata/motion/pipeline.py src/kata/cli.py tests/test_motion_pipeline.py tests/test_motion_cli.py
git commit -m "Phase 4: 3D viewer, report and kata motion CLI" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Pilot on the replays already on disk (checkpoint with the user)

Runs on the training machine (RTX 5080, `E:\kata-data`). No network. Uses the 1,256 accepted Phase 1 pilot replays plus the pilot's left-handed ones; replays on the 93 deselected pilot Tier B maps have no Phase 2 features and drop out of training automatically.

**Files:**
- Modify: `docs/phases/phase-4-motion.md` (§16 pilot findings; constants only with the user's approval)
- Possibly modify: `src/kata/motion/prepare.py`, `src/kata/motion/geometry.py` (approved constants; a changed track parameter bumps `MOTION_VERSION`)

- [ ] **Step 1: Environment.** `uv sync --extra audio`, then `uv run pytest -q`.
Expected: `232 passed, 3 skipped` or more (real-data tests may run now that `KATA_DATA` exists), and `uv run python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

- [ ] **Step 2: Prepare the pilot tracks.**

Run: `uv run kata motion prepare --pilot`
Expected: `tracks` with most `ok`; `left_handed_recovered` around 70 (5.4 % of 1,340); a few rejects. A `blade_check` or `frame_check` count above a few percent means a convention is wrong: go to Step 3 before anything else.

- [ ] **Step 3: Read the checks.** `uv run kata motion report --pilot --name phase4-pilot`, section 2:
  - **Blade check** percentiles. Expected median a few centimetres if the blade axis (+z, 1 m) and offsets are right. If the whole distribution sits far off, try the other local axes and applying the BSOR saber offsets (`replay_aux`), and record which one fits. Set `BLADE_MAX_M` from the accepted right-handed distribution (about its p99, rounded up).
  - **Frame flags** of the recovered left-handed replays: which (world mirrored, hands swapped) combination the game produces. Expect one combination for all of them; a mix means the measurement is unreliable.
  - **Hand height vs. player height** per layer: a slope clearly above 0 (say > 0.2 m/m with a tight fit) means y should be normalised by `height_median` (spec §3.2).

- [ ] **Step 4: Throughput and I/O.**

Run: `uv run kata motion train pilot-speed --pilot --hours 0.25` (the default 4 workers), and the same with `--workers 0` as run `pilot-speed0`.
Expected: `steps_per_s` in `runs/motion/pilot-speed/log.jsonl`; spec §6.3 assumed ~2 steps/s (0.5 s/step). Watch `nvidia-smi` while it runs: GPU utilisation well below 90 % with 4 workers means data loading is the bottleneck, so copy the train songs' features to the internal SSD (spec §6.6) and measure again. Note peak GPU memory.

- [ ] **Step 5: Overfit one song.** Pick one pilot train song with several replays:

```bash
uv run python -c "
import polars as pl
t = pl.read_parquet(r'E:\kata-data\processed\motion_tracks_pilot.parquet').filter((pl.col('status') == 'ok') & (pl.col('split') == 'train'))
print(t.group_by('audio_sha256').len().sort('len', descending=True).head(3))
"
uv run kata motion train pilot-overfit --pilot --only-song <sha> --total-steps 3000 --hours 1
```
Expected: in `runs/motion/pilot-overfit/log.jsonl` the loss falls steadily and the last `eval` has `val_timing_f` close to 1 (the quick check runs on the song itself in overfit mode). If not, the pipeline cannot learn; stop and debug before Step 6.

- [ ] **Step 6: Pilot run (one night).**

```bash
uv run kata motion train pilot-1 --pilot --hours 12
uv run kata motion tune pilot-1
uv run kata motion sample pilot-1 --split val
uv run kata motion evaluate pilot-1 --split val
uv run kata motion viewer pilot-1 --split val
uv run kata motion report --pilot --name phase4-pilot --run pilot-1
```
Expected: the commands finish; the report shows criteria values on pilot val (no bars apply yet; the pilot has ~10 val songs); the viewer pages open (Task 10 Step 8). The user looks at a few pages.

- [ ] **Step 7: Record findings in spec §16**, covering every **[verify in pilot]** / **[tune in pilot]** / **[verify at setup]** item: blade axis and offsets, `BLADE_MAX_M`, the left-handed frame transform, height normalisation, steps/s and GPU memory, data-loading bottleneck and fix, overfit result, guidance chosen, pilot criteria values, the user's impression of the viewer pages.

- [ ] **Step 8: Checkpoint.** Present the findings and any proposed changes to the user. Apply only what the user approves; log each change in spec §14. A changed track parameter means a new `MOTION_VERSION` (Global Constraints) and a re-run of `prepare --pilot`.

- [ ] **Step 9: Commit**

```bash
git add docs/phases/phase-4-motion.md
git commit -m "Phase 4: pilot findings (spec §16)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

(plus any approved code change, with its tests passing.)

---

### Task 12: Tier A replay fetch (only when the user says go)

Network: ~3–4 h at the Phase 1 rate limits, no GPU, so it can run during the day. **Ask the user before starting**; do not start it on the strength of an earlier approval.

**Files:**
- Modify: `docs/phases/phase-4-motion.md` (§16: fetch numbers)

- [ ] **Step 1: Confirm with the user** that the pilot checkpoint (Task 11 Step 8) is closed and the fetch may start. Check free space on E: (≥ ~120 GB: ~62 GB replays, ~12 GB tracks, margin above the 50 GB floor).

- [ ] **Step 2: Fetch and parse**

```bash
uv run kata data crawl-leaderboards --tier A
uv run kata data select-replays
uv run kata data download-replays
uv run kata data parse-replays
uv run kata data build
```
Expected: leaderboards only for Tier A hashes; `select-replays` ~22k replays; `download-replays` resumable (re-run after an interruption); `parse-replays` ~94 % `ok`, left-handed ~5 % rejected as `left_handed,…` (recovered next step).

- [ ] **Step 3: Prepare the full tracks**

Run: `uv run kata motion prepare`
Expected: ≥ 99 % of accepted + recoverable replays become tracks or carry a reason (criterion 8); `left_handed_recovered` ~1k; `motion/v1/` ~12 GB; `stats.json` from train only.

- [ ] **Step 4: Record** counts, sizes and timings in spec §16 and commit

```bash
git add docs/phases/phase-4-motion.md
git commit -m "Phase 4: Tier A replays fetched and prepared" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Training nights (base run, ablations, final run)

One run per night (`--hours 12`), the PC free during the day. Each night's command is started by the user or on their instruction.

**Files:**
- Modify: `docs/phases/phase-4-motion.md` (§16 or a new §18 section with the ablation table, if the user wants it in the spec rather than only in the report)

- [ ] **Step 1: Base run (night 1).** `uv run kata motion train base --hours 12`
Expected: ~80k steps if throughput matches the pilot; `best.pt` written; quick `val_timing_f` rising over the night in `log.jsonl`.

- [ ] **Step 2: Ablations (nights 2–3, two 6 h runs per night).** Same `--total-steps` as base so the lr schedules match:

```bash
uv run kata motion train abl-w320 --window 320 --hours 6
uv run kata motion train abl-mirror --mirror-aug --hours 6
uv run kata motion train abl-nostems --no-stems --hours 6
```
Compare each with base's `eval` entry at the same step (report section 3 lists last and best quick val F and the rate ratio per run). An ablation is adopted only if it improves criterion 1 or 2 on val without making another worse (spec §6.5); for criterion 2, run `tune`/`sample --limit 50`/`evaluate` on the two runs being compared.

- [ ] **Step 3: Final run (nights 4–5).** `uv run kata motion train final-1 <adopted options> --hours 12`; the next night, if the last quick val F is still rising, `uv run kata motion train final-1 --resume --hours 12`.

- [ ] **Step 4: Val evaluation of the final run**

```bash
uv run kata motion tune final-1
uv run kata motion sample final-1 --split val
uv run kata motion evaluate final-1 --split val
uv run kata motion viewer final-1 --split val
```
Expected: criteria 2–6 values on val in `runs/motion/final-1/eval_val.json` and the report. If a bar fails on val, discuss with the user before touching test.

- [ ] **Step 5: Commit** any spec notes

```bash
git add docs/phases/phase-4-motion.md
git commit -m "Phase 4: training runs and ablations" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Acceptance on test

**Files:**
- Modify: `docs/phases/phase-4-motion.md` (§18 build results, status line), `docs/PLAN.md` (Phase 4 ✅)

- [ ] **Step 1: Mark the final run and evaluate test once**

```bash
uv run kata motion final final-1
uv run kata motion sample final-1 --split test
uv run kata motion evaluate final-1 --split test
```
Expected: criteria 1–6 and 11 in `runs/motion/final-1/eval_test.json`. A second `evaluate --split test` refuses (criterion 9); never pass `--force` without the user's approval.

- [ ] **Step 2: Visual check (criterion 7).** `uv run kata motion viewer final-1 --split test` writes 6 blind pages spread over tempo. The user opens each (Task 10 Step 8), picks the song's audio file, guesses A/B, reveals, and judges whether the generated clip is plausible. Record the 6 judgements and blind guesses.

- [ ] **Step 3: Report.** `uv run kata motion report --name phase4`; check it has everything criterion 12 asks: all §7.2 metrics per label for val and test next to real replays and the baseline, the ablation table, and timings.

- [ ] **Step 4: Acceptance.** Fill spec §18 with a table of criteria 1–12 and their results, plus notes on any criterion that needs one. Set the status line to DONE and mark Phase 4 ✅ in `docs/PLAN.md` with a one-line results summary.

- [ ] **Step 5: Commit**

```bash
git add docs/phases/phase-4-motion.md docs/PLAN.md
git commit -m "Phase 4: v1 motion model build results" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

