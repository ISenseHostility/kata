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


def swept_plane_distance(p, base, tip, prev_tip) -> np.ndarray:
    """Distance from points p to the plane through the blade base-tip and the previous tip, all
    (..., 3): the game's cut plane, on which it records the cut point. Where the blade has not
    swept (no plane), the distance to the blade segment instead."""
    n = np.cross(tip - base, prev_tip - base)
    norm = np.linalg.norm(n, axis=-1)
    swept = norm > 1e-6
    d = np.abs(((p - base) * n).sum(-1)) / np.where(swept, norm, 1.0)
    return np.where(swept, d, point_segment_distance(p, base, tip))
