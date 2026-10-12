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
