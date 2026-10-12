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
