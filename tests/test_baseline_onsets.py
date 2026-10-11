import numpy as np

from kata.audio.store import CURVE_COLUMNS
from kata.baseline.onsets import loudness_boost, mapped_span, onset_strength, select, slot_strengths


def _curves(T=3000):
    c = np.zeros((T, 12), np.float32)
    c[:, CURVE_COLUMNS.index("rms_mix")] = -20
    return c


def test_onset_strength_weights_normalised_curves():
    c = _curves(1000)
    c[::50, CURVE_COLUMNS.index("onset_drums")] = 4.0     # 2 % of frames, so p99 is the peak
    c[25::50, CURVE_COLUMNS.index("onset_mix")] = 9.0
    s = onset_strength(c)
    assert s[0] == 1.0 and s[25] == 0.5


def test_loudness_boost_is_clipped_per_segment():
    c = _curves(1000)
    c[500:, CURVE_COLUMNS.index("rms_mix")] = -5
    b = loudness_boost(c, [(0, 5), (5, 10)])
    assert b[100] < 1 < b[700] and b.min() >= 0.5 and b.max() <= 1.5


def test_slot_strength_max_pools_30_ms():
    s = np.zeros(200)
    s[103] = 1.0
    np.testing.assert_array_equal(slot_strengths(s, np.array([1.0, 1.1])), [1.0, 0.0])


def test_select_respects_gap_and_count_and_prefers_strong_slots():
    t = np.arange(0, 10, 0.125)
    s = np.ones_like(t)
    s[::4] = 5
    idx = select(t, s, n_target=30, min_gap_s=0.25)
    assert np.all(np.diff(t[idx]) >= 0.25 - 1e-9) and len(idx) <= 30
    assert set(np.flatnonzero(s == 5)) <= set(idx.tolist())
    assert np.all(np.diff(idx) > 0)


def test_mapped_span_keeps_lead_in_and_out():
    t = np.arange(0, 60, 0.25)
    m = mapped_span(t, np.ones_like(t), 60.0)
    assert t[m].min() >= 1.5 and t[m].max() <= 58.0
