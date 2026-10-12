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
