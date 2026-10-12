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
