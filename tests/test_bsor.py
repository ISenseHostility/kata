import os
from pathlib import Path

import numpy as np
import pytest

from kata.data import bsor
from kata.data.align import align


def make_replay(n_frames=50) -> bsor.Replay:
    frames = np.zeros(n_frames, dtype=bsor.FRAME_DTYPE)
    frames["t"] = np.arange(n_frames) / 90.0
    frames["fps"] = 90
    frames["head_py"] = 1.5
    frames["head_qw"] = frames["left_qw"] = frames["right_qw"] = 1.0
    frames["right_px"] = np.linspace(0, 0.5, n_frames)
    cut = {f: 0 for f in bsor.CUT_FIELDS} | {"speed_ok": True, "saber_type": 1, "saber_speed": 3.5}
    notes = [
        {"note_id": 32018, "event_time": 1.0, "spawn_time": 1.001, "event_type": bsor.EVENT_GOOD, **cut},
        {"note_id": 30101, "event_time": 2.0, "spawn_time": 2.0, "event_type": bsor.EVENT_MISS},
        {"note_id": 72018, "event_time": 2.1, "spawn_time": 2.1, "event_type": bsor.EVENT_GOOD, **cut},
        {"note_id": 1, "event_time": 2.5, "spawn_time": 2.5, "event_type": bsor.EVENT_BOMB},
    ]
    info = {k: f"<{k}>" for k in bsor.INFO_STRINGS_1 + bsor.INFO_STRINGS_2}
    info.update(score=1234, jumpDistance=15.0, leftHanded=False, height=0.0, startTime=0.0,
                failTime=0.0, speed=0.0, modifiers="")
    return bsor.Replay(info, frames, notes, walls=[(1, 0.1, 3.0, 3.0)], heights=[(1.7, 0.0)],
                       pauses=[], saber_offsets=tuple(float(i) for i in range(14)),
                       custom_data={"k": b"\x01\x02"})


def test_roundtrip():
    original = make_replay()
    parsed = bsor.parse(bsor.write(original))
    assert parsed.info["score"] == 1234 and parsed.info["hash"] == "<hash>"
    np.testing.assert_array_equal(parsed.frames, original.frames)
    assert [n["note_id"] for n in parsed.notes] == [32018, 30101, 72018, 1]
    assert parsed.notes[0]["saber_speed"] == pytest.approx(3.5)
    assert "saber_speed" not in parsed.notes[1]  # misses carry no cut info
    assert parsed.walls[0][0] == 1 and parsed.heights == [(pytest.approx(1.7), 0.0)]
    assert parsed.saber_offsets == tuple(float(i) for i in range(14))
    assert parsed.custom_data == {"k": b"\x01\x02"}


def test_rejects_bad_magic_and_truncation():
    with pytest.raises(bsor.BsorError, match="magic"):
        bsor.parse(b"\x00" * 16)
    with pytest.raises(bsor.BsorError):
        bsor.parse(bsor.write(make_replay())[:300])


def test_decode_note_id():
    assert bsor.decode_note_id(32018) == (3, 2, 0, 1, 8)


def test_align_skips_bombs_and_chain_links():
    # map: one note matching 32018 (x2 y0 blue dot) at 1.0 s, one matching 30101 at 2.0 s
    res = align([2, 0], [0, 1], [1, 0], [8, 1], [1.0, 2.0], make_replay().notes)
    assert res.n_considered == 2  # chain element (7xxxx) and bomb skipped
    assert res.n_matched == 2
    assert res.matched_idx.tolist() == [0, 1, -1, -1]
    assert res.residual_ms[0] == pytest.approx(1.0, abs=1e-3)


def test_align_rejects_far_or_mismatched_notes():
    res = align([2], [0], [0], [8], [1.0], make_replay().notes)  # wrong colour
    assert res.n_matched == 0


REAL = Path(os.environ.get("KATA_DATA", r"E:\kata-data")) / "raw" / "replays"


@pytest.mark.skipif(not REAL.exists() or not any(REAL.rglob("*.bsor")),
                    reason="no downloaded replays (real-data test)")
def test_real_replays_parse_to_eof():
    for path in list(REAL.rglob("*.bsor"))[:5]:
        r = bsor.parse(path.read_bytes())
        assert len(r.frames) > 1000
        dt = np.diff(r.frames["t"])
        assert 50 < 1 / np.median(dt[dt > 0]) < 200


@pytest.mark.parametrize("legacy", [False, True])
def test_non_ascii_info_strings_with_legacy_lengths(legacy):
    original = make_replay()
    original.info.update(playerName="ぽみTEA", songName="Tokimeki★Meteostrike", mapper="Idlebob 🎵")
    parsed = bsor.parse(bsor.write(original, legacy_lengths=legacy))
    assert parsed.info["playerName"] == "ぽみTEA"
    assert parsed.info["songName"] == "Tokimeki★Meteostrike"
    assert parsed.info["mapper"] == "Idlebob 🎵"
    assert parsed.info["platform"] == "<platform>"
    np.testing.assert_array_equal(parsed.frames, original.frames)
