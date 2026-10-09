import json

import numpy as np
import pytest

from kata.data.mapformat import (MapFormatError, load_info, parse_difficulty, tempo_for)
from kata.data.mapformat.tempo import TempoMap

# One pattern, written in all three formats. Expected canonical tuples below.
NOTES = [(1.0, 1, 0, 0, 1, 0), (1.0, 2, 0, 1, 1, 0), (2.5, 3, 2, 1, 5, 0)]
BOMBS = [(2.0, 0, 2)]
OBSTACLES = [(3.0, 2.0, 0, 0, 1, 5), (4.0, 1.0, 0, 2, 4, 3)]
ARCS = [(0, 1.0, 1, 0, 1, 1.0, 2.0, 1, 1, 0, 1.5, 0)]
CHAINS = [(1, 1.0, 2, 0, 1, 1.5, 2, 2, 4, 0.8)]

V2 = {
    "_version": "2.6.0",
    "_notes": [
        {"_time": 2.5, "_lineIndex": 3, "_lineLayer": 2, "_type": 1, "_cutDirection": 5},
        {"_time": 1, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
        {"_time": 1, "_lineIndex": 2, "_lineLayer": 0, "_type": 1, "_cutDirection": 1},
        {"_time": 2, "_lineIndex": 0, "_lineLayer": 2, "_type": 3, "_cutDirection": 0},
    ],
    "_obstacles": [
        {"_time": 3, "_lineIndex": 0, "_type": 0, "_duration": 2, "_width": 1},
        {"_time": 4, "_lineIndex": 0, "_type": 1, "_duration": 1, "_width": 4},
    ],
    "_sliders": [{
        "_colorType": 0, "_headTime": 1, "_headLineIndex": 1, "_headLineLayer": 0,
        "_headControlPointLengthMultiplier": 1, "_headCutDirection": 1, "_tailTime": 2,
        "_tailLineIndex": 1, "_tailLineLayer": 1, "_tailControlPointLengthMultiplier": 1.5,
        "_tailCutDirection": 0, "_sliderMidAnchorMode": 0}],
    "_events": [{"_time": 0, "_type": 1, "_value": 1}],
}

V3 = {
    "version": "3.3.0",
    # Zero-valued keys omitted, as real v3 writers do.
    "colorNotes": [{"b": 1, "x": 1, "d": 1}, {"b": 1, "x": 2, "c": 1, "d": 1},
                   {"b": 2.5, "x": 3, "y": 2, "c": 1, "d": 5}],
    "bombNotes": [{"b": 2, "y": 2}],
    "obstacles": [{"b": 3, "d": 2, "w": 1, "h": 5}, {"b": 4, "d": 1, "y": 2, "w": 4, "h": 3}],
    "sliders": [{"b": 1, "x": 1, "d": 1, "mu": 1, "tb": 2, "tx": 1, "ty": 1, "tmu": 1.5}],
    "burstSliders": [{"c": 1, "b": 1, "x": 2, "d": 1, "tb": 1.5, "tx": 2, "ty": 2, "sc": 4, "s": 0.8}],
    "bpmEvents": [],
}

V4 = {
    "version": "4.0.0",
    "colorNotes": [{"b": 1, "r": 0, "i": 0}, {"b": 1, "i": 1}, {"b": 2.5, "i": 2}],
    # index 3 is used only as an arc tail, never placed as a note
    "colorNotesData": [{"x": 1, "y": 0, "c": 0, "d": 1, "a": 0}, {"x": 2, "y": 0, "c": 1, "d": 1},
                       {"x": 3, "y": 2, "c": 1, "d": 5}, {"x": 1, "y": 1, "c": 0, "d": 0}],
    "bombNotes": [{"b": 2, "i": 0}],
    "bombNotesData": [{"x": 0, "y": 2}],
    "obstacles": [{"b": 3, "i": 0}, {"b": 4, "i": 1}],
    "obstaclesData": [{"d": 2, "x": 0, "y": 0, "w": 1, "h": 5}, {"d": 1, "x": 0, "y": 2, "w": 4, "h": 3}],
    "arcs": [{"hb": 1, "tb": 2, "hi": 0, "ti": 3, "ai": 0}],
    "arcsData": [{"m": 1, "tm": 1.5, "a": 0}],
    "chains": [{"hb": 1, "tb": 1.5, "i": 1, "ci": 0}],
    "chainsData": [{"tx": 2, "ty": 2, "c": 4, "s": 0.8}],
}


@pytest.mark.parametrize("data", [V2, V3, V4], ids=["v2", "v3", "v4"])
def test_formats_produce_identical_canonical_output(data):
    bm = parse_difficulty(data)
    assert bm.notes == NOTES
    assert bm.bombs == BOMBS
    assert bm.obstacles == OBSTACLES
    assert bm.arcs == ARCS
    assert bm.chains == ([] if data is V2 else CHAINS)


def test_v2_bpm_events_and_legacy_custom_bpm_changes():
    data = {**V2, "_events": [{"_time": 4, "_type": 100, "_floatValue": 60}],
            "_customData": {"_BPMChanges": [{"_time": 8, "_BPM": 200}]}}
    bm = parse_difficulty(data)
    assert bm.bpm_events == [(4.0, 60.0)]
    assert "legacy_customdata_bpmchanges_ignored" in bm.warnings


@pytest.mark.parametrize("note", [
    {"_time": 1, "_lineIndex": 1500, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},  # ME precision
    {"_time": 1, "_lineIndex": 1, "_lineLayer": 3, "_type": 0, "_cutDirection": 1},  # off grid
    {"_time": 1, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1090},  # ME rotation
    {"_time": 1, "_lineIndex": 1, "_lineLayer": 0, "_type": 2, "_cutDirection": 1},  # unused type
])
def test_v2_rejects_modded_or_invalid_notes(note):
    with pytest.raises(MapFormatError):
        parse_difficulty({"_version": "2.0.0", "_notes": [note]})


def test_v4_rejects_rotation_lanes_and_bad_indices():
    with pytest.raises(MapFormatError, match="rotation"):
        parse_difficulty({**V4, "colorNotes": [{"b": 1, "r": 1, "i": 0}]})
    with pytest.raises(MapFormatError, match="out of range"):
        parse_difficulty({**V4, "colorNotes": [{"b": 1, "i": 9}]})


def test_tempo_constant_and_changes():
    t = TempoMap.from_bpm_events(120, [(4, 60)])
    np.testing.assert_allclose(t.to_seconds([0, 2, 4, 6]), [0, 1, 2, 4])
    # an event at beat 0 overrides the Info BPM
    t = TempoMap.from_bpm_events(100, [(0, 120)])
    np.testing.assert_allclose(t.to_seconds([2]), [1.0])


def test_tempo_from_v4_audio_regions():
    f = 44100
    regions = [{"si": 0, "ei": 2 * f, "sb": 0, "eb": 4}, {"si": 2 * f, "ei": 4 * f, "sb": 4, "eb": 6}]
    t = TempoMap.from_audio_regions(120, f, regions)
    np.testing.assert_allclose(t.to_seconds([0, 2, 4, 5, 6]), [0, 1, 2, 3, 4])


def _write(folder, name, data):
    (folder / name).write_text(json.dumps(data), encoding="utf-8")


def test_load_v2_info(tmp_path):
    _write(tmp_path, "Info.dat", {
        "_version": "2.1.0", "_songFilename": "song.egg", "_beatsPerMinute": 150,
        "_songTimeOffset": 0, "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
                {"_difficulty": "ExpertPlus", "_difficultyRank": 9, "_beatmapFilename": "EP.dat",
                 "_noteJumpMovementSpeed": 18, "_noteJumpStartBeatOffset": -0.5}]}]})
    info = load_info(tmp_path)
    assert info.bpm == 150 and info.song_filename == "song.egg"
    d = info.difficulties[0]
    assert (d.characteristic, d.difficulty, d.rank, d.filename, d.njs) == ("Standard", "ExpertPlus", 9, "EP.dat", 18)


def test_load_v4_info_uses_audio_data_tempo(tmp_path):
    _write(tmp_path, "info.dat", {  # lower-case name: lookup must be case-insensitive
        "version": "4.0.1", "audio": {"songFilename": "song.ogg", "bpm": 120,
                                      "audioDataFilename": "AudioData.dat"},
        "difficultyBeatmaps": [{"characteristic": "Standard", "difficulty": "Hard",
                                "beatmapDataFilename": "Hard.dat", "noteJumpMovementSpeed": 16}]})
    _write(tmp_path, "AudioData.dat", {"version": "4.0.0", "songFrequency": 1000,
                                       "bpmData": [{"si": 0, "ei": 4000, "sb": 0, "eb": 4}]})
    info = load_info(tmp_path)
    assert info.difficulties[0].rank == 5
    tempo, source = tempo_for(info, parse_difficulty(V4))
    assert source == "audiodata"
    np.testing.assert_allclose(tempo.to_seconds([2]), [2.0])  # 60 BPM from samples, not Info's 120


def test_offgrid_arc_anchor_is_allowed_with_warning():
    data = {**V3, "sliders": [{"b": 1, "x": 1, "d": 1, "tb": 2, "tx": 4, "ty": 3}]}
    bm = parse_difficulty(data)
    assert bm.arcs[0][7:9] == (4, 3)
    assert "offgrid_arc_or_chain_anchor" in bm.warnings
    with pytest.raises(MapFormatError, match="precision"):
        parse_difficulty({**V3, "sliders": [{"b": 1, "x": 1, "tb": 2, "tx": 1500}]})
    with pytest.raises(MapFormatError, match="off-grid"):  # notes stay strict
        parse_difficulty({**V3, "colorNotes": [{"b": 1, "x": 4}]})


def test_audio_fallback_to_single_audio_file(tmp_path):
    from kata.data.mapformat import find_audio
    (tmp_path / "Info.dat").write_text("{}")
    (tmp_path / "song.egg").write_bytes(b"OggS")
    path, fallback = find_audio(tmp_path, ".egg")
    assert path.name == "song.egg" and fallback
