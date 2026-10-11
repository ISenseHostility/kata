import json
import zipfile

import numpy as np
import pytest

from kata.baseline.export import EMPTY_V3, DifficultyOut, njs_offset_for, round_trip, write_map, write_png
from kata.baseline.grid import Grid
from kata.baseline.patterns import place
from kata.validate.rules import reaction_time_s


@pytest.mark.parametrize("njs, rt, bpm", [(16, 0.6, 120), (20, 0.5, 174), (10, 1.0, 90)])
def test_njs_offset_inverts_the_reaction_time(njs, rt, bpm):
    assert reaction_time_s(njs, njs_offset_for(njs, rt, bpm), bpm) == pytest.approx(rt, abs=1e-3)


def test_png_is_valid(tmp_path):
    write_png(tmp_path / "c.png", size=8)
    assert (tmp_path / "c.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_map_round_trips_through_the_phase1_parser(tmp_path):
    grid = Grid(np.concatenate([[0.0], 0.731 + 0.5 * np.arange(200)]), 120.0, False)
    t, b = grid.slots()
    notes = place(t[8:400:2], b[8:400:2], np.ones(196), np.array([]), "Expert", 0.4, np.random.default_rng(0))
    diffs = {"Expert": DifficultyOut(notes, 16.0, -0.2), "Easy": DifficultyOut(notes.head(40), 10.0, 0.0)}
    audio = tmp_path / "src.egg"
    audio.write_bytes(b"OggS fake")
    zp = write_map(tmp_path / "out", audio, {"song_name": "Song", "song_author": "Artist"}, grid, diffs, 100.0)
    round_trip(tmp_path / "out", diffs)
    info = json.loads((tmp_path / "out" / "Info.dat").read_text())
    for key in ("_version", "_songName", "_songSubName", "_songAuthorName", "_levelAuthorName", "_beatsPerMinute",
                "_songTimeOffset", "_shuffle", "_shufflePeriod", "_previewStartTime", "_previewDuration",
                "_songFilename", "_coverImageFilename", "_environmentName", "_difficultyBeatmapSets"):
        assert key in info
    ranks = [d["_difficultyRank"] for d in info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]]
    assert ranks == [1, 7]
    diff = json.loads((tmp_path / "out" / "ExpertStandard.dat").read_text())
    assert diff["version"] == "3.2.0" and all(k in diff for k in EMPTY_V3)
    with zipfile.ZipFile(zp) as z:
        assert sorted(z.namelist()) == ["EasyStandard.dat", "ExpertStandard.dat", "Info.dat", "cover.png", "song.egg"]
