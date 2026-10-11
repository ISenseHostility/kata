import json
import zipfile

import polars as pl

from kata.config import Paths
from kata.data.beatsaver import extract_map, is_audio
from kata.data.lowrated import LOWRATED_PILOT, calib_half, parse_lowrated, select_lowrated, year_quotas


def test_year_quotas_follow_the_share_and_redistribute_short_years():
    q = year_quotas({2019: 0.1, 2022: 0.5, 2024: 0.4}, {2019: 1000, 2022: 1000, 2024: 50}, 1000)
    assert q[2024] == 50                                  # short year gives everything it has
    assert sum(q.values()) == 1000
    assert q[2022] > q[2019] > 0
    assert year_quotas({2020: 1.0}, {2018: 3, 2020: 2}, 10) == {2018: 3, 2020: 2}   # pool smaller than n


def test_calib_half_depends_only_on_the_normalised_song_name():
    assert calib_half("Camellia", "Ghost") == calib_half("camellia", "GHOST (Extended)")
    halves = {calib_half("artist", f"song {i}") for i in range(50)}
    assert halves == {"fit", "holdout"}


def test_extract_map_can_drop_audio(tmp_path):
    zp = tmp_path / "m.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for name in ("Info.dat", "Expert.dat", "song.egg", ".egg", "x.wav", "cover.jpg"):
            z.writestr(name, b"{}")
    assert sorted(extract_map(zp, tmp_path / "a", drop_audio=True)) == ["Expert.dat", "Info.dat"]
    assert ".egg" in extract_map(zp, tmp_path / "b")
    assert is_audio("SONG.OGG") and not is_audio("song.dat")


def _catalog(n_low=30):
    maps, diffs = [], []
    for i in range(n_low + 3):
        low = i < n_low
        mid = f"m{i}"
        maps.append({"map_id": mid, "hash": f"h{i}", "name": "x", "song_name": f"Song {i}", "song_sub_name": "",
                     "song_author": "Artist", "level_author": "L", "uploader_id": 1, "uploader_name": "u",
                     "meta_bpm": 120.0, "meta_duration_s": 120.0, "uploaded_at": f"{2018 + i % 6}-05-01T00:00:00Z",
                     "created_at": "2020-01-01T00:00:00Z", "curated_at": None, "upvotes": 10 if low else 500,
                     "downvotes": 30 if low else 1, "rating": 0.3 if low else 0.99, "ranked_ss": False,
                     "ranked_bl": False, "qualified_ss": False, "qualified_bl": False, "automapper": False,
                     "declared_ai": "None", "sage_score": 0, "tags": [], "version_state": "Published",
                     "download_url": f"https://cdn/{i}.zip", "raw_json": "{}"})
        diffs.append({"map_id": mid, "hash": f"h{i}", "characteristic": "Standard", "difficulty": "Expert",
                      "njs": 16.0, "njs_offset": 0.0, "notes": 60, "bombs": 0, "obstacles": 0, "nps": 3.0,
                      "length_beats": 200.0, "seconds": 100.0, "events": 0, "chroma": False, "me": False, "ne": False,
                      "cinema": False, "parity_errors": 3, "parity_warns": 0, "parity_resets": 0, "ss_stars": None,
                      "bl_stars": None, "max_score": 0, "environment": "Default"})
    return pl.DataFrame(maps, infer_schema_length=None), pl.DataFrame(diffs, infer_schema_length=None)


def _write_map_folder(folder, n_notes=60):
    folder.mkdir(parents=True)
    (folder / "Info.dat").write_text(json.dumps({
        "_version": "2.1.0", "_songName": "s", "_songFilename": "song.egg", "_beatsPerMinute": 120,
        "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_difficultyRank": 7, "_beatmapFilename": "Expert.dat",
             "_noteJumpMovementSpeed": 16, "_noteJumpStartBeatOffset": 0}]}]}))
    notes = [{"b": 4 + i, "x": 3, "y": 0 if i % 2 == 0 else 2, "c": 1, "d": 1 if i % 2 == 0 else 0, "a": 0}
             for i in range(n_notes)]
    (folder / "Expert.dat").write_text(json.dumps({"version": "3.0.0", "colorNotes": notes}))


def test_select_and_parse_lowrated(tmp_path):
    p = Paths(tmp_path).ensure()
    maps, diffs = _catalog()
    maps.write_parquet(p.processed / "metadata_catalog.parquet")
    diffs.write_parquet(p.processed / "metadata_catalog_diffs.parquet")
    pl.DataFrame({"map_id": ["m31"], "hash": ["h31"]}).write_parquet(p.processed / "selected_maps.parquet")
    pl.DataFrame({"hash": ["h31", "h32"], "map_tier": ["A", "A"],
                  "uploaded_at": ["2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"]}).write_parquet(p.processed / "maps.parquet")
    s = select_lowrated(p, n=10)
    sel = pl.read_parquet(p.lowrated / "selected_maps.parquet")
    assert s["pool_maps"] == 30 and sel.height == 10 and set(sel["map_tier"]) == {"D"}
    assert sel["year"].is_in([2022, 2023]).sum() == 10          # Tier A's years are preferred
    assert int(sel["pilot"].sum()) == min(LOWRATED_PILOT, 10)
    # pretend the maps were downloaded without audio
    from kata.data.state import State
    st = State(p.state_db)
    for h in sel["hash"]:
        _write_map_folder(p.maps / h)
        st.mark("download-lowrated", h, "ok")
    st.close()
    out = parse_lowrated(p)
    m = pl.read_parquet(p.lowrated / "maps.parquet")
    d = pl.read_parquet(p.lowrated / "difficulties.parquet")
    assert out["diff_status"] == {"ok": 10}
    assert set(m["audio_status"]) == {"dropped"} and set(m["calib_half"]) <= {"fit", "holdout"}
    assert set(d["tier"]) == {"D"}
    assert pl.read_parquet(p.lowrated / "notes.parquet").height == 600
    assert not (p.processed / "notes.parquet").exists()          # Tier D never touches the main tables
