import zipfile

import polars as pl
import pytest

from kata.data.beatsaver import extract_map
from kata.data.pipeline import normalise_title, song_groups, split_for


def test_normalise_title():
    assert normalise_title("Ghost (feat. Someone) [Extended Mix]") == "ghost"
    assert normalise_title("Ghost ft. Someone") == "ghost"
    assert normalise_title("  RēVOLUTION!!  ") == "rēvolution"
    assert normalise_title("千本桜") == "千本桜"


def test_song_groups_union_names_and_audio():
    maps = pl.DataFrame({
        "hash": ["a", "b", "c", "d"],
        "song_author": ["Camellia", "camellia", "Someone", "Other"],
        "song_name": ["Ghost", "GHOST (Extended)", "Totally Different Title", "Unrelated"],
        "audio_sha256": ["x1", "x2", "x2", "x9"],
    })
    g = song_groups(maps)
    assert g["a"] == g["b"]  # same normalised name
    assert g["b"] == g["c"]  # same audio bytes
    assert g["d"] != g["a"]
    assert g["a"] == min("name:camellia|ghost", "name:someone|totally different title")


def test_split_is_deterministic_and_roughly_90_5_5():
    splits = [split_for(f"name:artist|song {i}") for i in range(5000)]
    assert splits == [split_for(f"name:artist|song {i}") for i in range(5000)]
    assert 0.87 < splits.count("train") / 5000 < 0.93
    assert splits.count("val") > 150 and splits.count("test") > 150


def _zip(path, files: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


def test_extract_map_keeps_only_files_next_to_info(tmp_path):
    zp = tmp_path / "m.zip"
    _zip(zp, {"sub/Info.dat": b"{}", "sub/ExpertPlus.dat": b"{}", "sub/song.egg": b"OggS",
              "sub/Hard.ben": b"{}", "sub/.egg": b"OggS", "sub/cover.jpg": b"jpg",
              "sub/nested/evil.dat": b"x", "../escape.dat": b"x"})
    kept = extract_map(zp, tmp_path / "out")
    assert sorted(kept) == [".egg", "ExpertPlus.dat", "Hard.ben", "Info.dat", "song.egg"]
    assert not (tmp_path / "escape.dat").exists()


def test_extract_map_requires_info(tmp_path):
    zp = tmp_path / "m.zip"
    _zip(zp, {"song.egg": b"OggS"})
    with pytest.raises(ValueError, match="Info.dat"):
        extract_map(zp, tmp_path / "out")
