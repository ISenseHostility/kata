"""Map folder + zip export (spec §8.5): Info.dat v2.1.0, difficulties v3.2.0."""

from __future__ import annotations

import json
import shutil
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from kata.baseline.grid import Grid
from kata.data.mapformat import DIFFICULTY_RANKS, load_info, parse_difficulty, read_json, tempo_for

INFO_VERSION, DIFFICULTY_VERSION = "2.1.0", "3.2.0"
LEVEL_AUTHOR = "kata-baseline v1"
COVER = "cover.png"
SONG = "song.egg"
# Every top-level key BeatSaver's v3.2 validator requires; all empty except notes and BPM events.
EMPTY_V3 = ("rotationEvents", "bombNotes", "obstacles", "sliders", "burstSliders", "waypoints",
            "basicBeatmapEvents", "colorBoostBeatmapEvents", "lightColorEventBoxGroups",
            "lightRotationEventBoxGroups", "lightTranslationEventBoxGroups")


@dataclass
class DifficultyOut:
    notes: pl.DataFrame      # beat, time_s, x, y, color, cut_dir, angle_offset
    njs: float
    njs_offset: float


def filename(label: str) -> str:
    return f"{label}Standard.dat"


def njs_offset_for(njs: float, reaction_time_s: float, bpm: float) -> float:
    """NJS offset (beats) that gives this reaction time, inverting the game's HJD rule (§3.4)."""
    hjd = 4.0
    while njs * (60.0 / bpm) * hjd > 17.999:
        hjd /= 2.0
    target = max(reaction_time_s * bpm / 60.0, 0.25)
    return round(target - hjd, 4)


def difficulty_json(notes: pl.DataFrame, grid: Grid) -> dict:
    beats = grid.to_beats(notes["time_s"].to_numpy())
    data = {"version": DIFFICULTY_VERSION,
            "bpmEvents": [{"b": b, "m": m} for b, m in grid.bpm_events()],
            "colorNotes": [{"b": float(b), "x": int(r["x"]), "y": int(r["y"]), "c": int(r["color"]),
                            "d": int(r["cut_dir"]), "a": int(r["angle_offset"])}
                           for b, r in zip(beats, notes.iter_rows(named=True))],
            "basicEventTypesWithKeywords": {"d": []},
            "useNormalEventsAsCompatibleEvents": True}
    data.update({k: [] for k in EMPTY_V3})
    return data


def info_json(meta: dict, grid: Grid, diffs: dict[str, DifficultyOut], duration_s: float) -> dict:
    return {
        "_version": INFO_VERSION,
        "_songName": meta.get("song_name") or "", "_songSubName": meta.get("song_sub_name") or "",
        "_songAuthorName": meta.get("song_author") or "", "_levelAuthorName": LEVEL_AUTHOR,
        "_beatsPerMinute": round(grid.bpm, 6), "_songTimeOffset": 0, "_shuffle": 0, "_shufflePeriod": 0.5,
        "_previewStartTime": round(0.3 * duration_s, 3), "_previewDuration": 10,
        "_songFilename": SONG, "_coverImageFilename": COVER, "_environmentName": "DefaultEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": label, "_difficultyRank": DIFFICULTY_RANKS[label],
                "_beatmapFilename": filename(label), "_noteJumpMovementSpeed": d.njs,
                "_noteJumpStartBeatOffset": d.njs_offset}
                for label, d in sorted(diffs.items(), key=lambda kv: DIFFICULTY_RANKS[kv[0]])]}]}


def write_png(path: Path, size: int = 256, rgb: tuple[int, int, int] = (40, 44, 52)) -> None:
    """A solid-colour PNG, written with the standard library only."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def write_map(folder: Path, audio: Path, meta: dict, grid: Grid, diffs: dict[str, DifficultyOut],
              duration_s: float) -> Path:
    """Write the folder and a zip next to it (same name + .zip); returns the zip path."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(audio, folder / SONG)
    write_png(folder / COVER)
    (folder / "Info.dat").write_text(json.dumps(info_json(meta, grid, diffs, duration_s), indent=1))
    for label, d in diffs.items():
        (folder / filename(label)).write_text(json.dumps(difficulty_json(d.notes, grid)))
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.iterdir()):
            z.write(f, f.name)
    return zip_path


def round_trip(folder: Path, diffs: dict[str, DifficultyOut], tol_s: float = 1e-4) -> None:
    """Re-parse with the Phase 1 parser; notes and times must come back unchanged (§8.5)."""
    info = load_info(folder)
    refs = {d.difficulty: d for d in info.difficulties}
    for label, d in diffs.items():
        bm = parse_difficulty(read_json(folder / refs[label].filename))
        tempo, _ = tempo_for(info, bm)
        got = [(n[1], n[2], n[3], n[4]) for n in bm.notes]
        want = [tuple(r) for r in d.notes.sort("time_s", "color", "x", "y")
                .select("x", "y", "color", "cut_dir").iter_rows()]
        if sorted(got) != sorted(want):
            raise AssertionError(f"{label}: notes differ after round trip")
        t = np.sort(tempo.to_seconds([n[0] for n in bm.notes]))
        err = np.abs(t - np.sort(d.notes["time_s"].to_numpy())).max() if len(t) else 0.0
        if err > tol_s:
            raise AssertionError(f"{label}: times differ by {err * 1000:.3f} ms after round trip")
