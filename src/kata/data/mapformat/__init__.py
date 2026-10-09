"""Load an extracted map folder (Info.dat + difficulty files) into canonical form."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from kata.data.mapformat import v2, v3, v4
from kata.data.mapformat.canonical import Beatmap, MapFormatError
from kata.data.mapformat.tempo import TempoMap

DIFFICULTY_RANKS = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}

__all__ = ["Beatmap", "MapFormatError", "TempoMap", "MapInfo", "DifficultyRef",
           "read_json", "load_info", "parse_difficulty", "tempo_for"]


@dataclass
class DifficultyRef:
    characteristic: str
    difficulty: str
    rank: int
    filename: str
    njs: float
    njs_offset: float


@dataclass
class MapInfo:
    folder: Path
    info_version: str
    song_filename: str
    bpm: float
    song_time_offset: float = 0.0
    audio_data_filename: str | None = None
    difficulties: list[DifficultyRef] = field(default_factory=list)


def read_json(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8-sig"))


def find_file(folder: Path, name: str) -> Path:
    """Case-insensitive lookup; zips are authored on case-insensitive filesystems."""
    p = folder / name
    if p.exists():
        return p
    lower = name.lower()
    for candidate in folder.iterdir():
        if candidate.name.lower() == lower:
            return candidate
    raise MapFormatError(f"missing file {name!r}")


AUDIO_SUFFIXES = (".egg", ".ogg", ".wav")


def find_audio(folder: Path, name: str) -> tuple[Path, bool]:
    """Audio named in Info.dat; falls back to the folder's only audio file (returns fallback flag)."""
    try:
        return find_file(folder, name), False
    except MapFormatError:
        candidates = [p for p in folder.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES]
        if len(candidates) == 1:
            return candidates[0], True
        raise


def load_info(folder: Path) -> MapInfo:
    data = read_json(find_file(folder, "Info.dat"))
    if "_difficultyBeatmapSets" in data:
        info = MapInfo(folder=folder, info_version=str(data.get("_version", "2.0.0")),
                       song_filename=data["_songFilename"], bpm=float(data["_beatsPerMinute"]),
                       song_time_offset=float(data.get("_songTimeOffset", 0) or 0))
        for s in data["_difficultyBeatmapSets"]:
            for d in s.get("_difficultyBeatmaps") or []:
                name = d["_difficulty"]
                info.difficulties.append(DifficultyRef(
                    s["_beatmapCharacteristicName"], name,
                    int(d.get("_difficultyRank", DIFFICULTY_RANKS.get(name, 0))),
                    d["_beatmapFilename"], float(d.get("_noteJumpMovementSpeed", 0) or 0),
                    float(d.get("_noteJumpStartBeatOffset", 0) or 0)))
        return info
    if "difficultyBeatmaps" in data:
        audio = data.get("audio") or {}
        info = MapInfo(folder=folder, info_version=str(data.get("version", "4.0.0")),
                       song_filename=audio["songFilename"], bpm=float(audio["bpm"]),
                       audio_data_filename=audio.get("audioDataFilename"))
        for d in data["difficultyBeatmaps"]:
            name = d["difficulty"]
            info.difficulties.append(DifficultyRef(
                d["characteristic"], name, DIFFICULTY_RANKS.get(name, 0), d["beatmapDataFilename"],
                float(d.get("noteJumpMovementSpeed", 0) or 0),
                float(d.get("noteJumpStartBeatOffset", 0) or 0)))
        return info
    raise MapFormatError("unrecognised Info.dat layout")


def parse_difficulty(data: dict) -> Beatmap:
    version = str(data.get("version") or data.get("_version") or "")
    if version.startswith("4"):
        return v4.parse(data)
    if version.startswith("3") or "colorNotes" in data:
        return v3.parse(data)
    if version.startswith("2") or "_notes" in data:
        return v2.parse(data)
    raise MapFormatError(f"unknown difficulty format version {version!r}")


def tempo_for(info: MapInfo, beatmap: Beatmap) -> tuple[TempoMap, str]:
    """Return the tempo map the game would use, plus a label for where it came from.

    `_songTimeOffset` is recorded on the map but not applied: modern game versions ignore
    it. The replay alignment check (§6.3) is what confirms or refutes this.
    """
    if info.audio_data_filename and not beatmap.bpm_events:
        try:
            audio = read_json(find_file(info.folder, info.audio_data_filename))
        except MapFormatError:
            audio = None
        if audio and audio.get("bpmData"):
            return (TempoMap.from_audio_regions(info.bpm, int(audio.get("songFrequency", 0)),
                                                audio["bpmData"]), "audiodata")
    source = "bpm_events" if beatmap.bpm_events else "info_bpm"
    return TempoMap.from_bpm_events(info.bpm, beatmap.bpm_events), source
