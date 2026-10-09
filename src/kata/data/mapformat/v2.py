"""Difficulty format v2 (`_notes`, `_obstacles`, `_sliders`, `_events`)."""

from __future__ import annotations

from kata.data.mapformat.canonical import (Beatmap, MapFormatError, check_anchor, check_color,
                                           check_cut, check_grid, check_obstacle)

BPM_EVENT_TYPE = 100
# v2 obstacle `_type` -> (y, h) in v3 terms.
OBSTACLE_TYPES = {0: (0, 5), 1: (2, 3)}


def parse(data: dict) -> Beatmap:
    bm = Beatmap(format_version=str(data.get("_version", "2.0.0")))
    for i, n in enumerate(data.get("_notes") or []):
        what = f"_notes[{i}]"
        t = n.get("_type")
        x, y = check_grid(n.get("_lineIndex"), n.get("_lineLayer"), what)
        beat = float(n["_time"])
        if t == 3:
            bm.bombs.append((beat, x, y))
        elif t in (0, 1):
            bm.notes.append((beat, x, y, check_color(t, what),
                             check_cut(n.get("_cutDirection"), what), 0))
        else:
            raise MapFormatError(f"{what}: unsupported note type {t!r}")

    for i, o in enumerate(data.get("_obstacles") or []):
        what = f"_obstacles[{i}]"
        t = o.get("_type")
        if t not in OBSTACLE_TYPES:
            raise MapFormatError(f"{what}: unsupported obstacle type {t!r}")
        y, h = OBSTACLE_TYPES[t]
        x, w = o.get("_lineIndex", 0), o.get("_width", 1)
        check_obstacle(x, y, w, h, what)
        bm.obstacles.append((float(o["_time"]), float(o.get("_duration", 0)), int(x), y, int(w), h))

    for i, s in enumerate(data.get("_sliders") or []):  # v2.6 arcs
        what = f"_sliders[{i}]"
        hx, hy = check_anchor(s.get("_headLineIndex"), s.get("_headLineLayer"), what, bm)
        tx, ty = check_anchor(s.get("_tailLineIndex"), s.get("_tailLineLayer"), what, bm)
        bm.arcs.append((
            check_color(s.get("_colorType"), what), float(s["_headTime"]), hx, hy,
            check_cut(s.get("_headCutDirection"), what),
            float(s.get("_headControlPointLengthMultiplier", 1.0)),
            float(s["_tailTime"]), tx, ty, check_cut(s.get("_tailCutDirection"), what),
            float(s.get("_tailControlPointLengthMultiplier", 1.0)),
            int(s.get("_sliderMidAnchorMode", 0))))

    for e in data.get("_events") or []:
        if e.get("_type") == BPM_EVENT_TYPE:
            bm.bpm_events.append((float(e["_time"]), float(e.get("_floatValue", 0))))

    custom = data.get("_customData") or {}
    if custom.get("_BPMChanges") or custom.get("_bpmChanges"):
        # Editor-only grid helpers; the game never read them, so neither do we.
        bm.warn("legacy_customdata_bpmchanges_ignored")
    return bm.sort()
