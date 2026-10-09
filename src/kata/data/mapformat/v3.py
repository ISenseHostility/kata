"""Difficulty format v3 (`colorNotes`, `bombNotes`, `obstacles`, `sliders`, `burstSliders`)."""

from __future__ import annotations

from kata.data.mapformat.canonical import (Beatmap, check_anchor, check_color, check_cut,
                                           check_grid, check_obstacle)


def parse(data: dict) -> Beatmap:
    bm = Beatmap(format_version=str(data.get("version", "3.0.0")))
    # v3 writers commonly omit keys whose value is 0.
    for i, n in enumerate(data.get("colorNotes") or []):
        what = f"colorNotes[{i}]"
        x, y = check_grid(n.get("x", 0), n.get("y", 0), what)
        bm.notes.append((float(n.get("b", 0)), x, y, check_color(n.get("c", 0), what),
                         check_cut(n.get("d", 0), what), int(n.get("a", 0))))

    for i, n in enumerate(data.get("bombNotes") or []):
        x, y = check_grid(n.get("x", 0), n.get("y", 0), f"bombNotes[{i}]")
        bm.bombs.append((float(n.get("b", 0)), x, y))

    for i, o in enumerate(data.get("obstacles") or []):
        x, y, w, h = (o.get(k, 0) for k in ("x", "y", "w", "h"))
        check_obstacle(x, y, w, h, f"obstacles[{i}]")
        bm.obstacles.append((float(o.get("b", 0)), float(o.get("d", 0)), int(x), int(y), int(w), int(h)))

    for i, s in enumerate(data.get("sliders") or []):
        what = f"sliders[{i}]"
        hx, hy = check_anchor(s.get("x", 0), s.get("y", 0), what, bm)
        tx, ty = check_anchor(s.get("tx", 0), s.get("ty", 0), what, bm)
        bm.arcs.append((check_color(s.get("c", 0), what), float(s.get("b", 0)), hx, hy,
                        check_cut(s.get("d", 0), what), float(s.get("mu", 1.0)),
                        float(s.get("tb", 0)), tx, ty, check_cut(s.get("tc", 0), what),
                        float(s.get("tmu", 1.0)), int(s.get("m", 0))))

    for i, s in enumerate(data.get("burstSliders") or []):
        what = f"burstSliders[{i}]"
        hx, hy = check_anchor(s.get("x", 0), s.get("y", 0), what, bm)
        tx, ty = check_anchor(s.get("tx", 0), s.get("ty", 0), what, bm)
        bm.chains.append((check_color(s.get("c", 0), what), float(s.get("b", 0)), hx, hy,
                          check_cut(s.get("d", 0), what), float(s.get("tb", 0)), tx, ty,
                          int(s.get("sc", 1)), float(s.get("s", 1.0))))

    for e in data.get("bpmEvents") or []:
        bm.bpm_events.append((float(e.get("b", 0)), float(e.get("m", 0))))

    custom = data.get("customData") or {}
    if custom.get("BPMChanges") or custom.get("bpmChanges"):
        bm.warn("legacy_customdata_bpmchanges_ignored")
    return bm.sort()
