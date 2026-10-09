"""Difficulty format v4: timing arrays that index into deduplicated `*Data` arrays."""

from __future__ import annotations

from kata.data.mapformat.canonical import (Beatmap, MapFormatError, check_anchor, check_color,
                                           check_cut, check_grid, check_obstacle)


def _lookup(table: list, index, what: str) -> dict:
    if not isinstance(index, int) or not 0 <= index < len(table):
        raise MapFormatError(f"{what}: data index {index!r} out of range ({len(table)})")
    return table[index]


def _check_lane(obj: dict, key: str, what: str) -> None:
    if obj.get(key, 0) != 0:
        raise MapFormatError(f"{what}: rotation lane {obj.get(key)} (360/90 content)")


def parse(data: dict) -> Beatmap:
    bm = Beatmap(format_version=str(data.get("version", "4.0.0")))
    note_data = data.get("colorNotesData") or []

    def note_fields(index, what: str, anchor: bool = False) -> tuple[int, int, int, int, int]:
        d = _lookup(note_data, index, what)
        if anchor:
            x, y = check_anchor(d.get("x", 0), d.get("y", 0), what, bm)
        else:
            x, y = check_grid(d.get("x", 0), d.get("y", 0), what)
        return x, y, check_color(d.get("c", 0), what), check_cut(d.get("d", 0), what), int(d.get("a", 0))

    for i, n in enumerate(data.get("colorNotes") or []):
        what = f"colorNotes[{i}]"
        _check_lane(n, "r", what)
        x, y, c, d, a = note_fields(n.get("i", 0), what)
        bm.notes.append((float(n.get("b", 0)), x, y, c, d, a))

    bomb_data = data.get("bombNotesData") or []
    for i, n in enumerate(data.get("bombNotes") or []):
        what = f"bombNotes[{i}]"
        _check_lane(n, "r", what)
        d = _lookup(bomb_data, n.get("i", 0), what)
        x, y = check_grid(d.get("x", 0), d.get("y", 0), what)
        bm.bombs.append((float(n.get("b", 0)), x, y))

    obstacle_data = data.get("obstaclesData") or []
    for i, o in enumerate(data.get("obstacles") or []):
        what = f"obstacles[{i}]"
        _check_lane(o, "r", what)
        d = _lookup(obstacle_data, o.get("i", 0), what)
        x, y, w, h = (d.get(k, 0) for k in ("x", "y", "w", "h"))
        check_obstacle(x, y, w, h, what)
        bm.obstacles.append((float(o.get("b", 0)), float(d.get("d", 0)), int(x), int(y), int(w), int(h)))

    arc_data = data.get("arcsData") or []
    for i, a in enumerate(data.get("arcs") or []):
        what = f"arcs[{i}]"
        _check_lane(a, "hr", what)
        _check_lane(a, "tr", what)
        hx, hy, hc, hd, _ = note_fields(a.get("hi", 0), what, anchor=True)
        tx, ty, _, td, _ = note_fields(a.get("ti", 0), what, anchor=True)
        meta = _lookup(arc_data, a.get("ai", 0), what)
        bm.arcs.append((hc, float(a.get("hb", 0)), hx, hy, hd, float(meta.get("m", 1.0)),
                        float(a.get("tb", 0)), tx, ty, td, float(meta.get("tm", 1.0)),
                        int(meta.get("a", 0))))

    chain_data = data.get("chainsData") or []
    for i, ch in enumerate(data.get("chains") or []):
        what = f"chains[{i}]"
        _check_lane(ch, "hr", what)
        _check_lane(ch, "tr", what)
        hx, hy, hc, hd, _ = note_fields(ch.get("i", 0), what, anchor=True)
        meta = _lookup(chain_data, ch.get("ci", 0), what)
        tx, ty = check_anchor(meta.get("tx", 0), meta.get("ty", 0), what, bm)
        bm.chains.append((check_color(hc, what), float(ch.get("hb", 0)), hx, hy, hd,
                          float(ch.get("tb", 0)), tx, ty, int(meta.get("c", 1)),
                          float(meta.get("s", 1.0))))
    return bm.sort()
