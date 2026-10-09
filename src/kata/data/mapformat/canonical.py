"""Format-independent beatmap representation (docs/phases/phase-1-data.md §5)."""

from __future__ import annotations

from dataclasses import dataclass, field

NOTE_COLS = ("beat", "x", "y", "color", "cut_dir", "angle_offset")
BOMB_COLS = ("beat", "x", "y")
OBSTACLE_COLS = ("beat", "duration", "x", "y", "w", "h")
ARC_COLS = ("color", "beat", "x", "y", "cut_dir", "head_mult",
            "tail_beat", "tail_x", "tail_y", "tail_cut_dir", "tail_mult", "mid_anchor")
CHAIN_COLS = ("color", "beat", "x", "y", "cut_dir",
              "tail_beat", "tail_x", "tail_y", "slice_count", "squish")


class MapFormatError(ValueError):
    """The difficulty cannot be represented canonically (modded, malformed, ...)."""


@dataclass
class Beatmap:
    format_version: str
    notes: list[tuple] = field(default_factory=list)
    bombs: list[tuple] = field(default_factory=list)
    obstacles: list[tuple] = field(default_factory=list)
    arcs: list[tuple] = field(default_factory=list)
    chains: list[tuple] = field(default_factory=list)
    bpm_events: list[tuple[float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def sort(self) -> "Beatmap":
        self.notes.sort(key=lambda n: (n[0], n[1], n[2], n[3]))
        self.bombs.sort()
        self.obstacles.sort()
        self.arcs.sort(key=lambda a: (a[1], a[2], a[3], a[0]))
        self.chains.sort(key=lambda c: (c[1], c[2], c[3], c[0]))
        return self

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)


def check_grid(x, y, what: str) -> tuple[int, int]:
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise MapFormatError(f"{what}: non-numeric position {x!r},{y!r}")
    if x != int(x) or y != int(y) or not (0 <= x <= 3 and 0 <= y <= 2):
        raise MapFormatError(f"{what}: off-grid position x={x} y={y}")
    return int(x), int(y)


def check_color(c, what: str) -> int:
    if c not in (0, 1):
        raise MapFormatError(f"{what}: invalid color {c!r}")
    return int(c)


def check_cut(d, what: str) -> int:
    if d not in range(9):
        raise MapFormatError(f"{what}: invalid cut direction {d!r}")
    return int(d)


def check_anchor(x, y, what: str, bm: "Beatmap") -> tuple[int, int]:
    """Arc/chain endpoints: vanilla (even ranked) maps place these past the grid edge."""
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))) or x != int(x) or y != int(y):
        raise MapFormatError(f"{what}: non-integer anchor {x!r},{y!r}")
    if abs(x) >= 1000 or abs(y) >= 1000:
        raise MapFormatError(f"{what}: precision-placement anchor x={x} y={y}")
    if not (0 <= x <= 3 and 0 <= y <= 2):
        bm.warn("offgrid_arc_or_chain_anchor")
    return int(x), int(y)


def check_obstacle(x, y, w, h, what: str) -> None:
    # Vanilla maps may push walls past the grid edge, so only reject modded magnitudes.
    if any(abs(v) >= 1000 for v in (x, y, w, h)):
        raise MapFormatError(f"{what}: precision-placement wall x={x} y={y} w={w} h={h}")
