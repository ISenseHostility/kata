"""Per-hand swing model (docs/phases/phase-3-validator.md §3).

Notes of one colour are grouped into swings (§3.2); a Viterbi pass then gives every swing a
parity (forehand/backhand), a direction for all-dot swings, and a wrist rotation (§3.3).
Angles are in degrees in the player's view: 0 = down, +90 = right, 180 = up, -90 = left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RED, BLUE = 0, 1
FOREHAND, BACKHAND = 0, 1
DOT = 8
CUT_ANGLE = {0: 180.0, 1: 0.0, 2: -90.0, 3: 90.0, 4: -135.0, 5: 135.0, 6: -45.0, 7: 45.0}
DOT_OPTIONS = (0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0)
# Wrist rotation range per hand (MapCheck, spec §3.3). Rotations outside it are infeasible.
ROT_RANGE = {RED: (-155.0, 195.0), BLUE: (-195.0, 155.0)}
# Column of each hand's own middle lane, for bomb resets (bs-parity / MapCheck).
BOMB_COLUMN = {RED: 1, BLUE: 2}


# §3.3: a same-parity pair at least this far apart is a "slow" reset. Measured on all Tier A
# difficulties (2026-10-10): the 5th percentile of the gap between consecutive same-hand
# down-arrows, per difficulty label.
T_RESET_S = {"Easy": 0.60, "Normal": 0.47, "Hard": 0.36, "Expert": 0.30, "ExpertPlus": 0.20}


@dataclass(frozen=True)
class SwingParams:
    merge_s: float = 0.070      # §3.2: a note closer than this to the previous one joins its swing ...
    same_time_s: float = 0.005  # ... unless it is later than this and its arrow turns >= split_deg
    split_deg: float = 90.0
    t_reset_s: float = 0.30     # the caller sets it per difficulty from T_RESET_S
    w_unjust: float = 10.0      # DP cost of an unjustified reset (and of ignoring a bomb)
    w_slow: float = 0.5         # DP cost of a slow reset
    w_rot: float = 1.0          # DP cost per (rotation change / 180)^2
    rebound_deg: float = 180.0  # E1: rotation change of at least this (BL AutoModder ParityWarningAngle)
    inverted_deg: float = 135.0 # E1: |rotation| above this is an inverted swing (BL AutoModder)


@dataclass
class Swing:
    hand: int
    t_start: float
    t_end: float
    beat_start: float
    notes: list[int]                 # row positions in the notes array of this difficulty
    xs: list[int]
    ys: list[int]
    angles: list[float]              # arrow angles; dots excluded
    times: list[float]
    is_chain: bool = False
    # filled by orient()
    parity: int = FOREHAND
    angle: float = math.nan
    rotation: float = 0.0
    d_rot: float = 0.0
    reset: str | None = None
    forced: int | None = None
    entry: tuple[int, int] = (0, 0)
    exit: tuple[int, int] = (0, 0)

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    @property
    def spread(self) -> float:
        a = self.angles
        return max((angle_diff(p, q) for i, p in enumerate(a) for q in a[i + 1:]), default=0.0)

    @property
    def kind(self) -> str:
        if self.is_chain:
            return "chain"
        if len(self.notes) == 1:
            return "single"
        if max(self.times) - min(self.times) > 1e-3:
            return "slider"
        cells = sorted(zip(self.xs, self.ys))
        gap = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(cells, cells[1:]))
        return "stack" if gap <= 1 else "window"


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, in [0, 180]."""
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def circular_mean(angles: list[float]) -> float:
    r = np.radians(angles)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def rotation(theta: float, parity: int, hand: int) -> float | None:
    """Wrist rotation of a swing in direction theta: forehand neutral = down, backhand neutral = up.
    Returns None when it falls outside the hand's range."""
    lo, hi = ROT_RANGE[hand]
    r = theta if parity == FOREHAND else theta - 180.0
    r = (r - lo) % 360.0 + lo
    return r if r <= hi + 1e-9 else None


def note_angle(cut_dir: int, angle_offset: float) -> float | None:
    """Swing direction of a note; None for dots."""
    if cut_dir == DOT:
        return None
    return (CUT_ANGLE[cut_dir] + angle_offset + 180.0) % 360.0 - 180.0


def build_swings(t, beat, x, y, color, cut_dir, angle_offset, chain_head=None,
                 params: SwingParams = SwingParams()) -> dict[int, list[Swing]]:
    """Group one difficulty's notes into swings per hand (§3.2). Arrays must be sorted by time.

    A note joins the current swing when it is within merge_s of the previous note, unless it comes
    more than same_time_s later with an arrow >= split_deg away from one already in the swing (a
    fast stream, not a slider; MapCheck's rule). Same-time arrows that disagree stay in one swing
    and are judged by H3/E3."""
    chain_head = np.zeros(len(t), bool) if chain_head is None else chain_head
    out: dict[int, list[Swing]] = {RED: [], BLUE: []}
    for hand in (RED, BLUE):
        idx = np.flatnonzero(color == hand)
        cur: Swing | None = None
        last_t = -math.inf
        for i in idx:
            ti = float(t[i])
            a = note_angle(int(cut_dir[i]), float(angle_offset[i]))
            turns = (a is not None and cur is not None and ti - last_t > params.same_time_s
                     and any(angle_diff(a, b) >= params.split_deg for b in cur.angles))
            if cur is None or ti - last_t > params.merge_s or turns:
                cur = Swing(hand, ti, ti, float(beat[i]), [], [], [], [], [])
                out[hand].append(cur)
            cur.notes.append(int(i))
            cur.xs.append(int(x[i]))
            cur.ys.append(int(y[i]))
            cur.times.append(ti)
            cur.t_end = ti
            cur.is_chain |= bool(chain_head[i])
            if a is not None:
                cur.angles.append(a)
            last_t = ti
    return out


def _forced(swings: list[Swing], hand: int, bomb_t, bomb_x, bomb_y) -> list[int | None]:
    """Parity forced by the last bomb in the hand's own middle column, top or bottom row, that lies
    between the previous swing and this one: bottom row -> forehand, top row -> backhand."""
    col = BOMB_COLUMN[hand]
    m = (bomb_x == col) & ((bomb_y == 0) | (bomb_y == 2))
    bt, by = bomb_t[m], bomb_y[m]
    forced: list[int | None] = []
    prev_end = -math.inf
    for s in swings:
        lo = np.searchsorted(bt, prev_end, side="right")
        hi = np.searchsorted(bt, s.t_start, side="left")
        forced.append(None if hi <= lo else (FOREHAND if by[hi - 1] == 0 else BACKHAND))
        prev_end = s.t_end
    return forced


def _states(s: Swing) -> list[tuple[int, float, float]]:
    """(parity, direction, rotation) options for a swing; all-dot swings may take any direction."""
    thetas = [circular_mean(s.angles)] if s.angles else list(DOT_OPTIONS)
    out = []
    for th in thetas:
        for p in (FOREHAND, BACKHAND):
            r = rotation(th, p, s.hand)
            if r is not None:
                out.append((p, th, r))
    return out


def orient(swings: list[Swing], bomb_t, bomb_x, bomb_y, params: SwingParams = SwingParams()) -> None:
    """Viterbi over one hand's swings: minimise strain + rotation change + reset costs (§3.3).
    Fills parity, angle, rotation, d_rot, forced, reset, entry and exit in place."""
    if not swings:
        return
    hand = swings[0].hand
    forced = _forced(swings, hand, bomb_t, bomb_x, bomb_y)
    states = [_states(s) for s in swings]
    cost = [((st[2] / 180.0) ** 2 + (0.1 if st[0] == BACKHAND else 0.0)
             + (params.w_unjust if forced[0] is not None and st[0] != forced[0] else 0.0))
            for st in states[0]]
    back: list[list[int]] = [[]]
    for i in range(1, len(swings)):
        gap = swings[i].t_start - swings[i - 1].t_end
        new_cost, new_back = [], []
        for p, _, r in states[i]:
            base = (r / 180.0) ** 2
            if forced[i] is not None and p != forced[i]:
                base += params.w_unjust
            best, arg = math.inf, 0
            for j, (pp, _, rp) in enumerate(states[i - 1]):
                d = abs(r - rp)
                c = cost[j] + params.w_rot * (d / 180.0) ** 2
                if d >= params.rebound_deg:
                    c += params.w_unjust
                if p == pp and forced[i] != p:
                    c += params.w_slow if gap >= params.t_reset_s else params.w_unjust
                if c < best:
                    best, arg = c, j
            new_cost.append(base + best)
            new_back.append(arg)
        cost, back = new_cost, back + [new_back]
    k = int(np.argmin(cost))
    for i in range(len(swings) - 1, -1, -1):
        s = swings[i]
        s.parity, s.angle, s.rotation = states[i][k]
        s.forced = forced[i]
        if i:
            k = back[i][k]
    for i, s in enumerate(swings):
        _entry_exit(s)
        if i == 0:
            continue
        prev = swings[i - 1]
        s.d_rot = abs(s.rotation - prev.rotation)
        if s.parity == prev.parity:
            gap = s.t_start - prev.t_end
            s.reset = ("bomb" if s.forced == s.parity
                       else "slow" if gap >= params.t_reset_s else "unjustified")


def _entry_exit(s: Swing) -> None:
    """First and last cell along the swing: by time, then by position along the cut direction."""
    th = math.radians(s.angle)
    dx, dy = math.sin(th), -math.cos(th)
    order = sorted(range(s.n_notes), key=lambda k: (round(s.times[k], 3), s.xs[k] * dx + s.ys[k] * dy))
    s.entry = (s.xs[order[0]], s.ys[order[0]])
    s.exit = (s.xs[order[-1]], s.ys[order[-1]])
