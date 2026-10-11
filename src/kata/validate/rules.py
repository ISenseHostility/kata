"""Rules (spec §4): one difficulty -> swing table, per-swing issues and per-map metrics.

Hard rules (H1-H3) feed the gate; error rules (E1-E3) and load rules (L1-L3) feed the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from kata.validate.swings import BLUE, RED, T_RESET_S, Swing, SwingParams, build_swings, orient

HARD_RULES = ("H1", "H2", "H3")
ERROR_RULES = ("E1", "E2", "E3")
LOAD_RULES = ("L1", "L2", "L3")
SCORED_RULES = ERROR_RULES + LOAD_RULES
LABELS = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
# E2 windows by difficulty in seconds (MapCheck): (t_min, cap).
VISION_WINDOW = {"Easy": (0.025, 1.2), "Normal": (0.05, 1.0), "Hard": (0.08, 0.75),
                 "Expert": (0.1, 0.625), "ExpertPlus": (0.1, 0.5)}
VISION_BOMB_T_MIN = 0.2
# The game's NJS when a map leaves it at 0.
DEFAULT_NJS = {"Easy": 10.0, "Normal": 10.0, "Hard": 10.0, "Expert": 12.0, "ExpertPlus": 16.0}
OVERLAP_M = 0.5            # H1: same cell closer than this in depth (BL R3A)
BOMB_BOMB_S = 0.02         # H1: two bombs in one cell closer than this (ScoreSaber)
STRAIN_TAIL = 0.05         # L2: mean of the top 5 % of per-swing strain
SPEED_RUN = 8              # L1: swings per rolling window
LOCAL_BEATS, LOCAL_STEP, LOCAL_MIN_SWINGS = 16.0, 4.0, 8   # §5.3


@dataclass
class MapInput:
    """One difficulty in canonical (Phase 1) form. Frames carry the Phase 1 column names."""
    difficulty: str
    njs: float
    njs_offset: float
    bpm: float
    notes: pl.DataFrame                  # beat, time_s, x, y, color, cut_dir, angle_offset
    bombs: pl.DataFrame | None = None    # beat, time_s, x, y
    walls: pl.DataFrame | None = None    # time_s, end_time_s, x, y, w, h
    chains: pl.DataFrame | None = None   # color, beat, time_s, x, y, cut_dir


@dataclass
class Analysis:
    swings: pl.DataFrame
    issues: pl.DataFrame
    metrics: dict[str, float]            # rule -> per-map metric, plus n_swings, nps, hard counts
    local: pl.DataFrame                  # window_start_beat, n_swings, one column per scored rule


ISSUE_SCHEMA = {"rule": pl.String, "severity": pl.String, "hand": pl.Int8, "swing_idx": pl.Int32,
                "time_s": pl.Float64, "value": pl.Float64, "threshold": pl.Float64}


def reaction_time_s(njs: float, njs_offset: float, bpm: float) -> float:
    """Half jump duration in seconds, as the game computes it (spec §3.4)."""
    hjd = 4.0
    while njs * (60.0 / bpm) * hjd > 17.999:
        hjd /= 2.0
    return max(hjd + njs_offset, 0.25) * 60.0 / bpm


def effective_njs(m: MapInput) -> float:
    return m.njs if m.njs > 0 else DEFAULT_NJS[m.difficulty]


def nps(note_times) -> float:
    """Notes per second over the mapped span (spec §4.3); identical for every map."""
    t = np.asarray(note_times, dtype=np.float64)
    if len(t) == 0:
        return 0.0
    return len(t) / max(float(t.max() - t.min()), 1.0)


def _col(df: pl.DataFrame | None, name: str, dtype=np.float64) -> np.ndarray:
    if df is None or df.height == 0:
        return np.zeros(0, dtype)
    return df[name].to_numpy().astype(dtype)


def _notes_with_chain_heads(m: MapInput) -> dict[str, np.ndarray]:
    n = m.notes.sort("time_s", "color", "x", "y", maintain_order=True)
    cols = {k: _col(n, k) for k in ("time_s", "beat", "x", "y", "color", "cut_dir", "angle_offset")}
    cols["chain_head"] = np.zeros(n.height, bool)
    if m.chains is not None and m.chains.height:
        add = {k: [] for k in cols}
        for c in m.chains.iter_rows(named=True):
            hit = np.flatnonzero((np.abs(cols["time_s"] - c["time_s"]) < 1e-3) & (cols["color"] == c["color"])
                                 & (cols["x"] == c["x"]) & (cols["y"] == c["y"]))
            if hit.size:
                cols["chain_head"][hit] = True
                continue
            for k, v in (("time_s", c["time_s"]), ("beat", c["beat"]), ("x", c["x"]), ("y", c["y"]),
                         ("color", c["color"]), ("cut_dir", c["cut_dir"]), ("angle_offset", 0.0),
                         ("chain_head", True)):
                add[k].append(v)
        if add["time_s"]:
            cols = {k: np.concatenate([cols[k], np.asarray(add[k], dtype=cols[k].dtype)]) for k in cols}
            order = np.argsort(cols["time_s"], kind="stable")
            cols = {k: v[order] for k, v in cols.items()}
    for k in ("x", "y", "color", "cut_dir"):
        cols[k] = cols[k].astype(np.int64)
    return cols


def _issue(rule, severity, hand, swing_idx, time_s, value, threshold) -> dict:
    return {"rule": rule, "severity": severity, "hand": hand, "swing_idx": swing_idx,
            "time_s": float(time_s), "value": float(value), "threshold": float(threshold)}


def _overlaps(m: MapInput, notes, note_swing, bomb_t, bomb_x, bomb_y) -> list[dict]:
    """H1: in one cell, a note within OVERLAP_M (depth = Δt · NJS) of another object, or two bombs
    closer than BOMB_BOMB_S (ScoreSaber; bomb tunnels are legal)."""
    njs = effective_njs(m)
    t = np.concatenate([notes["time_s"], bomb_t])
    cell = np.concatenate([notes["x"] * 3 + notes["y"], bomb_x * 3 + bomb_y])
    hand = np.concatenate([notes["color"], np.full(len(bomb_t), -1)])
    sidx = np.concatenate([note_swing, np.full(len(bomb_t), -1)])
    order = np.lexsort((t, cell))
    t, cell, hand, sidx = t[order], cell[order], hand[order], sidx[order]
    dt = t[1:] - t[:-1]
    both_bombs = (hand[1:] < 0) & (hand[:-1] < 0)
    close = (cell[1:] == cell[:-1]) & np.where(both_bombs, dt < BOMB_BOMB_S, dt * njs < OVERLAP_M)
    return [_issue("H1", "hard", hand[i + 1], sidx[i + 1], t[i + 1], dt[i] * njs, OVERLAP_M)
            for i in np.flatnonzero(close)]


def _in_walls(m: MapInput, notes, note_swing) -> list[dict]:
    """H2: a note strictly inside a wall's cells and time span."""
    out = []
    if m.walls is None or m.walls.height == 0:
        return out
    t = notes["time_s"]
    for w in m.walls.iter_rows(named=True):
        if w["w"] <= 0 or w["end_time_s"] - w["time_s"] <= 0 or w["h"] <= 0:
            continue
        lo = np.searchsorted(t, w["time_s"] + 1e-3, side="right")
        hi = np.searchsorted(t, w["end_time_s"] - 1e-3, side="left")
        for i in range(lo, hi):
            if w["x"] <= notes["x"][i] < w["x"] + w["w"] and w["y"] <= notes["y"][i] < w["y"] + w["h"]:
                out.append(_issue("H2", "hard", notes["color"][i], note_swing[i], t[i], 0, 0))
    return out


def _vision(m: MapInput, notes, note_swing, bomb_t, bomb_x, bomb_y) -> list[dict]:
    """E2: a note on a centre blocker's half arriving within (t_min, t_max] after it."""
    t_min, cap = VISION_WINDOW[m.difficulty]
    t_max = min(reaction_time_s(effective_njs(m), m.njs_offset, m.bpm), cap)
    t, x, y = notes["time_s"], notes["x"], notes["y"]
    blockers = [(t[i], x[i], t_min, i) for i in np.flatnonzero(((x == 1) | (x == 2)) & (y == 1))]
    blockers += [(bt, bx, VISION_BOMB_T_MIN, -1) for bt, bx, by in zip(bomb_t, bomb_x, bomb_y)
                 if bx in (1, 2) and by == 1]
    blocked: dict[int, float] = {}
    for bt, bx, lo_s, src in blockers:
        lo = np.searchsorted(t, bt + lo_s, side="right")
        hi = np.searchsorted(t, bt + t_max, side="right")
        for i in range(lo, hi):
            same_half = x[i] <= 1 if bx == 1 else x[i] >= 2
            same_swing = src >= 0 and note_swing[i] == note_swing[src] and notes["color"][i] == notes["color"][src]
            if same_half and not same_swing:
                blocked.setdefault(i, t[i] - bt)
    return [_issue("E2", "error", notes["color"][i], note_swing[i], t[i], dt, t_max)
            for i, dt in sorted(blocked.items())]


def _speed(swings: list[Swing]) -> np.ndarray:
    """Per swing: (1 + grid distance from the previous swing's exit) / start-to-start Δt."""
    v = np.full(len(swings), np.nan)
    for i in range(1, len(swings)):
        a, b = swings[i - 1], swings[i]
        d = math.dist(a.exit, b.entry)
        v[i] = (1.0 + d) / max(b.t_start - a.t_start, 1e-3)
    return v


def _peak_run(v: np.ndarray) -> float:
    """Peak mean of SPEED_RUN consecutive values (all of them when there are fewer)."""
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0.0
    if len(v) <= SPEED_RUN:
        return float(v.mean())
    c = np.concatenate([[0.0], np.cumsum(v)])
    return float(((c[SPEED_RUN:] - c[:-SPEED_RUN]) / SPEED_RUN).max())


def _metrics(hand, speed, strain, calm_reset, errors: dict[str, int], minutes: float) -> dict[str, float]:
    """Per-map (or per-window) values of the scored rules; arrays are per swing in time order."""
    n = max(len(hand), 1)
    out = {r: errors.get(r, 0) * 100.0 / n for r in ERROR_RULES}
    out["L1"] = max(_peak_run(speed[hand == h]) for h in (RED, BLUE))
    s = np.sort(strain[~np.isnan(strain)])
    out["L2"] = float(s[-max(1, int(round(len(s) * STRAIN_TAIL))):].mean()) if len(s) else 0.0
    out["L3"] = float(calm_reset.sum()) / max(minutes, 1.0 / 60)
    return out


def analyse(m: MapInput, params: SwingParams | None = None) -> Analysis:
    params = params or SwingParams(t_reset_s=T_RESET_S[m.difficulty])
    notes = _notes_with_chain_heads(m)
    bombs = m.bombs.sort("time_s") if m.bombs is not None and m.bombs.height else None
    bomb_t, bomb_x, bomb_y = _col(bombs, "time_s"), _col(bombs, "x", np.int64), _col(bombs, "y", np.int64)
    by_hand = build_swings(notes["time_s"], notes["beat"], notes["x"], notes["y"], notes["color"],
                           notes["cut_dir"], notes["angle_offset"], notes["chain_head"], params)
    note_swing = np.full(len(notes["time_s"]), -1, np.int64)
    rows, issues = [], []
    for hand, swings in by_hand.items():
        orient(swings, bomb_t, bomb_x, bomb_y, params)
        speed = _speed(swings)
        for k, s in enumerate(swings):
            note_swing[s.notes] = k
            rows.append({"hand": hand, "swing_idx": k, "t_start": s.t_start, "t_end": s.t_end,
                         "beat_start": s.beat_start, "kind": s.kind, "n_notes": s.n_notes,
                         "spread": s.spread, "parity": s.parity, "angle": s.angle, "rotation": s.rotation,
                         "d_rot": s.d_rot, "strain": (s.d_rot / 180.0) ** 2 if k else math.nan,
                         "reset": s.reset, "forced": s.forced, "speed": speed[k],
                         "entry_x": s.entry[0], "entry_y": s.entry[1], "exit_x": s.exit[0],
                         "exit_y": s.exit[1], "note_rows": s.notes})
            if s.spread >= 90:
                issues.append(_issue("H3", "hard", hand, k, s.t_start, s.spread, 90))
            elif s.spread > 45:
                issues.append(_issue("E3", "error", hand, k, s.t_start, s.spread, 45))
            if (s.reset == "unjustified" or s.d_rot >= params.rebound_deg
                    or abs(s.rotation) > params.inverted_deg):
                issues.append(_issue("E1", "error", hand, k, s.t_start, s.d_rot, params.rebound_deg))
    issues += _overlaps(m, notes, note_swing, bomb_t, bomb_x, bomb_y)
    issues += _in_walls(m, notes, note_swing)
    issues += _vision(m, notes, note_swing, bomb_t, bomb_x, bomb_y)

    sw = (pl.DataFrame(rows, schema=SWING_SCHEMA) if rows else pl.DataFrame(schema=SWING_SCHEMA))
    sw = sw.sort("t_start", "hand")
    iss = pl.DataFrame(issues, schema=ISSUE_SCHEMA).sort("time_s", "rule")
    t = notes["time_s"]
    minutes = float(t.max() - t.min()) / 60.0 if len(t) else 0.0
    counts = dict(iss.group_by("rule").len().iter_rows())
    a = _swing_arrays(sw, iss)
    njs = effective_njs(m)
    metrics = {"n_swings": float(sw.height), "nps": nps(t), "njs": njs,
               "reaction_time_s": reaction_time_s(njs, m.njs_offset, m.bpm), "dt_p5": _dt_p5(sw),
               **{r: float(counts.get(r, 0)) for r in HARD_RULES},
               **_metrics(a["hand"], a["speed"], a["strain"], a["calm"], counts, minutes)}
    return Analysis(sw, iss, metrics, _local(a))


def _dt_p5(sw: pl.DataFrame) -> float:
    """5th percentile of start-to-start intervals between consecutive swings of one hand."""
    dt = (sw.sort("hand", "swing_idx").select(pl.col("t_start").diff().over("hand"))
          .to_series().drop_nulls().to_numpy())
    return float(np.percentile(dt, 5)) if len(dt) else float("nan")


def _swing_arrays(sw: pl.DataFrame, iss: pl.DataFrame) -> dict[str, np.ndarray]:
    """Per-swing numpy columns (in sw's time order), including error-issue counts per rule."""
    a = {"hand": sw["hand"].to_numpy(), "speed": sw["speed"].to_numpy(), "strain": sw["strain"].to_numpy(),
         "beat": sw["beat_start"].to_numpy(), "t0": sw["t_start"].to_numpy(), "t1": sw["t_end"].to_numpy(),
         "calm": sw["reset"].is_in(["bomb", "slow"]).fill_null(False).to_numpy()}
    pos = {(h, k): i for i, (h, k) in enumerate(zip(a["hand"], sw["swing_idx"].to_numpy()))}
    for r in ERROR_RULES:
        c = np.zeros(sw.height, np.int64)
        for h, k in iss.filter((pl.col("rule") == r) & (pl.col("hand") >= 0)).select("hand", "swing_idx").iter_rows():
            c[pos[(h, k)]] += 1
        a[r] = c
    return a


def _local(a: dict[str, np.ndarray]) -> pl.DataFrame:
    """Rule metrics over windows of LOCAL_BEATS map beats, stepped by LOCAL_STEP (§5.3)."""
    rows = []
    if len(a["beat"]):
        order = np.argsort(a["beat"], kind="stable")
        a = {k: v[order] for k, v in a.items()}
        b = math.floor(a["beat"][0] / LOCAL_STEP) * LOCAL_STEP
        while b <= a["beat"][-1]:
            lo, hi = np.searchsorted(a["beat"], [b, b + LOCAL_BEATS], side="left")
            if hi - lo >= LOCAL_MIN_SWINGS:
                s = slice(lo, hi)
                minutes = (a["t1"][s].max() - a["t0"][s].min()) / 60.0
                errors = {r: int(a[r][s].sum()) for r in ERROR_RULES}
                rows.append({"window_start_beat": float(b), "n_swings": hi - lo,
                             **_metrics(a["hand"][s], a["speed"][s], a["strain"][s], a["calm"][s], errors, minutes)})
            b += LOCAL_STEP
    return pl.DataFrame(rows, schema={"window_start_beat": pl.Float64, "n_swings": pl.Int64,
                                      **{r: pl.Float64 for r in SCORED_RULES}})


SWING_SCHEMA = {
    "hand": pl.Int8, "swing_idx": pl.Int32, "t_start": pl.Float64, "t_end": pl.Float64,
    "beat_start": pl.Float64, "kind": pl.String, "n_notes": pl.Int32, "spread": pl.Float64,
    "parity": pl.Int8, "angle": pl.Float64, "rotation": pl.Float64, "d_rot": pl.Float64,
    "strain": pl.Float64, "reset": pl.String, "forced": pl.Int8, "speed": pl.Float64,
    "entry_x": pl.Int8, "entry_y": pl.Int8, "exit_x": pl.Int8, "exit_y": pl.Int8,
    "note_rows": pl.List(pl.Int64)}
