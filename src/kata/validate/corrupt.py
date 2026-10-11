"""Synthetic corruptions with known ground truth (spec §6.3).

Each corruption changes one clean difficulty in one place and says which rule must notice it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from kata.validate.rules import LOCAL_BEATS, Analysis, MapInput
from kata.validate.score import Result
from kata.validate.swings import T_RESET_S

# cut_dir after turning the swing direction by 180° and by +90°.
FLIP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5}
TURN_90 = {1: 3, 3: 0, 0: 2, 2: 1, 7: 5, 5: 4, 4: 6, 6: 7}
TURN_M90 = {v: k for k, v in TURN_90.items()}
C4_TARGET = 1.5   # C4 compresses the fastest run to this multiple of the stratum's L1 q99
CORRUPTIONS = {"C1": "E1", "C2": "E2", "C3": "H1", "C4": "L1", "C5": "L2", "C6": "H3", "C7": "E3"}
SORT = ("time_s", "color", "x", "y")


@dataclass(frozen=True)
class Injection:
    corruption: str
    rule: str
    hand: int        # -1 when the injected object is a bomb
    time_s: float
    beat: float
    end_s: float | None = None   # C4/C5 change several swings; issues up to here count as "near"


def _sorted_notes(m: MapInput) -> pl.DataFrame:
    return m.notes.sort(*SORT, maintain_order=True)


def _with_notes(m: MapInput, notes: pl.DataFrame) -> MapInput:
    return replace(m, notes=notes.sort(*SORT, maintain_order=True))


def _with_bomb(m: MapInput, t: float, beat: float, x: int, y: int) -> MapInput:
    row = pl.DataFrame({"beat": [beat], "time_s": [t], "x": [x], "y": [y]})
    bombs = row if m.bombs is None or m.bombs.height == 0 else pl.concat(
        [m.bombs.select("beat", "time_s", "x", "y"), row], how="vertical_relaxed")
    return replace(m, bombs=bombs.sort("time_s"))


def _swings(clean: Analysis, hand: int | None = None) -> pl.DataFrame:
    sw = clean.swings if hand is None else clean.swings.filter(pl.col("hand") == hand)
    return sw.sort("hand", "swing_idx")


def corrupt(kind: str, m: MapInput, clean: Analysis, rng: np.random.Generator,
            l1_q99: float | None = None) -> tuple[Injection, MapInput] | None:
    """One injection of `kind` at a random eligible place; None if the map has no eligible place.
    `clean` must be analyse() of m with chains=None, so note_rows index _sorted_notes(m).
    C4 needs the stratum's L1 q99 from the reference."""
    n = _sorted_notes(m)
    sw = _swings(clean)
    rule = CORRUPTIONS[kind]
    t_reset = T_RESET_S[m.difficulty]

    if kind == "C1":   # flip one arrow between two close neighbours of the same hand
        ok = sw.with_columns(gp=pl.col("t_start") - pl.col("t_end").shift(1).over("hand"),
                             gn=pl.col("t_start").shift(-1).over("hand") - pl.col("t_end")).filter(
            (pl.col("n_notes") == 1) & (pl.col("gp") < t_reset) & (pl.col("gn") < t_reset))
        ok = ok.filter(pl.col("note_rows").list.first().map_elements(lambda i: n["cut_dir"][i] < 8,
                                                                     return_dtype=pl.Boolean))
        if ok.height == 0:
            return None
        s = ok.row(int(rng.integers(ok.height)), named=True)
        i = s["note_rows"][0]
        notes = n.with_columns(pl.when(pl.int_range(pl.len()) == i).then(pl.col("cut_dir").replace(FLIP))
                               .otherwise(pl.col("cut_dir")).alias("cut_dir"))
        return Injection(kind, rule, s["hand"], s["t_start"], s["beat_start"]), _with_notes(m, notes)

    if kind in ("C2", "C3"):
        t, x, y = n["time_s"].to_numpy(), n["x"].to_numpy(), n["y"].to_numpy()
        if kind == "C2":   # a centre bomb 250 ms before a note on its half
            cand = [i for i in range(len(t)) if t[i] > 0.5 and not np.any(
                (np.abs(t - (t[i] - 0.25)) < 0.1) & ((x == 1) | (x == 2)) & (y == 1))]
        else:              # a bomb in a note's outer-lane cell, 10 ms after it
            cand = [i for i in range(len(t)) if x[i] in (0, 3)]
        if not cand:
            return None
        i = cand[int(rng.integers(len(cand)))]
        r = n.row(i, named=True)
        spb = 60.0 / m.bpm
        if kind == "C2":
            bx = 1 if r["x"] <= 1 else 2
            mi = _with_bomb(m, r["time_s"] - 0.25, r["beat"] - 0.25 / spb, bx, 1)
            return Injection(kind, rule, r["color"], r["time_s"], r["beat"]), mi
        mi = _with_bomb(m, r["time_s"] + 0.01, r["beat"] + 0.01 / spb, r["x"], r["y"])
        return Injection(kind, rule, -1, r["time_s"] + 0.01, r["beat"]), mi

    if kind == "C4":   # compress a random 8-swing run to C4_TARGET x the stratum's L1 q99
        if l1_q99 is None:
            return None
        runs = []
        for hand in (0, 1):
            h = _swings(clean, hand)
            v = h["speed"].fill_nan(0).fill_null(0).to_numpy()
            t = h["t_start"].to_numpy()
            for k in range(1, h.height - 7):
                f = C4_TARGET * l1_q99 / max(v[k:k + 8].mean(), 1e-9)
                # eligible: not already beyond the target, and no gap shrinks into slider range
                if f > 1.0 and np.diff(t[k - 1:k + 8]).min() / f >= 0.075:
                    runs.append((hand, k, f))
        if not runs:
            return None
        hand, k, f = runs[int(rng.integers(len(runs)))]
        run = _swings(clean, hand).slice(k, 8)
        t0, b0, t1 = run["t_start"][0], run["beat_start"][0], run["t_end"][-1]
        sel = (pl.col("color") == hand) & (pl.col("time_s") >= t0) & (pl.col("time_s") <= t1)
        notes = n.with_columns(
            pl.when(sel).then(t0 + (pl.col("time_s") - t0) / f).otherwise(pl.col("time_s")).alias("time_s"),
            pl.when(sel).then(b0 + (pl.col("beat") - b0) / f).otherwise(pl.col("beat")).alias("beat"))
        return Injection(kind, rule, hand, t0, b0, t1), _with_notes(m, notes)

    if kind == "C5":   # zig-zag 4 consecutive single-arrow swings of one hand by +90°/-90°
        hand = int(rng.integers(2))
        h = _swings(clean, hand)
        single = h["n_notes"].to_numpy() == 1
        arrow = np.array([n["cut_dir"][rows[0]] < 8 for rows in h["note_rows"]], bool) if h.height else np.zeros(0, bool)
        good = single & arrow
        # only zig-zags too fast for a reset: a slower one is a legal slow reset, not a defect (§18)
        fast = h["t_start"].to_numpy()[1:] - h["t_end"].to_numpy()[:-1] < t_reset
        starts = [k for k in range(h.height - 3) if good[k:k + 4].all() and fast[k:k + 3].all()]
        if not starts:
            return None
        k = starts[int(rng.integers(len(starts)))]
        rows = [h["note_rows"][j][0] for j in range(k, k + 4)]
        i = pl.int_range(pl.len())
        notes = n.with_columns(pl.when(i.is_in(rows[0::2])).then(pl.col("cut_dir").replace(TURN_90))
                               .when(i.is_in(rows[1::2])).then(pl.col("cut_dir").replace(TURN_M90))
                               .otherwise(pl.col("cut_dir")).alias("cut_dir"))
        end = h["t_end"][min(k + 4, h.height - 1)]   # the zig-zag and the swing that leaves it
        return Injection(kind, rule, hand, h["t_start"][k], h["beat_start"][k], end), _with_notes(m, notes)

    if kind in ("C6", "C7"):   # turn one arrow of a same-time multi-note swing by 90° / by 60°
        ok = sw.filter(pl.col("kind").is_in(["stack", "window"]) & (pl.col("spread") == 0))
        ok = ok.filter(pl.col("note_rows").map_elements(lambda r: all(n["cut_dir"][i] < 8 for i in r),
                                                        return_dtype=pl.Boolean))
        if ok.height == 0:
            return None
        s = ok.row(int(rng.integers(ok.height)), named=True)
        i = s["note_rows"][-1]
        hit = pl.int_range(pl.len()) == i
        if kind == "C6":
            notes = n.with_columns(pl.when(hit).then(pl.col("cut_dir").replace(TURN_90)).otherwise(pl.col("cut_dir")))
        else:
            notes = n.with_columns(pl.when(hit).then(pl.col("angle_offset") + 60).otherwise(pl.col("angle_offset")))
        return Injection(kind, rule, s["hand"], s["t_start"], s["beat_start"]), _with_notes(m, notes)
    raise ValueError(f"unknown corruption {kind}")


def _window(inj: Injection, res: Result) -> tuple[float, float]:
    """Time span of the injected swing's neighbours (±1 swing of the same hand), or ±0.5 s for bombs."""
    if inj.hand < 0:
        return inj.time_s - 0.5, inj.time_s + 0.5
    h = res.swings.filter(pl.col("hand") == inj.hand).sort("swing_idx")
    t0, t1 = h["t_start"].to_numpy(), h["t_end"].to_numpy()
    j = int(np.clip(np.searchsorted(t0, inj.time_s + 1e-6) - 1, 0, max(len(t0) - 1, 0)))
    lo, hi = max(j - 1, 0), min(j + 1, len(t0) - 1)
    return (t0[lo] - 1e-6, t1[hi] + 1e-6) if len(t0) else (inj.time_s - 0.5, inj.time_s + 0.5)


def _dropped(clean: Result, bad: Result, rule: str, beat: float) -> bool:
    """A window containing `beat` whose sub-score for `rule` is < 0.5 in `bad` but was not in `clean`."""
    def below(res: Result) -> set[float]:
        if res.local.height == 0:
            return set()
        w = res.local.filter((pl.col("window_start_beat") <= beat) & (pl.col("window_start_beat") + LOCAL_BEATS > beat)
                             & (pl.col(f"s_{rule}") < 0.5))
        return set(w["window_start_beat"].to_list())
    return bool(below(bad) - below(clean))


def evaluate(inj: Injection, clean: Result, bad: Result) -> tuple[bool, int]:
    """(detected, false alarms). Hard/error rules: a new issue of the rule within ±1 swing, on the
    injected hand (any hand for bombs). Load rules: that rule's local sub-score drops below 0.5 in
    a window containing the injection. C5 (L2) also counts L3 there or a new E1 issue within ±1
    swing: the DP reads a fast zig-zag as rotation strain or a parity break, and a slower one as
    resets (§6.3). Issues and windows already present in the clean map never count."""
    lo, hi = _window(inj, bad)
    if inj.end_s is not None:
        hi = max(hi, inj.end_s + 1e-6)
    near = (pl.col("time_s") >= lo) & (pl.col("time_s") <= hi)
    own = near & (pl.col("hand") == inj.hand) if inj.hand >= 0 else near
    key = pl.col("rule") + "@" + pl.col("time_s").round(3).cast(pl.String)
    old = list(set(clean.issues.select(key).to_series().to_list()))
    fresh = bad.issues.with_columns(k=key).filter(~pl.col("k").is_in(old))
    if inj.rule.startswith("L"):
        # C5's zig-zag reads as rotation strain, as resets, or as parity breaks, depending on tempo.
        rules_ = ("L2", "L3") if inj.rule == "L2" else (inj.rule,)
        hit = any(_dropped(clean, bad, r, inj.beat) for r in rules_) or (
            inj.rule == "L2" and fresh.filter(own & (pl.col("rule") == "E1")).height > 0)
    else:
        hit = fresh.filter(own & (pl.col("rule") == inj.rule)).height > 0
    return hit, fresh.filter(~near).height
