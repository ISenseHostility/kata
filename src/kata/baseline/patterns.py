"""Hands, directions and positions (spec §8.3): alternating, parity-correct, no vision blockers."""

from __future__ import annotations

import numpy as np
import polars as pl

from kata.validate.swings import BACKHAND, BLUE, CUT_ANGLE, FOREHAND, RED, angle_diff

DIR_OF_ANGLE = {a: d for d, a in CUT_ANGLE.items()}
DOUBLE_LABELS = ("Hard", "Expert", "ExpertPlus")
CROSSOVER_LABELS = ("Expert", "ExpertPlus")
STRAIGHT_ONLY = ("Easy", "Normal")
P_STRAIGHT, P_MID_ROW, P_CROSSOVER = 0.6, 0.3, 0.05
DOUBLE_RATIO = 2.0       # double if strength >= this × the median strength of the selected slots
DOWNBEAT_TOL_S = 0.03
NOTE_SCHEMA = {"beat": pl.Float64, "time_s": pl.Float64, "x": pl.Int64, "y": pl.Int64,
               "color": pl.Int64, "cut_dir": pl.Int64, "angle_offset": pl.Int64}


def _norm(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class Hand:
    def __init__(self, color: int):
        self.color = color
        self.parity = BACKHAND        # so the first swing is a forehand
        self.angle = 180.0
        self.last_t = -np.inf

    def directions(self, label: str, rng: np.random.Generator) -> list[float]:
        """Candidate angles for the next swing (opposite parity), most preferred first."""
        base = 0.0 if self.parity == BACKHAND else 180.0
        opposite = _norm(self.angle + 180.0)
        options = [base] if label in STRAIGHT_ONLY else [base, _norm(base + 45), _norm(base - 45)]
        options = [a for a in options if angle_diff(a, opposite) <= 45.0 + 1e-9] or [base]
        if len(options) > 1 and rng.random() > P_STRAIGHT:
            options = options[1:] + options[:1] if rng.random() < 0.5 else options[::-1]
        return options

    def swing(self, t: float, angle: float) -> None:
        self.parity = FOREHAND if self.parity == BACKHAND else BACKHAND
        self.angle, self.last_t = angle, t


def position(hand: Hand, label: str, rng: np.random.Generator, crossover: bool = True) -> tuple[int, int]:
    """Row by parity (forehand low, backhand high, sometimes the middle row); lane by hand. The
    middle row only uses the outer lanes, so the baseline never places a vision blocker. Doubles
    pass crossover=False, so the two hands never share a cell."""
    low = hand.parity == FOREHAND   # parity already flipped to this swing's
    y = 1 if rng.random() < P_MID_ROW else (0 if low else 2)
    lanes = (0, 1) if hand.color == RED else (2, 3)
    if y == 1:
        lanes = (0,) if hand.color == RED else (3,)
    elif crossover and label in CROSSOVER_LABELS and rng.random() < P_CROSSOVER:
        lanes = (2,) if hand.color == RED else (1,)
    return int(lanes[int(rng.integers(len(lanes)))]), y


def place(times: np.ndarray, beats: np.ndarray, strengths: np.ndarray, downbeats: np.ndarray,
          label: str, min_gap_hand_s: float, rng: np.random.Generator) -> pl.DataFrame:
    """Notes for the selected slots: alternating hands, doubles on strong downbeats (Hard and up)."""
    hands = {RED: Hand(RED), BLUE: Hand(BLUE)}
    rows, nxt = [], BLUE
    strong = DOUBLE_RATIO * (np.median(strengths) if len(strengths) else 0.0)
    for i, (t, b, s) in enumerate(zip(times, beats, strengths)):
        on_downbeat = len(downbeats) and np.min(np.abs(downbeats - t)) <= DOWNBEAT_TOL_S
        both_free = all(t - h.last_t >= min_gap_hand_s for h in hands.values())
        nxt_t = times[i + 1] if i + 1 < len(times) else np.inf
        double = (label in DOUBLE_LABELS and on_downbeat and both_free and nxt_t - t >= min_gap_hand_s
                  and s >= strong and hands[RED].parity == hands[BLUE].parity)
        for c in ((RED, BLUE) if double else (nxt,)):
            h = hands[c]
            a = h.directions(label, rng)[0]
            h.swing(t, a)
            x, y = position(h, label, rng, crossover=not double)
            rows.append((float(b), float(t), x, y, c, DIR_OF_ANGLE[a], 0))
        nxt = BLUE if (double or nxt == RED) else RED
    return pl.DataFrame(rows, schema=NOTE_SCHEMA, orient="row").sort("time_s", "color", "x", "y")
