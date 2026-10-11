"""Sub-scores, the combined score, the gate and validate() (spec §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from kata.validate.reference import SCORE_FLOOR, WEIGHTS, Reference
from kata.validate.rules import HARD_RULES, SCORED_RULES, Analysis, MapInput, analyse


def sub_score(m: float, a: dict[str, float]) -> float:
    """1 up to q90, 0.5 at q99, 0 at `zero`; linear in between (§5.1)."""
    if m <= a["q90"]:
        return 1.0
    if m <= a["q99"]:
        return 1.0 - 0.5 * (m - a["q90"]) / (a["q99"] - a["q90"])
    if m < a["zero"]:
        return 0.5 * (a["zero"] - m) / (a["zero"] - a["q99"])
    return 0.0


def sub_scores_np(m: np.ndarray, q90: np.ndarray, q99: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """Vectorised sub_score."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.select([m <= q90, m <= q99, m < zero],
                         [1.0, 1.0 - 0.5 * (m - q90) / (q99 - q90), 0.5 * (zero - m) / (zero - q99)], 0.0)


def score_frame(df: pl.DataFrame, ref: Reference, weights: dict[str, float] = WEIGHTS) -> pl.DataFrame:
    """Add s_<rule> and score to a frame that has a `stratum` column and one column per scored rule."""
    strata = df["stratum"].unique().to_list()
    cols = {}
    for r in SCORED_RULES:
        an = {s: ref.anchors(r, s) for s in strata}
        q = {k: df["stratum"].replace_strict({s: a[k] for s, a in an.items()}, return_dtype=pl.Float64).to_numpy()
             for k in ("q90", "q99", "zero")}
        cols[f"s_{r}"] = sub_scores_np(df[r].to_numpy().astype(np.float64), q["q90"], q["q99"], q["zero"])
    total = sum(weights.values())
    logs = sum(w * np.log(np.maximum(cols[f"s_{r}"], SCORE_FLOOR)) for r, w in weights.items()) / total
    return df.with_columns(*(pl.Series(k, v) for k, v in cols.items()), pl.Series("score", 100.0 * np.exp(logs)))


def combine(subs: dict[str, float], weights: dict[str, float] = WEIGHTS) -> float:
    """100 × weighted geometric mean, each sub-score floored at SCORE_FLOOR (§5.2)."""
    total = sum(weights.values())
    return 100.0 * math.exp(sum(w * math.log(max(subs[r], SCORE_FLOOR)) for r, w in weights.items()) / total)


def score_metrics(metrics: dict[str, float], stratum: str, ref: Reference,
                  weights: dict[str, float] = WEIGHTS) -> tuple[float, dict[str, float]]:
    subs = {r: sub_score(metrics[r], ref.anchors(r, stratum)) for r in SCORED_RULES}
    return combine(subs, weights), subs


@dataclass
class Result:
    passed: bool
    score: float
    sub_scores: dict[str, float]
    metrics: dict[str, float]
    stratum: str
    swings: pl.DataFrame
    issues: pl.DataFrame
    local: pl.DataFrame          # window_start_beat, n_swings, rule metrics, s_<rule>, score


def score_local(local: pl.DataFrame, stratum: str, ref: Reference,
                weights: dict[str, float] = WEIGHTS) -> pl.DataFrame:
    if local.height == 0:
        return local
    return score_frame(local.with_columns(pl.lit(stratum).alias("stratum")), ref, weights).drop("stratum")


def score_analysis(a: Analysis, difficulty: str, ref: Reference, weights: dict[str, float] = WEIGHTS) -> Result:
    stratum = ref.stratum(difficulty, a.metrics["nps"])
    total, subs = score_metrics(a.metrics, stratum, ref, weights)
    passed = all(a.metrics[r] == 0 for r in HARD_RULES)
    return Result(passed, total, subs, a.metrics, stratum, a.swings, a.issues,
                  score_local(a.local, stratum, ref, weights))


def validate(m: MapInput, ref: Reference, weights: dict[str, float] = WEIGHTS) -> Result:
    """Gate, score, per-rule breakdown, issues and local scores for one difficulty. Pure: no I/O."""
    return score_analysis(analyse(m), m.difficulty, ref, weights)
