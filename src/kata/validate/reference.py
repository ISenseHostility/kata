"""Strata, Tier A reference quantiles and the versioned validator directory (spec §4.2-4.3, §9)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from kata.validate import rules
from kata.validate.rules import ERROR_RULES, HARD_RULES, LABELS, LOAD_RULES
from kata.validate.swings import ROT_RANGE, T_RESET_S, SwingParams

VALIDATOR_VERSION = "v1"
NPS_BAND = 2.0
MIN_STRATUM_MAPS = 100
ERROR_Q90_MIN, ERROR_Q99_GAP = 0.2, 1.0   # §5.1: minimum anchor spacing for error rules
LOAD_Q99_RATIO = 1.05                     # load rules: q99 at least this × q90, so anchors never collapse
SCORE_FLOOR = 0.01                        # §5.2: sub-scores are floored here in the geometric mean
WEIGHTS = {r: 1.0 / 6 for r in ERROR_RULES + LOAD_RULES}   # D5: equal until you approve a change


def validator_config() -> dict:
    """Every parameter that changes a score. A different value needs a new VALIDATOR_VERSION."""
    return {"version": VALIDATOR_VERSION, "swing": asdict(SwingParams()), "t_reset_s": T_RESET_S,
            "rot_range": {str(k): v for k, v in ROT_RANGE.items()},
            "vision_window": rules.VISION_WINDOW, "vision_bomb_t_min": rules.VISION_BOMB_T_MIN,
            "default_njs": rules.DEFAULT_NJS, "overlap_m": rules.OVERLAP_M,
            "bomb_bomb_s": rules.BOMB_BOMB_S, "speed_run": rules.SPEED_RUN,
            "strain_tail": rules.STRAIN_TAIL, "local": [rules.LOCAL_BEATS, rules.LOCAL_STEP, rules.LOCAL_MIN_SWINGS],
            "nps_band": NPS_BAND, "min_stratum_maps": MIN_STRATUM_MAPS,
            "error_anchor_min": [ERROR_Q90_MIN, ERROR_Q99_GAP], "load_anchor_ratio": LOAD_Q99_RATIO,
            "spread_deg": {"H3": rules.H3_DEG, "E3": rules.E3_DEG}, "score_floor": SCORE_FLOOR,
            "rules": {"hard": list(HARD_RULES), "error": list(ERROR_RULES), "load": list(LOAD_RULES)},
            "weights": WEIGHTS}


class ValidatorDir:
    """$KATA_DATA/validator/<version>/: config.json (pinned) and reference.json."""

    def __init__(self, root: Path):
        self.dir = Path(root) / VALIDATOR_VERSION

    @property
    def config_path(self) -> Path:
        return self.dir / "config.json"

    @property
    def reference_path(self) -> Path:
        return self.dir / "reference.json"

    def open_for_write(self, config: dict | None = None) -> None:
        config = validator_config() if config is None else config
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            self.check(config)
            return
        self.config_path.write_text(json.dumps(config, sort_keys=True, indent=1))

    def check(self, config: dict | None = None) -> None:
        """Refuse to read or write this version's tables when the code's config is not the pinned one."""
        config = validator_config() if config is None else config
        if not self.config_path.exists():
            raise FileNotFoundError(f"{self.config_path} is missing; run `kata validate analyse` first")
        if json.loads(self.config_path.read_text()) != json.loads(json.dumps(config, sort_keys=True)):
            raise ValueError(f"validator config differs from {self.config_path}; bump VALIDATOR_VERSION")

    def write_reference(self, ref: "Reference") -> None:
        self.reference_path.write_text(json.dumps(ref.data, indent=1, sort_keys=True))

    def load_reference(self) -> "Reference":
        self.check()
        return Reference(json.loads(self.reference_path.read_text()))


def band(nps: float) -> int:
    return int(max(nps, 0.0) // NPS_BAND)


def merge_bands(counts: dict[int, int], min_maps: int = MIN_STRATUM_MAPS) -> list[list[int]]:
    """Contiguous [lo_band, hi_band, n] cells: the smallest cell under min_maps merges into its
    smaller neighbour until every cell has min_maps (or one cell is left)."""
    lo, hi = min(counts), max(counts)
    cells = [[b, b, counts.get(b, 0)] for b in range(lo, hi + 1)]
    while len(cells) > 1 and min(c[2] for c in cells) < min_maps:
        i = min(range(len(cells)), key=lambda k: (cells[k][2], k))
        if i == 0:
            j = 1
        elif i == len(cells) - 1:
            j = i - 1
        else:
            j = i - 1 if cells[i - 1][2] <= cells[i + 1][2] else i + 1
        a, b = sorted((i, j))
        cells[a:b + 1] = [[cells[a][0], cells[b][1], cells[a][2] + cells[b][2]]]
    return cells


def stratum_key(label: str, cells: list[list[int]], nps: float) -> str:
    b = band(nps)
    cell = next((c for c in cells if c[0] <= b <= c[1]), cells[0] if b < cells[0][0] else cells[-1])
    if len(cells) == 1:
        return f"{label}|all"
    return f"{label}|{cell[0] * NPS_BAND:g}-{(cell[1] + 1) * NPS_BAND:g}"


def _anchors(values: np.ndarray, error: bool) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    q90, q99 = (float(np.quantile(v, 0.9)), float(np.quantile(v, 0.99))) if len(v) else (0.0, 0.0)
    if error:
        q90 = max(q90, ERROR_Q90_MIN)
        q99 = max(q99, q90 + ERROR_Q99_GAP)
    else:
        q99 = max(q99, q90 * LOAD_Q99_RATIO + 1e-9)
    return {"q90": q90, "q99": q99, "zero": q99 + 2 * (q99 - q90)}


class Reference:
    """Tier A train quantiles per rule (pooled for error rules, per stratum for load rules),
    the strata themselves, and the baseline mapper's targets."""

    def __init__(self, data: dict):
        self.data = data

    @property
    def strata(self) -> dict[str, list[list[int]]]:
        return self.data["strata"]

    def stratum(self, label: str, nps: float) -> str:
        return stratum_key(label, self.strata[label], nps)

    def anchors(self, rule: str, stratum: str) -> dict[str, float]:
        r = self.data["rules"][rule]
        return r["pooled"] if "pooled" in r else r["strata"][stratum]

    @property
    def baseline(self) -> dict:
        return self.data["baseline"]

    @classmethod
    def build(cls, metrics: pl.DataFrame) -> "Reference":
        """metrics: one row per Tier A train difficulty with columns difficulty, nps, njs,
        reaction_time_s, dt_p5 and one column per scored rule."""
        strata = {}
        for label in LABELS:
            sub = metrics.filter(pl.col("difficulty") == label)
            counts = dict(sub.group_by(pl.col("nps").map_elements(band, return_dtype=pl.Int64))
                          .len().iter_rows()) or {0: 0}
            strata[label] = merge_bands(counts)
        m = metrics.with_columns(pl.struct("difficulty", "nps").map_elements(
            lambda r: stratum_key(r["difficulty"], strata[r["difficulty"]], r["nps"]),
            return_dtype=pl.String).alias("stratum"))
        rules_ = {r: {"kind": "error", "pooled": _anchors(m[r].to_numpy(), True)} for r in ERROR_RULES}
        groups = m.partition_by("stratum", as_dict=True)
        for r in LOAD_RULES:
            rules_[r] = {"kind": "load", "strata": {k[0]: _anchors(g[r].to_numpy(), False)
                                                     for k, g in groups.items()}}
        by_label = m.group_by("difficulty").agg(pl.col("nps").median(), pl.col("njs").median(),
                                                pl.col("reaction_time_s").median())
        baseline = {"nps": dict(by_label.select("difficulty", "nps").iter_rows()),
                    "njs": dict(by_label.select("difficulty", "njs").iter_rows()),
                    "reaction_time_s": dict(by_label.select("difficulty", "reaction_time_s").iter_rows()),
                    "min_gap_s": {k[0]: float(g["dt_p5"].median()) for k, g in groups.items()}}
        counts = {k[0]: g.height for k, g in groups.items()}
        return cls({"version": VALIDATOR_VERSION, "n_maps": metrics.height, "strata": strata,
                    "stratum_sizes": counts, "rules": rules_, "baseline": baseline})
