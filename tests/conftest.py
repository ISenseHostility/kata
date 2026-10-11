"""Shared synthetic fixtures for the Phase 3 tests (validator + baseline)."""
import polars as pl
import pytest


def notes_df(rows) -> pl.DataFrame:
    """rows: (time_s, x, y, color, cut_dir[, angle_offset]); beat = time_s * 2 (120 BPM)."""
    out = [(t * 2.0, t, x, y, c, d, (r[0] if r else 0)) for t, x, y, c, d, *r in rows]
    return pl.DataFrame(out, schema={"beat": pl.Float64, "time_s": pl.Float64, "x": pl.Int64, "y": pl.Int64,
                                     "color": pl.Int64, "cut_dir": pl.Int64, "angle_offset": pl.Int64},
                        orient="row")


def bombs_df(rows) -> pl.DataFrame:
    """rows: (time_s, x, y)."""
    return pl.DataFrame([(t * 2.0, t, x, y) for t, x, y in rows],
                        schema={"beat": pl.Float64, "time_s": pl.Float64, "x": pl.Int64, "y": pl.Int64},
                        orient="row")


def stream(n_per_hand: int = 64, step: float = 0.09, t0: float = 2.0, stack_every: int = 8) -> pl.DataFrame:
    """Clean Expert+ stream: hands alternate every `step` s, forehand-down at the bottom and backhand-up
    at the top, red in lane 0 and blue in lane 3; every `stack_every`-th red swing is a 2-note stack."""
    rows = []
    for i in range(2 * n_per_hand):
        t = t0 + i * step
        c = i % 2
        k = i // 2
        down = k % 2 == 0
        x, y, d = (0 if c == 0 else 3), (0 if down else 2), (1 if down else 0)
        rows.append((t, x, y, c, d))
        if c == 0 and k % stack_every == 4:
            rows.append((t, 1, y, c, d))
    return notes_df(rows)


@pytest.fixture
def toy_ref() -> "Reference":  # noqa: F821
    """A hand-built reference: error anchors at the spacing minimum, load anchors per label."""
    from kata.validate.reference import Reference   # imported here: Task 1 runs before reference.py exists
    from kata.validate.rules import ERROR_RULES, LABELS, LOAD_RULES
    err = {"q90": 0.2, "q99": 1.2, "zero": 3.2}
    load = {"L1": {"q90": 17.0, "q99": 18.0, "zero": 20.0}, "L2": {"q90": 0.05, "q99": 0.1, "zero": 0.2},
            "L3": {"q90": 1.0, "q99": 2.0, "zero": 4.0}}
    strata = {label: [[0, 10, 500]] for label in LABELS}
    rules = {r: {"kind": "error", "pooled": err} for r in ERROR_RULES}
    rules.update({r: {"kind": "load", "strata": {f"{label}|all": load[r] for label in LABELS}} for r in LOAD_RULES})
    baseline = {"nps": {"Easy": 2.4, "Normal": 3.4, "Hard": 4.4, "Expert": 5.9, "ExpertPlus": 7.6},
                "njs": {"Easy": 10, "Normal": 12, "Hard": 14, "Expert": 16, "ExpertPlus": 18},
                "reaction_time_s": {label: 0.6 for label in LABELS},
                "min_gap_s": {f"{label}|all": g for label, g in
                              zip(LABELS, (0.35, 0.3, 0.2, 0.15, 0.12))}}
    return Reference({"version": "v1", "n_maps": 500, "strata": strata, "stratum_sizes": {},
                      "rules": rules, "baseline": baseline})
