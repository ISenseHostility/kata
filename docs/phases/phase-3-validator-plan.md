# Phase 3 — Validator + Rule-Based Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `kata.validate` (a playability validator with a hard gate, a calibrated 0–100 score, per-swing issues and local scores), `kata.baseline` (an audio-only, notes-only mapper for all five difficulties that exports ArcViewer-loadable maps), and the Tier D + synthetic-corruption calibration that proves the score means something. Then run them on the pilot and on the full dataset.

**Architecture:** One per-hand swing model (`swings.py`: swing formation + a Viterbi pass for parity, dot directions and wrist rotation) feeds every rule (`rules.py`). `reference.py` turns Tier A train metrics into strata and quantile anchors; `score.py` turns metrics into sub-scores, a geometric-mean score and a gate. `corrupt.py` injects known defects. The baseline (`grid` → `onsets` → `patterns` → `generate` → `export`) reuses the same validator for its self-check. Stages follow the Phase 1/2 conventions (`_with_state`, deterministic table rebuilds) and Tier D reuses the Phase 1 download/parse code with audio dropped.

**Tech Stack:** Python 3.13 (uv), numpy, polars, duckdb, typer, pytest. No new dependencies. Everything runs on CPU; the baseline reads Phase 2 features through `kata.audio.store`, which does not import torch.

**Spec:** `docs/phases/phase-3-validator.md` (approved 2026-10-10; prototype corrections in §17). Read §3 (swing model), §4 (rules), §5 (scoring), §6 (calibration) and §8 (baseline) before starting.

**Provenance:** every code block in this plan ran in a throwaway prototype on 2026-10-10: 134 tests passing (Phases 1–3), `ruff --select F` clean, and the whole stage chain run on the real dataset in a scratch data root (spec §17). Copy the code as written; if something fails, it is an environment difference worth reporting, not a placeholder.

## Global Constraints

- Install with `uv sync` (the `audio` extra is **not** needed for Phase 3) and run with `uv run …`. `kata/cli.py` imports Phase 3 modules inside each command, like Phase 2.
- Validator constants live in code and are pinned per version: `VALIDATOR_VERSION = "v1"`; `validator/v1/config.json` is written once and a different config raises `ValueError("... bump VALIDATOR_VERSION")` (spec §9).
- Angles: degrees in the player's view, 0 = down, +90 = right, 180 = up, −90 = left; `cut_dir` 0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right, 6 down-left, 7 down-right, 8 dot. `angle_offset` turns counter-clockwise from down towards right.
- Wrist ranges: red −155° … +195°, blue −195° … +155°. Merge window 70 ms; same-time 5 ms; split at ≥ 90°.
- `T_RESET_S = {"Easy": 0.60, "Normal": 0.47, "Hard": 0.36, "Expert": 0.30, "ExpertPlus": 0.20}`.
- Rules H1–H3 (gate), E1–E3 and L1–L3 (score), equal weights 1/6, sub-score anchors q90 → 1, q99 → 0.5, q99 + 2(q99 − q90) → 0, floor 0.01; error anchors at least q90 0.2 and q99 q90 + 1.0 issues per 100 swings.
- Tier D: rating < 0.6, ≥ 20 votes, not curated, not ranked; 2,000 maps, 200 pilot; audio deleted at extraction; tables only under `processed/lowrated/`.
- Baseline export: `Info.dat` v2.1.0, difficulty files v3.2.0 with every top-level key; output under `$KATA_DATA/generated/baseline/v1/{sha}/` plus `{sha}.zip`. Generated maps are never uploaded (copyrighted audio).
- No real maps, audio or features are committed (Phase 1 D6). Test fixtures are synthetic; real-data checks read `KATA_DATA` and are run by hand.
- Every stage that downloads checks the 50 GB free-space floor (`kata.data.pipeline._check_disk`).
- Commit after every task. End each commit message with the attribution lines from the session's system reminder.

## Review Focus

1. **Re-running a stage after a crash or with different worker scheduling.** Expected: byte-identical tables. The prototype first iterated a Python `set` of sources, whose order changes per process, and row order differed between runs. Pinned by the four-table equality check in `test_stages_end_to_end` (Task 8).
2. **A map with NJS 0, or with only one hand, or with no notes.** Expected: the game's default NJS is used; metrics stay finite; no crash. Pinned by `test_njs_zero_falls_back_to_the_game_default` and `test_one_handed_and_empty_maps_do_not_crash` (Task 1).
3. **Changing a validator parameter after scores exist.** Expected: refusal with "bump VALIDATOR_VERSION", never two configs in one version directory. Pinned by `test_validator_dir_pins_its_config` (Task 2) and `test_reference_refuses_a_changed_validator_config` (Task 8).
4. **Tier D reaching a Phase 2 or Phase 4 consumer.** Expected: Tier D tables exist only under `processed/lowrated/`, and the refactored Phase 1 parser writes byte-identical tables for Tiers A–C. Pinned by `test_select_and_parse_lowrated` (Task 7) and the real-data parse comparison in Task 7 Step 1/Step 6.
5. **Bomb tunnels, doubles and crossovers.** Expected: legal bomb tunnels never fail the gate; the baseline never puts both hands in one cell. Pinned by `test_overlap_h1_uses_depth_for_notes_and_20_ms_for_bomb_pairs` (Task 1) and `test_placement_alternates_without_any_rule_issue` (Task 5).

## File Structure

| File | Responsibility |
|------|----------------|
| `src/kata/validate/__init__.py` | package docstring |
| `src/kata/validate/swings.py` | angles, wrist rotation, swing formation, Viterbi orientation, reset labels (spec §3) |
| `src/kata/validate/rules.py` | `MapInput`, `analyse()`: swing table, issues H1–H3/E1–E3, metrics L1–L3, local windows (§4, §5.3) |
| `src/kata/validate/reference.py` | config pinning, strata, Tier A anchors, baseline targets (§4.2–4.3, §9) |
| `src/kata/validate/score.py` | sub-scores, geometric mean, gate, `validate()`, vectorised `score_frame` (§5) |
| `src/kata/validate/corrupt.py` | C1–C7 injections and detection (§6.3) |
| `src/kata/validate/pipeline.py` | stages `analyse`, `reference`, `score`, `corrupt`, `report` |
| `src/kata/validate/report.py` | Phase 3 Markdown report, AUC helpers |
| `src/kata/baseline/__init__.py` | package docstring |
| `src/kata/baseline/grid.py` | beat fit, piecewise grid, lead-in, `bpmEvents`, slots (§8.1) |
| `src/kata/baseline/onsets.py` | onset strength, loudness boost, slot strengths, greedy selection (§8.2) |
| `src/kata/baseline/patterns.py` | hands, directions, positions, doubles (§8.3) |
| `src/kata/baseline/generate.py` | one song → five validated difficulties with repair (§8) |
| `src/kata/baseline/export.py` | Info.dat, v3 difficulty files, cover, zip, round trip (§8.4–8.5) |
| `src/kata/baseline/pipeline.py` | `generate` stage and `baseline_eval` (§8.6) |
| `src/kata/data/lowrated.py` | Tier D select / download / parse (§6.1) |
| `src/kata/config.py` (modify) | `Paths.lowrated`, `Paths.validator`, `Paths.generated` |
| `src/kata/data/beatsaver.py` (modify) | `extract_map(..., drop_audio)`, `download_map(..., drop_audio)` |
| `src/kata/data/pipeline.py` (modify) | `_filter_catalog`, `_parse_selection`, audio flag in `_parse_one_map`, `lowrated_*` views |
| `src/kata/cli.py` (modify) | `kata data *-lowrated`, `kata validate …`, `kata baseline generate` |
| `tests/conftest.py` | synthetic note/bomb frames, a clean stream, a hand-built reference |
| `tests/test_validate_*.py`, `tests/test_baseline_*.py`, `tests/test_lowrated.py` | one test module per source module |

Task order follows the dependencies: 1 → 2 → 3 (validator core) → 4 → 5 → 6 (baseline core, needs 1–2) → 7 (Tier D, Phase 1 changes) → 8 (validator stages, needs 7) → 9 (baseline stage + CLI) → 10 (pilot, checkpoint) → 11 (full run).

---

### Task 1: Swing model and rules

**Files:**
- Create: `src/kata/validate/__init__.py`, `src/kata/validate/swings.py`, `src/kata/validate/rules.py`
- Test: `tests/conftest.py`, `tests/test_validate_swings.py`, `tests/test_validate_rules.py`

**Interfaces:**
- Consumes: nothing from Phase 3. Note frames use the Phase 1 column names (`beat, time_s, x, y, color, cut_dir, angle_offset`).
- Produces (swings): `RED, BLUE, FOREHAND, BACKHAND, DOT, CUT_ANGLE, T_RESET_S, ROT_RANGE`; `SwingParams` (frozen dataclass); `Swing`; `angle_diff(a, b) -> float`; `note_angle(cut_dir, angle_offset) -> float | None`; `rotation(theta, parity, hand) -> float | None`; `build_swings(t, beat, x, y, color, cut_dir, angle_offset, chain_head=None, params=SwingParams()) -> dict[int, list[Swing]]`; `orient(swings, bomb_t, bomb_x, bomb_y, params) -> None`.
- Produces (rules): `HARD_RULES, ERROR_RULES, LOAD_RULES, SCORED_RULES, LABELS, VISION_WINDOW, DEFAULT_NJS, OVERLAP_M, BOMB_BOMB_S, SPEED_RUN, STRAIN_TAIL, LOCAL_BEATS, LOCAL_STEP, LOCAL_MIN_SWINGS, SWING_SCHEMA, ISSUE_SCHEMA`; `MapInput(difficulty, njs, njs_offset, bpm, notes, bombs=None, walls=None, chains=None)`; `Analysis(swings, issues, metrics, local)`; `analyse(m: MapInput, params: SwingParams | None = None) -> Analysis`; `reaction_time_s(njs, njs_offset, bpm) -> float`; `nps(times) -> float`. `metrics` keys: `n_swings, nps, njs, reaction_time_s, dt_p5, H1, H2, H3, E1, E2, E3, L1, L2, L3`.
- Produces (tests): fixtures `notes_df(rows)`, `bombs_df(rows)`, `stream(n_per_hand=64, step=0.09, t0=2.0, stack_every=8)`, `toy_ref` (used by Tasks 2–9).

- [ ] **Step 1: Write the failing tests** — `tests/conftest.py`:

```python
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
```

`tests/test_validate_swings.py`:

```python
import numpy as np
import pytest

from conftest import bombs_df, notes_df
from kata.validate.rules import MapInput, analyse
from kata.validate.swings import (BACKHAND, BLUE, FOREHAND, RED, SwingParams, angle_diff, build_swings,
                                  note_angle, rotation)


def _swings(rows, params=SwingParams()):
    n = notes_df(rows).sort("time_s")
    return build_swings(*(n[c].to_numpy() for c in ("time_s", "beat", "x", "y", "color", "cut_dir", "angle_offset")),
                        params=params)


def test_note_angles_follow_the_players_view():
    assert note_angle(1, 0) == 0.0          # down
    assert note_angle(0, 0) == 180.0 or note_angle(0, 0) == -180.0
    assert note_angle(3, 0) == 90.0         # right
    assert note_angle(6, 0) == -45.0        # down-left
    assert note_angle(1, 45) == 45.0        # angle offset turns counter-clockwise from down towards right
    assert note_angle(8, 0) is None
    assert angle_diff(170, -170) == 20


def test_rotation_is_relative_to_each_paritys_neutral_and_respects_wrist_limits():
    assert rotation(0, FOREHAND, BLUE) == 0
    assert rotation(180, BACKHAND, BLUE) == 0
    assert rotation(-90, BACKHAND, BLUE) == 90
    assert rotation(160, FOREHAND, BLUE) is None      # blue range is [-195, 155]
    assert rotation(-160, FOREHAND, RED) is None      # red mirrors it
    assert rotation(-180, FOREHAND, BLUE) == -180


def test_swing_kinds():
    sw = _swings([(1.0, 1, 0, 1, 1), (1.0, 2, 0, 1, 1),          # stack
                  (2.0, 0, 0, 1, 1), (2.0, 3, 0, 1, 1),          # window
                  (3.0, 3, 2, 1, 1), (3.02, 3, 1, 1, 1), (3.04, 3, 0, 1, 1),   # slider
                  (4.0, 3, 0, 1, 1)])[BLUE]
    assert [s.kind for s in sw] == ["stack", "window", "slider", "single"]
    assert sw[2].n_notes == 3


def test_merge_window_and_fast_stream_split():
    sw = _swings([(1.0, 3, 0, 1, 1), (1.1, 3, 2, 1, 0)])[BLUE]      # 100 ms apart -> two swings
    assert len(sw) == 2
    sw = _swings([(1.0, 3, 0, 1, 1), (1.065, 3, 2, 1, 0)])[BLUE]    # 65 ms but the arrow turns 180°
    assert len(sw) == 2
    sw = _swings([(1.0, 3, 0, 1, 1), (1.0, 2, 0, 1, 3)])[BLUE]      # same time, 90° apart: one swing
    assert len(sw) == 1 and sw[0].spread == 90


def _map(rows, bombs=(), difficulty="Expert"):
    return MapInput(difficulty, 16.0, 0.0, 120.0, notes_df(rows), bombs_df(list(bombs)) if bombs else None)


def test_alternating_swings_get_alternating_parity_and_no_resets():
    a = analyse(_map([(1 + 0.5 * i, 3, 0 if i % 2 == 0 else 2, 1, 1 if i % 2 == 0 else 0) for i in range(8)]))
    assert a.swings["parity"].to_list() == [FOREHAND, BACKHAND] * 4
    assert a.swings["reset"].null_count() == 8
    assert a.issues.height == 0


def test_fast_double_down_is_a_parity_break_but_a_slow_one_is_a_reset():
    fast = analyse(_map([(1.0, 3, 0, 1, 1), (1.2, 3, 0, 1, 1)]))        # Expert reset time is 0.30 s
    assert "E1" in fast.issues["rule"].to_list()
    slow = analyse(_map([(1.0, 3, 0, 1, 1), (2.0, 3, 0, 1, 1)]))
    assert slow.swings["reset"].to_list() == [None, "slow"]
    assert slow.issues.height == 0


def test_bottom_bomb_in_the_hands_column_justifies_a_reset():
    a = analyse(_map([(1.0, 3, 0, 1, 1), (1.25, 3, 0, 1, 1)], bombs=[(1.12, 2, 0)]))
    assert a.swings["reset"].to_list() == [None, "bomb"]
    assert a.swings["forced"].to_list()[1] == FOREHAND
    assert a.issues.height == 0
    other_column = analyse(_map([(1.0, 3, 0, 1, 1), (1.25, 3, 0, 1, 1)], bombs=[(1.12, 0, 0)]))
    assert "E1" in other_column.issues["rule"].to_list()


def test_dots_take_the_direction_that_continues_the_flow():
    a = analyse(_map([(1.0, 3, 0, 1, 1), (1.4, 3, 2, 1, 8), (1.8, 3, 0, 1, 8)]))
    assert a.swings["parity"].to_list() == [FOREHAND, BACKHAND, FOREHAND]
    assert a.swings["rotation"].abs().max() == 0
    assert a.issues.height == 0


@pytest.mark.parametrize("arrow", [0, 2, 3, 4, 5, 6, 7])
def test_a_flipped_arrow_in_a_fast_stream_is_an_e1_issue(arrow):
    flip = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5}
    rows = [(1.0 + 0.2 * i, 3, 0, 1, arrow if i % 2 else flip[arrow]) for i in range(6)]
    rows[3] = (rows[3][0], 3, 0, 1, flip[rows[3][4]])
    a = analyse(_map(rows, difficulty="ExpertPlus"))
    assert "E1" in a.issues["rule"].to_list()
    assert np.isfinite(a.metrics["L2"])
```

`tests/test_validate_rules.py`:

```python
import numpy as np
import polars as pl
import pytest

from conftest import bombs_df, notes_df, stream
from kata.validate.rules import MapInput, _peak_run, analyse, nps, reaction_time_s


def _map(rows, bombs=(), walls=None, chains=None, difficulty="Expert", njs=16.0):
    return MapInput(difficulty, njs, 0.0, 120.0, notes_df(rows), bombs_df(list(bombs)) if bombs else None,
                    walls, chains)


def _rules(a) -> list[str]:
    return a.issues["rule"].to_list()


def test_reaction_time_follows_the_games_half_jump_rule():
    assert reaction_time_s(18, 0, 120) == pytest.approx(0.5)      # hjd 4 -> 2 -> 1 beat
    assert reaction_time_s(18, 0.5, 120) == pytest.approx(0.75)
    assert reaction_time_s(10, -5, 120) == pytest.approx(0.125)   # floor of 0.25 beat


def test_overlap_h1_uses_depth_for_notes_and_20_ms_for_bomb_pairs():
    assert _rules(analyse(_map([(1.0, 0, 0, 0, 1)], bombs=[(1.01, 0, 0)]))) == ["H1"]   # 0.16 m
    assert _rules(analyse(_map([(1.0, 0, 0, 0, 1), (1.05, 0, 0, 1, 1)]))) == []         # 0.8 m
    assert _rules(analyse(_map([(3.0, 3, 0, 1, 1)], bombs=[(1.0, 0, 0), (1.03, 0, 0)]))) == []   # bomb tunnel
    assert _rules(analyse(_map([(3.0, 3, 0, 1, 1)], bombs=[(1.0, 0, 0), (1.01, 0, 0)]))) == ["H1"]


def test_njs_zero_falls_back_to_the_game_default():
    a = analyse(_map([(1.0, 0, 0, 0, 1)], bombs=[(1.04, 0, 0)], njs=0.0))   # Expert default 12 -> 0.48 m
    assert _rules(a) == ["H1"]


def test_note_inside_a_wall_is_h2():
    walls = pl.DataFrame({"time_s": [1.0], "end_time_s": [2.0], "x": [1], "y": [0], "w": [2], "h": [5]})
    assert _rules(analyse(_map([(1.5, 2, 0, 1, 1)], walls=walls))) == ["H2"]
    assert _rules(analyse(_map([(1.5, 3, 0, 1, 1)], walls=walls))) == []
    assert _rules(analyse(_map([(2.5, 2, 0, 1, 1)], walls=walls))) == []


def test_swing_spread_h3_and_e3():
    assert _rules(analyse(_map([(1.0, 2, 0, 1, 1), (1.0, 3, 0, 1, 3)]))) == ["H3"]           # 90°
    assert _rules(analyse(_map([(1.0, 2, 0, 1, 1), (1.0, 3, 0, 1, 7)]))) == []               # 45°
    assert _rules(analyse(_map([(1.0, 2, 0, 1, 1), (1.0, 3, 0, 1, 1, 60)]))) == ["E3"]       # 60°


def test_vision_block_e2_same_half_window_and_bomb_minimum():
    blocker = (1.0, 1, 1, 0, 1)
    assert _rules(analyse(_map([blocker, (1.2, 1, 0, 1, 1)]))) == ["E2"]
    assert _rules(analyse(_map([blocker, (1.2, 3, 0, 1, 1)]))) == []        # other half
    assert _rules(analyse(_map([blocker, (1.05, 0, 0, 1, 1)]))) == []       # under Expert's 0.1 s
    assert _rules(analyse(_map([blocker, (1.7, 1, 0, 1, 1)]))) == []        # past the 0.625 s cap
    assert _rules(analyse(_map([(1.15, 2, 0, 1, 1)], bombs=[(1.0, 2, 1)]))) == []   # bomb: 0.2 s minimum
    assert _rules(analyse(_map([(1.25, 2, 0, 1, 1)], bombs=[(1.0, 2, 1)]))) == ["E2"]


def test_chain_heads_count_once():
    chain = pl.DataFrame({"color": [1], "beat": [2.0], "time_s": [1.0], "x": [3], "y": [0], "cut_dir": [1]})
    with_note = analyse(_map([(1.0, 3, 0, 1, 1)], chains=chain))
    alone = analyse(_map([(3.0, 0, 0, 0, 1)], chains=chain))
    assert with_note.swings.height == 1 and with_note.swings["kind"][0] == "chain"
    assert alone.swings.height == 2


def test_load_metrics():
    assert _peak_run(np.array([np.nan, 1, 1, 1, 1, 1, 1, 1, 9, 1])) == pytest.approx(16 / 8)
    assert _peak_run(np.array([np.nan, 2.0, 4.0])) == 3.0
    assert nps([1.0, 2.0, 3.0]) == 1.5
    assert nps([5.0]) == 1.0


def test_clean_stream_metrics_and_local_windows():
    a = analyse(MapInput("ExpertPlus", 18.0, 0.0, 120.0, stream()))
    assert a.issues.height == 0
    assert a.metrics["n_swings"] == 128
    # each hand swings every 0.18 s between the bottom and top rows (2 cells; a stack adds at most one)
    assert 3 / 0.18 <= a.metrics["L1"] <= (1 + 5 ** 0.5) / 0.18
    assert a.metrics["L2"] == 0.0 and a.metrics["L3"] == 0.0
    assert a.local.height > 0 and (a.local["n_swings"] >= 8).all()


def test_one_handed_and_empty_maps_do_not_crash():
    one = analyse(_map([(1.0 + 0.5 * i, 3, 0 if i % 2 == 0 else 2, 1, 1 if i % 2 == 0 else 0) for i in range(10)]))
    assert one.metrics["n_swings"] == 10 and np.isfinite(one.metrics["L1"])
    empty = analyse(MapInput("Easy", 10.0, 0.0, 120.0, notes_df([])))
    assert empty.metrics["n_swings"] == 0 and empty.issues.height == 0 and empty.local.height == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_validate_swings.py tests/test_validate_rules.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'kata.validate'`.

- [ ] **Step 3: Implement** — `src/kata/validate/__init__.py`:

```python
"""Phase 3 playability validator (docs/phases/phase-3-validator.md)."""
```

`src/kata/validate/swings.py`:

```python
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
```

`src/kata/validate/rules.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_validate_swings.py tests/test_validate_rules.py -q`
Expected: `25 passed`.

- [ ] **Step 5: Real-data smoke check (by hand, needs `KATA_DATA`)**

```bash
uv run python -c "
import polars as pl, time
from kata.validate.rules import MapInput, analyse
P = r'E:\kata-data\processed'
d = pl.read_parquet(P + '/difficulties.parquet').filter((pl.col('tier') == 'A') & (pl.col('parse_status') == 'ok')).head(20)
m = pl.read_parquet(P + '/maps.parquet').select('hash', 'info_bpm')
n = pl.read_parquet(P + '/notes.parquet')
for r in d.join(m, on='hash').iter_rows(named=True):
    k = (pl.col('hash') == r['hash']) & (pl.col('difficulty') == r['difficulty'])
    t = time.perf_counter(); a = analyse(MapInput(r['difficulty'], r['njs'] or 0, r['njs_offset'] or 0, r['info_bpm'], n.filter(k)))
    print(r['difficulty'], int(a.metrics['n_swings']), {h: a.metrics[h] for h in ('H1', 'H2', 'H3')}, f'{(time.perf_counter() - t) * 1000:.0f} ms')
"
```
Expected: no hard-rule hits on almost all of these ranked difficulties; tens of milliseconds each.

- [ ] **Step 6: Commit**

```bash
git add src/kata/validate/__init__.py src/kata/validate/swings.py src/kata/validate/rules.py tests/conftest.py tests/test_validate_swings.py tests/test_validate_rules.py
git commit -m "Phase 3: swing model and playability rules"
```

---

### Task 2: Reference and scoring

**Files:**
- Create: `src/kata/validate/reference.py`, `src/kata/validate/score.py`
- Test: `tests/test_validate_reference.py`, `tests/test_validate_score.py`

**Interfaces:**
- Consumes: Task 1 (`rules`, `swings`).
- Produces (reference): `VALIDATOR_VERSION = "v1"`, `NPS_BAND`, `MIN_STRATUM_MAPS`, `WEIGHTS`; `validator_config() -> dict`; `ValidatorDir(root)` with `.dir`, `.config_path`, `.reference_path`, `.open_for_write(config=None)`, `.write_reference(ref)`, `.load_reference() -> Reference`; `band(nps) -> int`; `merge_bands(counts, min_maps) -> list[[lo, hi, n]]`; `stratum_key(label, cells, nps) -> str`; `Reference(data)` with `.strata`, `.stratum(label, nps)`, `.anchors(rule, stratum) -> {"q90","q99","zero"}`, `.baseline` (`nps`, `njs`, `reaction_time_s` per label; `min_gap_s` per stratum), `.data`, and `Reference.build(metrics: pl.DataFrame) -> Reference` (rows: one per Tier A train difficulty with `difficulty, nps, njs, reaction_time_s, dt_p5` and every scored rule).
- Produces (score): `SCORE_FLOOR`; `sub_score(m, anchors) -> float`; `sub_scores_np(m, q90, q99, zero) -> np.ndarray`; `score_frame(df, ref, weights=WEIGHTS) -> pl.DataFrame` (adds `s_<rule>` and `score`; needs a `stratum` column); `combine(subs, weights) -> float`; `score_metrics(metrics, stratum, ref, weights) -> (score, subs)`; `Result(passed, score, sub_scores, metrics, stratum, swings, issues, local)`; `score_local(local, stratum, ref, weights)`; `score_analysis(a, difficulty, ref, weights) -> Result`; `validate(m: MapInput, ref: Reference, weights=WEIGHTS) -> Result`.

- [ ] **Step 1: Write the failing tests** — `tests/test_validate_reference.py`:

```python
import json

import polars as pl
import pytest

from kata.validate.reference import (Reference, ValidatorDir, _anchors, merge_bands, stratum_key,
                                     validator_config)
from kata.validate.rules import ERROR_RULES, LOAD_RULES


def test_merge_bands_merges_the_smallest_cell_into_its_smaller_neighbour():
    assert merge_bands({0: 50, 1: 200, 2: 30, 3: 150}) == [[0, 1, 250], [2, 3, 180]]
    assert merge_bands({1: 40, 3: 30}) == [[1, 3, 70]]           # gaps count as empty bands
    assert merge_bands({2: 500}) == [[2, 2, 500]]


def test_stratum_key_clamps_to_the_outer_cells():
    cells = [[0, 1, 250], [2, 3, 180]]
    assert stratum_key("Hard", cells, 3.9) == "Hard|0-4"
    assert stratum_key("Hard", cells, 4.0) == "Hard|4-8"
    assert stratum_key("Hard", cells, 30.0) == "Hard|4-8"
    assert stratum_key("Hard", [[0, 5, 80]], 3.0) == "Hard|all"


def test_error_anchors_keep_a_minimum_spacing():
    assert _anchors([0.0] * 50, error=True) == {"q90": 0.2, "q99": 1.2, "zero": 3.2}
    a = _anchors(list(range(101)), error=False)
    assert a["q90"] == pytest.approx(90) and a["q99"] == pytest.approx(99) and a["zero"] == pytest.approx(117)


def _metrics(n=400):
    rows = []
    for i in range(n):
        label = "Expert" if i % 2 else "Hard"
        rows.append({"difficulty": label, "nps": 3.0 + (i % 8), "njs": 16.0, "reaction_time_s": 0.6,
                     "dt_p5": 0.15, **{r: float(i % 10) / 10 for r in ERROR_RULES + LOAD_RULES}})
    return pl.DataFrame(rows)


def test_reference_build_has_strata_anchors_and_baseline_targets():
    ref = Reference.build(_metrics())
    assert set(ref.strata) == {"Easy", "Normal", "Hard", "Expert", "ExpertPlus"}
    assert all(sum(c[2] for c in cells) >= 100 or len(cells) == 1 for cells in ref.strata.values())
    s = ref.stratum("Expert", 6.0)
    assert set(ref.anchors("L1", s)) == {"q90", "q99", "zero"}
    assert ref.anchors("E1", "anything") == ref.data["rules"]["E1"]["pooled"]
    assert ref.baseline["njs"]["Expert"] == 16.0 and ref.baseline["min_gap_s"][s] == 0.15
    assert Reference(json.loads(json.dumps(ref.data))).anchors("L1", s) == ref.anchors("L1", s)


def test_validator_dir_pins_its_config(tmp_path):
    vd = ValidatorDir(tmp_path)
    vd.open_for_write()
    vd.open_for_write()                                   # same config: fine
    changed = {**validator_config(), "overlap_m": 0.4}
    with pytest.raises(ValueError, match="bump VALIDATOR_VERSION"):
        vd.open_for_write(changed)
```

`tests/test_validate_score.py`:

```python
import pytest

from conftest import bombs_df, stream
from kata.validate.rules import MapInput
from kata.validate.score import combine, sub_score, validate

A = {"q90": 1.0, "q99": 2.0, "zero": 4.0}


@pytest.mark.parametrize("m, s", [(0.0, 1.0), (1.0, 1.0), (1.5, 0.75), (2.0, 0.5), (3.0, 0.25), (4.0, 0.0), (9.0, 0.0)])
def test_sub_score_is_piecewise_linear_between_the_anchors(m, s):
    assert sub_score(m, A) == pytest.approx(s)


def test_combine_is_a_floored_geometric_mean():
    ones = {r: 1.0 for r in ("E1", "E2", "E3", "L1", "L2", "L3")}
    assert combine(ones) == pytest.approx(100.0)
    assert combine({**ones, "E2": 0.0}) == pytest.approx(100 * 0.01 ** (1 / 6))
    assert combine({**ones, "L1": 0.5}) == pytest.approx(100 * 0.5 ** (1 / 6))


def test_validate_gates_hard_errors_but_still_scores(toy_ref):
    clean = validate(MapInput("ExpertPlus", 18.0, 0.0, 120.0, stream()), toy_ref)
    assert clean.passed and clean.score == pytest.approx(100.0) and clean.stratum == "ExpertPlus|all"
    assert {"s_E1", "score"} <= set(clean.local.columns)
    n = stream()
    t, x, y = n["time_s"][10], n["x"][10], n["y"][10]
    bad = validate(MapInput("ExpertPlus", 18.0, 0.0, 120.0, n, bombs_df([(t + 0.005, x, y)])), toy_ref)
    assert not bad.passed and bad.metrics["H1"] == 1 and bad.score == pytest.approx(100.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_validate_reference.py tests/test_validate_score.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.validate.reference'` (and `.score`).

- [ ] **Step 3: Implement** — `src/kata/validate/reference.py`:

```python
"""Strata, Tier A reference quantiles and the versioned validator directory (spec §4.2-4.3, §9)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from kata.validate import rules
from kata.validate.rules import ERROR_RULES, LABELS, LOAD_RULES
from kata.validate.swings import ROT_RANGE, T_RESET_S, SwingParams

VALIDATOR_VERSION = "v1"
NPS_BAND = 2.0
MIN_STRATUM_MAPS = 100
ERROR_Q90_MIN, ERROR_Q99_GAP = 0.2, 1.0   # §5.1: minimum anchor spacing for error rules
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
            "error_anchor_min": [ERROR_Q90_MIN, ERROR_Q99_GAP], "weights": WEIGHTS}


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
        text = json.dumps(config, sort_keys=True, indent=1)
        if self.config_path.exists():
            if json.loads(self.config_path.read_text()) != json.loads(text):
                raise ValueError(f"validator config differs from {self.config_path}; bump VALIDATOR_VERSION")
            return
        self.config_path.write_text(text)

    def write_reference(self, ref: "Reference") -> None:
        self.reference_path.write_text(json.dumps(ref.data, indent=1, sort_keys=True))

    def load_reference(self) -> "Reference":
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
        q99 = max(q99, q90 * 1.05 + 1e-9)
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
```

`src/kata/validate/score.py`:

```python
"""Sub-scores, the combined score, the gate and validate() (spec §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from kata.validate.reference import WEIGHTS, Reference
from kata.validate.rules import HARD_RULES, SCORED_RULES, Analysis, MapInput, analyse

SCORE_FLOOR = 0.01


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_validate_reference.py tests/test_validate_score.py -q`
Expected: `14 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/validate/reference.py src/kata/validate/score.py tests/test_validate_reference.py tests/test_validate_score.py
git commit -m "Phase 3: strata, Tier A reference anchors and scoring"
```

---

### Task 3: Synthetic corruptions

**Files:**
- Create: `src/kata/validate/corrupt.py`
- Test: `tests/test_validate_corrupt.py`

**Interfaces:**
- Consumes: Task 1 (`analyse`, `MapInput`, `LOCAL_BEATS`, `T_RESET_S`), Task 2 (`Result`).
- Produces: `CORRUPTIONS = {"C1": "E1", "C2": "E2", "C3": "H1", "C4": "L1", "C5": "L2", "C6": "H3", "C7": "E3"}`; `C4_TARGET = 1.5`; `Injection(corruption, rule, hand, time_s, beat, end_s=None)`; `corrupt(kind, m, clean: Analysis, rng, l1_q99=None) -> (Injection, MapInput) | None` (`clean` is `analyse(m)` with `chains=None`); `evaluate(inj, clean: Result, bad: Result) -> (detected: bool, false_alarms: int)`.

- [ ] **Step 1: Write the failing test** — `tests/test_validate_corrupt.py`:

```python
import numpy as np
import pytest

from conftest import stream
from kata.validate.corrupt import CORRUPTIONS, corrupt, evaluate
from kata.validate.rules import MapInput, analyse
from kata.validate.score import score_analysis


@pytest.mark.parametrize("kind", sorted(CORRUPTIONS))
def test_every_corruption_is_detected_on_a_clean_stream_without_false_alarms(kind, toy_ref):
    m = MapInput("ExpertPlus", 18.0, 0.0, 120.0, stream())
    a = analyse(m)
    clean = score_analysis(a, m.difficulty, toy_ref)
    assert clean.passed and clean.issues.height == 0
    rng = np.random.default_rng(7)
    for _ in range(5):
        inj, bad = corrupt(kind, m, a, rng, toy_ref.anchors("L1", clean.stratum)["q99"])
        assert inj.rule == CORRUPTIONS[kind]
        hit, false_alarms = evaluate(inj, clean, score_analysis(analyse(bad), bad.difficulty, toy_ref))
        assert hit, inj
        assert false_alarms == 0, inj


def test_no_eligible_place_returns_none(toy_ref):
    m = MapInput("ExpertPlus", 18.0, 0.0, 120.0, stream(n_per_hand=4, stack_every=99))
    assert corrupt("C6", m, analyse(m), np.random.default_rng(0)) is None
    assert corrupt("C4", m, analyse(m), np.random.default_rng(0), l1_q99=None) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_validate_corrupt.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.validate.corrupt'`.

- [ ] **Step 3: Implement** — `src/kata/validate/corrupt.py`:

```python
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
        starts = [k for k in range(h.height - 3) if good[k:k + 4].all()]
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


def evaluate(inj: Injection, clean: Result, bad: Result) -> tuple[bool, int]:
    """(detected, false alarms). Hard/error rules: an issue of the rule within ±1 swing.
    Load rules: that rule's local sub-score < 0.5 in a window containing the injection. C5 (L2)
    also counts L3 there or an E1 issue within ±1 swing: the DP reads a fast zig-zag as rotation
    strain or a parity break, and a slower one as resets (§6.3)."""
    lo, hi = _window(inj, bad)
    if inj.end_s is not None:
        hi = max(hi, inj.end_s + 1e-6)
    near = (pl.col("time_s") >= lo) & (pl.col("time_s") <= hi)
    if inj.rule.startswith("L"):
        # C5's zig-zag reads as rotation strain, as resets, or as parity breaks, depending on tempo.
        rules_ = ("L2", "L3") if inj.rule == "L2" else (inj.rule,)
        w = bad.local.filter((pl.col("window_start_beat") <= inj.beat)
                             & (pl.col("window_start_beat") + LOCAL_BEATS > inj.beat)) if bad.local.height else bad.local
        hit = any(w.height and w.filter(pl.col(f"s_{r}") < 0.5).height for r in rules_) or (
            inj.rule == "L2" and bad.issues.filter(near & (pl.col("rule") == "E1")).height > 0)
    else:
        hit = bad.issues.filter(near & (pl.col("rule") == inj.rule)).height > 0
    key = pl.col("rule") + "@" + pl.col("time_s").round(3).cast(pl.String)
    old = set(clean.issues.select(key).to_series().to_list())
    new = bad.issues.filter(~near).with_columns(k=key).filter(~pl.col("k").is_in(list(old)))
    return hit, new.height
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_validate_corrupt.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/validate/corrupt.py tests/test_validate_corrupt.py
git commit -m "Phase 3: synthetic corruptions with ground truth"
```

---

### Task 4: Baseline tempo grid and onset selection

**Files:**
- Create: `src/kata/baseline/__init__.py`, `src/kata/baseline/grid.py`, `src/kata/baseline/onsets.py`
- Test: `tests/test_baseline_grid.py`, `tests/test_baseline_onsets.py`

**Interfaces:**
- Consumes: `kata.data.mapformat.tempo.TempoMap` (Phase 1), `kata.audio.frames.FPS`, `kata.audio.store.CURVE_COLUMNS` (Phase 2; no torch import).
- Produces (grid): `STEADY_P95_S = 0.030`, `SLOW_BPM = 110.0`; `Grid(times, bpm, piecewise)` with `.to_beats(t)`, `.bpm_events() -> [(beat, bpm)]`, `.tempo_map() -> TempoMap`, `.slots() -> (times, beats)`; `fit_line(beats) -> (t0, period, p95)`; `fit_grid(beats, duration_s, segments=None) -> Grid`.
- Produces (onsets): `ONSET_WEIGHTS`; `onset_strength(curves) -> np.ndarray`; `loudness_boost(curves, segments) -> np.ndarray`; `slot_strengths(frame_strength, slot_times) -> np.ndarray`; `mapped_span(slot_times, strengths, duration_s) -> np.ndarray[bool]`; `select(slot_times, strengths, n_target, min_gap_s) -> np.ndarray[int]` (time-ordered indices).

- [ ] **Step 1: Write the failing tests** — `tests/test_baseline_grid.py`:

```python
import numpy as np
import pytest

from kata.baseline.grid import Grid, fit_grid, fit_line


def _beats(t0, period, n, jitter=0.0, drop=(), seed=0):
    b = t0 + np.arange(n) * period + np.random.default_rng(seed).uniform(-jitter, jitter, n)
    return np.delete(b, list(drop))


def test_fit_line_survives_missed_beats_and_jitter():
    t0, period, p95 = fit_line(_beats(0.23, 0.5, 400, jitter=0.01, drop=(5, 100, 101)))
    assert period == pytest.approx(0.5, abs=1e-4)
    assert t0 == pytest.approx(0.23, abs=0.003)
    assert p95 < 0.011


def test_steady_grid_puts_map_beat_one_on_the_first_audio_beat_after_half_a_period():
    g = fit_grid(_beats(0.23, 0.5, 200), duration_s=100.0)
    assert not g.piecewise and g.bpm == pytest.approx(120)
    assert g.times[0] == 0 and g.times[1] == pytest.approx(0.73)       # 0.23 < 0.25 is skipped
    assert g.bpm_events()[0] == (0.0, pytest.approx(60 / 0.73))
    assert g.bpm_events()[1] == (1.0, pytest.approx(120))
    k = np.arange(1, 50)
    np.testing.assert_allclose(g.to_beats(0.73 + 0.5 * (k - 1)), k, atol=1e-9)
    np.testing.assert_allclose(g.tempo_map().to_seconds(k), 0.73 + 0.5 * (k - 1), atol=1e-9)
    assert g.to_beats(200.0) == pytest.approx(1 + (200 - 0.73) / 0.5)   # extrapolates past the grid


def test_unsteady_song_is_fitted_per_segment():
    a = _beats(0.5, 0.5, 60)                            # 120 BPM until 30 s
    b = a[-1] + 0.6 + np.arange(80) * (60 / 140)        # then 140 BPM
    g = fit_grid(np.concatenate([a, b]), duration_s=b[-1] + 1, segments=[(0, 30.25), (30.25, b[-1] + 1)])
    assert g.piecewise
    k = g.to_beats(np.concatenate([a[5:-5], b[5:-5]]))
    np.testing.assert_allclose(k, np.round(k), atol=0.02)
    assert len(g.bpm_events()) >= 3


@pytest.mark.parametrize("bpm, per_beat", [(120, 4), (100, 8)])
def test_slots_are_quarter_beats_or_eighths_for_slow_songs(bpm, per_beat):
    period = 60 / bpm
    g = Grid(np.concatenate([[0.0], period * np.arange(1, 6)]), bpm, False)
    t, b = g.slots()
    assert b[0] == 1 and b[1] == pytest.approx(1 + 1 / per_beat) and b[-1] == 5
    np.testing.assert_allclose(np.diff(t), period / per_beat)
```

`tests/test_baseline_onsets.py`:

```python
import numpy as np

from kata.audio.store import CURVE_COLUMNS
from kata.baseline.onsets import loudness_boost, mapped_span, onset_strength, select, slot_strengths


def _curves(T=3000):
    c = np.zeros((T, 12), np.float32)
    c[:, CURVE_COLUMNS.index("rms_mix")] = -20
    return c


def test_onset_strength_weights_normalised_curves():
    c = _curves(1000)
    c[::50, CURVE_COLUMNS.index("onset_drums")] = 4.0     # 2 % of frames, so p99 is the peak
    c[25::50, CURVE_COLUMNS.index("onset_mix")] = 9.0
    s = onset_strength(c)
    assert s[0] == 1.0 and s[25] == 0.5


def test_loudness_boost_is_clipped_per_segment():
    c = _curves(1000)
    c[500:, CURVE_COLUMNS.index("rms_mix")] = -5
    b = loudness_boost(c, [(0, 5), (5, 10)])
    assert b[100] < 1 < b[700] and b.min() >= 0.5 and b.max() <= 1.5


def test_slot_strength_max_pools_30_ms():
    s = np.zeros(200)
    s[103] = 1.0
    np.testing.assert_array_equal(slot_strengths(s, np.array([1.0, 1.1])), [1.0, 0.0])


def test_select_respects_gap_and_count_and_prefers_strong_slots():
    t = np.arange(0, 10, 0.125)
    s = np.ones_like(t)
    s[::4] = 5
    idx = select(t, s, n_target=30, min_gap_s=0.25)
    assert np.all(np.diff(t[idx]) >= 0.25 - 1e-9) and len(idx) <= 30
    assert set(np.flatnonzero(s == 5)) <= set(idx.tolist())
    assert np.all(np.diff(idx) > 0)


def test_mapped_span_keeps_lead_in_and_out():
    t = np.arange(0, 60, 0.25)
    m = mapped_span(t, np.ones_like(t), 60.0)
    assert t[m].min() >= 1.5 and t[m].max() <= 58.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_baseline_grid.py tests/test_baseline_onsets.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.baseline'`.

- [ ] **Step 3: Implement** — `src/kata/baseline/__init__.py`:

```python
"""Phase 3 rule-based baseline mapper (docs/phases/phase-3-validator.md §8)."""
```

`src/kata/baseline/grid.py`:

```python
"""Tempo grid from audio beats (spec §8.1): map beat k+1 is fitted audio beat k.

The grid is a list of times, one per map beat, starting with map beat 0 at t = 0. Beat 0 -> 1 is a
lead-in whose length puts map beat 1 on the first fitted audio beat at least half a period in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kata.data.mapformat.tempo import TempoMap

STEADY_P95_S = 0.030   # residual p95 above this -> fit per structure segment (measured: Beat This
                       # beats on 20 ms frames give 15-30 ms on steady songs)
SLOW_BPM = 110.0       # below this the slots are 1/8 of a fitted beat, so slow songs reach Expert+ density
MIN_SEGMENT_BEATS = 4


@dataclass
class Grid:
    times: np.ndarray        # time of map beat k, k = 0 .. len-1; times[0] = 0
    bpm: float               # Info BPM: the main fitted tempo
    piecewise: bool

    def to_beats(self, t) -> np.ndarray:
        """Map beat of time(s) t; linear beyond the last grid time at the last tempo."""
        t = np.asarray(t, dtype=np.float64)
        k = np.arange(len(self.times), dtype=np.float64)
        last = (t - self.times[-1]) / (self.times[-1] - self.times[-2]) + k[-1]
        return np.where(t > self.times[-1], last, np.interp(t, self.times, k))

    def bpm_events(self) -> list[tuple[float, float]]:
        """(beat, bpm) at every tempo change, including the lead-in at beat 0."""
        bpm = 60.0 / np.diff(self.times)
        out = [(0.0, float(bpm[0]))]
        for k in range(1, len(bpm)):
            if abs(bpm[k] - out[-1][1]) > 1e-6 * bpm[k]:
                out.append((float(k), float(bpm[k])))
        return out

    def tempo_map(self) -> TempoMap:
        return TempoMap.from_bpm_events(self.bpm, self.bpm_events())

    def slots(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, beats) of every 1/4 beat (1/8 below SLOW_BPM) from map beat 1 to the last grid time."""
        per = 8 if self.bpm < SLOW_BPM else 4
        t, k = self.times[1:], np.arange(1, len(self.times), dtype=np.float64)
        q = np.arange(per) / per
        times = (t[:-1, None] + np.diff(t)[:, None] * q).ravel()
        beats = (k[:-1, None] + q).ravel()
        return np.append(times, t[-1]), np.append(beats, k[-1])


def fit_line(beats: np.ndarray) -> tuple[float, float, float]:
    """(t0, period, residual p95) of t = t0 + k * period. Indices step by each interval divided by
    the median interval, rounded, so a missed beat leaves a gap instead of shifting later indices
    (rounding (b - b0) / median instead drifts by whole beats over a long song)."""
    b = np.sort(np.asarray(beats, dtype=np.float64))
    steps = np.maximum(np.round(np.diff(b) / np.median(np.diff(b))), 1)
    k = np.concatenate([[0.0], np.cumsum(steps)])
    period, t0 = np.polyfit(k, b, 1)
    resid = np.abs(b - (t0 + k * period))
    return float(t0), float(period), float(np.percentile(resid, 95))


def _line_times(t0: float, period: float, start: float, end: float) -> np.ndarray:
    j0 = int(np.ceil((start - t0) / period - 1e-9))
    j1 = int(np.floor((end - t0) / period + 1e-9))
    return t0 + np.arange(j0, j1 + 1) * period


def fit_grid(beats, duration_s: float, segments: list[tuple[float, float]] | None = None) -> Grid:
    """Grid covering [0, duration_s] from Beat This beats; piecewise per segment if unsteady."""
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    if len(beats) < MIN_SEGMENT_BEATS:
        raise ValueError(f"only {len(beats)} beats")
    t0, period, p95 = fit_line(beats)
    piecewise = p95 > STEADY_P95_S and bool(segments)
    if not piecewise:
        audio = _line_times(t0, period, 0.0, duration_s)
    else:
        parts, start = [], 0.0
        for k, (_, end) in enumerate(segments):
            last = k == len(segments) - 1
            seg_end = duration_s if last else end
            inside = beats[(beats >= start) & (beats < seg_end)]
            if len(inside) < MIN_SEGMENT_BEATS and not last:
                continue   # too short to fit: merged into the next segment
            if len(inside) < MIN_SEGMENT_BEATS:
                inside = beats[beats >= start - period * MIN_SEGMENT_BEATS]
            a, p, _ = fit_line(inside)
            parts.append(_line_times(a, p, start, seg_end - 1e-9))
            start = seg_end
        audio = np.concatenate(parts)
        keep = np.concatenate([[True], np.diff(audio) > 0.5 * period])
        audio = audio[keep]
    audio = audio[audio >= 0.5 * period]
    times = np.concatenate([[0.0], audio])
    return Grid(times, 60.0 / period, piecewise)
```

`src/kata/baseline/onsets.py`:

```python
"""Note times from onset curves (spec §8.2)."""

from __future__ import annotations

import numpy as np

from kata.audio.frames import FPS
from kata.audio.store import CURVE_COLUMNS

ONSET_WEIGHTS = {"onset_drums": 1.0, "onset_mix": 0.5, "onset_bass": 0.3, "onset_other": 0.3,
                 "onset_vocals": 0.3}
POOL_S = 0.03            # slot strength = max over ±30 ms
LEAD_IN_S, LEAD_OUT_S = 1.5, 2.0
ACTIVE_FRACTION = 0.2    # the mapped span runs from the first to the last slot this strong (× max)


def onset_strength(curves: np.ndarray) -> np.ndarray:
    """Weighted sum of the onset curves, each divided by its own 99th percentile."""
    out = np.zeros(curves.shape[0], np.float64)
    for name, w in ONSET_WEIGHTS.items():
        c = np.asarray(curves[:, CURVE_COLUMNS.index(name)], dtype=np.float64)
        p99 = np.percentile(c, 99)
        out += w * c / (p99 if p99 > 0 else 1.0)
    return out


def loudness_boost(curves: np.ndarray, segments: list[tuple[float, float]]) -> np.ndarray:
    """Per frame: clip(1 + 0.5 z, 0.5, 1.5), z = the segment's mean rms_mix z-scored within the song."""
    rms = np.asarray(curves[:, CURVE_COLUMNS.index("rms_mix")], dtype=np.float64)
    mu, sd = rms.mean(), rms.std() or 1.0
    boost = np.ones(len(rms))
    for start, end in segments:
        a, b = int(round(start * FPS)), min(int(round(end * FPS)), len(rms))
        if b > a:
            boost[a:b] = np.clip(1 + 0.5 * (rms[a:b].mean() - mu) / sd, 0.5, 1.5)
    return boost


def slot_strengths(frame_strength: np.ndarray, slot_times: np.ndarray) -> np.ndarray:
    r = int(round(POOL_S * FPS))
    centre = np.round(np.asarray(slot_times) * FPS).astype(np.int64)
    idx = np.clip(centre[:, None] + np.arange(-r, r + 1), 0, len(frame_strength) - 1)
    return frame_strength[idx].max(axis=1)


def mapped_span(slot_times: np.ndarray, strengths: np.ndarray, duration_s: float) -> np.ndarray:
    """Mask of slots inside [first, last] strong slot, at least LEAD_IN_S / LEAD_OUT_S from the ends."""
    ok = (slot_times >= LEAD_IN_S) & (slot_times <= duration_s - LEAD_OUT_S)
    strong = np.flatnonzero(ok & (strengths >= ACTIVE_FRACTION * strengths[ok].max())) if ok.any() else []
    if len(strong) == 0:
        return np.zeros(len(slot_times), bool)
    return ok & (np.arange(len(slot_times)) >= strong[0]) & (np.arange(len(slot_times)) <= strong[-1])


def select(slot_times: np.ndarray, strengths: np.ndarray, n_target: int, min_gap_s: float) -> np.ndarray:
    """Greedy by strength: indices (time-ordered) of up to n_target slots at least min_gap_s apart."""
    chosen: list[float] = []
    picked: list[int] = []
    for i in np.argsort(-strengths, kind="stable"):
        t = slot_times[i]
        k = np.searchsorted(chosen, t)
        if (k > 0 and t - chosen[k - 1] < min_gap_s - 1e-9) or (k < len(chosen) and chosen[k] - t < min_gap_s - 1e-9):
            continue
        chosen.insert(k, t)
        picked.insert(k, int(i))
        if len(picked) >= n_target:
            break
    return np.asarray(picked, dtype=np.int64)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_baseline_grid.py tests/test_baseline_onsets.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/baseline/__init__.py src/kata/baseline/grid.py src/kata/baseline/onsets.py tests/test_baseline_grid.py tests/test_baseline_onsets.py
git commit -m "Phase 3: baseline tempo grid and onset selection"
```

---

### Task 5: Baseline patterns and map export

**Files:**
- Create: `src/kata/baseline/patterns.py`, `src/kata/baseline/export.py`
- Test: `tests/test_baseline_patterns.py`, `tests/test_baseline_export.py`

**Interfaces:**
- Consumes: Task 1 (`swings` constants, `analyse`, `reaction_time_s`), Task 4 (`Grid`), Phase 1 (`kata.data.mapformat`: `DIFFICULTY_RANKS`, `load_info`, `parse_difficulty`, `read_json`, `tempo_for`).
- Produces (patterns): `NOTE_SCHEMA`; `place(times, beats, strengths, downbeats, label, min_gap_hand_s, rng) -> pl.DataFrame` (note frame sorted by `time_s, color, x, y`).
- Produces (export): `INFO_VERSION = "2.1.0"`, `DIFFICULTY_VERSION = "3.2.0"`, `EMPTY_V3`; `DifficultyOut(notes, njs, njs_offset)`; `filename(label) -> str`; `njs_offset_for(njs, reaction_time_s, bpm) -> float`; `difficulty_json(notes, grid) -> dict`; `info_json(meta, grid, diffs, duration_s) -> dict`; `write_png(path, size=256, rgb=...)`; `write_map(folder, audio, meta, grid, diffs, duration_s) -> Path` (the zip); `round_trip(folder, diffs, tol_s=1e-4) -> None` (raises `AssertionError`).

- [ ] **Step 1: Write the failing tests** — `tests/test_baseline_patterns.py`:

```python
import numpy as np
import polars as pl
import pytest

from kata.baseline.patterns import place
from kata.validate.rules import LABELS, MapInput, analyse


@pytest.mark.parametrize("label", LABELS)
def test_placement_alternates_without_any_rule_issue(label):
    # 2 s bars: a strong downbeat, then 4 notes, then a breath, so both hands are free and on the
    # same parity at every downbeat (the double conditions)
    times = (2.0 + 2.0 * np.arange(40)[:, None] + np.array([0, 0.5, 0.75, 1.0, 1.25])).ravel()
    strength = np.tile([5.0, 1, 1, 1, 1], 40)
    notes = place(times, times * 2, strength, times[::5], label, 0.4, np.random.default_rng(1))
    a = analyse(MapInput(label, 16.0, 0.0, 120.0, notes))
    assert a.issues.height == 0
    assert notes.filter(pl.col("x").is_in([1, 2]) & (pl.col("y") == 1)).height == 0   # no vision blockers
    doubles = notes.group_by("time_s").len().filter(pl.col("len") > 1).height
    assert (doubles > 0) == (label in ("Hard", "Expert", "ExpertPlus"))
    if label in ("Easy", "Normal"):
        assert set(notes["cut_dir"].unique().to_list()) <= {0, 1}


def test_placement_is_deterministic_per_seed():
    times = 2.0 + np.arange(50) * 0.25
    a = place(times, times * 2, np.ones(50), np.array([]), "Expert", 0.4, np.random.default_rng(3))
    b = place(times, times * 2, np.ones(50), np.array([]), "Expert", 0.4, np.random.default_rng(3))
    assert a.equals(b)
```

`tests/test_baseline_export.py`:

```python
import json
import zipfile

import numpy as np
import pytest

from kata.baseline.export import EMPTY_V3, DifficultyOut, njs_offset_for, round_trip, write_map, write_png
from kata.baseline.grid import Grid
from kata.baseline.patterns import place
from kata.validate.rules import reaction_time_s


@pytest.mark.parametrize("njs, rt, bpm", [(16, 0.6, 120), (20, 0.5, 174), (10, 1.0, 90)])
def test_njs_offset_inverts_the_reaction_time(njs, rt, bpm):
    assert reaction_time_s(njs, njs_offset_for(njs, rt, bpm), bpm) == pytest.approx(rt, abs=1e-3)


def test_png_is_valid(tmp_path):
    write_png(tmp_path / "c.png", size=8)
    assert (tmp_path / "c.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_map_round_trips_through_the_phase1_parser(tmp_path):
    grid = Grid(np.concatenate([[0.0], 0.731 + 0.5 * np.arange(200)]), 120.0, False)
    t, b = grid.slots()
    notes = place(t[8:400:2], b[8:400:2], np.ones(196), np.array([]), "Expert", 0.4, np.random.default_rng(0))
    diffs = {"Expert": DifficultyOut(notes, 16.0, -0.2), "Easy": DifficultyOut(notes.head(40), 10.0, 0.0)}
    audio = tmp_path / "src.egg"
    audio.write_bytes(b"OggS fake")
    zp = write_map(tmp_path / "out", audio, {"song_name": "Song", "song_author": "Artist"}, grid, diffs, 100.0)
    round_trip(tmp_path / "out", diffs)
    info = json.loads((tmp_path / "out" / "Info.dat").read_text())
    for key in ("_version", "_songName", "_songSubName", "_songAuthorName", "_levelAuthorName", "_beatsPerMinute",
                "_songTimeOffset", "_shuffle", "_shufflePeriod", "_previewStartTime", "_previewDuration",
                "_songFilename", "_coverImageFilename", "_environmentName", "_difficultyBeatmapSets"):
        assert key in info
    ranks = [d["_difficultyRank"] for d in info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]]
    assert ranks == [1, 7]
    diff = json.loads((tmp_path / "out" / "ExpertStandard.dat").read_text())
    assert diff["version"] == "3.2.0" and all(k in diff for k in EMPTY_V3)
    with zipfile.ZipFile(zp) as z:
        assert sorted(z.namelist()) == ["EasyStandard.dat", "ExpertStandard.dat", "Info.dat", "cover.png", "song.egg"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_baseline_patterns.py tests/test_baseline_export.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.baseline.patterns'` (and `.export`).

- [ ] **Step 3: Implement** — `src/kata/baseline/patterns.py`:

```python
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
```

`src/kata/baseline/export.py`:

```python
"""Map folder + zip export (spec §8.5): Info.dat v2.1.0, difficulties v3.2.0."""

from __future__ import annotations

import json
import shutil
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from kata.baseline.grid import Grid
from kata.data.mapformat import DIFFICULTY_RANKS, load_info, parse_difficulty, read_json, tempo_for

INFO_VERSION, DIFFICULTY_VERSION = "2.1.0", "3.2.0"
LEVEL_AUTHOR = "kata-baseline v1"
COVER = "cover.png"
SONG = "song.egg"
# Every top-level key BeatSaver's v3.2 validator requires; all empty except notes and BPM events.
EMPTY_V3 = ("rotationEvents", "bombNotes", "obstacles", "sliders", "burstSliders", "waypoints",
            "basicBeatmapEvents", "colorBoostBeatmapEvents", "lightColorEventBoxGroups",
            "lightRotationEventBoxGroups", "lightTranslationEventBoxGroups")


@dataclass
class DifficultyOut:
    notes: pl.DataFrame      # beat, time_s, x, y, color, cut_dir, angle_offset
    njs: float
    njs_offset: float


def filename(label: str) -> str:
    return f"{label}Standard.dat"


def njs_offset_for(njs: float, reaction_time_s: float, bpm: float) -> float:
    """NJS offset (beats) that gives this reaction time, inverting the game's HJD rule (§3.4)."""
    hjd = 4.0
    while njs * (60.0 / bpm) * hjd > 17.999:
        hjd /= 2.0
    target = max(reaction_time_s * bpm / 60.0, 0.25)
    return round(target - hjd, 4)


def difficulty_json(notes: pl.DataFrame, grid: Grid) -> dict:
    beats = grid.to_beats(notes["time_s"].to_numpy())
    data = {"version": DIFFICULTY_VERSION,
            "bpmEvents": [{"b": b, "m": m} for b, m in grid.bpm_events()],
            "colorNotes": [{"b": float(b), "x": int(r["x"]), "y": int(r["y"]), "c": int(r["color"]),
                            "d": int(r["cut_dir"]), "a": int(r["angle_offset"])}
                           for b, r in zip(beats, notes.iter_rows(named=True))],
            "basicEventTypesWithKeywords": {"d": []},
            "useNormalEventsAsCompatibleEvents": True}
    data.update({k: [] for k in EMPTY_V3})
    return data


def info_json(meta: dict, grid: Grid, diffs: dict[str, DifficultyOut], duration_s: float) -> dict:
    return {
        "_version": INFO_VERSION,
        "_songName": meta.get("song_name") or "", "_songSubName": meta.get("song_sub_name") or "",
        "_songAuthorName": meta.get("song_author") or "", "_levelAuthorName": LEVEL_AUTHOR,
        "_beatsPerMinute": round(grid.bpm, 6), "_songTimeOffset": 0, "_shuffle": 0, "_shufflePeriod": 0.5,
        "_previewStartTime": round(0.3 * duration_s, 3), "_previewDuration": 10,
        "_songFilename": SONG, "_coverImageFilename": COVER, "_environmentName": "DefaultEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": label, "_difficultyRank": DIFFICULTY_RANKS[label],
                "_beatmapFilename": filename(label), "_noteJumpMovementSpeed": d.njs,
                "_noteJumpStartBeatOffset": d.njs_offset}
                for label, d in sorted(diffs.items(), key=lambda kv: DIFFICULTY_RANKS[kv[0]])]}]}


def write_png(path: Path, size: int = 256, rgb: tuple[int, int, int] = (40, 44, 52)) -> None:
    """A solid-colour PNG, written with the standard library only."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def write_map(folder: Path, audio: Path, meta: dict, grid: Grid, diffs: dict[str, DifficultyOut],
              duration_s: float) -> Path:
    """Write the folder and a zip next to it (same name + .zip); returns the zip path."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(audio, folder / SONG)
    write_png(folder / COVER)
    (folder / "Info.dat").write_text(json.dumps(info_json(meta, grid, diffs, duration_s), indent=1))
    for label, d in diffs.items():
        (folder / filename(label)).write_text(json.dumps(difficulty_json(d.notes, grid)))
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.iterdir()):
            z.write(f, f.name)
    return zip_path


def round_trip(folder: Path, diffs: dict[str, DifficultyOut], tol_s: float = 1e-4) -> None:
    """Re-parse with the Phase 1 parser; notes and times must come back unchanged (§8.5)."""
    info = load_info(folder)
    refs = {d.difficulty: d for d in info.difficulties}
    for label, d in diffs.items():
        bm = parse_difficulty(read_json(folder / refs[label].filename))
        tempo, _ = tempo_for(info, bm)
        got = [(n[1], n[2], n[3], n[4]) for n in bm.notes]
        want = [tuple(r) for r in d.notes.sort("time_s", "color", "x", "y")
                .select("x", "y", "color", "cut_dir").iter_rows()]
        if sorted(got) != sorted(want):
            raise AssertionError(f"{label}: notes differ after round trip")
        t = np.sort(tempo.to_seconds([n[0] for n in bm.notes]))
        err = np.abs(t - np.sort(d.notes["time_s"].to_numpy())).max() if len(t) else 0.0
        if err > tol_s:
            raise AssertionError(f"{label}: times differ by {err * 1000:.3f} ms after round trip")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_baseline_patterns.py tests/test_baseline_export.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/baseline/patterns.py src/kata/baseline/export.py tests/test_baseline_patterns.py tests/test_baseline_export.py
git commit -m "Phase 3: baseline patterns and v3 map export"
```

---

### Task 6: Baseline generation for one song

**Files:**
- Create: `src/kata/baseline/generate.py`
- Test: `tests/test_baseline_generate.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 5.
- Produces: `REPAIR_ROUNDS = 3`; `SongInput(sha, duration_s, curves, beats, downbeats, segments)`; `seed_for(sha, label) -> int`; `generate(song, ref, labels=LABELS) -> (Grid, {label: DifficultyOut}, {label: Result})`.

- [ ] **Step 1: Write the failing test** — `tests/test_baseline_generate.py`:

```python
import numpy as np
import pytest

from kata.audio.frames import FPS
from kata.audio.store import CURVE_COLUMNS
from kata.baseline.generate import SongInput, generate


def test_generate_gives_five_passing_difficulties_with_rising_density(toy_ref):
    dur = 90.0
    curves = np.zeros((int(dur * FPS) + 1, 12), np.float32)
    curves[:, CURVE_COLUMNS.index("rms_mix")] = -20
    rng = np.random.default_rng(0)
    for k in range(int((dur - 1) / 0.125)):              # onsets on every 1/4 of a 120 BPM beat
        curves[int(round((0.5 + 0.125 * k) * FPS)), CURVE_COLUMNS.index("onset_drums")] = rng.uniform(0.3, 1)
    beats = 0.5 + 0.5 * np.arange(int(dur / 0.5) - 1)
    song = SongInput("ab" * 32, dur, curves, beats, beats[::4], [(0.0, 45.0), (45.0, dur)])
    grid, out, results = generate(song, toy_ref)
    assert grid.bpm == pytest.approx(120)
    counts = [out[label].notes.height for label in ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")]
    assert counts == sorted(counts) and counts[0] > 50
    assert all(r.passed and r.issues.height == 0 for r in results.values())
    again = generate(song, toy_ref)[1]
    assert all(out[k].notes.equals(again[k].notes) for k in out)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_baseline_generate.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.baseline.generate'`.

- [ ] **Step 3: Implement** — `src/kata/baseline/generate.py`:

```python
"""One song's audio features -> five validated difficulties (spec §8)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import polars as pl

from kata.baseline import onsets
from kata.baseline.export import DifficultyOut, njs_offset_for
from kata.baseline.grid import Grid, fit_grid
from kata.baseline.patterns import place
from kata.validate.reference import Reference
from kata.validate.rules import LABELS, MapInput, analyse
from kata.validate.score import Result, score_analysis

REPAIR_ROUNDS = 3


@dataclass
class SongInput:
    sha: str
    duration_s: float
    curves: np.ndarray                    # (T, 12) Phase 2 curves
    beats: np.ndarray                     # Beat This beat times (s)
    downbeats: np.ndarray
    segments: list[tuple[float, float]]   # audio_segments (start_s, end_s)


def seed_for(sha: str, label: str) -> int:
    return int(hashlib.md5(f"{sha}|{label}".encode()).hexdigest()[:8], 16)


def _repair(m: MapInput, ref: Reference) -> tuple[MapInput, Result]:
    """Drop the notes of every swing with a hard or error issue, re-check, up to REPAIR_ROUNDS times."""
    res = score_analysis(analyse(m), m.difficulty, ref)
    for _ in range(REPAIR_ROUNDS):
        bad = res.issues.filter(pl.col("severity").is_in(["hard", "error"]) & (pl.col("hand") >= 0))
        if bad.height == 0:
            break
        keys = bad.select(pl.col("hand").cast(pl.Int8), pl.col("swing_idx").cast(pl.Int32)).unique()
        rows = res.swings.join(keys, on=["hand", "swing_idx"], how="semi")["note_rows"].explode().to_list()
        notes = m.notes.sort("time_s", "color", "x", "y", maintain_order=True)
        m = MapInput(m.difficulty, m.njs, m.njs_offset, m.bpm,
                     notes.filter(~pl.int_range(pl.len()).is_in(rows)))
        res = score_analysis(analyse(m), m.difficulty, ref)
    return m, res


def generate(song: SongInput, ref: Reference, labels=LABELS) -> tuple[Grid, dict[str, DifficultyOut], dict[str, Result]]:
    grid = fit_grid(song.beats, song.duration_s, song.segments)
    slot_t, slot_b = grid.slots()
    frame = onsets.onset_strength(song.curves) * onsets.loudness_boost(song.curves, song.segments)
    strength = onsets.slot_strengths(frame, slot_t)
    span = onsets.mapped_span(slot_t, strength, song.duration_s)
    slot_t, slot_b, strength = slot_t[span], slot_b[span], strength[span]
    span_s = float(slot_t[-1] - slot_t[0]) if len(slot_t) else 0.0
    out, results = {}, {}
    for label in labels:
        target_nps = ref.baseline["nps"][label]
        gap = ref.baseline["min_gap_s"][ref.stratum(label, target_nps)]
        idx = onsets.select(slot_t, strength, int(round(target_nps * span_s)), gap / 2.0)
        notes = place(slot_t[idx], slot_b[idx], strength[idx], song.downbeats, label, gap,
                      np.random.default_rng(seed_for(song.sha, label)))
        njs = float(ref.baseline["njs"][label])
        offset = njs_offset_for(njs, ref.baseline["reaction_time_s"][label], grid.bpm)
        m, res = _repair(MapInput(label, njs, offset, grid.bpm, notes), ref)
        out[label] = DifficultyOut(m.notes, njs, offset)
        results[label] = res
    return grid, out, results
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_baseline_generate.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/kata/baseline/generate.py tests/test_baseline_generate.py
git commit -m "Phase 3: one song to five validated baseline difficulties"
```

---

### Task 7: Tier D (Phase 1 extensions)

**Files:**
- Create: `src/kata/data/lowrated.py`
- Modify: `src/kata/config.py`, `src/kata/data/beatsaver.py`, `src/kata/data/pipeline.py`
- Test: `tests/test_lowrated.py`

**Interfaces:**
- Consumes: Phase 1 `kata.data.pipeline` (`_with_state`, `_gather_items`, `normalise_title`, `MAP_FILTERS`, `SELECT_SQL`), `kata.data.beatsaver`, `kata.data.state.State`.
- Produces: `Paths.lowrated` (`processed/lowrated`), `Paths.validator` (`root/validator`), `Paths.generated` (`root/generated`), all created by `Paths.ensure()`; `beatsaver.AUDIO_NAMES`, `beatsaver.is_audio(name)`, `extract_map(zip_path, dest, drop_audio=False)`, `download_map(client, paths, hash, url, drop_audio=False)`; `pipeline._filter_catalog(con, maps, diffs) -> funnel` (builds `fm`, `fd`); `pipeline._parse_selection(paths, sel_diffs, sel_maps, downloaded, out_dir, audio=True) -> summary`; `pipeline._parse_one_map(folder, hash, rows, audio=True)`; `lowrated.LOWRATED_MAX_RATING/MIN_VOTES/SAMPLE/PILOT`, `year_quotas(share, available, n)`, `calib_half(song_author, song_name) -> "fit" | "holdout"`, stages `select_lowrated(paths, n=2000)`, `download_lowrated(paths, pilot=False)` (async), `parse_lowrated(paths, pilot=False)`.

- [ ] **Step 1: Before touching `pipeline.py`, save the current parser output on real maps (needs `KATA_DATA`)**

Save this as `cmp_parse.py` in a scratch directory outside the repo (`<scratch>` below):

```python
import pickle, sys, polars as pl
from kata.data.pipeline import _parse_one_map
sel = pl.read_parquet(r"E:\kata-data\processed\selected_diffs.parquet")
hashes = sel["hash"].unique().sort()[:: 250].to_list()[:25]
out = []
for h in hashes:
    rows = sel.filter(pl.col("hash") == h).to_dicts()
    r = _parse_one_map(rf"E:\kata-data\raw\maps\{h}", h, rows)
    out.append({k: (v.to_dicts() if isinstance(v, pl.DataFrame) else v) for k, v in r.items()})
pickle.dump(out, open(sys.argv[1], "wb"))
print(len(out))
```

Run: `uv run python <scratch>/cmp_parse.py <scratch>/old.pkl`
Expected: prints the number of maps parsed (22 on the v1 dataset).

- [ ] **Step 2: Write the failing test** — `tests/test_lowrated.py`:

```python
import json
import zipfile

import polars as pl

from kata.config import Paths
from kata.data.beatsaver import extract_map, is_audio
from kata.data.lowrated import LOWRATED_PILOT, calib_half, parse_lowrated, select_lowrated, year_quotas


def test_year_quotas_follow_the_share_and_redistribute_short_years():
    q = year_quotas({2019: 0.1, 2022: 0.5, 2024: 0.4}, {2019: 1000, 2022: 1000, 2024: 50}, 1000)
    assert q[2024] == 50                                  # short year gives everything it has
    assert sum(q.values()) == 1000
    assert q[2022] > q[2019] > 0
    assert year_quotas({2020: 1.0}, {2018: 3, 2020: 2}, 10) == {2018: 3, 2020: 2}   # pool smaller than n


def test_calib_half_depends_only_on_the_normalised_song_name():
    assert calib_half("Camellia", "Ghost") == calib_half("camellia", "GHOST (Extended)")
    halves = {calib_half("artist", f"song {i}") for i in range(50)}
    assert halves == {"fit", "holdout"}


def test_extract_map_can_drop_audio(tmp_path):
    zp = tmp_path / "m.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for name in ("Info.dat", "Expert.dat", "song.egg", ".egg", "x.wav", "cover.jpg"):
            z.writestr(name, b"{}")
    assert sorted(extract_map(zp, tmp_path / "a", drop_audio=True)) == ["Expert.dat", "Info.dat"]
    assert ".egg" in extract_map(zp, tmp_path / "b")
    assert is_audio("SONG.OGG") and not is_audio("song.dat")


def _catalog(n_low=30):
    maps, diffs = [], []
    for i in range(n_low + 3):
        low = i < n_low
        mid = f"m{i}"
        maps.append({"map_id": mid, "hash": f"h{i}", "name": "x", "song_name": f"Song {i}", "song_sub_name": "",
                     "song_author": "Artist", "level_author": "L", "uploader_id": 1, "uploader_name": "u",
                     "meta_bpm": 120.0, "meta_duration_s": 120.0, "uploaded_at": f"{2018 + i % 6}-05-01T00:00:00Z",
                     "created_at": "2020-01-01T00:00:00Z", "curated_at": None, "upvotes": 10 if low else 500,
                     "downvotes": 30 if low else 1, "rating": 0.3 if low else 0.99, "ranked_ss": False,
                     "ranked_bl": False, "qualified_ss": False, "qualified_bl": False, "automapper": False,
                     "declared_ai": "None", "sage_score": 0, "tags": [], "version_state": "Published",
                     "download_url": f"https://cdn/{i}.zip", "raw_json": "{}"})
        diffs.append({"map_id": mid, "hash": f"h{i}", "characteristic": "Standard", "difficulty": "Expert",
                      "njs": 16.0, "njs_offset": 0.0, "notes": 60, "bombs": 0, "obstacles": 0, "nps": 3.0,
                      "length_beats": 200.0, "seconds": 100.0, "events": 0, "chroma": False, "me": False, "ne": False,
                      "cinema": False, "parity_errors": 3, "parity_warns": 0, "parity_resets": 0, "ss_stars": None,
                      "bl_stars": None, "max_score": 0, "environment": "Default"})
    return pl.DataFrame(maps, infer_schema_length=None), pl.DataFrame(diffs, infer_schema_length=None)


def _write_map_folder(folder, n_notes=60):
    folder.mkdir(parents=True)
    (folder / "Info.dat").write_text(json.dumps({
        "_version": "2.1.0", "_songName": "s", "_songFilename": "song.egg", "_beatsPerMinute": 120,
        "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_difficultyRank": 7, "_beatmapFilename": "Expert.dat",
             "_noteJumpMovementSpeed": 16, "_noteJumpStartBeatOffset": 0}]}]}))
    notes = [{"b": 4 + i, "x": 3, "y": 0 if i % 2 == 0 else 2, "c": 1, "d": 1 if i % 2 == 0 else 0, "a": 0}
             for i in range(n_notes)]
    (folder / "Expert.dat").write_text(json.dumps({"version": "3.0.0", "colorNotes": notes}))


def test_select_and_parse_lowrated(tmp_path):
    p = Paths(tmp_path).ensure()
    maps, diffs = _catalog()
    maps.write_parquet(p.processed / "metadata_catalog.parquet")
    diffs.write_parquet(p.processed / "metadata_catalog_diffs.parquet")
    pl.DataFrame({"map_id": ["m31"], "hash": ["h31"]}).write_parquet(p.processed / "selected_maps.parquet")
    pl.DataFrame({"hash": ["h31", "h32"], "map_tier": ["A", "A"],
                  "uploaded_at": ["2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"]}).write_parquet(p.processed / "maps.parquet")
    s = select_lowrated(p, n=10)
    sel = pl.read_parquet(p.lowrated / "selected_maps.parquet")
    assert s["pool_maps"] == 30 and sel.height == 10 and set(sel["map_tier"]) == {"D"}
    assert sel["year"].is_in([2022, 2023]).sum() == 10          # Tier A's years are preferred
    assert int(sel["pilot"].sum()) == min(LOWRATED_PILOT, 10)
    # pretend the maps were downloaded without audio
    from kata.data.state import State
    st = State(p.state_db)
    for h in sel["hash"]:
        _write_map_folder(p.maps / h)
        st.mark("download-lowrated", h, "ok")
    st.close()
    out = parse_lowrated(p)
    m = pl.read_parquet(p.lowrated / "maps.parquet")
    d = pl.read_parquet(p.lowrated / "difficulties.parquet")
    assert out["diff_status"] == {"ok": 10}
    assert set(m["audio_status"]) == {"dropped"} and set(m["calib_half"]) <= {"fit", "holdout"}
    assert set(d["tier"]) == {"D"}
    assert pl.read_parquet(p.lowrated / "notes.parquet").height == 600
    assert not (p.processed / "notes.parquet").exists()          # Tier D never touches the main tables
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_lowrated.py -q`
Expected: `ModuleNotFoundError: No module named 'kata.data.lowrated'` (and `is_audio` import error).

- [ ] **Step 4: Implement the Phase 1 changes** — apply these diffs (`git apply` works on them as written):

`src/kata/config.py`:

```diff
diff --git a/src/kata/config.py b/src/kata/config.py
index 38c6314..2c899c9 100644
--- a/src/kata/config.py
+++ b/src/kata/config.py
@@ -62,6 +62,21 @@ class Paths:
         """Phase 2 feature store root (docs/phases/phase-2-audio.md §9)."""
         return self.root / "features"
 
+    @property
+    def lowrated(self) -> Path:
+        """Phase 3 Tier D tables (docs/phases/phase-3-validator.md §6.1)."""
+        return self.processed / "lowrated"
+
+    @property
+    def validator(self) -> Path:
+        """Phase 3 validator config and reference (docs/phases/phase-3-validator.md §9)."""
+        return self.root / "validator"
+
+    @property
+    def generated(self) -> Path:
+        """Phase 3 baseline maps (docs/phases/phase-3-validator.md §8.5)."""
+        return self.root / "generated"
+
     @property
     def state_db(self) -> Path:
         return self.root / "state" / "pipeline.duckdb"
@@ -77,6 +92,6 @@ class Paths:
     def ensure(self) -> "Paths":
         for p in (self.catalog_pages, self.maps, self.bl_leaderboards, self.bl_scores,
                   self.replays, self.processed, self.replay_frames, self.features,
-                  self.state_db.parent, self.reports):
+                  self.lowrated, self.validator, self.generated, self.state_db.parent, self.reports):
             p.mkdir(parents=True, exist_ok=True)
         return self
```

`src/kata/data/beatsaver.py`:

```diff
diff --git a/src/kata/data/beatsaver.py b/src/kata/data/beatsaver.py
index 09d172e..f3af1d6 100644
--- a/src/kata/data/beatsaver.py
+++ b/src/kata/data/beatsaver.py
@@ -18,6 +18,13 @@ COMPLETE_MARKER = "_COMPLETE"
 # Covers and videos are dropped; everything else is kept, because Info.dat may name
 # difficulty or audio files with any extension (seen: `*.ben`, a song literally named `.egg`).
 DROP_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".mp4", ".webm", ".mov", ".mkv"}
+# Tier D (Phase 3) keeps no audio: the validator reads only the .dat files.
+AUDIO_NAMES = {".egg", ".ogg", ".wav"}
+
+
+def is_audio(name: str) -> bool:
+    n = name.lower()
+    return PurePosixPath(n).suffix in AUDIO_NAMES or n in AUDIO_NAMES
 
 
 def _write_page(path: Path, docs: list[dict]) -> None:
@@ -137,7 +144,7 @@ def flatten_catalog(paths: Paths) -> tuple[pl.DataFrame, pl.DataFrame]:
     return maps_df, diffs_df
 
 
-def extract_map(zip_path: Path, dest: Path) -> list[str]:
+def extract_map(zip_path: Path, dest: Path, drop_audio: bool = False) -> list[str]:
     """Extract the files next to Info.dat (flattened; no path traversal possible)."""
     with zipfile.ZipFile(zip_path) as z:
         members = [i for i in z.infolist() if not i.is_dir()]
@@ -149,20 +156,21 @@ def extract_map(zip_path: Path, dest: Path) -> list[str]:
         kept = []
         for i in members:
             p = PurePosixPath(i.filename)
-            if p.parent != base or p.suffix.lower() in DROP_SUFFIXES:
+            if p.parent != base or p.suffix.lower() in DROP_SUFFIXES or (drop_audio and is_audio(p.name)):
                 continue
             (dest / p.name).write_bytes(z.read(i))
             kept.append(p.name)
     return kept
 
 
-async def download_map(client: PoliteClient, paths: Paths, map_hash: str, url: str) -> dict:
+async def download_map(client: PoliteClient, paths: Paths, map_hash: str, url: str,
+                       drop_audio: bool = False) -> dict:
     zip_path = paths.maps / "_zips" / f"{map_hash}.zip"
     sha = await client.download(url, zip_path)
     size = zip_path.stat().st_size
     dest = paths.maps / map_hash
     try:
-        files = extract_map(zip_path, dest)
+        files = extract_map(zip_path, dest, drop_audio)
     finally:
         zip_path.unlink(missing_ok=True)
     return {"zip_sha256": sha, "zip_bytes": size, "files": files}
```

`src/kata/data/pipeline.py` (pure refactor of `_select` and `_parse_maps`, plus the audio flag and the `lowrated_*` views):

```diff
diff --git a/src/kata/data/pipeline.py b/src/kata/data/pipeline.py
index 10c2f6e..5210329 100644
--- a/src/kata/data/pipeline.py
+++ b/src/kata/data/pipeline.py
@@ -116,10 +116,10 @@ MAP_FILTERS = [
 ]
 
 
-def _select(paths, state, summary):
-    maps, diffs = beatsaver.flatten_catalog(paths)
-    con = duckdb.connect()
-    con.register("m_all", maps.drop("raw_json"))
+def _filter_catalog(con, maps: pl.DataFrame, diffs: pl.DataFrame) -> dict:
+    """Build `fm` (maps passing MAP_FILTERS) and `fd` (their difficulties passing §3.2, with a
+    tier) in `con`; returns the funnel counts."""
+    con.register("m_all", maps.drop("raw_json") if "raw_json" in maps.columns else maps)
     con.register("d", diffs)
     funnel = {"catalog_maps": maps.height}
     where = []
@@ -129,6 +129,13 @@ def _select(paths, state, summary):
     con.execute(f"CREATE TABLE fm AS SELECT * FROM m_all WHERE {' AND '.join(where)}")
     con.execute(SELECT_SQL)
     funnel["maps_with_valid_standard_diff"] = con.execute("SELECT count(DISTINCT map_id) FROM fd").fetchone()[0]
+    return funnel
+
+
+def _select(paths, state, summary):
+    maps, diffs = beatsaver.flatten_catalog(paths)
+    con = duckdb.connect()
+    funnel = _filter_catalog(con, maps, diffs)
 
     map_tiers = con.execute("""
         SELECT map_id, any_value(hash) AS hash, min(tier) AS map_tier
@@ -199,16 +206,14 @@ OBJECT_TABLES = {
 }
 
 
-def _parse_one_map(folder: str, map_hash: str, diff_rows: list[dict]) -> dict:
+def _parse_one_map(folder: str, map_hash: str, diff_rows: list[dict], audio: bool = True) -> dict:
     """Worker: parse a downloaded map folder.
 
     Returns the map row, difficulty rows, and one DataFrame per object table (columnar, so
-    the full run does not hold millions of Python dicts).
+    the full run does not hold millions of Python dicts). audio=False (Tier D) skips the audio
+    checks: those maps were extracted without their audio.
     """
-    import soundfile as sf
-
-    from kata.data.mapformat import (MapFormatError, find_audio, find_file, load_info,
-                                     parse_difficulty, read_json, tempo_for)
+    from kata.data.mapformat import load_info
 
     out = {"map": {"hash": map_hash}, "difficulties": [], "tempo_maps": [],
            **{name: [] for name in OBJECT_TABLES}}
@@ -220,8 +225,20 @@ def _parse_one_map(folder: str, map_hash: str, diff_rows: list[dict]) -> dict:
         return _frames(out)
     m.update(info_format_version=info.info_version, info_bpm=info.bpm,
              song_time_offset=info.song_time_offset, parse_status="ok", parse_error=None)
+    if audio:
+        _audio_checks(m, Path(folder), info.song_filename)
+    else:
+        m.update(audio_status="dropped", audio_error=None)
+    _parse_difficulties(out, folder, map_hash, diff_rows, info)
+    return _frames(out)
+
+
+def _audio_checks(m: dict, folder: Path, song_filename: str) -> None:
+    import soundfile as sf
+
+    from kata.data.mapformat import find_audio
     try:
-        audio, m["audio_filename_fallback"] = find_audio(Path(folder), info.song_filename)
+        audio, m["audio_filename_fallback"] = find_audio(folder, song_filename)
         m["audio_file"] = audio.name
         m["audio_bytes"] = audio.stat().st_size
         m["audio_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
@@ -232,6 +249,10 @@ def _parse_one_map(folder: str, map_hash: str, diff_rows: list[dict]) -> dict:
     except Exception as e:
         m.update(audio_status="error", audio_error=f"{type(e).__name__}: {e}"[:300])
 
+
+def _parse_difficulties(out: dict, folder: str, map_hash: str, diff_rows: list[dict], info) -> None:
+    from kata.data.mapformat import MapFormatError, find_file, parse_difficulty, read_json, tempo_for
+
     refs = {(d.characteristic, d.difficulty): d for d in info.difficulties}
     for row in diff_rows:
         key = dict(hash=map_hash, characteristic=row["characteristic"], difficulty=row["difficulty"])
@@ -285,7 +306,6 @@ def _parse_one_map(folder: str, map_hash: str, diff_rows: list[dict]) -> dict:
             out[name].append(df)
         for seg in tempo.rows():
             out["tempo_maps"].append({**key, **seg})
-    return _frames(out)
 
 
 def _frames(out: dict) -> dict:
@@ -296,19 +316,28 @@ def _frames(out: dict) -> dict:
 
 def _parse_maps(paths, state, summary, pilot=False):
     sel_diffs = _subset(pl.read_parquet(paths.processed / "selected_diffs.parquet"), paths, pilot)
-    downloaded = state.done("download-map")
+    sel_maps = pl.read_parquet(paths.processed / "selected_maps.parquet").select("hash", "map_tier", "pilot")
+    summary.update(_parse_selection(paths, sel_diffs, sel_maps, state.done("download-map"), paths.processed))
+
+parse_maps = _with_state("parse-maps", _parse_maps)
+
+
+def _parse_selection(paths, sel_diffs: pl.DataFrame, sel_maps: pl.DataFrame, downloaded: set[str],
+                     out_dir: Path, audio: bool = True) -> dict:
+    """Parse the downloaded maps of a selection and write maps, difficulties, object and tempo
+    tables to out_dir. Returns the summary counts."""
+    summary: dict = {}
     by_hash: dict[str, list[dict]] = {}
     for r in sel_diffs.iter_rows(named=True):
         if r["hash"] in downloaded:
             by_hash.setdefault(r["hash"], []).append(r)
     results = []
     with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
-        futures = [pool.submit(_parse_one_map, str(paths.maps / h), h, rows) for h, rows in by_hash.items()]
+        futures = [pool.submit(_parse_one_map, str(paths.maps / h), h, rows, audio) for h, rows in by_hash.items()]
         for f in futures:
             results.append(f.result())
 
     catalog = pl.read_parquet(paths.processed / "metadata_catalog.parquet").drop("raw_json")
-    sel_maps = pl.read_parquet(paths.processed / "selected_maps.parquet").select("hash", "map_tier", "pilot")
     maps_df = (pl.DataFrame([r["map"] for r in results], infer_schema_length=None)
                .join(catalog, on="hash", how="left").join(sel_maps, on="hash", how="left"))
     diffs_df = pl.DataFrame([d for r in results for d in r["difficulties"]], infer_schema_length=None)
@@ -317,8 +346,9 @@ def _parse_maps(paths, state, summary, pilot=False):
                          "parity_warns", "parity_resets", "ss_stars", "bl_stars", "chroma", "cinema",
                          "environment"),
         on=["hash", "characteristic", "difficulty"], how="left")
-    maps_df.write_parquet(paths.processed / "maps.parquet")
-    diffs_df.write_parquet(paths.processed / "difficulties.parquet")
+    out_dir.mkdir(parents=True, exist_ok=True)
+    maps_df.write_parquet(out_dir / "maps.parquet")
+    diffs_df.write_parquet(out_dir / "difficulties.parquet")
 
     accepted = diffs_df.filter(pl.col("parse_status") == "ok").select("hash", "characteristic", "difficulty")
     for name in OBJECT_TABLES:
@@ -328,9 +358,9 @@ def _parse_maps(paths, state, summary, pilot=False):
             front = ["hash", "characteristic", "difficulty", "idx"]
             df = df.join(accepted, on=["hash", "characteristic", "difficulty"], how="semi").select(
                 front + [c for c in df.columns if c not in front])
-        df.write_parquet(paths.processed / f"{name}.parquet")
+        df.write_parquet(out_dir / f"{name}.parquet")
     tempo = pl.DataFrame([x for r in results for x in r["tempo_maps"]], infer_schema_length=None)
-    tempo.write_parquet(paths.processed / "tempo_maps.parquet")
+    tempo.write_parquet(out_dir / "tempo_maps.parquet")
 
     summary["maps"] = maps_df.height
     summary["map_parse_status"] = dict(maps_df.group_by("parse_status").len().iter_rows())
@@ -338,8 +368,7 @@ def _parse_maps(paths, state, summary, pilot=False):
     summary["diff_status"] = dict(diffs_df.group_by("parse_status").len().iter_rows())
     summary["format_versions"] = dict(diffs_df.group_by(pl.col("format_version").str.slice(0, 1))
                                       .len().sort("format_version").iter_rows())
-
-parse_maps = _with_state("parse-maps", _parse_maps)
+    return summary
 
 
 # --------------------------------------------------------------------------- leaderboards
@@ -611,6 +640,8 @@ def _build(paths, state, summary):
     con = duckdb.connect(str(paths.query_db))
     for f in sorted(paths.processed.glob("*.parquet")):
         con.execute(f"CREATE OR REPLACE VIEW {f.stem} AS SELECT * FROM read_parquet('{f.as_posix()}')")
+    for f in sorted(paths.lowrated.glob("*.parquet")):   # Phase 3 Tier D
+        con.execute(f"CREATE OR REPLACE VIEW lowrated_{f.stem} AS SELECT * FROM read_parquet('{f.as_posix()}')")
     frames_glob = (paths.replay_frames / "*" / "*.parquet").as_posix()
     if any(paths.replay_frames.glob("*/*.parquet")):
         con.execute(f"""CREATE OR REPLACE VIEW replay_frames AS
```

- [ ] **Step 5: Implement Tier D** — `src/kata/data/lowrated.py`:

```python
"""Tier D: a year-matched sample of low-rated maps for validator calibration
(docs/phases/phase-3-validator.md §6.1). Only the .dat files are kept; the tables go to
processed/lowrated/ so that no Phase 2 or Phase 4 consumer sees them by accident."""

from __future__ import annotations

import hashlib

import duckdb
import polars as pl

from kata.data import beatsaver
from kata.data.http import PoliteClient
from kata.data.pipeline import _filter_catalog, _gather_items, _parse_selection, _with_state, normalise_title

LOWRATED_MAX_RATING = 0.6
LOWRATED_MIN_VOTES = 20
LOWRATED_SAMPLE = 2000
LOWRATED_PILOT = 200
DOWNLOAD_STAGE = "download-lowrated"


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def year_quotas(share: dict[int, float], available: dict[int, int], n: int) -> dict[int, int]:
    """Maps to take per upload year: proportional to `share` (Tier A's year mix), capped by what
    each year has; slots a short year cannot fill go to the other years in proportion."""
    quotas = dict.fromkeys(available, 0)
    left = min(n, sum(available.values()))
    while left:
        open_ = [y for y in sorted(available) if quotas[y] < available[y]]
        w = {y: share.get(y, 0.0) for y in open_}
        if not any(w.values()):
            w = dict.fromkeys(open_, 1.0)
        total = sum(w.values())
        want = {y: left * w[y] / total for y in open_}
        step = {y: min(int(want[y]), available[y] - quotas[y]) for y in open_}
        if not any(step.values()):   # only fractions left: one map to the year that wants most
            step = {max(open_, key=lambda y: (want[y], -y)): 1}
        for y, k in step.items():
            quotas[y] += k
            left -= k
    return quotas


def calib_half(song_author: str | None, song_name: str | None) -> str:
    """'fit' or 'holdout' by the song's normalised name (Phase 1's name key)."""
    key = f"name:{normalise_title(song_author)}|{normalise_title(song_name)}"
    return "fit" if int(_md5(key), 16) % 2 == 0 else "holdout"


def _select_lowrated(paths, state, summary, n=LOWRATED_SAMPLE):
    P = paths.processed
    con = duckdb.connect()
    summary["funnel"] = _filter_catalog(con, pl.read_parquet(P / "metadata_catalog.parquet"),
                                        pl.read_parquet(P / "metadata_catalog_diffs.parquet"))
    con.register("sel", pl.read_parquet(P / "selected_maps.parquet").select("hash"))
    pool = con.execute(f"""
        SELECT map_id, hash, download_url, CAST(substr(uploaded_at, 1, 4) AS INTEGER) AS year,
               rating, upvotes + downvotes AS votes
        FROM fm
        WHERE rating < {LOWRATED_MAX_RATING} AND upvotes + downvotes >= {LOWRATED_MIN_VOTES}
          AND curated_at IS NULL AND NOT ranked_bl AND NOT ranked_ss
          AND hash NOT IN (SELECT hash FROM sel)
          AND map_id IN (SELECT map_id FROM fd)""").pl()
    pool = pool.with_columns(pl.col("map_id").map_elements(_md5, return_dtype=pl.String).alias("_order"))
    tier_a = (pl.read_parquet(P / "maps.parquet").filter(pl.col("map_tier") == "A")
              .select(pl.col("uploaded_at").str.slice(0, 4).cast(pl.Int64).alias("year")))
    share = {y: c / tier_a.height for y, c in tier_a.group_by("year").len().iter_rows()}
    available = dict(pool.group_by("year").len().iter_rows())
    quotas = year_quotas(share, available, n)
    picked = pl.concat([pool.filter(pl.col("year") == y).sort("_order").head(q) for y, q in quotas.items() if q])
    pilot_ids = picked.sort("_order").head(LOWRATED_PILOT)["map_id"]
    picked = picked.with_columns(pl.col("map_id").is_in(pilot_ids.implode()).alias("pilot"),
                                 pl.lit("D").alias("map_tier")).drop("_order").sort("map_id")
    diffs = (con.execute("SELECT * FROM fd").pl().join(picked.select("map_id"), on="map_id", how="semi")
             .with_columns(pl.lit("D").alias("tier")))
    paths.lowrated.mkdir(parents=True, exist_ok=True)
    picked.write_parquet(paths.lowrated / "selected_maps.parquet")
    diffs.write_parquet(paths.lowrated / "selected_diffs.parquet")
    summary.update(pool_maps=pool.height, sampled_maps=picked.height, sampled_diffs=diffs.height,
                   pilot_maps=int(picked["pilot"].sum()),
                   years={str(y): {"tier_a_share": round(share.get(y, 0.0), 3), "pool": available.get(y, 0),
                                   "sampled": quotas.get(y, 0)} for y in sorted(set(share) | set(available))})

select_lowrated = _with_state("select-lowrated", _select_lowrated)


async def _download_lowrated(paths, state, summary, pilot=False):
    sel = pl.read_parquet(paths.lowrated / "selected_maps.parquet")
    if pilot:
        sel = sel.filter(pl.col("pilot"))
    done = state.done(DOWNLOAD_STAGE)
    todo = [(r["hash"], r) for r in sel.iter_rows(named=True) if r["hash"] not in done]
    summary["todo"] = len(todo)
    async with PoliteClient() as client:
        await _gather_items(todo, lambda r: beatsaver.download_map(client, paths, r["hash"], r["download_url"],
                                                                   drop_audio=True),
                            state, DOWNLOAD_STAGE, paths)
        summary["requests"] = client.request_count
    summary["status"] = state.counts(DOWNLOAD_STAGE)
    summary["errors"] = state.errors(DOWNLOAD_STAGE, 10)

download_lowrated = _with_state(DOWNLOAD_STAGE, _download_lowrated)


def _parse_lowrated(paths, state, summary, pilot=False):
    sel_maps = pl.read_parquet(paths.lowrated / "selected_maps.parquet")
    if pilot:
        sel_maps = sel_maps.filter(pl.col("pilot"))
    sel_diffs = pl.read_parquet(paths.lowrated / "selected_diffs.parquet").join(
        sel_maps.select("hash"), on="hash", how="semi")
    summary.update(_parse_selection(paths, sel_diffs, sel_maps.select("hash", "map_tier", "pilot"),
                                    state.done(DOWNLOAD_STAGE), paths.lowrated, audio=False))
    maps = pl.read_parquet(paths.lowrated / "maps.parquet")
    maps = maps.with_columns(pl.struct("song_author", "song_name").map_elements(
        lambda r: calib_half(r["song_author"], r["song_name"]), return_dtype=pl.String).alias("calib_half"))
    maps.write_parquet(paths.lowrated / "maps.parquet")
    summary["calib_half"] = dict(maps.group_by("calib_half").len().iter_rows())

parse_lowrated = _with_state("parse-lowrated", _parse_lowrated)
```

- [ ] **Step 6: Run the tests and the parser comparison**

Run: `uv run pytest tests/test_lowrated.py tests/test_pipeline_units.py tests/test_mapformat.py -q`
Expected: all pass (4 new tests).

Run: `uv run python <scratch>/cmp_parse.py <scratch>/new.pkl && uv run python -c "import pickle; print(pickle.load(open(r'<scratch>/old.pkl','rb')) == pickle.load(open(r'<scratch>/new.pkl','rb')))"`
Expected: `True` — the refactor changes nothing for Tiers A–C.

- [ ] **Step 7: Dry-run the selection (no network)**

Run `select_lowrated` against a copy of `metadata_catalog*.parquet`, `selected_maps.parquet` and `maps.parquet` in a scratch `KATA_DATA` (`Paths(scratch).ensure()`, copy the four files into `processed/`).
Expected (v1 catalogue): `pool_maps` 5,676, `sampled_maps` 2,000, `sampled_diffs` 2,568, years 2022–2025 fully used (spec §17).

- [ ] **Step 8: Commit**

```bash
git add src/kata/config.py src/kata/data/beatsaver.py src/kata/data/pipeline.py src/kata/data/lowrated.py tests/test_lowrated.py
git commit -m "Phase 3: Tier D selection, audio-free download and parse"
```

---

### Task 8: Validator stages and report

**Files:**
- Create: `src/kata/validate/pipeline.py`, `src/kata/validate/report.py`
- Test: `tests/test_validate_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–3 and 7 (`Paths.lowrated`, `Paths.validator`), `kata.data.pipeline._with_state`, `kata.data.report.md_table`.
- Produces (pipeline): `KEY`, `difficulty_index(paths) -> pl.DataFrame` (Tiers A–C from `processed/`, D from `processed/lowrated/`, with `tier, split, year, source, njs, njs_offset, info_bpm, format_version, parity_errors`); stages `analyse_stage(paths)`, `reference_stage(paths)`, `score_stage(paths)`, `corrupt_stage(paths, pilot=False)`, `report_stage(paths, name="phase3")`; `corruption_summary(df) -> pl.DataFrame`. Tables as in spec §9.
- Produces (report): `auc(pos, neg) -> float`, `year_bin(year)`, `nps_bin(nps)`, `stratified_auc(a, d, col="score") -> (cells, weighted_mean)`, `write_report(paths, state, name) -> Path`.

- [ ] **Step 1: Write the failing test** — `tests/test_validate_pipeline.py`:

```python
"""Phase 3 validator stages on a tiny synthetic data root (Tier A train/val, Tier B, Tier D)."""
import polars as pl
import pytest

from conftest import stream
from kata.config import Paths
from kata.validate import pipeline as vp
from kata.validate.reference import ValidatorDir

KEY = ["hash", "characteristic", "difficulty"]
FLIP = {0: 1, 1: 0}


def _diff(h: str, broken: bool, n_per_hand: int = 48) -> pl.DataFrame:
    n = stream(n_per_hand=n_per_hand)
    if broken:   # every 4th note's arrow flipped: fast parity breaks
        n = n.with_columns(pl.when(pl.int_range(pl.len()) % 4 == 1).then(pl.col("cut_dir").replace(FLIP))
                           .otherwise(pl.col("cut_dir")).alias("cut_dir"))
    return n.with_columns(pl.lit(h).alias("hash"), pl.lit("Standard").alias("characteristic"),
                          pl.lit("ExpertPlus").alias("difficulty"), pl.int_range(pl.len()).alias("idx"))


def _write(folder, rows, split_col):
    maps = pl.DataFrame({"hash": [r[0] for r in rows], "info_bpm": [120.0] * len(rows),
                         split_col: [r[2] for r in rows], "pilot": [True] * len(rows),
                         "uploaded_at": ["2022-03-01T00:00:00Z"] * len(rows)})
    diffs = pl.DataFrame({"hash": [r[0] for r in rows], "characteristic": ["Standard"] * len(rows),
                          "difficulty": ["ExpertPlus"] * len(rows), "tier": [r[1] for r in rows],
                          "parse_status": ["ok"] * len(rows), "njs": [18.0] * len(rows), "njs_offset": [0.0] * len(rows),
                          "format_version": ["2.0.0"] * len(rows), "parity_errors": [0] * len(rows)})
    folder.mkdir(parents=True, exist_ok=True)
    maps.write_parquet(folder / "maps.parquet")
    diffs.write_parquet(folder / "difficulties.parquet")
    pl.concat([_diff(h, broken) for h, _, _, broken in rows]).write_parquet(folder / "notes.parquet")
    empty = {"hash": pl.String, "characteristic": pl.String, "difficulty": pl.String, "idx": pl.Int64,
             "beat": pl.Float64, "time_s": pl.Float64, "x": pl.Int64, "y": pl.Int64}
    pl.DataFrame(schema=empty).write_parquet(folder / "bombs.parquet")


@pytest.fixture
def paths(tmp_path) -> Paths:
    p = Paths(tmp_path).ensure()
    main = [(f"a{i}", "A", "train", False) for i in range(12)] + [("v0", "A", "val", False), ("v1", "A", "test", False),
                                                                   ("b0", "B", "train", False)]
    _write(p.processed, main, "split")
    _write(p.lowrated, [("d0", "D", "fit", True), ("d1", "D", "holdout", True), ("d2", "D", "holdout", True)],
           "calib_half")
    return p


def test_stages_end_to_end(paths):
    s = vp.analyse_stage(paths)
    assert s["difficulties"] == {"A/ok": 14, "B/ok": 1, "D/ok": 3}
    tables = ("validation_metrics", "validation_swings", "validation_issues", "validation_local_metrics")
    first = {t: pl.read_parquet(paths.processed / f"{t}.parquet") for t in tables}
    vp.analyse_stage(paths)                                    # rebuilds identical tables
    for t in tables:
        assert pl.read_parquet(paths.processed / f"{t}.parquet").equals(first[t]), t

    r = vp.reference_stage(paths)
    assert r["n_maps"] == 12                                   # Tier A train only
    assert ValidatorDir(paths.validator).reference_path.exists()

    vp.score_stage(paths)
    v = pl.read_parquet(paths.processed / "validation.parquet")
    a = v.filter(pl.col("tier") == "A")["score"]
    d = v.filter(pl.col("tier") == "D")["score"]
    assert a.min() > d.max()
    assert v.filter(pl.col("tier") == "A")["pass"].all() and not v.filter(pl.col("tier") == "D")["pass"].any()
    assert set(v["validator_version"]) == {"v1"}
    local = pl.read_parquet(paths.processed / "validation_local.parquet")
    assert {"score", "s_E1", "window_start_beat"} <= set(local.columns)

    c = vp.corrupt_stage(paths, pilot=True)
    res = pl.read_parquet(paths.processed / "corruption_results.parquet")
    assert set(res["hash"]) == {"v0", "v1"}                    # Tier A val + test only
    assert {row["corruption"] for row in c["by_corruption"]} >= {"C1", "C2", "C3", "C6", "C7"}

    out = vp.report_stage(paths, name="t")
    text = open(out["path"], encoding="utf-8").read()
    assert "Overall AUC" in text and "Synthetic corruptions" in text


def test_reference_refuses_a_changed_validator_config(paths):
    vp.analyse_stage(paths)
    ValidatorDir(paths.validator).config_path.write_text('{"version": "v1", "something": "else"}')
    with pytest.raises(ValueError, match="bump VALIDATOR_VERSION"):
        vp.reference_stage(paths)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_validate_pipeline.py -q`
Expected: `ImportError: cannot import name 'pipeline' from 'kata.validate'`.

- [ ] **Step 3: Implement** — `src/kata/validate/pipeline.py`:

```python
"""Phase 3 validator stages (docs/phases/phase-3-validator.md §10). Same conventions as Phases 1-2:
every stage runs inside `_with_state` and rebuilds its tables deterministically from its inputs."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from kata.config import Paths
from kata.data.pipeline import _with_state
from kata.validate.corrupt import CORRUPTIONS, corrupt, evaluate
from kata.validate.reference import Reference, ValidatorDir
from kata.validate.rules import HARD_RULES, MapInput, analyse
from kata.validate.score import score_analysis, score_frame

KEY = ["hash", "characteristic", "difficulty"]
OBJECTS = ("notes", "bombs", "obstacles", "chains")
BATCH = 150
INJECTIONS_PER_KIND = 5
PILOT_CORRUPT_DIFFS = 100


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def difficulty_index(paths: Paths) -> pl.DataFrame:
    """Every parsed difficulty of Tiers A/B/C (processed/) and D (processed/lowrated/), with what
    the validator and the calibration need. Tier D's `split` is its calibration half. (The pilot
    needs no subset here: analysis is cheap, and only the pilot Tier D maps are parsed.)"""
    cols = ["hash", "characteristic", "difficulty", "tier", "njs", "njs_offset", "format_version",
            "parity_errors"]
    parts = []
    for source, folder, split_col in (("main", paths.processed, "split"), ("lowrated", paths.lowrated, "calib_half")):
        if not (folder / "difficulties.parquet").exists():
            continue
        maps = pl.read_parquet(folder / "maps.parquet").select(
            "hash", "info_bpm", pl.col(split_col).alias("split"), "pilot",
            pl.col("uploaded_at").str.slice(0, 4).cast(pl.Int64).alias("year"))
        d = (pl.read_parquet(folder / "difficulties.parquet").filter(pl.col("parse_status") == "ok")
             .select(cols).join(maps, on="hash"))
        parts.append(d.with_columns(pl.lit(source).alias("source")))
    return pl.concat(parts, how="vertical_relaxed").sort(KEY)


def _inputs(folders: dict[str, str], rows: list[dict], chains: bool = True) -> list[tuple[dict, MapInput]]:
    """MapInput per row, reading only this batch's difficulties from each source's tables."""
    out = []
    for source in sorted({r["source"] for r in rows}):   # sorted: set order varies per process
        mine = [r for r in rows if r["source"] == source]
        keys = pl.DataFrame({k: [r[k] for r in mine] for k in KEY}).lazy()
        tabs = {}
        for name in OBJECTS:
            f = Path(folders[source]) / f"{name}.parquet"
            tabs[name] = (pl.scan_parquet(f).join(keys, on=KEY, how="semi").collect().partition_by(KEY, as_dict=True)
                          if f.exists() else {})
        empty = pl.DataFrame(schema={"beat": pl.Float64, "time_s": pl.Float64, "x": pl.Int64, "y": pl.Int64,
                                     "color": pl.Int64, "cut_dir": pl.Int64, "angle_offset": pl.Int64})
        for r in mine:
            k = (r["hash"], r["characteristic"], r["difficulty"])
            out.append((r, MapInput(r["difficulty"], r["njs"] or 0.0, r["njs_offset"] or 0.0, r["info_bpm"],
                                    tabs["notes"].get(k, empty), tabs["bombs"].get(k), tabs["obstacles"].get(k),
                                    tabs["chains"].get(k) if chains else None)))
    return out


def _with_key(df: pl.DataFrame, r: dict) -> pl.DataFrame:
    return df.with_columns(*(pl.lit(r[k]).alias(k) for k in KEY)).select(KEY + df.columns)


def _analyse_batch(folders: dict[str, str], rows: list[dict]) -> dict:
    metrics, swings, issues, local = [], [], [], []
    for r, m in _inputs(folders, rows):
        base = {k: r[k] for k in KEY + ["tier", "split", "year", "source", "format_version", "parity_errors"]}
        try:
            a = analyse(m)
        except Exception as e:  # recorded per difficulty; the stage keeps going
            metrics.append({**base, "status": "error", "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        metrics.append({**base, "status": "ok", "error": None, **a.metrics})
        swings.append(_with_key(a.swings.drop("note_rows"), r))
        issues.append(_with_key(a.issues, r))
        local.append(_with_key(a.local, r))
    cat = lambda xs: pl.concat(xs, how="vertical_relaxed") if xs else None  # noqa: E731
    return {"metrics": metrics, "swings": cat(swings), "issues": cat(issues), "local": cat(local)}


def _folders(paths: Paths) -> dict[str, str]:
    return {"main": str(paths.processed), "lowrated": str(paths.lowrated)}


def _analyse(paths, state, summary):
    ValidatorDir(paths.validator).open_for_write()
    rows = difficulty_index(paths).to_dicts()
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    out = {"metrics": [], "swings": [], "issues": [], "local": []}
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        for i, res in enumerate(pool.map(_analyse_batch, [_folders(paths)] * len(batches), batches)):
            out["metrics"] += res["metrics"]
            for k in ("swings", "issues", "local"):
                if res[k] is not None:
                    out[k].append(res[k])
            if (i + 1) % 20 == 0:
                print(f"  [validate-analyse] {(i + 1) * BATCH}/{len(rows)}", flush=True)
    P = paths.processed
    metrics = pl.DataFrame(out["metrics"], infer_schema_length=None).sort(KEY)
    metrics.write_parquet(P / "validation_metrics.parquet")
    for name, order in (("swings", ["hand", "swing_idx"]), ("issues", ["time_s", "rule", "hand", "swing_idx"]),
                        ("local", ["window_start_beat"])):
        table = pl.concat(out[name], how="vertical_relaxed").sort(KEY + order)
        table.write_parquet(P / ("validation_local_metrics.parquet" if name == "local" else f"validation_{name}.parquet"))
    summary["difficulties"] = dict(metrics.group_by("tier", "status").len().sort("tier", "status")
                                   .select(pl.concat_str("tier", "status", separator="/"), "len").iter_rows())
    summary["errors"] = metrics.filter(pl.col("status") != "ok").select(KEY + ["error"]).head(10).to_dicts()

analyse_stage = _with_state("validate-analyse", _analyse)


def _reference(paths, state, summary):
    m = pl.read_parquet(paths.processed / "validation_metrics.parquet").filter(
        (pl.col("status") == "ok") & (pl.col("tier") == "A") & (pl.col("split") == "train"))
    assert set(m["split"].unique().to_list()) == {"train"}, "reference must use Tier A train only"
    ref = Reference.build(m)
    vd = ValidatorDir(paths.validator)
    vd.open_for_write()
    vd.write_reference(ref)
    summary.update(n_maps=m.height, stratum_sizes=ref.data["stratum_sizes"])

reference_stage = _with_state("validate-reference", _reference)


def _score(paths, state, summary):
    ref = ValidatorDir(paths.validator).load_reference()
    P = paths.processed
    m = pl.read_parquet(P / "validation_metrics.parquet").filter(pl.col("status") == "ok")
    m = m.with_columns(pl.struct("difficulty", "nps").map_elements(
        lambda r: ref.stratum(r["difficulty"], r["nps"]), return_dtype=pl.String).alias("stratum"))
    v = score_frame(m, ref).with_columns(
        pl.all_horizontal(*(pl.col(h) == 0 for h in HARD_RULES)).alias("pass"),
        pl.lit(ref.data["version"]).alias("validator_version"))
    v.write_parquet(P / "validation.parquet")
    local = pl.read_parquet(P / "validation_local_metrics.parquet").join(m.select(KEY + ["stratum"]), on=KEY)
    score_frame(local, ref).write_parquet(P / "validation_local.parquet")
    summary["by_tier"] = (v.group_by("tier").agg(pl.len().alias("n"), pl.col("pass").mean().round(4).alias("pass"),
                                                 pl.col("score").median().round(2).alias("median_score"))
                          .sort("tier").to_dicts())

score_stage = _with_state("validate-score", _score)


def _seed(key: tuple) -> int:
    return int(hashlib.md5("|".join(key).encode()).hexdigest()[:8], 16)


def _corrupt_batch(folders: dict[str, str], ref_data: dict, rows: list[dict]) -> list[dict]:
    ref = Reference(ref_data)
    out = []
    for r, m in _inputs(folders, rows, chains=False):
        a = analyse(m)
        clean = score_analysis(a, m.difficulty, ref)
        k = (r["hash"], r["characteristic"], r["difficulty"])
        rng = np.random.default_rng(_seed(k))
        q99 = ref.anchors("L1", clean.stratum)["q99"]
        for kind, rule in CORRUPTIONS.items():
            for i in range(INJECTIONS_PER_KIND):
                x = corrupt(kind, m, a, rng, q99)
                if x is None:
                    out.append({**dict(zip(KEY, k)), "corruption": kind, "rule": rule, "i": i, "eligible": False,
                                "detected": None, "false_alarms": None, "n_swings": clean.metrics["n_swings"]})
                    break
                inj, bad = x
                hit, fa = evaluate(inj, clean, score_analysis(analyse(bad), bad.difficulty, ref))
                out.append({**dict(zip(KEY, k)), "corruption": kind, "rule": rule, "i": i, "eligible": True,
                            "detected": hit, "false_alarms": fa, "n_swings": clean.metrics["n_swings"]})
    return out


def _corrupt(paths, state, summary, pilot=False):
    ref = ValidatorDir(paths.validator).load_reference()
    idx = difficulty_index(paths).filter((pl.col("tier") == "A") & pl.col("split").is_in(["val", "test"]))
    rows = (idx.head(PILOT_CORRUPT_DIFFS) if pilot else idx).to_dicts()
    batches = [rows[i:i + 20] for i in range(0, len(rows), 20)]
    res = []
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        for part in pool.map(_corrupt_batch, [_folders(paths)] * len(batches), [ref.data] * len(batches), batches):
            res += part
    df = pl.DataFrame(res, infer_schema_length=None).sort(KEY + ["corruption", "i"])
    df.write_parquet(paths.processed / "corruption_results.parquet")
    summary["by_corruption"] = corruption_summary(df).to_dicts()

corrupt_stage = _with_state("validate-corrupt", _corrupt)


def corruption_summary(df: pl.DataFrame) -> pl.DataFrame:
    ok = df.filter(pl.col("eligible"))
    return (ok.group_by("corruption", "rule").agg(
        pl.len().alias("injections"), pl.col("detected").mean().round(4).alias("recall"),
        (pl.col("false_alarms").sum() / pl.col("n_swings").sum() * 100).round(4).alias("false_alarm_pct"))
        .sort("corruption"))


def _report(paths, state, summary, name="phase3"):
    from kata.validate.report import write_report
    summary["path"] = str(write_report(paths, state, name))

report_stage = _with_state("validate-report", _report)
```

`src/kata/validate/report.py`:

```python
"""Phase 3 Markdown report (spec §11, criterion 11)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from kata.config import Paths
from kata.data.report import md_table
from kata.validate.pipeline import corruption_summary
from kata.validate.rules import SCORED_RULES

YEAR_BINS = [(0, 2019, "≤2019"), (2020, 2021, "2020-21"), (2022, 2023, "2022-23"), (2024, 9999, "≥2024")]
NPS_BINS = [(0, 4, "0-4"), (4, 6, "4-6"), (6, 8, "6-8"), (8, 1e9, "8+")]
MIN_PER_SIDE = 20


def auc(pos, neg) -> float:
    """P(pos > neg) + ½ P(tie): Mann-Whitney U from average ranks."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def year_bin(year: int) -> str:
    return next(label for lo, hi, label in YEAR_BINS if lo <= year <= hi)


def nps_bin(nps: float) -> str:
    return next(label for lo, hi, label in NPS_BINS if lo <= nps < hi)


def stratified_auc(a: pl.DataFrame, d: pl.DataFrame, col: str = "score") -> tuple[pl.DataFrame, float]:
    """AUC within upload-year × NPS cells with >= MIN_PER_SIDE maps on each side, and the
    size-weighted mean over those cells (criterion 4)."""
    def cells(df):
        return df.with_columns(
            pl.col("year").map_elements(year_bin, return_dtype=pl.String).alias("years"),
            pl.col("nps").map_elements(nps_bin, return_dtype=pl.String).alias("nps_band"))
    a, d = cells(a), cells(d)
    rows = []
    for (years, band), ga in a.group_by("years", "nps_band"):
        gd = d.filter((pl.col("years") == years) & (pl.col("nps_band") == band))
        if ga.height >= MIN_PER_SIDE and gd.height >= MIN_PER_SIDE:
            rows.append({"years": years, "nps_band": band, "n_a": ga.height, "n_d": gd.height,
                         "auc": round(auc(ga[col], gd[col]), 3)})
    t = pl.DataFrame(rows, schema={"years": pl.String, "nps_band": pl.String, "n_a": pl.Int64,
                                   "n_d": pl.Int64, "auc": pl.Float64}).sort("years", "nps_band")
    w = (t["n_a"] + t["n_d"]).to_numpy()
    mean = float((t["auc"].to_numpy() * w).sum() / w.sum()) if t.height else float("nan")
    return t, mean


def write_report(paths: Paths, state, name: str) -> Path:
    P = paths.processed
    out = [f"# Phase 3 validator report — {name}", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data root `{paths.root}`.", ""]
    runs = state.con.execute("""
        SELECT stage, args, round(seconds, 1) AS seconds, summary FROM runs
        WHERE stage LIKE 'validate-%' OR stage LIKE 'baseline-%' OR stage LIKE '%lowrated%'
        QUALIFY row_number() OVER (PARTITION BY stage, args ORDER BY started_at DESC) = 1
        ORDER BY started_at""").pl()
    out += ["## 1. Stage timings (latest run of each)", "", md_table(runs.select("stage", "args", "seconds")), ""]
    last = {r["stage"]: json.loads(r["summary"] or "{}") for r in runs.iter_rows(named=True)}
    if "select-lowrated" in last:
        years = last["select-lowrated"].get("years", {})
        out += ["## 2. Tier D sample", "",
                f"- Pool {last['select-lowrated'].get('pool_maps')} maps, sampled "
                f"{last['select-lowrated'].get('sampled_maps')} ({last['select-lowrated'].get('sampled_diffs')} diffs)", "",
                md_table(pl.DataFrame([{"year": y, **v} for y, v in years.items()])) if years else "", ""]

    if not (P / "validation.parquet").exists():
        return _write(paths, name, out)
    v = pl.read_parquet(P / "validation.parquet")
    out += ["## 3. Gate and score per tier", "", md_table(v.group_by("tier").agg(
        pl.len().alias("n"), pl.col("pass").mean().alias("pass_rate"),
        pl.col("score").quantile(0.1).alias("q10"), pl.col("score").median().alias("median"),
        *(pl.col(f"s_{r}").mean().alias(f"mean_s_{r}") for r in SCORED_RULES)).sort("tier")), ""]

    a = v.filter((pl.col("tier") == "A") & pl.col("split").is_in(["val", "test"]))
    d = v.filter((pl.col("tier") == "D") & (pl.col("split") == "holdout"))
    if a.height and d.height:
        cells, mean = stratified_auc(a, d)
        per_rule = {r: round(auc(a[f"s_{r}"], d[f"s_{r}"]), 3) for r in SCORED_RULES}
        per_label = (v.filter(pl.col("tier").is_in(["A", "D"]))
                     .filter(((pl.col("tier") == "A") & pl.col("split").is_in(["val", "test"]))
                             | ((pl.col("tier") == "D") & (pl.col("split") == "holdout"))))
        label_rows = [{"difficulty": lab, "n_a": g.filter(pl.col("tier") == "A").height,
                       "n_d": g.filter(pl.col("tier") == "D").height,
                       "auc": round(auc(g.filter(pl.col("tier") == "A")["score"],
                                        g.filter(pl.col("tier") == "D")["score"]), 3)}
                      for (lab,), g in per_label.group_by("difficulty")]
        out += ["## 4. Separation: Tier A val+test vs. Tier D holdout (criterion 4)", "",
                f"- Gate pass: Tier A {a['pass'].mean():.4f} · Tier D {d['pass'].mean():.4f}",
                f"- **Overall AUC {auc(a['score'], d['score']):.3f}** (A {a.height}, D {d.height})",
                f"- **Per-stratum AUC, size-weighted mean {mean:.3f}** over {cells.height} cells "
                f"with ≥ {MIN_PER_SIDE} maps per side", "",
                md_table(cells), "", "Per-rule AUC (sub-scores):", "",
                md_table(pl.DataFrame([per_rule])), "", "Per difficulty label:", "",
                md_table(pl.DataFrame(label_rows).sort("difficulty")), ""]

    if (P / "corruption_results.parquet").exists():
        out += ["## 5. Synthetic corruptions (criterion 5)", "",
                md_table(corruption_summary(pl.read_parquet(P / "corruption_results.parquet"))), ""]

    if (P / "baseline_eval.parquet").exists():
        b = pl.read_parquet(P / "baseline_eval.parquet")
        q10 = (v.filter(pl.col("tier") == "A").group_by("stratum").agg(pl.col("score").quantile(0.1).alias("tier_a_q10")))
        bb = b.join(q10, on="stratum", how="left")
        out += ["## 6. Baseline (criterion 6)", "", md_table(bb.group_by("difficulty").agg(
            pl.len().alias("n"), pl.col("pass").mean().alias("pass_rate"), pl.col("score").median().alias("median_score"),
            pl.col("tier_a_q10").median().alias("tier_a_q10"), pl.col("nps").median().alias("median_nps"),
            pl.col("timing_f").median().alias("median_timing_f"), pl.col("round_trip").mean().alias("round_trip"))
            .sort("difficulty")), ""]

    v2 = v.filter(pl.col("format_version").str.starts_with("2") & pl.col("parity_errors").is_not_null())
    if v2.height:
        out += ["## 7. BeatSaver parity agreement (v2 only, diagnostic)", "", md_table(v2.group_by("tier").agg(
            pl.len().alias("n"), (pl.col("parity_errors") > 0).mean().alias("bs_errors"),
            (pl.col("E1") > 0).mean().alias("our_e1"),
            ((pl.col("parity_errors") > 0) == (pl.col("E1") > 0)).mean().alias("agree")).sort("tier")), ""]
    return _write(paths, name, out)


def _write(paths: Paths, name: str, out: list[str]) -> Path:
    path = paths.reports / f"{name}-{datetime.now():%Y-%m-%d}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run the test to verify it passes, several times**

Run: `for i in 1 2 3 4 5; do uv run pytest tests/test_validate_pipeline.py -q; done`
Expected: `2 passed` every time. (A flaky failure here means row order depends on process scheduling: Review Focus 1.)

- [ ] **Step 5: Commit**

```bash
git add src/kata/validate/pipeline.py src/kata/validate/report.py tests/test_validate_pipeline.py
git commit -m "Phase 3: validator stages and report"
```

---

### Task 9: Baseline stage and CLI

**Files:**
- Create: `src/kata/baseline/pipeline.py`
- Modify: `src/kata/cli.py`
- Test: `tests/test_baseline_pipeline.py`

**Interfaces:**
- Consumes: Tasks 2, 5, 6, 7 (`Paths.generated`), 8; Phase 2 `FeatureStore`, `f_measure`.
- Produces: `BASELINE_VERSION = "v1"`, `TIMING_TOL_S = 0.03`; `songs(paths, pilot=False) -> pl.DataFrame`; `generate_stage(paths, pilot=False)`; CLI commands `kata data select-lowrated | download-lowrated [--pilot] | parse-lowrated [--pilot]`, `kata validate analyse | reference | score | corrupt [--pilot] | report [--name] | pilot`, `kata baseline generate [--pilot]`.

- [ ] **Step 1: Write the failing test** — `tests/test_baseline_pipeline.py`:

```python
"""`kata baseline generate` on one synthetic val song with stored features."""
import numpy as np
import polars as pl

from kata.audio.frames import FPS
from kata.audio.store import CURVE_COLUMNS, FeatureStore
from kata.baseline import pipeline as bp
from kata.config import Paths
from kata.validate.reference import ValidatorDir

SHA = "cd" * 32


def test_generate_stage_writes_maps_and_eval(tmp_path, toy_ref):
    p = Paths(tmp_path).ensure()
    dur = 60.0
    T = int(dur * FPS) + 1
    curves = np.zeros((T, 12), np.float32)
    curves[:, CURVE_COLUMNS.index("rms_mix")] = -20
    for k in range(int((dur - 1) / 0.125)):
        curves[int(round((0.5 + 0.125 * k) * FPS)), CURVE_COLUMNS.index("onset_drums")] = 1.0 if k % 4 == 0 else 0.5
    arrays = {"mel": np.zeros((T, 128)), "stem_mel": np.zeros((T, 4, 64)), "chroma": np.zeros((T, 12)), "curves": curves}
    beats = 0.5 + 0.5 * np.arange(int(dur / 0.5) - 1)
    FeatureStore(p.features).write(SHA, arrays, beats, beats[::4])
    (p.maps / "h1").mkdir(parents=True)
    (p.maps / "h1" / "song.egg").write_bytes(b"OggS fake")
    pl.DataFrame({"hash": ["h1"], "audio_status": ["ok"], "split": ["val"], "pilot": [True], "audio_sha256": [SHA],
                  "audio_file": ["song.egg"], "song_name": ["Song"], "song_sub_name": [""],
                  "song_author": ["Artist"]}).write_parquet(p.processed / "maps.parquet")
    pl.DataFrame({"audio_sha256": [SHA], "status": ["ok"], "duration_s": [dur]}).write_parquet(
        p.processed / "audio_features.parquet")
    pl.DataFrame({"hash": ["h1"], "tier": ["A"], "parse_status": ["ok"]}).write_parquet(
        p.processed / "difficulties.parquet")
    pl.DataFrame({"audio_sha256": [SHA, SHA], "seg_idx": [0, 1], "start_s": [0.0, 30.0], "end_s": [30.0, dur]}
                 ).write_parquet(p.processed / "audio_segments.parquet")
    human = 2.0 + 0.5 * np.arange(100)
    pl.DataFrame({"hash": ["h1"] * 100, "difficulty": ["Expert"] * 100, "time_s": human}).write_parquet(
        p.processed / "notes.parquet")
    vd = ValidatorDir(p.validator)
    vd.open_for_write()
    vd.write_reference(toy_ref)

    s = bp.generate_stage(p)
    ev = pl.read_parquet(p.processed / "baseline_eval.parquet")
    assert s["songs"] == 1 and ev.height == 5
    assert ev["pass"].all() and ev["round_trip"].all()
    assert ev.filter(pl.col("difficulty") == "Expert")["timing_f"][0] > 0.3
    assert ev.filter(pl.col("difficulty") == "Easy")["timing_f"][0] is None    # no human Easy map
    folder = p.generated / "baseline" / bp.BASELINE_VERSION / SHA
    assert (folder / "Info.dat").exists() and folder.with_suffix(".zip").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_baseline_pipeline.py -q`
Expected: `ImportError: cannot import name 'pipeline' from 'kata.baseline'`.

- [ ] **Step 3: Implement** — `src/kata/baseline/pipeline.py`:

```python
"""`kata baseline generate` (spec §8.6): val + test songs -> map folders, zips and baseline_eval."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from kata.audio.beats import f_measure
from kata.audio.store import FeatureStore
from kata.baseline.export import round_trip, write_map
from kata.baseline.generate import SongInput, generate
from kata.config import Paths
from kata.data.pipeline import _check_disk, _with_state
from kata.validate.reference import Reference, ValidatorDir
from kata.validate.rules import SCORED_RULES

BASELINE_VERSION = "v1"
TIMING_TOL_S = 0.03


def songs(paths: Paths, pilot: bool = False) -> pl.DataFrame:
    """One row per val/test audio file with features: the map row whose metadata it uses (smallest
    hash), its duration, and the Tier A map hashes of that song for the timing comparison."""
    P = paths.processed
    maps = pl.read_parquet(P / "maps.parquet").filter(
        (pl.col("audio_status") == "ok") & pl.col("split").is_in(["val", "test"]))
    if pilot:
        maps = maps.filter(pl.col("pilot"))
    feats = pl.read_parquet(P / "audio_features.parquet").filter(pl.col("status") == "ok").select(
        "audio_sha256", "duration_s")
    tier_a = (pl.read_parquet(P / "difficulties.parquet").filter((pl.col("tier") == "A") & (pl.col("parse_status") == "ok"))
              .select("hash").unique())
    a_hashes = (maps.join(tier_a, on="hash").group_by("audio_sha256").agg(pl.col("hash").sort().alias("tier_a_hashes")))
    first = maps.sort("hash").group_by("audio_sha256", maintain_order=True).first()
    return (first.select("audio_sha256", "hash", "audio_file", "song_name", "song_sub_name", "song_author", "split")
            .join(feats, on="audio_sha256").join(a_hashes, on="audio_sha256", how="left").sort("audio_sha256"))


def _one(root: str, ref_data: dict, song: dict, segments: list[tuple[float, float]],
         human: dict[str, list[float]]) -> list[dict]:
    paths = Paths(Path(root))
    ref = Reference(ref_data)
    store = FeatureStore(paths.features)
    sha = song["audio_sha256"]
    beats, downbeats = store.events(sha)
    inp = SongInput(sha, song["duration_s"], np.asarray(store.load(sha, "curves"), np.float32),
                    beats, downbeats, segments)
    grid, out, results = generate(inp, ref)
    folder = paths.generated / "baseline" / BASELINE_VERSION / sha
    write_map(folder, paths.maps / song["hash"] / song["audio_file"], song, grid, out, song["duration_s"])
    try:
        round_trip(folder, out)
        ok = True
    except AssertionError:
        ok = False
    rows = []
    for label, res in results.items():
        t = np.unique(out[label].notes["time_s"].to_numpy())
        ref_t = human.get(label)
        rows.append({"audio_sha256": sha, "split": song["split"], "difficulty": label,
                     "n_notes": out[label].notes.height, "nps": res.metrics["nps"], "pass": res.passed,
                     "score": res.score, "stratum": res.stratum, **{f"s_{r}": res.sub_scores[r] for r in SCORED_RULES},
                     "timing_f": f_measure(np.unique(ref_t), t, TIMING_TOL_S) if ref_t else None,
                     "grid_bpm": grid.bpm, "piecewise": grid.piecewise, "round_trip": ok})
    return rows


def _generate(paths, state, summary, pilot=False):
    ref = ValidatorDir(paths.validator).load_reference()
    todo = songs(paths, pilot)
    segs = pl.read_parquet(paths.processed / "audio_segments.parquet").sort("audio_sha256", "seg_idx")
    seg_map = {k[0]: list(zip(g["start_s"], g["end_s"])) for k, g in segs.partition_by("audio_sha256", as_dict=True).items()}
    hashes = [h for hs in todo["tier_a_hashes"].drop_nulls() for h in hs]
    notes = (pl.scan_parquet(paths.processed / "notes.parquet").filter(pl.col("hash").is_in(hashes))
             .select("hash", "difficulty", "time_s").collect())
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = []
        for song in todo.iter_rows(named=True):
            hs = song["tier_a_hashes"] or []
            human = {}
            for label, g in notes.filter(pl.col("hash").is_in(hs[:1])).partition_by("difficulty", as_dict=True).items():
                human[label[0]] = g["time_s"].to_list()
            futures.append(pool.submit(_one, str(paths.root), ref.data, song, seg_map.get(song["audio_sha256"], []), human))
        for i, f in enumerate(futures):
            rows += f.result()
            if (i + 1) % 50 == 0:
                _check_disk(paths)
                print(f"  [baseline-generate] {i + 1}/{len(futures)}", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None).sort("audio_sha256", "difficulty")
    df.write_parquet(paths.processed / "baseline_eval.parquet")
    summary.update(songs=todo.height, maps_written=todo.height,
                   by_difficulty=df.group_by("difficulty").agg(
                       pl.col("pass").mean().alias("pass"), pl.col("score").median().alias("median_score"),
                       pl.col("timing_f").median().alias("median_timing_f")).sort("difficulty").to_dicts())

generate_stage = _with_state("baseline-generate", _generate)
```

`src/kata/cli.py`:

```diff
diff --git a/src/kata/cli.py b/src/kata/cli.py
index 557a196..884d59a 100644
--- a/src/kata/cli.py
+++ b/src/kata/cli.py
@@ -15,6 +15,10 @@ data = typer.Typer(no_args_is_help=True, help="Phase 1 data pipeline stages.")
 app.add_typer(data, name="data")
 audio = typer.Typer(no_args_is_help=True, help="Phase 2 audio feature stages (needs `uv sync --extra audio`).")
 app.add_typer(audio, name="audio")
+validate = typer.Typer(no_args_is_help=True, help="Phase 3 validator stages.")
+app.add_typer(validate, name="validate")
+baseline = typer.Typer(no_args_is_help=True, help="Phase 3 rule-based baseline mapper.")
+app.add_typer(baseline, name="baseline")
 
 PILOT = typer.Option(False, "--pilot", help="Restrict to the deterministic pilot subset.")
 
@@ -131,3 +135,87 @@ def audio_pilot():
     from kata.audio import pipeline as ap
     for stage, fn, kwargs in ap.PILOT_STAGES:
         _run(stage, fn, **kwargs)
+
+
+# --------------------------------------------------------------------------- Phase 3
+# Tier D lives in kata.data.lowrated; the validator and baseline import only numpy/polars, so
+# none of these commands needs the audio extra.
+
+@data.command("select-lowrated")
+def select_lowrated():
+    """Tier D: year-matched sample of low-rated maps from the stored catalogue."""
+    from kata.data import lowrated
+    _run("select-lowrated", lowrated.select_lowrated)
+
+
+@data.command("download-lowrated")
+def download_lowrated(pilot: bool = PILOT):
+    """Download Tier D maps without their audio (resumable)."""
+    from kata.data import lowrated
+    _run("download-lowrated", lowrated.download_lowrated, pilot=pilot)
+
+
+@data.command("parse-lowrated")
+def parse_lowrated(pilot: bool = PILOT):
+    """Parse Tier D into processed/lowrated/ and assign calibration halves."""
+    from kata.data import lowrated
+    _run("parse-lowrated", lowrated.parse_lowrated, pilot=pilot)
+
+
+@validate.command("analyse")
+def validate_analyse():
+    """Swings, issues and per-map / per-window metrics for every parsed difficulty (A-D)."""
+    from kata.validate import pipeline as vp
+    _run("validate-analyse", vp.analyse_stage)
+
+
+@validate.command("reference")
+def validate_reference():
+    """Tier A train quantiles -> validator/v1/reference.json."""
+    from kata.validate import pipeline as vp
+    _run("validate-reference", vp.reference_stage)
+
+
+@validate.command("score")
+def validate_score():
+    """Sub-scores, scores and the gate -> validation.parquet, validation_local.parquet."""
+    from kata.validate import pipeline as vp
+    _run("validate-score", vp.score_stage)
+
+
+@validate.command("corrupt")
+def validate_corrupt(pilot: bool = PILOT):
+    """Synthetic corruptions of Tier A val+test -> corruption_results.parquet."""
+    from kata.validate import pipeline as vp
+    _run("validate-corrupt", vp.corrupt_stage, pilot=pilot)
+
+
+@validate.command("report")
+def validate_report(name: str = typer.Option("phase3", help="Report file name prefix.")):
+    from kata.validate import pipeline as vp
+    _run("validate-report", vp.report_stage, name=name)
+
+
+@baseline.command("generate")
+def baseline_generate(pilot: bool = PILOT):
+    """Baseline maps for every val+test song -> generated/baseline/v1/, baseline_eval.parquet."""
+    from kata.baseline import pipeline as bp
+    _run("baseline-generate", bp.generate_stage, pilot=pilot)
+
+
+@validate.command("pilot")
+def validate_pilot():
+    """Phase 3 pilot: Tier D pilot maps, every validator stage and the baseline on the pilot subset."""
+    from kata.baseline import pipeline as bp
+    from kata.data import lowrated
+    from kata.validate import pipeline as vp
+    for stage, fn, kwargs in [("select-lowrated", lowrated.select_lowrated, {}),
+                              ("download-lowrated", lowrated.download_lowrated, {"pilot": True}),
+                              ("parse-lowrated", lowrated.parse_lowrated, {"pilot": True}),
+                              ("validate-analyse", vp.analyse_stage, {}),
+                              ("validate-reference", vp.reference_stage, {}),
+                              ("validate-score", vp.score_stage, {}),
+                              ("validate-corrupt", vp.corrupt_stage, {"pilot": True}),
+                              ("baseline-generate", bp.generate_stage, {"pilot": True}),
+                              ("validate-report", vp.report_stage, {"name": "phase3-pilot"})]:
+        _run(stage, fn, **kwargs)
```

- [ ] **Step 4: Run the whole suite and lint**

Run: `uv run pytest -q`
Expected: `134 passed, 7 skipped` without the audio extra (Phase 2 GPU/torch tests skip); more pass with `--extra audio`.

Run: `uvx ruff check --select F,E9 --line-length 120 src tests`
Expected: `All checks passed!`

Run: `uv run kata validate --help` and `uv run kata baseline --help`
Expected: the commands listed under **Interfaces**.

- [ ] **Step 5: Commit**

```bash
git add src/kata/baseline/pipeline.py src/kata/cli.py tests/test_baseline_pipeline.py
git commit -m "Phase 3: baseline generate stage and CLI"
```

---

### Task 10: Pilot run and findings (checkpoint with the user)

**Files:**
- Modify: `docs/phases/phase-3-validator.md` (add §18 Pilot findings; update criteria only with the user's approval)

- [ ] **Step 1: Run the pilot** (network: downloads 200 Tier D maps without audio, a few minutes at 4 req/s)

Run: `uv run kata validate pilot`
Expected: every stage prints its summary JSON; the last line names `reports/phase3-pilot-YYYY-MM-DD.md`.

- [ ] **Step 2: Read the report and record findings in spec §18**, covering every **[verify in pilot]** / **[tune in pilot]** item:
  - Tier D parse rate (criterion 1 bar: ≥ 99 % of difficulties) and year mix;
  - gate pass rate, Tier A val+test (criterion 3: ≥ 99 %);
  - overall and per-stratum AUC with the 200 pilot Tier D maps (criterion 4) and per-rule AUCs;
  - corruption recall (criterion 5); for C1 and C5, inspect missed injections as in spec §17 and say whether the rule or the corruption is at fault;
  - baseline: pass rate, median score against Tier A q10, timing F;
  - `analyse` time per difficulty (criterion 8: < 100 ms median).

- [ ] **Step 3: ArcViewer check.** Pick 5 pilot baseline zips covering slow, fast and unsteady songs (`baseline_eval.grid_bpm`, `piecewise`). The user opens each at https://allpoland.github.io/ArcViewer/ through the file picker and plays all five difficulties (criterion 7).

- [ ] **Step 4: Checkpoint.** Present the findings and any proposed changes (thresholds, weights, the C1/C5 bar) to the user. Apply only what the user approves; log each change in spec §15. A parameter change means a new `VALIDATOR_VERSION` (Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add docs/phases/phase-3-validator.md
git commit -m "Phase 3: pilot findings (spec §18)"
```

---

### Task 11: Full run and acceptance

**Files:**
- Modify: `docs/phases/phase-3-validator.md` (§19 build results, status), `docs/PLAN.md` (Phase 3 ✅)

- [ ] **Step 1: Tier D, full sample** (network, ~2,000 downloads at 4 concurrent, under an hour)

Run: `uv run kata data download-lowrated && uv run kata data parse-lowrated && uv run kata data build`
Expected: parse summary `diff_status` with ≥ 99 % `ok`; `processed/lowrated/` under 1 GB; `lowrated_*` views in `kata.duckdb`.

- [ ] **Step 2: Validator chain**

Run: `uv run kata validate analyse && uv run kata validate reference && uv run kata validate score && uv run kata validate corrupt`
Expected: `analyse` about 2 minutes, 0 errors; `reference` built from Tier A train only; `corrupt` about 5 minutes.

- [ ] **Step 3: Baseline**

Run: `uv run kata baseline generate`
Expected: one folder + zip per val/test song under `generated/baseline/v1/` (~500 songs, a few GB, mostly copied audio).

- [ ] **Step 4: Report and idempotency**

Run: `uv run kata validate report`, then re-run `uv run kata validate analyse` and compare the four `validation_*` tables byte for byte with the previous run (criterion 10).

- [ ] **Step 5: Acceptance.** Fill spec §19 with a table of criteria 1–11 and their results, plus notes on any criterion that needs one. Set the status line to DONE and mark Phase 3 ✅ in `docs/PLAN.md` with a one-line results summary.

- [ ] **Step 6: Commit**

```bash
git add docs/phases/phase-3-validator.md docs/PLAN.md
git commit -m "Phase 3: v1 validator and baseline build results"
```
