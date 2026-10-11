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
