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
