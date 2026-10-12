"""Phase 4 stages on a tiny synthetic data root."""
import json

import numpy as np
import polars as pl
import pytest

from kata.config import Paths
from kata.data.pipeline import leaderboard_diffs
from kata.motion import pipeline as mp
from kata.motion.prepare import MotionStore
from motion_synth import write_root

KEY = ["hash", "characteristic", "difficulty"]


@pytest.fixture
def paths(tmp_path) -> Paths:
    p = Paths(tmp_path).ensure()
    write_root(p)
    return p


def test_prepare_stage_tracks_rejects_stats_and_idempotency(paths):
    s = mp.prepare_stage(paths)
    t = pl.read_parquet(paths.processed / "motion_tracks.parquet")
    assert t["score_id"].to_list() == ["1", "2", "4", "5"]              # hash mismatch never considered
    assert s["tracks"] == {"ok": 3, "rejected": 1}
    assert s["reject_reasons"] == {"blade_check": 1}
    assert s["left_handed_recovered"] == 1
    ok = t.filter(pl.col("status") == "ok")
    assert set(ok["split"]) == {"train", "val"}
    assert ok["nps"].n_unique() == 1 and ok["hps"][0] == pytest.approx(ok["nps"][0])   # no stacks
    store = MotionStore(paths.motion)
    assert all(store.has(sid) for sid in ok["score_id"])
    stats = json.loads(store.stats_path.read_text())
    assert stats["n_tracks"] == 2                                     # train only: 1 and 2
    first = t
    mp.prepare_stage(paths)
    assert pl.read_parquet(paths.processed / "motion_tracks.parquet").equals(first)


def test_leaderboard_crawl_can_be_limited_to_one_tier(paths):
    assert sorted(leaderboard_diffs(paths)["hash"]) == ["h1", "h2", "h3"]
    assert sorted(leaderboard_diffs(paths, tier="A")["hash"]) == ["h1", "h2"]
    assert np.array_equal(leaderboard_diffs(paths, tier="B")["hash"].to_numpy(), ["h3"])
