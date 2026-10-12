"""Phase 4 stages on a tiny synthetic data root."""
import json
import re

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


TINY_RUN = dict(batch=2, warmup=1, total_steps=2, eval_every=2, eval_diffs=1, ddim_steps=2, workers=0)


@pytest.fixture
def trained(paths) -> Paths:
    """A tiny run 'r1' trained for two steps on CPU, its guidance tuned and the val split sampled."""
    pytest.importorskip("torch")
    from kata.motion.model import ModelConfig
    from kata.motion.train import RunConfig
    mp.prepare_stage(paths)
    s = mp.train_stage(paths, run="r1", config=RunConfig(model=ModelConfig(d=32, heads=4, enc_layers=1,
                                                                             dec_layers=1), **TINY_RUN),
                       device="cpu")
    assert s["step"] == 2
    t = mp.tune_stage(paths, run="r1", n=1, values=(1.0, 2.0), device="cpu")
    assert t["guidance"] in (1.0, 2.0) and len(t["table"]) == 2
    assert json.loads((paths.motion_runs / "r1" / "guidance.json").read_text())["guidance"] == t["guidance"]
    s = mp.sample_stage(paths, run="r1", split="val", device="cpu")
    assert s["jobs"] == 3 and s["targets"] == 1                     # full seeds 0 and 1, label-only seed 0
    return paths


def test_sampling_resumes_evaluation_replaces_and_generate_works(trained):
    from kata.motion.sample import generate
    paths = trained
    assert mp.sample_stage(paths, run="r1", split="val", device="cpu")["jobs"] == 0   # resumable
    e = mp.evaluate_stage(paths, run="r1", split="val")
    assert [c["criterion"] for c in e["criteria"]] == [2, 3, 4, 5, 6]   # no baseline_eval: criterion 1 skipped
    ev = pl.read_parquet(paths.processed / "motion_eval.parquet")
    assert set(ev["source"]) == {"generated", "real"} and ev.filter(pl.col("source") == "generated").height == 3
    assert ev.filter(pl.col("source") == "real").height == 1
    mp.evaluate_stage(paths, run="r1", split="val")                  # replaces, never duplicates
    assert pl.read_parquet(paths.processed / "motion_eval.parquet").height == ev.height
    assert (paths.motion_runs / "r1" / "eval_val.json").exists()
    (paths.motion_runs / "r1" / "eval_test.json").write_text("{}")
    with pytest.raises(ValueError, match="already evaluated on test"):     # test is evaluated once
        mp.evaluate_stage(paths, run="r1", split="test")
    with pytest.raises(FileNotFoundError):
        mp.final_stage(paths, run="nope")
    mp.final_stage(paths, run="r1")
    track = generate("b" * 64, "Expert", 2.0, paths=paths, device="cpu")
    assert track.left_rot.shape == (len(track.t), 3, 3) and track.t[1] == pytest.approx(0.02)


def test_viewer_pages_and_report(trained):
    paths = trained
    mp.evaluate_stage(paths, run="r1", split="val")
    mp.final_stage(paths, run="r1")
    v = mp.viewer_stage(paths, run="r1", split="val", n=1)
    page = open(v["pages"][0], encoding="utf-8").read()
    assert "three@0.160.0" in page and '"blind":true' in page
    data = json.loads(re.search(r"const DATA = (\{.*?\});$", page, re.M).group(1))
    assert data["clips"][0]["notes"] == data["clips"][1]["notes"] != []     # notes never reveal the real clip
    r = mp.report_stage(paths, name="t")
    text = open(r["path"], encoding="utf-8").read()
    assert "Acceptance criteria" in text and "Blade check" in text and "| r1 |" in text
