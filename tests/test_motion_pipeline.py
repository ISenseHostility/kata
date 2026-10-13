"""Phase 4 stages on a tiny synthetic data root."""
import base64
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


def test_tracks_are_scaled_to_a_head_height():
    from kata.motion.prepare import HEAD_Y_REF_M
    from kata.motion.sample import to_track
    motion = np.zeros((5, 27), np.float32)
    for s in (0, 9, 18):                                   # identity rot6 for head, left, right
        motion[:, s + 3] = motion[:, s + 7] = 1.0
    motion[:, 0:3] = [0.1, 1.6, 0.2]
    motion[:, 9:12] = [0.3, 1.0, 0.4]
    ref = to_track(motion, np.array([]), np.array([]))
    tall = to_track(motion, np.array([]), np.array([]), head_height=1.1 * HEAD_Y_REF_M)
    assert ref.head_pos[0, 1] == pytest.approx(1.6) and ref.left_pos[0, 1] == pytest.approx(1.0)
    assert tall.head_pos[0, 1] == pytest.approx(1.76) and tall.left_pos[0, 1] == pytest.approx(1.1)
    assert tall.left_pos[0, 0] == pytest.approx(0.3) and tall.left_pos[0, 2] == pytest.approx(0.4)


def test_viewer_pages_and_report(trained):
    paths = trained
    mp.evaluate_stage(paths, run="r1", split="val")
    mp.final_stage(paths, run="r1")
    v = mp.viewer_stage(paths, run="r1", split="val", n=1)
    page = open(v["pages"][0], encoding="utf-8").read()
    assert "three@0.160.0" in page and '"blind":true' in page
    data = json.loads(re.search(r"const DATA = (\{.*?\});$", page, re.M).group(1))
    # Only the real clip was played against the map, so its sabers meet the notes: in blind mode the
    # notes stay hidden until Reveal, and neither clip carries them.
    assert data["notes"] != [] and all("notes" not in c for c in data["clips"])
    assert "showNotes = !DATA.blind" in page
    r = mp.report_stage(paths, name="t")
    text = open(r["path"], encoding="utf-8").read()
    assert "Acceptance criteria" in text and "Blade check" in text and "| r1 |" in text
    assert "Hand height at hits vs. median head height" in text


def test_viewer_pages_carry_the_song_audio(trained):
    # The pages stay local (§12), so they carry the song: no file to pick before playing.
    paths = trained
    page = open(mp.viewer_stage(paths, run="r1", split="val", n=1)["pages"][0], encoding="utf-8").read()
    assert '<audio id="audio"></audio>' in page and '<label id="pick">' in page   # no audio on disk: picker
    song = pl.read_parquet(paths.processed / "maps.parquet").filter(pl.col("split") == "val").row(0, named=True)
    (paths.maps / song["hash"]).mkdir(parents=True, exist_ok=True)
    (paths.maps / song["hash"] / song["audio_file"]).write_bytes(b"OggS fake vorbis")
    page = open(mp.viewer_stage(paths, run="r1", split="val", n=1)["pages"][0], encoding="utf-8").read()
    assert f'<audio id="audio" src="data:audio/ogg;base64,{base64.b64encode(b"OggS fake vorbis").decode()}">' in page
    assert 'id="pick" hidden' in page


def _page_labels(path) -> list:
    page = open(path, encoding="utf-8").read()
    return json.loads(re.search(r"const DATA = (\{.*?\});$", page, re.M).group(1))["labels"]


def test_blind_order_is_not_fixed_by_the_page_index(trained):
    orders = {tuple(_page_labels(mp.viewer_stage(trained, run="r1", split="val", n=1)["pages"][0]))
              for _ in range(20)}
    assert orders == {("generated", "real"), ("real", "generated")}


def test_stale_samples_are_regenerated(trained):
    paths = trained
    run_dir = paths.motion_runs / "r1"
    g = json.loads((run_dir / "guidance.json").read_text())
    (run_dir / "guidance.json").write_text(json.dumps({**g, "guidance": g["guidance"] + 1.0}))
    assert mp.sample_stage(paths, run="r1", split="val", device="cpu")["jobs"] == 3       # new guidance
    assert mp.sample_stage(paths, run="r1", split="val", device="cpu")["jobs"] == 0
    import os
    best = run_dir / "ckpt" / "best.pt"
    st = best.stat()
    os.utime(best, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))                          # a new best.pt
    assert mp.sample_stage(paths, run="r1", split="val", device="cpu")["jobs"] == 3


def test_evaluate_refuses_missing_or_stale_samples(trained):
    paths = trained
    run_dir = paths.motion_runs / "r1"
    sample = next((paths.generated / "motion" / "r1" / "val").glob("*_full_1.npz"))
    sample.unlink()
    with pytest.raises(FileNotFoundError, match="1 of 3 samples"):
        mp.evaluate_stage(paths, run="r1", split="val")
    assert not (run_dir / "eval_val.json").exists()
    mp.sample_stage(paths, run="r1", split="val", device="cpu")
    g = json.loads((run_dir / "guidance.json").read_text())
    (run_dir / "guidance.json").write_text(json.dumps({**g, "guidance": g["guidance"] + 1.0}))
    with pytest.raises(FileNotFoundError, match="3 of 3 samples"):
        mp.evaluate_stage(paths, run="r1", split="val")
