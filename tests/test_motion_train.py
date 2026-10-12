"""Runs, checkpoints, exact resume and the time limit (spec §6.3-6.4), on CPU with a tiny model."""
import json

import pytest

torch = pytest.importorskip("torch")

from kata.config import Paths  # noqa: E402
from kata.motion import pipeline as mp  # noqa: E402
from kata.motion.model import ModelConfig  # noqa: E402
from kata.motion.train import Run, RunConfig, lr_factor, train  # noqa: E402
from motion_synth import write_root  # noqa: E402

TINY = RunConfig(model=ModelConfig(d=32, heads=4, enc_layers=1, dec_layers=1, dropout=0.1), batch=2, warmup=2,
                 total_steps=6, eval_every=3, eval_diffs=2, ddim_steps=2, workers=0, log_every=1)


@pytest.fixture
def paths(tmp_path) -> Paths:
    p = Paths(tmp_path).ensure()
    write_root(p)
    mp.prepare_stage(p)
    return p


def _state(paths, run_id):
    return torch.load(Run(paths, run_id).ckpt_dir / "last.pt", weights_only=True)


def test_resumed_training_is_identical_to_straight_training(paths):
    a = train(paths, "straight", TINY, device="cpu", max_steps=4)
    assert a["step"] == 4 and a["stopped"] == "max_steps"
    train(paths, "resumed", TINY, device="cpu", max_steps=2)
    b = train(paths, "resumed", resume=True, device="cpu", max_steps=4)
    assert b["step"] == 4
    sa, sb = _state(paths, "straight"), _state(paths, "resumed")
    for part in ("model", "ema"):
        assert all(torch.equal(sa[part][k], sb[part][k]) for k in sa[part]), part
    assert torch.equal(sa["rng"]["cpu"], sb["rng"]["cpu"])


def test_micro_batches_train_like_the_full_batch(paths):
    # Batch 64 does not fit a 16 GB GPU; micro-batches of 32 must give the same steps (pilot, §16).
    no_dropout = ModelConfig(**{**TINY.model.__dict__, "dropout": 0.0})
    cfg = RunConfig(**{**TINY.__dict__, "model": no_dropout})
    train(paths, "full", cfg, device="cpu", max_steps=3)
    train(paths, "micro", RunConfig(**{**cfg.__dict__, "micro_batch": 1}), device="cpu", max_steps=3)
    sa, sb = _state(paths, "full"), _state(paths, "micro")
    assert all(torch.allclose(sa["model"][k], sb["model"][k], atol=1e-5) for k in sa["model"])
    assert torch.equal(sa["rng"]["cpu"], sb["rng"]["cpu"])


def test_run_guards_and_logs(paths):
    train(paths, "r", TINY, device="cpu", max_steps=3)
    with pytest.raises(ValueError, match="resume"):
        train(paths, "r", TINY, device="cpu")
    with pytest.raises(FileNotFoundError):
        train(paths, "missing", resume=True, device="cpu")
    other = RunConfig(**{**TINY.__dict__, "lr": 1e-3})
    with pytest.raises(ValueError, match="different config"):
        Run.create(paths, "r", other)
    run = Run(paths, "r")
    rows = [json.loads(line) for line in run.log_path.read_text().splitlines()]
    assert [r["step"] for r in rows if "loss" in r] == [1, 2, 3]
    ev = [r["eval"] for r in rows if "eval" in r]
    assert len(ev) == 1 and ev[0]["n"] == 1 and 0.0 <= ev[0]["val_timing_f"] <= 1.0
    assert (run.ckpt_dir / "best.pt").exists()
    assert run.load_model("best").config == TINY.model
    assert run.motion_stats_path.exists() and run.audio_stats_path.exists()


def test_time_limit_stops_cleanly_with_a_checkpoint(paths):
    s = train(paths, "t", TINY, device="cpu", hours=0.0)
    assert s["stopped"] == "time_limit" and s["step"] == 1
    assert _state(paths, "t")["step"] == 1


def test_lr_schedule():
    f = lr_factor(RunConfig(warmup=10, total_steps=110, min_lr_frac=0.1))
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(60) == pytest.approx(0.55) and f(110) == pytest.approx(0.1)


def test_overfit_mode_trains_and_checks_on_the_chosen_song_only(paths):
    cfg = RunConfig(**{**TINY.__dict__, "only_songs": ("a" * 64,)})
    assert train(paths, "o", cfg, device="cpu", max_steps=2)["step"] == 2
    run = Run(paths, "o")
    assert run.config().only_songs == ("a" * 64,)                 # survives the JSON round trip
    assert [sha for sha, _ in run.sampler("train").songs] == ["a" * 64]
    assert run.sampler("val").songs == []
    ev = [json.loads(x)["eval"] for x in run.log_path.read_text().splitlines() if '"eval"' in x]
    assert ev[-1]["n"] == 1                                       # the quick check ran on the train song
