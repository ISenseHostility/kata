"""`kata motion …` commands are wired to their stages."""
from typer.testing import CliRunner

from kata.cli import app
from kata.config import Paths
from motion_synth import write_root


def test_motion_commands_are_registered():
    out = CliRunner().invoke(app, ["motion", "--help"]).output
    for cmd in ("prepare", "train", "tune", "sample", "evaluate", "final", "viewer", "report"):
        assert cmd in out
    assert "--tier" in CliRunner().invoke(app, ["data", "crawl-leaderboards", "--help"]).output


def test_motion_prepare_runs_from_the_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("KATA_DATA", str(tmp_path))
    write_root(Paths(tmp_path).ensure())
    res = CliRunner().invoke(app, ["motion", "prepare"])
    assert res.exit_code == 0, res.output
    assert '"ok": 3' in res.output and (tmp_path / "processed" / "motion_tracks.parquet").exists()


def test_train_runs_batch_64_as_micro_batches_of_32_by_default(monkeypatch):
    # Batch 64 needs ~16.8 GiB; the 16 GB training GPU fits 32 per pass (pilot, spec §16).
    from kata.motion import pipeline as mp
    seen = []
    monkeypatch.setattr(mp, "train_stage", lambda paths, **kw: seen.append(kw["config"]) or {})
    assert CliRunner().invoke(app, ["motion", "train", "r"]).exit_code == 0
    assert CliRunner().invoke(app, ["motion", "train", "r", "--micro-batch", "16"]).exit_code == 0
    assert (seen[0].batch, seen[0].micro_batch, seen[1].micro_batch) == (64, 32, 16)
