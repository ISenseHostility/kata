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
