"""`kata` command line. Data stages live under `kata data …`."""

from __future__ import annotations

import asyncio
import json

import typer

from kata.config import Paths
from kata.data import pipeline

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
data = typer.Typer(no_args_is_help=True, help="Phase 1 data pipeline stages.")
app.add_typer(data, name="data")

PILOT = typer.Option(False, "--pilot", help="Restrict to the deterministic pilot subset.")


def _run(stage: str, fn, *args, **kwargs) -> None:
    paths = Paths().ensure()
    result = fn(paths, *args, **kwargs)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    typer.echo(f"[{stage}] " + json.dumps(result, indent=1, default=str))


@data.command("crawl-catalog")
def crawl_catalog():
    """Crawl all BeatSaver map metadata (resumable)."""
    _run("crawl-catalog", pipeline.crawl_catalog)


@data.command("select")
def select():
    """Flatten the catalogue and assign quality tiers (+ pilot subset)."""
    _run("select", pipeline.select)


@data.command("download-maps")
def download_maps(pilot: bool = PILOT, limit: int = typer.Option(0, help="Max maps (0 = all).")):
    _run("download-maps", pipeline.download_maps, pilot=pilot, limit=limit)


@data.command("parse-maps")
def parse_maps(pilot: bool = PILOT):
    _run("parse-maps", pipeline.parse_maps, pilot=pilot)


@data.command("crawl-leaderboards")
def crawl_leaderboards(pilot: bool = PILOT):
    _run("crawl-leaderboards", pipeline.crawl_leaderboards, pilot=pilot)


@data.command("select-replays")
def select_replays(pilot: bool = PILOT, k: int = typer.Option(3, help="Replays per difficulty.")):
    _run("select-replays", pipeline.select_replays, pilot=pilot, k=k)


@data.command("download-replays")
def download_replays(pilot: bool = PILOT):
    _run("download-replays", pipeline.download_replays, pilot=pilot)


@data.command("parse-replays")
def parse_replays(pilot: bool = PILOT):
    _run("parse-replays", pipeline.parse_replays, pilot=pilot)


@data.command("build")
def build():
    """Song groups, splits, DuckDB views."""
    _run("build", pipeline.build)


@data.command("report")
def report(name: str = typer.Option("phase1", help="Report file name prefix.")):
    _run("report", pipeline.report, name=name)


@data.command("pilot")
def pilot():
    """Run every stage on the pilot subset, then write the pilot report."""
    for stage, fn, kwargs in pipeline.PILOT_STAGES:
        _run(stage, fn, **kwargs)
