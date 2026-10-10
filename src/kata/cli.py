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
audio = typer.Typer(no_args_is_help=True, help="Phase 2 audio feature stages (needs `uv sync --extra audio`).")
app.add_typer(audio, name="audio")

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


# --------------------------------------------------------------------------- Phase 2
# kata.audio needs the optional `audio` extra, so it is imported inside each command.

@audio.command("features")
def audio_features(pilot: bool = PILOT, limit: int = typer.Option(0, help="Max songs (0 = all)."),
                   device: str = typer.Option("cuda", help="Torch device.")):
    """Stems, spectral features, beats per unique audio file (GPU, resumable)."""
    from kata.audio import pipeline as ap
    _run("audio-features", ap.features, pilot=pilot, limit=limit, device=device)


@audio.command("structure")
def audio_structure(pilot: bool = PILOT):
    """Segment boundaries + repetition labels from the stored features."""
    from kata.audio import pipeline as ap
    _run("audio-structure", ap.structure, pilot=pilot)


@audio.command("sync")
def audio_sync(pilot: bool = PILOT):
    """Map <-> audio lag and map-grid vs. audio-beat agreement per difficulty."""
    from kata.audio import pipeline as ap
    _run("audio-sync", ap.sync, pilot=pilot)


@audio.command("stats")
def audio_stats(pilot: bool = PILOT):
    """Train-split normalisation statistics -> features/v1/stats.json."""
    from kata.audio import pipeline as ap
    _run("audio-stats", ap.stats, pilot=pilot)


@audio.command("report")
def audio_report(name: str = typer.Option("phase2", help="Report file name prefix.")):
    from kata.audio import pipeline as ap
    _run("audio-report", ap.report, name=name)


@audio.command("pilot")
def audio_pilot():
    """Run every Phase 2 stage on the pilot subset, then write the pilot report."""
    from kata.audio import pipeline as ap
    for stage, fn, kwargs in ap.PILOT_STAGES:
        _run(stage, fn, **kwargs)
