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
validate = typer.Typer(no_args_is_help=True, help="Phase 3 validator stages.")
app.add_typer(validate, name="validate")
baseline = typer.Typer(no_args_is_help=True, help="Phase 3 rule-based baseline mapper.")
app.add_typer(baseline, name="baseline")

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


# --------------------------------------------------------------------------- Phase 3
# Tier D lives in kata.data.lowrated; the validator and baseline import only numpy/polars, so
# none of these commands needs the audio extra.

@data.command("select-lowrated")
def select_lowrated():
    """Tier D: year-matched sample of low-rated maps from the stored catalogue."""
    from kata.data import lowrated
    _run("select-lowrated", lowrated.select_lowrated)


@data.command("download-lowrated")
def download_lowrated(pilot: bool = PILOT):
    """Download Tier D maps without their audio (resumable)."""
    from kata.data import lowrated
    _run("download-lowrated", lowrated.download_lowrated, pilot=pilot)


@data.command("parse-lowrated")
def parse_lowrated(pilot: bool = PILOT):
    """Parse Tier D into processed/lowrated/ and assign calibration halves."""
    from kata.data import lowrated
    _run("parse-lowrated", lowrated.parse_lowrated, pilot=pilot)


@validate.command("analyse")
def validate_analyse():
    """Swings, issues and per-map / per-window metrics for every parsed difficulty (A-D)."""
    from kata.validate import pipeline as vp
    _run("validate-analyse", vp.analyse_stage)


@validate.command("reference")
def validate_reference():
    """Tier A train quantiles -> validator/v1/reference.json."""
    from kata.validate import pipeline as vp
    _run("validate-reference", vp.reference_stage)


@validate.command("score")
def validate_score():
    """Sub-scores, scores and the gate -> validation.parquet, validation_local.parquet."""
    from kata.validate import pipeline as vp
    _run("validate-score", vp.score_stage)


@validate.command("corrupt")
def validate_corrupt(pilot: bool = PILOT):
    """Synthetic corruptions of Tier A val+test -> corruption_results.parquet."""
    from kata.validate import pipeline as vp
    _run("validate-corrupt", vp.corrupt_stage, pilot=pilot)


@validate.command("report")
def validate_report(name: str = typer.Option("phase3", help="Report file name prefix.")):
    from kata.validate import pipeline as vp
    _run("validate-report", vp.report_stage, name=name)


@baseline.command("generate")
def baseline_generate(pilot: bool = PILOT):
    """Baseline maps for every val+test song -> generated/baseline/v1/, baseline_eval.parquet."""
    from kata.baseline import pipeline as bp
    _run("baseline-generate", bp.generate_stage, pilot=pilot)


@validate.command("pilot")
def validate_pilot():
    """Phase 3 pilot: Tier D pilot maps, every validator stage and the baseline on the pilot subset."""
    from kata.baseline import pipeline as bp
    from kata.data import lowrated
    from kata.validate import pipeline as vp
    for stage, fn, kwargs in [("select-lowrated", lowrated.select_lowrated, {}),
                              ("download-lowrated", lowrated.download_lowrated, {"pilot": True}),
                              ("parse-lowrated", lowrated.parse_lowrated, {"pilot": True}),
                              ("validate-analyse", vp.analyse_stage, {}),
                              ("validate-reference", vp.reference_stage, {}),
                              ("validate-score", vp.score_stage, {}),
                              ("validate-corrupt", vp.corrupt_stage, {"pilot": True}),
                              ("baseline-generate", bp.generate_stage, {"pilot": True}),
                              ("validate-report", vp.report_stage, {"name": "phase3-pilot"})]:
        _run(stage, fn, **kwargs)
