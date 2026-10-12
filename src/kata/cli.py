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
motion = typer.Typer(no_args_is_help=True, help="Phase 4 motion model (all but `prepare` need `uv sync --extra audio`).")
app.add_typer(motion, name="motion")

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
def crawl_leaderboards(pilot: bool = PILOT,
                       tier: str = typer.Option(None, help="Only this tier's difficulties (Phase 4: A).")):
    _run("crawl-leaderboards", pipeline.crawl_leaderboards, pilot=pilot, tier=tier)


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
def validate_report(name: str = typer.Option("phase3", help="Report file name prefix."),
                    calibration: str = typer.Option("holdout", help="Separation sets: 'holdout' (final) or "
                                                    "'fit' (tuning halves, for pilot decisions).")):
    """Phase 3 Markdown report under reports/."""
    from kata.validate import pipeline as vp
    _run("validate-report", vp.report_stage, name=name, calibration=calibration)


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
                              ("validate-report", vp.report_stage, {"name": "phase3-pilot", "calibration": "fit"})]:
        _run(stage, fn, **kwargs)


# --------------------------------------------------------------------------- Phase 4
# kata.motion.prepare is numpy-only; training, sampling and evaluation import torch inside each stage.

RUN = typer.Argument(..., help="Run id: a folder under runs/motion/.")
DEVICE = typer.Option(None, help="Torch device (default: cuda when available).")


@motion.command("prepare")
def motion_prepare(pilot: bool = PILOT):
    """Replays -> 50 fps motion tracks, motion_tracks.parquet and train stats (spec §3)."""
    from kata.motion import pipeline as mp
    _run("motion-prepare", mp.prepare_stage, pilot=pilot)


@motion.command("train")
def motion_train(run: str = RUN, resume: bool = typer.Option(False, help="Continue from ckpt/last.pt."),
                 hours: float = typer.Option(None, help="Stop cleanly after this many hours."),
                 device: str = typer.Option("cuda", help="Torch device."),
                 pilot: bool = typer.Option(False, help="Train on the pilot tracks (new runs only)."),
                 window: int = typer.Option(640, help="Window in 50 fps frames (new runs only)."),
                 mirror_aug: bool = typer.Option(False, help="Mirror augmentation (new runs only)."),
                 stems: bool = typer.Option(True, help="Use stem features (new runs only)."),
                 total_steps: int = typer.Option(80_000, help="Steps the lr schedule spans (new runs only)."),
                 batch: int = typer.Option(64, help="Windows per step (new runs only)."),
                 workers: int = typer.Option(4, help="DataLoader worker processes (new runs only)."),
                 only_song: list[str] = typer.Option([], help="Overfit check: train on this audio sha only.")):
    """Train a run overnight; resume it the next night with --resume (spec §6.4)."""
    from kata.motion import pipeline as mp
    from kata.motion.train import RunConfig
    config = None if resume else RunConfig(window=window, mirror_aug=mirror_aug, use_stems=stems, pilot=pilot,
                                           total_steps=total_steps, batch=batch, workers=workers,
                                           only_songs=tuple(only_song))
    _run("motion-train", mp.train_stage, run=run, config=config, resume=resume, hours=hours, device=device)


@motion.command("tune")
def motion_tune(run: str = RUN, device: str = DEVICE):
    """Pick the guidance weight on 50 val difficulties -> runs/motion/{run}/guidance.json (spec §7.1)."""
    from kata.motion import pipeline as mp
    _run("motion-tune", mp.tune_stage, run=run, device=device)


@motion.command("sample")
def motion_sample(run: str = RUN, split: str = typer.Option("val", help="val or test."),
                  limit: int = typer.Option(None, help="Only this many difficulties, evenly spread."),
                  device: str = DEVICE):
    """Whole songs for every difficulty of the split -> generated/motion/{run}/{split}/ (resumable)."""
    from kata.motion import pipeline as mp
    _run("motion-sample", mp.sample_stage, run=run, split=split, limit=limit, device=device)


@motion.command("evaluate")
def motion_evaluate(run: str = RUN, split: str = typer.Option("val", help="val or test (test: once, §8)."),
                    device: str = DEVICE,
                    force: bool = typer.Option(False, help="Re-evaluate test (only with approval).")):
    """Metrics and acceptance criteria -> motion_eval.parquet, runs/motion/{run}/eval_{split}.json."""
    from kata.motion import pipeline as mp
    _run("motion-evaluate", mp.evaluate_stage, run=run, split=split, device=device, force=force)


@motion.command("final")
def motion_final(run: str = RUN):
    """Mark the run generate() uses by default."""
    from kata.motion import pipeline as mp
    _run("motion-final", mp.final_stage, run=run)


@motion.command("viewer")
def motion_viewer(run: str = RUN, split: str = typer.Option("test", help="val or test."),
                  n: int = typer.Option(6, help="Songs, spread over tempo."),
                  blind: bool = typer.Option(True, help="Show clips as A/B until revealed.")):
    """Generated vs. real 3D viewer pages -> reports/motion-viewer/ (spec §7.3)."""
    from kata.motion import pipeline as mp
    _run("motion-viewer", mp.viewer_stage, run=run, split=split, n=n, blind=blind)


@motion.command("report")
def motion_report(name: str = typer.Option("phase4", help="Report file name prefix."),
                  run: str = typer.Option(None, help="Run to report (default: the final run)."),
                  pilot: bool = PILOT):
    """Phase 4 Markdown report under reports/."""
    from kata.motion import pipeline as mp
    _run("motion-report", mp.report_stage, name=name, run=run, pilot=pilot)
