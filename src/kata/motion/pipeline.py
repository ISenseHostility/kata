"""Phase 4 stages (docs/phases/phase-4-motion.md §10-§11). Same conventions as Phases 1-3: every
stage runs inside `_with_state`; tables are rebuilt deterministically from their inputs."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from kata.config import Paths
from kata.data.pipeline import _with_state
from kata.motion.prepare import MotionStore, Rejected, channel_stats, motion_config, prepare_replay
from kata.validate.rules import nps

KEY = ["hash", "characteristic", "difficulty"]
# Phase 1 rejects a left-handed replay as "left_handed" and, because its note ids are mirrored,
# usually also on alignment. Those are the ones prepare recovers (spec §3.5, D7).
RECOVERABLE = {"left_handed", "align_match_rate", "align_residual"}


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _recoverable(reason: str | None) -> bool:
    parts = set((reason or "").split(","))
    return "left_handed" in parts and parts <= RECOVERABLE


def _prepare_one(root: str, score_id: str, bsor_path: str, notes: dict, left_handed: bool) -> dict:
    from kata.data import bsor
    store = MotionStore(Paths(Path(root)).motion)
    try:
        p = prepare_replay(bsor.parse(Path(bsor_path).read_bytes()), notes, left_handed)
    except Rejected as e:
        return {"status": "rejected", "reject_reason": str(e)}
    except Exception as e:  # recorded per replay; the stage keeps going
        return {"status": "error", "reject_reason": f"{type(e).__name__}: {e}"[:300]}
    store.write(score_id, p.data, p.valid)
    return {"status": "ok", "reject_reason": None, "frame0": p.frame0, "hps": nps(p.hit_times), **p.info}


def _prepare(paths, state, summary, pilot=False):
    suffix = "_pilot" if pilot else ""
    P = paths.processed
    replays = pl.read_parquet(P / f"replays{suffix}.parquet")
    replays = replays.filter((pl.col("status") == "ok")
                             | pl.col("reject_reason").map_elements(_recoverable, return_dtype=pl.Boolean))
    maps = pl.read_parquet(P / "maps.parquet").select("hash", "split", "audio_sha256")
    diffs = (pl.read_parquet(P / "difficulties.parquet").filter(pl.col("parse_status") == "ok")
             .select(KEY + ["tier"]))
    replays = replays.join(maps, on="hash").join(diffs, on=KEY).sort("score_id")
    notes = pl.read_parquet(P / "notes.parquet").join(replays.select(KEY).unique(), on=KEY, how="semi")
    by_diff = {k: {c: g[c].to_numpy() for c in ("x", "y", "color", "cut_dir", "time_s")}
               for k, g in notes.partition_by(KEY, as_dict=True).items()}
    store = MotionStore(paths.motion)
    store.open_for_write(motion_config())

    rows = []
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        futures = []
        for r in replays.iter_rows(named=True):
            sid = str(r["score_id"])
            futures.append((r, pool.submit(_prepare_one, str(paths.root), sid,
                                           str(paths.replays / r["hash"] / f"{sid}.bsor"),
                                           by_diff[(r["hash"], r["characteristic"], r["difficulty"])],
                                           bool(r["left_handed"]))))
        for i, (r, f) in enumerate(futures):
            key = (r["hash"], r["characteristic"], r["difficulty"])
            rows.append({"score_id": str(r["score_id"]), **{k: r[k] for k in KEY}, "label": r["difficulty"],
                         "split": r["split"], "audio_sha256": r["audio_sha256"], "tier": r["tier"],
                         "nps": nps(by_diff[key]["time_s"]), "height_median": r["height_median"],
                         "motion_version": store.version, **f.result()})
            if (i + 1) % 1000 == 0:
                print(f"  [motion-prepare] {i + 1}/{len(futures)}", flush=True)
    tracks = pl.DataFrame(rows, infer_schema_length=None).sort("score_id")
    tracks.write_parquet(P / f"motion_tracks{suffix}.parquet")

    train = tracks.filter((pl.col("status") == "ok") & (pl.col("split") == "train"))["score_id"].to_list()
    store.stats_path.write_text(json.dumps(channel_stats(store, train)))
    summary["tracks"] = dict(tracks.group_by("status").len().sort("status").iter_rows())
    summary["reject_reasons"] = dict(tracks.filter(pl.col("status") != "ok").group_by("reject_reason").len()
                                     .sort("reject_reason").iter_rows())
    ok = tracks.filter(pl.col("status") == "ok")
    summary["left_handed_recovered"] = ok.filter(pl.col("mirrored")).height
    summary["stats_train_tracks"] = len(train)

prepare_stage = _with_state("motion-prepare", _prepare)


# --------------------------------------------------------------------------- training and sampling
# These stages import torch (the `audio` extra) inside the function, like the Phase 2 stages.

GUIDANCE_VALUES = (1.0, 1.5, 2.0, 3.0, 4.0)
TUNE_DIFFS = 50
SPEED_SONGS = 3


def _train(paths, state, summary, run, config=None, resume=False, hours=None, device="cuda", max_steps=None):
    from kata.motion.train import train
    summary.update(train(paths, run, config, resume, hours, device, max_steps))

train_stage = _with_state("motion-train", _train)


def _sample_dir(paths: Paths, run: str, split: str) -> Path:
    return paths.generated / "motion" / run / split


def _tune(paths, state, summary, run, n=TUNE_DIFFS, values=GUIDANCE_VALUES, device=None):
    """Guidance weight on val (§7.1): best median timing F among the values passing criteria 2 and 4."""
    from kata.motion.evaluate import criteria, evaluate_samples
    from kata.motion.sample import Generator, eval_targets, sample_targets, spread
    gen = Generator(paths, run, device=device)
    targets = spread(eval_targets(paths, gen.run, "val"), n)
    table = []
    for w in values:
        d = _sample_dir(paths, run, "tune") / f"w{w}"
        sample_targets(gen, targets, d, w, seeds=(0,), label_only=False)
        rows, pooled = evaluate_samples(paths, run, "val", targets, d)
        crit = {c["criterion"]: c for c in criteria(rows, pooled)}
        f = rows.filter(pl.col("source") == "generated")["timing_f"].median()
        table.append({"guidance": w, "median_f": f, "plausible": crit[2]["pass"], "control": crit[4]["pass"]})
    passing = [r for r in table if r["plausible"] and r["control"]]
    best = max(passing or table, key=lambda r: r["median_f"] or 0.0)
    note = None if passing else "no value passes criteria 2 and 4; best timing F chosen"
    gen.run.guidance_path.write_text(json.dumps({"guidance": best["guidance"], "table": table, "note": note},
                                                indent=1))
    summary.update(guidance=best["guidance"], table=table, note=note, n_targets=targets.height)

tune_stage = _with_state("motion-tune", _tune)


def _sample(paths, state, summary, run, split="val", limit=None, device=None, batch_jobs=16):
    from kata.motion.sample import Generator, eval_targets, sample_targets, spread
    gen = Generator(paths, run, device=device)
    targets = spread(eval_targets(paths, gen.run, split), limit)
    summary.update(sample_targets(gen, targets, _sample_dir(paths, run, split), gen.run.guidance(),
                                  batch_jobs=batch_jobs), targets=targets.height, guidance=gen.run.guidance())

sample_stage = _with_state("motion-sample", _sample)


def _evaluate(paths, state, summary, run, split="val", limit=None, device=None, force=False):
    """Criteria for one split. Test is evaluated once per run (criterion 9); force only with approval."""
    import time

    from kata.motion.evaluate import criteria, evaluate_samples
    from kata.motion.sample import ckpt_fingerprint, durations, eval_targets, generate, spread, stale_samples
    from kata.motion.train import Run
    if split == "test" and (paths.motion_runs / run / "eval_test.json").exists() and not force:
        raise ValueError(f"run {run} was already evaluated on test; re-running needs force=True and your approval")
    run_obj = Run(paths, run)
    targets = spread(eval_targets(paths, run_obj, split), limit)
    stale = stale_samples(_sample_dir(paths, run, split), targets, run_obj.guidance(), ckpt_fingerprint(run_obj))
    if stale:                                                # never score (or spend test on) a partial set
        raise FileNotFoundError(f"{len(stale)} of {3 * targets.height} samples missing or stale (guidance "
                                f"{run_obj.guidance()}, ckpt/best.pt) in {_sample_dir(paths, run, split)}; "
                                f"run `kata motion sample {run} --split {split}` first")
    rows, pooled = evaluate_samples(paths, run, split, targets, _sample_dir(paths, run, split))
    base_path = paths.processed / "baseline_eval.parquet"
    baseline = pl.read_parquet(base_path).filter(pl.col("split") == split) if base_path.exists() else None
    crit = criteria(rows, pooled, baseline)
    speed = []
    if split == "test":                                      # criterion 11, on the first few test songs
        dur = durations(paths)
        for r in targets.head(SPEED_SONGS).iter_rows(named=True):
            t0 = time.monotonic()
            generate(r["audio_sha256"], r["label"], r["nps"], run=run, paths=paths, device=device)
            speed.append((time.monotonic() - t0) * 180.0 / dur[r["audio_sha256"]])
        crit.append({"criterion": 11, "name": "speed", "value": f"max {max(speed):.1f} s per 3-min song",
                     "pass": max(speed) <= 60.0})
    f = paths.processed / "motion_eval.parquet"
    if f.exists():
        old = pl.read_parquet(f).filter(~((pl.col("run") == run) & (pl.col("split") == split)))
        rows = pl.concat([old, rows], how="diagonal_relaxed")
    rows.sort("run", "split", "hash", "characteristic", "difficulty", "source", "mode", "seed", "score_id",
              nulls_last=True).write_parquet(f)
    out = {"criteria": crit, "pooled": pooled, "speed": speed, "targets": targets.height}
    (paths.motion_runs / run / f"eval_{split}.json").write_text(json.dumps(out, indent=1))
    summary.update(criteria=[{k: c[k] for k in ("criterion", "value", "pass")} for c in crit])

evaluate_stage = _with_state("motion-evaluate", _evaluate)


def _final(paths, state, summary, run):
    if not (paths.motion_runs / run / "ckpt" / "best.pt").exists():
        raise FileNotFoundError(f"run {run} has no best checkpoint")
    (paths.motion_runs / "final").write_text(run)
    summary["final"] = run

final_stage = _with_state("motion-final", _final)


def _viewer(paths, state, summary, run, split="test", n=6, blind=True):
    """Generated (full, seed 0) next to the first real replay for n songs spread over tempo (§7.3).
    Both clips share the map notes; in blind mode they appear only after Reveal (viewer.page)."""
    import numpy as np

    from kata.motion.evaluate import real_hits
    from kata.motion.sample import eval_targets, load_sample, sample_path, spread
    from kata.motion.train import Run, tracks_table
    from kata.motion.viewer import clip, write_page
    targets = eval_targets(paths, Run(paths, run), split)
    tempo = paths.processed / "audio_tempo.parquet"
    if tempo.exists():
        targets = targets.join(pl.read_parquet(tempo).select("audio_sha256", "est_bpm"), on="audio_sha256",
                               how="left").sort("est_bpm", nulls_last=True)
    targets = targets.unique("audio_sha256", keep="first", maintain_order=True)
    store = MotionStore(paths.motion)
    meta = tracks_table(paths, Run(paths, run).config().pilot).select("score_id", "frame0")
    frame0 = dict(zip(meta["score_id"], meta["frame0"]))
    notes = pl.read_parquet(paths.processed / "notes.parquet")
    written = []
    for i, r in enumerate(spread(targets, n).iter_rows(named=True)):
        key = (r["hash"], r["characteristic"], r["difficulty"])
        path = sample_path(_sample_dir(paths, run, split), key, "full", 0)
        if not path.exists():
            continue
        s = load_sample(path)
        sid = r["score_ids"][0]
        raw = np.asarray(store.load(sid), np.float32)
        kn = notes.filter((pl.col("hash") == key[0]) & (pl.col("characteristic") == key[1])
                          & (pl.col("difficulty") == key[2]))
        note_list = [[float(t), int(x), int(y), int(c), int(d)] for t, x, y, c, d in
                     kn.select("time_s", "x", "y", "color", "cut_dir").iter_rows()]
        real = clip(raw[:, :27], real_hits(raw, frame0[sid]), frame0[sid], note_list)
        generated = clip(s["motion"], (s["hits_left"], s["hits_right"]), notes=note_list)
        out = paths.reports / "motion-viewer" / f"{run}_{split}_{i:02d}_{key[0]}_{key[2]}.html"
        write_page(out, f"{key[0]} {key[2]} ({run})", generated, real, ("generated", "real"), blind)
        written.append(str(out))
    summary.update(pages=written)

viewer_stage = _with_state("motion-viewer", _viewer)


def _report(paths, state, summary, name="phase4", run=None, pilot=False):
    from kata.motion.report import write_report
    summary["path"] = str(write_report(paths, state, name, run, pilot))

report_stage = _with_state("motion-report", _report)
