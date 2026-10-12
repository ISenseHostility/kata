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
