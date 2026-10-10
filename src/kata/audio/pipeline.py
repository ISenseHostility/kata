"""Phase 2 stages (docs/phases/phase-2-audio.md §11). Same conventions as kata.data.pipeline:
every stage runs inside `_with_state`; `features` is resumable per audio_sha256, the others
rebuild their tables from the stored features each run."""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from kata.audio.beats import beat_in_bar, grid_agreement, tempo_stats
from kata.audio.frames import beat_grid_times, tempo_from_rows
from kata.audio.store import ARRAYS, CURVE_COLUMNS, FeatureStore
from kata.audio.structure import segment
from kata.audio.sync import estimate_lag
from kata.config import Paths
from kata.data.pipeline import _check_disk, _with_state

FEATURES_STAGE = "audio-features"
STRUCTURE_STAGE = "audio-structure"
STATS_MAX_SONGS = 500
DIFF_KEY = ["hash", "characteristic", "difficulty"]


def songs(paths: Paths, pilot: bool = False) -> pl.DataFrame:
    """One row per unique audio file: audio_sha256, hash (smallest map hash using it), audio_file."""
    maps = pl.read_parquet(paths.processed / "maps.parquet").filter(pl.col("audio_status") == "ok")
    if pilot:
        maps = maps.filter(pl.col("pilot"))
    return (maps.sort("hash").group_by("audio_sha256", maintain_order=True).first()
            .select("audio_sha256", "hash", "audio_file").sort("audio_sha256"))


def ok_songs(paths: Paths) -> pl.DataFrame:
    f = paths.processed / "audio_features.parquet"
    if not f.exists():
        return pl.DataFrame({"audio_sha256": [], "duration_s": []},
                            schema={"audio_sha256": pl.String, "duration_s": pl.Float64})
    return pl.read_parquet(f).filter(pl.col("status") == "ok").select("audio_sha256", "duration_s")


# --------------------------------------------------------------------------- features

def write_feature_tables(paths: Paths, state, store: FeatureStore) -> dict:
    """audio_features / audio_beats / audio_tempo, rebuilt from the state DB and the stored events."""
    rows = state.con.execute("SELECT key, status, error, info FROM items WHERE stage = ? ORDER BY key",
                             [FEATURES_STAGE]).fetchall()
    feats, beat_parts, tempo = [], [], []
    for sha, status, error, info in rows:
        feats.append({"audio_sha256": sha, "status": status, "error": error, **json.loads(info or "{}")})
        if status != "ok":
            continue
        b, d = store.events(sha)
        pos, is_db = beat_in_bar(b, d)
        beat_parts.append(pl.DataFrame({"audio_sha256": [sha] * len(b), "idx": np.arange(len(b)),
                                        "time_s": b, "beat_in_bar": pos, "is_downbeat": is_db}))
        tempo.append({"audio_sha256": sha, "n_beats": len(b), "n_downbeats": len(d), **tempo_stats(b, d)})
    P = paths.processed
    pl.DataFrame(feats, infer_schema_length=None).write_parquet(P / "audio_features.parquet")
    (pl.concat(beat_parts) if beat_parts else pl.DataFrame()).write_parquet(P / "audio_beats.parquet")
    pl.DataFrame(tempo, infer_schema_length=None).write_parquet(P / "audio_tempo.parquet")
    return {"songs_ok": len(tempo), "songs_error": len(feats) - len(tempo)}


def _features(paths, state, summary, pilot=False, limit=0, device="cuda", extractor=None):
    from kata.audio import decode
    from kata.audio.extract import Extractor

    all_songs = songs(paths, pilot)
    store = FeatureStore(paths.features)
    ex = extractor if extractor is not None else Extractor(device)
    store.open_for_write(ex.config)
    done = state.done(FEATURES_STAGE)
    todo = [r for r in all_songs.iter_rows(named=True)
            if not (r["audio_sha256"] in done and store.has(r["audio_sha256"]))]
    if limit:
        todo = todo[:limit]
    counts = {"ok": 0, "error": 0}
    # Decode + resample the next songs on CPU threads while the GPU works on the current one.
    with ThreadPoolExecutor(max_workers=2) as pool:
        queue: deque = deque()
        it = iter(todo)

        def submit():
            r = next(it, None)
            if r is not None:
                queue.append((r, pool.submit(decode.load, paths.maps / r["hash"] / r["audio_file"])))

        for _ in range(3):
            submit()
        n = 0
        while queue:
            r, fut = queue.popleft()
            submit()
            sha = r["audio_sha256"]
            try:
                f = ex(fut.result())
                store.write(sha, f.arrays, f.beats, f.downbeats)
                state.mark(FEATURES_STAGE, sha, "ok", info=f.scalars)
                counts["ok"] += 1
            except Exception as e:  # recorded per song; the stage keeps going
                state.mark(FEATURES_STAGE, sha, "error", error=f"{type(e).__name__}: {e}"[:500])
                counts["error"] += 1
            n += 1
            if n % 50 == 0 or n == len(todo):
                _check_disk(paths)
                print(f"  [{FEATURES_STAGE}] {n}/{len(todo)}", flush=True)
    summary.update(songs=all_songs.height, processed=len(todo), **counts,
                   status=state.counts(FEATURES_STAGE), **write_feature_tables(paths, state, store))

features = _with_state(FEATURES_STAGE, _features)
