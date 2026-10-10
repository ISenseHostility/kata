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


# --------------------------------------------------------------------------- structure

def _structure_one(features_root: str, sha: str, duration_s: float) -> list[dict]:
    store = FeatureStore(Path(features_root))
    b, d = store.events(sha)
    segs = segment(np.asarray(store.load(sha, "chroma"), np.float32),
                   np.asarray(store.load(sha, "mel"), np.float32), b, d, duration_s)
    n_labels = len({s["label"] for s in segs})
    return [{"audio_sha256": sha, "seg_idx": i, **s, "n_labels": n_labels} for i, s in enumerate(segs)]


def _structure(paths, state, summary, pilot=False):
    todo = ok_songs(paths)
    if pilot:
        todo = todo.join(songs(paths, pilot=True).select("audio_sha256"), on="audio_sha256", how="semi")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = {r["audio_sha256"]: pool.submit(_structure_one, str(paths.features), r["audio_sha256"],
                                                  r["duration_s"]) for r in todo.iter_rows(named=True)}
        for sha, fut in futures.items():
            try:
                rows += fut.result()
                state.mark(STRUCTURE_STAGE, sha, "ok")
            except Exception as e:
                state.mark(STRUCTURE_STAGE, sha, "error", error=f"{type(e).__name__}: {e}"[:500])
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(paths.processed / "audio_segments.parquet")
    summary.update(songs=todo.height, segments=len(rows), status=state.counts(STRUCTURE_STAGE))

structure = _with_state(STRUCTURE_STAGE, _structure)


# --------------------------------------------------------------------------- sync

def _sync(paths, state, summary, pilot=False):
    from kata.audio.spectral import SpectralExtractor
    from kata.audio.sync import measure_onset_bias

    P = paths.processed
    maps = pl.read_parquet(P / "maps.parquet", columns=["hash", "audio_sha256", "pilot"])
    if pilot:
        maps = maps.filter(pl.col("pilot"))
    maps = maps.join(ok_songs(paths), on="audio_sha256", how="semi").select("hash", "audio_sha256")
    diffs = (pl.read_parquet(P / "difficulties.parquet", columns=DIFF_KEY + ["tier", "parse_status"])
             .filter(pl.col("parse_status") == "ok").join(maps, on="hash"))
    notes = (pl.read_parquet(P / "notes.parquet", columns=DIFF_KEY + ["beat", "time_s"])
             .join(maps.select("hash"), on="hash", how="semi"))
    tempo = pl.read_parquet(P / "tempo_maps.parquet").join(maps.select("hash"), on="hash", how="semi")
    notes_by = notes.partition_by(DIFF_KEY, as_dict=True)
    tempo_by = tempo.partition_by(DIFF_KEY, as_dict=True)
    bias = measure_onset_bias(SpectralExtractor("cpu"))
    store = FeatureStore(paths.features)
    sync_rows, grid_rows = [], []
    for (sha,), group in diffs.group_by("audio_sha256"):
        curves = np.asarray(store.load(sha, "curves"), np.float32)
        beats, _ = store.events(sha)
        for d in group.iter_rows(named=True):
            key = tuple(d[k] for k in DIFF_KEY)
            n = notes_by.get(key)
            if n is None or n.height == 0:
                continue
            base = {**dict(zip(DIFF_KEY, key)), "tier": d["tier"], "audio_sha256": sha}
            times = n["time_s"].to_numpy()
            for src, col in (("mix", "onset_mix"), ("drums", "onset_drums")):
                lag = estimate_lag(curves[:, CURVE_COLUMNS.index(col)], times, bias)
                sync_rows.append({**base, "src": src, "n_notes": int(len(np.unique(times))), **lag})
            rows = tempo_by[key].to_dicts()
            beat = n["beat"].to_numpy()
            grid = beat_grid_times(tempo_from_rows(rows), math.ceil(beat.min()), math.floor(beat.max()))
            map_bpm = min(rows, key=lambda r: r["seg_idx"])["bpm"]
            grid_rows.append({**base, "map_bpm": map_bpm, "n_grid": len(grid), **grid_agreement(grid, beats)})
    pl.DataFrame(sync_rows, infer_schema_length=None).write_parquet(P / "map_audio_sync.parquet")
    pl.DataFrame(grid_rows, infer_schema_length=None).write_parquet(P / "beat_agreement.parquet")
    summary.update(onset_bias_ms=bias * 1000, difficulties=len(grid_rows))

sync = _with_state("audio-sync", _sync)


# --------------------------------------------------------------------------- stats

def _stats(paths, state, summary, pilot=False):
    maps = pl.read_parquet(paths.processed / "maps.parquet", columns=["audio_sha256", "split", "pilot"])
    if pilot:
        maps = maps.filter(pl.col("pilot"))
    train = maps.filter(pl.col("split") == "train").join(ok_songs(paths), on="audio_sha256", how="semi")
    shas = sorted(set(train["audio_sha256"]))
    if len(shas) > STATS_MAX_SONGS:
        shas = sorted(random.Random(0).sample(shas, STATS_MAX_SONGS))
    store = FeatureStore(paths.features)
    acc: dict[str, list] = {}
    for sha in shas:
        for name in ARRAYS:
            a = np.asarray(store.load(sha, name), dtype=np.float64)
            flat = a.reshape(a.shape[0], -1)
            s = acc.setdefault(name, [np.zeros(flat.shape[1]), np.zeros(flat.shape[1]), 0, a.shape[1:]])
            s[0] += flat.sum(axis=0)
            s[1] += np.square(flat).sum(axis=0)
            s[2] += flat.shape[0]
    arrays = {}
    for name, (s1, s2, n, shape) in acc.items():
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
        arrays[name] = {"shape": list(shape), "mean": mean.reshape(shape).tolist(),
                        "std": std.reshape(shape).tolist(), "n_frames": int(n)}
    out = {"split": "train", "n_songs": len(shas), "songs": shas, "arrays": arrays}
    store.stats_path.write_text(json.dumps(out))
    summary.update(n_songs=len(shas), path=str(store.stats_path))

stats = _with_state("audio-stats", _stats)


# --------------------------------------------------------------------------- report

def _report(paths, state, summary, name="phase2"):
    from kata.audio.report import write_report
    summary["path"] = str(write_report(paths, state, name))

report = _with_state("audio-report", _report)

PILOT_STAGES = [
    ("audio-features", features, {"pilot": True}),
    ("audio-structure", structure, {"pilot": True}),
    ("audio-sync", sync, {"pilot": True}),
    ("audio-stats", stats, {"pilot": True}),
    ("audio-report", report, {"name": "phase2-pilot"}),
]
