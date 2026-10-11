"""`kata baseline generate` (spec §8.6): val + test songs -> map folders, zips and baseline_eval."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from kata.audio.beats import f_measure
from kata.audio.store import FeatureStore
from kata.baseline.export import round_trip, write_map
from kata.baseline.generate import SongInput, generate
from kata.config import Paths
from kata.data.pipeline import _check_disk, _with_state
from kata.validate.reference import Reference, ValidatorDir
from kata.validate.rules import SCORED_RULES

BASELINE_VERSION = "v1"
TIMING_TOL_S = 0.03


def songs(paths: Paths, pilot: bool = False) -> pl.DataFrame:
    """One row per val/test audio file with features: the map row whose metadata it uses (smallest
    hash), its duration, and the Tier A map hashes of that song for the timing comparison."""
    P = paths.processed
    maps = pl.read_parquet(P / "maps.parquet").filter(
        (pl.col("audio_status") == "ok") & pl.col("split").is_in(["val", "test"]))
    if pilot:
        maps = maps.filter(pl.col("pilot"))
    feats = pl.read_parquet(P / "audio_features.parquet").filter(pl.col("status") == "ok").select(
        "audio_sha256", "duration_s")
    tier_a = (pl.read_parquet(P / "difficulties.parquet").filter((pl.col("tier") == "A") & (pl.col("parse_status") == "ok"))
              .select("hash").unique())
    a_hashes = (maps.join(tier_a, on="hash").group_by("audio_sha256").agg(pl.col("hash").sort().alias("tier_a_hashes")))
    first = maps.sort("hash").group_by("audio_sha256", maintain_order=True).first()
    return (first.select("audio_sha256", "hash", "audio_file", "song_name", "song_sub_name", "song_author", "split")
            .join(feats, on="audio_sha256").join(a_hashes, on="audio_sha256", how="left").sort("audio_sha256"))


def _one(root: str, ref_data: dict, song: dict, segments: list[tuple[float, float]],
         human: dict[str, list[float]]) -> list[dict]:
    paths = Paths(Path(root))
    ref = Reference(ref_data)
    store = FeatureStore(paths.features)
    sha = song["audio_sha256"]
    beats, downbeats = store.events(sha)
    inp = SongInput(sha, song["duration_s"], np.asarray(store.load(sha, "curves"), np.float32),
                    beats, downbeats, segments)
    grid, out, results = generate(inp, ref)
    folder = paths.generated / "baseline" / BASELINE_VERSION / sha
    write_map(folder, paths.maps / song["hash"] / song["audio_file"], song, grid, out, song["duration_s"])
    try:
        round_trip(folder, out)
        ok = True
    except AssertionError:
        ok = False
    rows = []
    for label, res in results.items():
        t = np.unique(out[label].notes["time_s"].to_numpy())
        ref_t = human.get(label)
        rows.append({"audio_sha256": sha, "split": song["split"], "difficulty": label,
                     "n_notes": out[label].notes.height, "nps": res.metrics["nps"], "pass": res.passed,
                     "score": res.score, "stratum": res.stratum, **{f"s_{r}": res.sub_scores[r] for r in SCORED_RULES},
                     "timing_f": f_measure(np.unique(ref_t), t, TIMING_TOL_S) if ref_t else None,
                     "grid_bpm": grid.bpm, "piecewise": grid.piecewise, "round_trip": ok})
    return rows


def _generate(paths, state, summary, pilot=False):
    ref = ValidatorDir(paths.validator).load_reference()
    todo = songs(paths, pilot)
    segs = pl.read_parquet(paths.processed / "audio_segments.parquet").sort("audio_sha256", "seg_idx")
    seg_map = {k[0]: list(zip(g["start_s"], g["end_s"])) for k, g in segs.partition_by("audio_sha256", as_dict=True).items()}
    hashes = [h for hs in todo["tier_a_hashes"].drop_nulls() for h in hs]
    notes = (pl.scan_parquet(paths.processed / "notes.parquet").filter(pl.col("hash").is_in(hashes))
             .select("hash", "difficulty", "time_s").collect())
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = []
        for song in todo.iter_rows(named=True):
            hs = song["tier_a_hashes"] or []
            human = {}
            for label, g in notes.filter(pl.col("hash").is_in(hs[:1])).partition_by("difficulty", as_dict=True).items():
                human[label[0]] = g["time_s"].to_list()
            futures.append(pool.submit(_one, str(paths.root), ref.data, song, seg_map.get(song["audio_sha256"], []), human))
        for i, f in enumerate(futures):
            rows += f.result()
            if (i + 1) % 50 == 0:
                _check_disk(paths)
                print(f"  [baseline-generate] {i + 1}/{len(futures)}", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None).sort("audio_sha256", "difficulty")
    df.write_parquet(paths.processed / "baseline_eval.parquet")
    summary.update(songs=todo.height, maps_written=todo.height,
                   by_difficulty=df.group_by("difficulty").agg(
                       pl.col("pass").mean().alias("pass"), pl.col("score").median().alias("median_score"),
                       pl.col("timing_f").median().alias("median_timing_f")).sort("difficulty").to_dicts())

generate_stage = _with_state("baseline-generate", _generate)
