"""Tier D: a year-matched sample of low-rated maps for validator calibration
(docs/phases/phase-3-validator.md §6.1). Only the .dat files are kept; the tables go to
processed/lowrated/ so that no Phase 2 or Phase 4 consumer sees them by accident."""

from __future__ import annotations

import hashlib

import duckdb
import polars as pl

from kata.data import beatsaver
from kata.data.http import PoliteClient
from kata.data.pipeline import _filter_catalog, _gather_items, _parse_selection, _with_state, normalise_title

LOWRATED_MAX_RATING = 0.6
LOWRATED_MIN_VOTES = 20
LOWRATED_SAMPLE = 2000
LOWRATED_PILOT = 200
DOWNLOAD_STAGE = "download-lowrated"


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def year_quotas(share: dict[int, float], available: dict[int, int], n: int) -> dict[int, int]:
    """Maps to take per upload year: proportional to `share` (Tier A's year mix), capped by what
    each year has; slots a short year cannot fill go to the other years in proportion."""
    quotas = dict.fromkeys(available, 0)
    left = min(n, sum(available.values()))
    while left:
        open_ = [y for y in sorted(available) if quotas[y] < available[y]]
        w = {y: share.get(y, 0.0) for y in open_}
        if not any(w.values()):
            w = dict.fromkeys(open_, 1.0)
        total = sum(w.values())
        want = {y: left * w[y] / total for y in open_}
        step = {y: min(int(want[y]), available[y] - quotas[y]) for y in open_}
        if not any(step.values()):   # only fractions left: one map to the year that wants most
            step = {max(open_, key=lambda y: (want[y], -y)): 1}
        for y, k in step.items():
            quotas[y] += k
            left -= k
    return quotas


def calib_half(song_author: str | None, song_name: str | None) -> str:
    """'fit' or 'holdout' by the song's normalised name (Phase 1's name key)."""
    key = f"name:{normalise_title(song_author)}|{normalise_title(song_name)}"
    return "fit" if int(_md5(key), 16) % 2 == 0 else "holdout"


def _select_lowrated(paths, state, summary, n=LOWRATED_SAMPLE):
    P = paths.processed
    con = duckdb.connect()
    summary["funnel"] = _filter_catalog(con, pl.read_parquet(P / "metadata_catalog.parquet"),
                                        pl.read_parquet(P / "metadata_catalog_diffs.parquet"))
    con.register("sel", pl.read_parquet(P / "selected_maps.parquet").select("hash"))
    pool = con.execute(f"""
        SELECT map_id, hash, download_url, CAST(substr(uploaded_at, 1, 4) AS INTEGER) AS year,
               rating, upvotes + downvotes AS votes
        FROM fm
        WHERE rating < {LOWRATED_MAX_RATING} AND upvotes + downvotes >= {LOWRATED_MIN_VOTES}
          AND curated_at IS NULL AND NOT ranked_bl AND NOT ranked_ss
          AND hash NOT IN (SELECT hash FROM sel)
          AND map_id IN (SELECT map_id FROM fd)""").pl()
    pool = pool.with_columns(pl.col("map_id").map_elements(_md5, return_dtype=pl.String).alias("_order"))
    tier_a = (pl.read_parquet(P / "maps.parquet").filter(pl.col("map_tier") == "A")
              .select(pl.col("uploaded_at").str.slice(0, 4).cast(pl.Int64).alias("year")))
    share = {y: c / tier_a.height for y, c in tier_a.group_by("year").len().iter_rows()}
    available = dict(pool.group_by("year").len().iter_rows())
    quotas = year_quotas(share, available, n)
    picked = pl.concat([pool.filter(pl.col("year") == y).sort("_order").head(q) for y, q in quotas.items() if q])
    pilot_ids = picked.sort("_order").head(LOWRATED_PILOT)["map_id"]
    picked = picked.with_columns(pl.col("map_id").is_in(pilot_ids.implode()).alias("pilot"),
                                 pl.lit("D").alias("map_tier")).drop("_order").sort("map_id")
    diffs = (con.execute("SELECT * FROM fd").pl().join(picked.select("map_id"), on="map_id", how="semi")
             .with_columns(pl.lit("D").alias("tier")))
    paths.lowrated.mkdir(parents=True, exist_ok=True)
    picked.write_parquet(paths.lowrated / "selected_maps.parquet")
    diffs.write_parquet(paths.lowrated / "selected_diffs.parquet")
    summary.update(pool_maps=pool.height, sampled_maps=picked.height, sampled_diffs=diffs.height,
                   pilot_maps=int(picked["pilot"].sum()),
                   years={str(y): {"tier_a_share": round(share.get(y, 0.0), 3), "pool": available.get(y, 0),
                                   "sampled": quotas.get(y, 0)} for y in sorted(set(share) | set(available))})

select_lowrated = _with_state("select-lowrated", _select_lowrated)


async def _download_lowrated(paths, state, summary, pilot=False):
    sel = pl.read_parquet(paths.lowrated / "selected_maps.parquet")
    if pilot:
        sel = sel.filter(pl.col("pilot"))
    done = state.done(DOWNLOAD_STAGE)
    todo = [(r["hash"], r) for r in sel.iter_rows(named=True) if r["hash"] not in done]
    summary["todo"] = len(todo)
    async with PoliteClient() as client:
        await _gather_items(todo, lambda r: beatsaver.download_map(client, paths, r["hash"], r["download_url"],
                                                                   drop_audio=True),
                            state, DOWNLOAD_STAGE, paths)
        summary["requests"] = client.request_count
    summary["status"] = state.counts(DOWNLOAD_STAGE)
    summary["errors"] = state.errors(DOWNLOAD_STAGE, 10)

download_lowrated = _with_state(DOWNLOAD_STAGE, _download_lowrated)


def _parse_lowrated(paths, state, summary, pilot=False):
    sel_maps = pl.read_parquet(paths.lowrated / "selected_maps.parquet")
    if pilot:
        sel_maps = sel_maps.filter(pl.col("pilot"))
    sel_diffs = pl.read_parquet(paths.lowrated / "selected_diffs.parquet").join(
        sel_maps.select("hash"), on="hash", how="semi")
    summary.update(_parse_selection(paths, sel_diffs, sel_maps.select("hash", "map_tier", "pilot"),
                                    state.done(DOWNLOAD_STAGE), paths.lowrated, audio=False))
    maps = pl.read_parquet(paths.lowrated / "maps.parquet")
    maps = maps.with_columns(pl.struct("song_author", "song_name").map_elements(
        lambda r: calib_half(r["song_author"], r["song_name"]), return_dtype=pl.String).alias("calib_half"))
    maps.write_parquet(paths.lowrated / "maps.parquet")
    summary["calib_half"] = dict(maps.group_by("calib_half").len().iter_rows())

parse_lowrated = _with_state("parse-lowrated", _parse_lowrated)
