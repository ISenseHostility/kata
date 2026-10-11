"""Stage orchestration (docs/phases/phase-1-data.md §8).

Download stages are resumable through `State`; parse/build stages are pure functions of
`raw/` and rebuild their outputs from scratch, so re-running them is always safe.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from kata.config import MIN_FREE_BYTES, Paths
from kata.data import beatleader, beatsaver
from kata.data.http import PoliteClient
from kata.data.mapformat.canonical import ARC_COLS, BOMB_COLS, CHAIN_COLS, NOTE_COLS, OBSTACLE_COLS
from kata.data.state import State

PILOT_PER_TIER = 100
# Tier B (D8): curated AND rating > 0.95 AND upvotes > 100.
TIER_B_MIN_RATING = 0.95
TIER_B_MIN_UPVOTES = 100
ALIGN_MIN_MATCH_RATE = 0.98
ALIGN_MAX_P95_MS = 2.0


def _with_state(stage: str, fn):
    """Run fn(state, summary) inside a recorded run; returns the summary."""
    def wrapper(paths: Paths, **kwargs):
        state = State(paths.state_db)
        args = " ".join(f"{k}={v}" for k, v in kwargs.items())
        if asyncio.iscoroutinefunction(fn):
            async def run():
                try:
                    with state.run(stage, args) as summary:
                        await fn(paths, state, summary, **kwargs)
                    return summary
                finally:
                    state.close()
            return run()
        try:
            with state.run(stage, args) as summary:
                fn(paths, state, summary, **kwargs)
            return summary
        finally:
            state.close()
    wrapper.__name__ = fn.__name__
    return wrapper


def _check_disk(paths: Paths) -> None:
    free = shutil.disk_usage(paths.root).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"only {free / 1024**3:.0f} GB free on the data drive; stopping")


async def _gather_items(items, worker, state: State, stage: str, paths: Paths, log=print,
                        chunk: int = 200) -> None:
    """Run worker(item) -> info for each (key, item); record ok/error per key."""
    total = len(items)
    for start in range(0, total, chunk):
        _check_disk(paths)

        async def one(key, item):
            try:
                state.mark(stage, key, "ok", info=await worker(item))
            except Exception as e:  # recorded per item; the stage keeps going
                state.mark(stage, key, "error", error=f"{type(e).__name__}: {e}"[:500])

        await asyncio.gather(*(one(k, it) for k, it in items[start:start + chunk]))
        log(f"  [{stage}] {min(start + chunk, total)}/{total}")


def _subset(df: pl.DataFrame, paths: Paths, pilot: bool) -> pl.DataFrame:
    if not pilot:
        return df
    sel = pl.read_parquet(paths.processed / "selected_maps.parquet").filter(pl.col("pilot"))
    return df.join(sel.select("hash"), on="hash", how="semi")


# --------------------------------------------------------------------------- catalogue

async def _crawl_catalog(paths, state, summary):
    async with PoliteClient() as client:
        summary.update(await beatsaver.crawl_catalog(client, paths))
        summary["requests"] = client.request_count

crawl_catalog = _with_state("crawl-catalog", _crawl_catalog)


SELECT_SQL = f"""
CREATE TABLE fd AS
SELECT d.*, m.ranked_bl, m.ranked_ss, m.curated_at, m.rating, m.upvotes,
  CASE WHEN (m.ranked_bl AND d.bl_stars IS NOT NULL) OR (m.ranked_ss AND d.ss_stars IS NOT NULL) THEN 'A'
       WHEN m.curated_at IS NOT NULL AND m.rating > {TIER_B_MIN_RATING} AND m.upvotes > {TIER_B_MIN_UPVOTES} THEN 'B'
       ELSE 'C' END AS tier
FROM d JOIN fm m USING (map_id)
WHERE d.characteristic = 'Standard' AND NOT d.me AND NOT d.ne AND d.notes >= 50 AND d.seconds >= 30;
"""

MAP_FILTERS = [
    ("has_published_version", "hash IS NOT NULL AND version_state = 'Published'"),
    ("not_automapper", "NOT automapper"),
    ("not_declared_ai", "declared_ai = 'None'"),
    ("duration_30_900s", "meta_duration_s BETWEEN 30 AND 900"),
]


def _filter_catalog(con, maps: pl.DataFrame, diffs: pl.DataFrame) -> dict:
    """Build `fm` (maps passing MAP_FILTERS) and `fd` (their difficulties passing §3.2, with a
    tier) in `con`; returns the funnel counts."""
    con.register("m_all", maps.drop("raw_json") if "raw_json" in maps.columns else maps)
    con.register("d", diffs)
    funnel = {"catalog_maps": maps.height}
    where = []
    for name, cond in MAP_FILTERS:
        where.append(cond)
        funnel[name] = con.execute(f"SELECT count(*) FROM m_all WHERE {' AND '.join(where)}").fetchone()[0]
    con.execute(f"CREATE TABLE fm AS SELECT * FROM m_all WHERE {' AND '.join(where)}")
    con.execute(SELECT_SQL)
    funnel["maps_with_valid_standard_diff"] = con.execute("SELECT count(DISTINCT map_id) FROM fd").fetchone()[0]
    return funnel


def _select(paths, state, summary):
    maps, diffs = beatsaver.flatten_catalog(paths)
    con = duckdb.connect()
    funnel = _filter_catalog(con, maps, diffs)

    map_tiers = con.execute("""
        SELECT map_id, any_value(hash) AS hash, min(tier) AS map_tier
        FROM fd GROUP BY map_id""").pl()
    selected = map_tiers.filter(pl.col("map_tier").is_in(["A", "B"])).join(
        maps.select("map_id", "download_url", "created_at"), on="map_id")
    # Deterministic pilot: first N per tier by md5(map_id).
    selected = selected.with_columns(
        pl.col("map_id").map_elements(lambda s: hashlib.md5(s.encode()).hexdigest(),
                                      return_dtype=pl.String).alias("_order"))
    pilot_ids = pl.concat([
        selected.filter(pl.col("map_tier") == t).sort("_order").head(PILOT_PER_TIER).select("map_id")
        for t in ("A", "B")])
    selected = selected.with_columns(pl.col("map_id").is_in(pilot_ids["map_id"].implode()).alias("pilot")).drop("_order")
    selected.write_parquet(paths.processed / "selected_maps.parquet")

    sel_diffs = con.execute("SELECT * FROM fd").pl().join(selected.select("map_id"), on="map_id", how="semi")
    sel_diffs.write_parquet(paths.processed / "selected_diffs.parquet")

    funnel["selected_maps"] = selected.height
    summary["funnel"] = funnel
    summary["maps_by_tier"] = dict(selected.group_by("map_tier").len().sort("map_tier").iter_rows())
    summary["diffs_by_tier"] = dict(sel_diffs.group_by("tier").len().sort("tier").iter_rows())
    summary["pilot_maps"] = int(selected["pilot"].sum())
    # Distribution for tuning the Tier B thresholds: non-A maps by rating x upvotes.
    summary["tier_b_tuning_grid"] = con.execute("""
        WITH x AS (SELECT map_id, any_value(rating) r, any_value(upvotes) u,
                          bool_or(curated_at IS NOT NULL) cur, min(tier) t FROM fd GROUP BY map_id)
        SELECT CASE WHEN r >= 0.9 THEN '>=0.90' WHEN r >= 0.85 THEN '0.85-0.90'
                    WHEN r >= 0.8 THEN '0.80-0.85' WHEN r >= 0.75 THEN '0.75-0.80'
                    WHEN r >= 0.7 THEN '0.70-0.75' ELSE '<0.70' END AS rating,
          count(*) FILTER (WHERE u >= 10) AS "up>=10", count(*) FILTER (WHERE u >= 30) AS "up>=30",
          count(*) FILTER (WHERE u >= 50) AS "up>=50", count(*) FILTER (WHERE u >= 100) AS "up>=100",
          count(*) FILTER (WHERE cur) AS curated
        FROM x WHERE t <> 'A' GROUP BY 1 ORDER BY min(r) DESC""").pl().to_dicts()

select = _with_state("select", _select)


# --------------------------------------------------------------------------- maps

async def _download_maps(paths, state, summary, pilot=False, limit=0):
    sel = pl.read_parquet(paths.processed / "selected_maps.parquet")
    if pilot:
        sel = sel.filter(pl.col("pilot"))
    done = state.done("download-map")
    todo = [(r["hash"], r) for r in sel.iter_rows(named=True) if r["hash"] not in done]
    if limit:
        todo = todo[:limit]
    summary["todo"] = len(todo)
    async with PoliteClient() as client:
        await _gather_items(todo, lambda r: beatsaver.download_map(client, paths, r["hash"], r["download_url"]),
                            state, "download-map", paths)
        summary["requests"] = client.request_count
    summary["status"] = state.counts("download-map")
    summary["errors"] = state.errors("download-map", 10)

download_maps = _with_state("download-maps", _download_maps)


# name -> (columns, beat column index, tail-beat column name or None)
OBJECT_TABLES = {
    "notes": (NOTE_COLS, 0, None),
    "bombs": (BOMB_COLS, 0, None),
    "obstacles": (OBSTACLE_COLS, 0, None),
    "arcs": (ARC_COLS, 1, "tail_beat"),
    "chains": (CHAIN_COLS, 1, "tail_beat"),
}


def _parse_one_map(folder: str, map_hash: str, diff_rows: list[dict], audio: bool = True) -> dict:
    """Worker: parse a downloaded map folder.

    Returns the map row, difficulty rows, and one DataFrame per object table (columnar, so
    the full run does not hold millions of Python dicts). audio=False (Tier D) skips the audio
    checks: those maps were extracted without their audio.
    """
    from kata.data.mapformat import load_info

    out = {"map": {"hash": map_hash}, "difficulties": [], "tempo_maps": [],
           **{name: [] for name in OBJECT_TABLES}}
    m = out["map"]
    try:
        info = load_info(Path(folder))
    except Exception as e:
        m.update(parse_status="error", parse_error=f"info: {type(e).__name__}: {e}"[:300])
        return _frames(out)
    m.update(info_format_version=info.info_version, info_bpm=info.bpm,
             song_time_offset=info.song_time_offset, parse_status="ok", parse_error=None)
    if audio:
        _audio_checks(m, Path(folder), info.song_filename)
    else:
        m.update(audio_status="dropped", audio_error=None)
    _parse_difficulties(out, folder, map_hash, diff_rows, info)
    return _frames(out)


def _audio_checks(m: dict, folder: Path, song_filename: str) -> None:
    import soundfile as sf

    from kata.data.mapformat import find_audio
    try:
        audio, m["audio_filename_fallback"] = find_audio(folder, song_filename)
        m["audio_file"] = audio.name
        m["audio_bytes"] = audio.stat().st_size
        m["audio_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
        with open(audio, "rb") as f:
            ai = sf.info(f)
        m.update(audio_duration_s=ai.duration, audio_sample_rate=ai.samplerate,
                 audio_channels=ai.channels, audio_status="ok", audio_error=None)
    except Exception as e:
        m.update(audio_status="error", audio_error=f"{type(e).__name__}: {e}"[:300])


def _parse_difficulties(out: dict, folder: str, map_hash: str, diff_rows: list[dict], info) -> None:
    from kata.data.mapformat import MapFormatError, find_file, parse_difficulty, read_json, tempo_for

    refs = {(d.characteristic, d.difficulty): d for d in info.difficulties}
    for row in diff_rows:
        key = dict(hash=map_hash, characteristic=row["characteristic"], difficulty=row["difficulty"])
        drow = {**key, "tier": row["tier"], "format_version": None, "parse_status": "ok",
                "reject_reason": None, "parse_warnings": []}
        out["difficulties"].append(drow)
        ref = refs.get((row["characteristic"], row["difficulty"]))
        if ref is None:
            drow.update(parse_status="rejected", reject_reason="missing_in_info")
            continue
        drow.update(difficulty_rank=ref.rank, njs=ref.njs, njs_offset=ref.njs_offset)
        try:
            bm = parse_difficulty(read_json(find_file(Path(folder), ref.filename)))
            tempo, tempo_source = tempo_for(info, bm)
        except (MapFormatError, ValueError, KeyError, TypeError) as e:
            drow.update(parse_status="rejected", reject_reason=f"format: {type(e).__name__}: {e}"[:300])
            continue
        drow.update(format_version=bm.format_version, tempo_source=tempo_source,
                    parse_warnings=bm.warnings, n_notes=len(bm.notes), n_bombs=len(bm.bombs),
                    n_obstacles=len(bm.obstacles), n_arcs=len(bm.arcs), n_chains=len(bm.chains),
                    n_bpm_changes=len(tempo.segments) - 1)
        expected = (row["notes"], row["bombs"], row["obstacles"])
        parsed = (len(bm.notes), len(bm.bombs), len(bm.obstacles))
        if (parsed != expected and parsed[:2] == expected[:2]
                and row["obstacles"] in (row["notes"], row["notes"] + row["bombs"])):
            # BeatSaver quirk: its obstacle count sometimes repeats the note (+ bomb) count.
            bm.warn("beatsaver_obstacle_count_equals_notes")
            drow["parse_warnings"] = bm.warnings
            expected = parsed
        if parsed != expected:
            drow.update(parse_status="rejected",
                        reject_reason=f"count_mismatch: parsed {(len(bm.notes), len(bm.bombs), len(bm.obstacles))} "
                                      f"vs beatsaver {expected}")
            continue
        if bm.notes:
            times = tempo.to_seconds([n[0] for n in bm.notes])
            drow["duration_s"] = float(times[-1] - times[0])
        for name, (cols, beat_col, tail_col) in OBJECT_TABLES.items():
            rows = getattr(bm, name)
            if not rows:
                continue
            df = pl.DataFrame(rows, schema=list(cols), orient="row")
            df = df.with_columns(
                pl.lit(map_hash).alias("hash"), pl.lit(row["characteristic"]).alias("characteristic"),
                pl.lit(row["difficulty"]).alias("difficulty"), pl.int_range(pl.len()).alias("idx"),
                pl.Series("time_s", tempo.to_seconds(df[cols[beat_col]].to_numpy())))
            if name == "obstacles":
                df = df.with_columns(pl.Series("end_time_s", tempo.to_seconds((df["beat"] + df["duration"]).to_numpy())))
            elif tail_col is not None:
                df = df.with_columns(pl.Series("end_time_s", tempo.to_seconds(df[tail_col].to_numpy())))
            out[name].append(df)
        for seg in tempo.rows():
            out["tempo_maps"].append({**key, **seg})


def _frames(out: dict) -> dict:
    for name in OBJECT_TABLES:
        out[name] = pl.concat(out[name], how="diagonal_relaxed") if out[name] else None
    return out


def _parse_maps(paths, state, summary, pilot=False):
    sel_diffs = _subset(pl.read_parquet(paths.processed / "selected_diffs.parquet"), paths, pilot)
    sel_maps = pl.read_parquet(paths.processed / "selected_maps.parquet").select("hash", "map_tier", "pilot")
    summary.update(_parse_selection(paths, sel_diffs, sel_maps, state.done("download-map"), paths.processed))

parse_maps = _with_state("parse-maps", _parse_maps)


def _parse_selection(paths, sel_diffs: pl.DataFrame, sel_maps: pl.DataFrame, downloaded: set[str],
                     out_dir: Path, audio: bool = True) -> dict:
    """Parse the downloaded maps of a selection and write maps, difficulties, object and tempo
    tables to out_dir. Returns the summary counts."""
    summary: dict = {}
    by_hash: dict[str, list[dict]] = {}
    for r in sel_diffs.iter_rows(named=True):
        if r["hash"] in downloaded:
            by_hash.setdefault(r["hash"], []).append(r)
    results = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = [pool.submit(_parse_one_map, str(paths.maps / h), h, rows, audio) for h, rows in by_hash.items()]
        for f in futures:
            results.append(f.result())

    catalog = pl.read_parquet(paths.processed / "metadata_catalog.parquet").drop("raw_json")
    maps_df = (pl.DataFrame([r["map"] for r in results], infer_schema_length=None)
               .join(catalog, on="hash", how="left").join(sel_maps, on="hash", how="left"))
    diffs_df = pl.DataFrame([d for r in results for d in r["difficulties"]], infer_schema_length=None)
    diffs_df = diffs_df.join(
        sel_diffs.select("hash", "characteristic", "difficulty", "nps", "seconds", "parity_errors",
                         "parity_warns", "parity_resets", "ss_stars", "bl_stars", "chroma", "cinema",
                         "environment"),
        on=["hash", "characteristic", "difficulty"], how="left")
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_df.write_parquet(out_dir / "maps.parquet")
    diffs_df.write_parquet(out_dir / "difficulties.parquet")

    accepted = diffs_df.filter(pl.col("parse_status") == "ok").select("hash", "characteristic", "difficulty")
    for name in OBJECT_TABLES:
        parts = [r[name] for r in results if r[name] is not None]
        df = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
        if df.height:
            front = ["hash", "characteristic", "difficulty", "idx"]
            df = df.join(accepted, on=["hash", "characteristic", "difficulty"], how="semi").select(
                front + [c for c in df.columns if c not in front])
        df.write_parquet(out_dir / f"{name}.parquet")
    tempo = pl.DataFrame([x for r in results for x in r["tempo_maps"]], infer_schema_length=None)
    tempo.write_parquet(out_dir / "tempo_maps.parquet")

    summary["maps"] = maps_df.height
    summary["map_parse_status"] = dict(maps_df.group_by("parse_status").len().iter_rows())
    summary["audio_status"] = dict(maps_df.group_by("audio_status").len().iter_rows())
    summary["diff_status"] = dict(diffs_df.group_by("parse_status").len().iter_rows())
    summary["format_versions"] = dict(diffs_df.group_by(pl.col("format_version").str.slice(0, 1))
                                      .len().sort("format_version").iter_rows())
    return summary


# --------------------------------------------------------------------------- leaderboards

async def _crawl_leaderboards(paths, state, summary, pilot=False):
    diffs = pl.read_parquet(paths.processed / "difficulties.parquet").filter(pl.col("parse_status") == "ok")
    diffs = _subset(diffs, paths, pilot)
    hashes = sorted(set(diffs["hash"]))
    done = state.done("bl-leaderboards")
    async with PoliteClient() as client:
        todo = [(h, h) for h in hashes if h not in done]
        summary["leaderboard_hashes_todo"] = len(todo)
        await _gather_items(todo, lambda h: _count(beatleader.fetch_leaderboards(client, paths, h)),
                            state, "bl-leaderboards", paths)
        lb_rows = []
        for h in hashes:
            p = beatleader.leaderboards_path(paths, h)
            if p.exists():
                lb_rows.extend(beatleader.flatten_leaderboards(h, beatleader.load(p)))
        lbs = pl.DataFrame(lb_rows, infer_schema_length=None).join(
            diffs.select("hash", "characteristic", "difficulty"),
            on=["hash", "characteristic", "difficulty"], how="semi")
        lbs.write_parquet(paths.processed / ("leaderboards_pilot.parquet" if pilot else "leaderboards.parquet"))
        done = state.done("bl-scores")
        todo = [(i, i) for i in lbs["leaderboard_id"] if i not in done]
        summary["score_leaderboards_todo"] = len(todo)
        await _gather_items(todo, lambda i: _count(beatleader.fetch_scores(client, paths, i), "scores"),
                            state, "bl-scores", paths)
        summary["requests"] = client.request_count
    score_rows = []
    for i in lbs["leaderboard_id"]:
        p = beatleader.scores_path(paths, i)
        if p.exists():
            score_rows.extend(beatleader.flatten_scores(beatleader.load(p)))
    scores = pl.DataFrame(score_rows, infer_schema_length=None)
    scores.write_parquet(paths.processed / ("scores_pilot.parquet" if pilot else "scores.parquet"))
    summary["leaderboards"] = lbs.height
    summary["diffs_with_leaderboard_pct"] = round(100 * lbs.height / max(1, diffs.height), 1)
    summary["scores"] = scores.height
    summary["errors"] = state.errors("bl-leaderboards", 5) + state.errors("bl-scores", 5)


async def _count(coro, key=None) -> dict:
    result = await coro
    return {"n": len(result[key] if key else result)}

crawl_leaderboards = _with_state("crawl-leaderboards", _crawl_leaderboards)


# --------------------------------------------------------------------------- replays

def _select_replays(paths, state, summary, pilot=False, k=3):
    suffix = "_pilot" if pilot else ""
    lbs = pl.read_parquet(paths.processed / f"leaderboards{suffix}.parquet")
    scores = pl.read_parquet(paths.processed / f"scores{suffix}.parquet")
    diffs = pl.read_parquet(paths.processed / "difficulties.parquet").select(
        "hash", "characteristic", "difficulty", "n_notes")
    s = scores.join(lbs, on="leaderboard_id").join(diffs, on=["hash", "characteristic", "difficulty"])
    funnel = {"scores": s.height}
    for name, expr in [
        ("no_modifiers", pl.col("modifiers") == ""),
        ("no_pauses", pl.col("pauses") == 0),
        ("human_permanent_player", ~pl.col("player_bot") & ~pl.col("player_temporary")),
        ("few_mistakes", (pl.col("missed") + pl.col("bad_cuts"))
         <= pl.max_horizontal(pl.lit(3), (pl.col("n_notes") * 0.01).floor())),
    ]:
        s = s.filter(expr)
        funnel[name] = s.height

    n_lb = s["leaderboard_id"].n_unique()
    cap = max(1, math.ceil(0.01 * k * n_lb))
    per_player: dict[str, int] = {}
    picks = []
    for lb_id, group in sorted(s.group_by("leaderboard_id"), key=lambda g: g[0][0]):
        rows = group.sort("rank").to_dicts()
        rng = random.Random(lb_id[0])
        best, rest = rows[0], rows[1:]
        rng.shuffle(rest)
        chosen: list[tuple[dict, str]] = []
        for row, why in [(best, "top_acc")] + [(r, "random_top_half") for r in rest]:
            if len(chosen) == k:
                break
            pid = row["player_id"]
            if per_player.get(pid, 0) >= cap or any(c[0]["player_id"] == pid for c in chosen):
                continue
            chosen.append((row, why))
            per_player[pid] = per_player.get(pid, 0) + 1
        picks.extend({**row, "pick_reason": why} for row, why in chosen)
    sel = pl.DataFrame(picks, infer_schema_length=None)
    sel.write_parquet(paths.processed / f"replay_selection{suffix}.parquet")
    funnel["selected"] = sel.height
    summary.update(funnel=funnel, leaderboards_with_candidates=n_lb, player_cap=cap,
                   distinct_players=sel["player_id"].n_unique(),
                   leaderboards_short_of_k=int((sel.group_by("leaderboard_id").len()["len"] < k).sum()))

select_replays = _with_state("select-replays", _select_replays)


async def _download_replays(paths, state, summary, pilot=False):
    sel = pl.read_parquet(paths.processed / f"replay_selection{'_pilot' if pilot else ''}.parquet")
    done = state.done("replay-download")
    todo = [(str(r["score_id"]), r) for r in sel.iter_rows(named=True) if str(r["score_id"]) not in done]
    summary["todo"] = len(todo)
    async with PoliteClient() as client:
        await _gather_items(todo, lambda r: beatleader.download_replay(client, paths, r),
                            state, "replay-download", paths)
        summary["requests"] = client.request_count
    summary["status"] = state.counts("replay-download")
    summary["errors"] = state.errors("replay-download", 10)

download_replays = _with_state("download-replays", _download_replays)


def _parse_one_replay(bsor_path: str, frames_path: str, sel: dict, notes: dict) -> dict:
    from kata.data import bsor
    from kata.data.align import align

    row = {k: sel[k] for k in ("score_id", "player_id", "leaderboard_id", "hash", "characteristic",
                               "difficulty", "accuracy", "rank", "modifiers", "pauses", "full_combo",
                               "missed", "bad_cuts", "pick_reason")}
    row.update(status="ok", reject_reason=None)
    events_out: list[dict] = []
    try:
        r = bsor.parse(Path(bsor_path).read_bytes())
    except Exception as e:
        row.update(status="rejected", reject_reason=f"bsor: {type(e).__name__}: {e}"[:300])
        return {"replay": row, "events": events_out}
    info = r.info
    t = r.frames["t"].astype(np.float64)
    dt = np.diff(t)
    heights = [h for h, _ in r.heights if h > 0]
    row.update(platform=info["platform"], hmd=info["hmd"], controller=info["controller"],
               tracking_system=info["trackingSystem"], game_version=info["gameVersion"],
               bsor_version=info["version"], left_handed=bool(info["leftHanded"]),
               jump_distance=info["jumpDistance"], height_median=float(np.median(heights)) if heights else None,
               n_frames=len(r.frames), fps_median=float(1 / np.median(dt[dt > 0])) if len(dt) else None,
               frame_dt_max_ms=float(dt.max() * 1000) if len(dt) else None,
               non_monotonic_frames=int((dt <= 0).sum()), n_pauses=len(r.pauses),
               n_wall_hits=len(r.walls), has_saber_offsets=r.saber_offsets is not None,
               bytes=Path(bsor_path).stat().st_size)
    if r.saber_offsets is not None:
        row["saber_offsets"] = list(r.saber_offsets)

    reasons = []
    if info["hash"].lower() != sel["hash"]:
        reasons.append("hash_mismatch")
    if info["difficulty"] != sel["difficulty"] or info["mode"] != sel["characteristic"]:
        reasons.append("difficulty_mismatch")
    if info["leftHanded"]:
        reasons.append("left_handed")
    if info["modifiers"]:
        reasons.append("modifiers")
    # Pause entries in the first/last second are routine (BeatLeader does not count them);
    # only a pause mid-song breaks the motion stream.
    song_end = float(t.max()) if len(t) else 0.0
    mid_pauses = [p for p in r.pauses if 1.0 < p[1] < song_end - 1.0]
    row["n_mid_pauses"] = len(mid_pauses)
    if mid_pauses:
        reasons.append("mid_song_pause")

    res = align(notes["x"], notes["y"], notes["color"], notes["cut_dir"], notes["time_s"], r.notes)
    row.update(align_events=res.n_considered, align_matched=res.n_matched,
               align_match_rate=res.match_rate, align_p50_ms=res.residual_quantile(0.5),
               align_p95_ms=res.residual_quantile(0.95), align_max_ms=res.residual_quantile(1.0))
    if res.match_rate < ALIGN_MIN_MATCH_RATE:
        reasons.append("align_match_rate")
    elif not res.residual_quantile(0.95) <= ALIGN_MAX_P95_MS:
        reasons.append("align_residual")
    if reasons:
        row.update(status="rejected", reject_reason=",".join(reasons))

    for e, m, resid in zip(r.notes, res.matched_idx.tolist(), res.residual_ms.tolist()):
        events_out.append({"score_id": sel["score_id"], **e, "matched_note_idx": m,
                           "residual_ms": None if math.isnan(resid) else resid})
    if not reasons and not Path(frames_path).exists():
        Path(frames_path).parent.mkdir(parents=True, exist_ok=True)
        pl.from_numpy(r.frames).write_parquet(frames_path)
    return {"replay": row, "events": events_out}


def _parse_replays(paths, state, summary, pilot=False):
    suffix = "_pilot" if pilot else ""
    sel = pl.read_parquet(paths.processed / f"replay_selection{suffix}.parquet")
    downloaded = state.done("replay-download")
    sel = sel.filter(pl.col("score_id").cast(pl.String).is_in(list(downloaded)))
    notes = pl.read_parquet(paths.processed / "notes.parquet")
    by_diff = {k: g for k, g in notes.group_by(["hash", "characteristic", "difficulty"])}
    results = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = []
        for r in sel.iter_rows(named=True):
            g = by_diff[(r["hash"], r["characteristic"], r["difficulty"])]
            note_arrays = {c: g[c].to_numpy() for c in ("x", "y", "color", "cut_dir", "time_s")}
            futures.append(pool.submit(
                _parse_one_replay, str(paths.replays / r["hash"] / f"{r['score_id']}.bsor"),
                str(paths.replay_frames / f"hash={r['hash']}" / f"{r['score_id']}.parquet"), r, note_arrays))
        results = [f.result() for f in futures]
    replays = pl.DataFrame([r["replay"] for r in results], infer_schema_length=None)
    events = pl.DataFrame([e for r in results for e in r["events"]], infer_schema_length=None)
    replays.write_parquet(paths.processed / f"replays{suffix}.parquet")
    events.write_parquet(paths.processed / f"replay_note_events{suffix}.parquet")
    summary["replays"] = replays.height
    summary["status"] = dict(replays.group_by("status").len().iter_rows())
    summary["reject_reasons"] = dict(replays.filter(pl.col("status") != "ok")
                                     .group_by("reject_reason").len().iter_rows())

parse_replays = _with_state("parse-replays", _parse_replays)


# --------------------------------------------------------------------------- build + report

_BRACKETS = re.compile(r"[\(\[\{【（].*?[\)\]\}】）]")
_FEAT = re.compile(r"\b(feat|ft)\.?\s.*$")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalise_title(s: str | None) -> str:
    s = (s or "").lower()
    s = _BRACKETS.sub(" ", s)
    s = _FEAT.sub(" ", s)
    return _NON_WORD.sub(" ", s).strip()


def song_groups(maps: pl.DataFrame) -> dict[str, str]:
    """hash -> group key. Union of: same normalised (author, title), same audio sha256."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    name_key = {}
    for r in maps.iter_rows(named=True):
        key = f"name:{normalise_title(r['song_author'])}|{normalise_title(r['song_name'])}"
        name_key[r["hash"]] = key
        union(f"map:{r['hash']}", key)
        if r.get("audio_sha256"):
            union(f"map:{r['hash']}", f"audio:{r['audio_sha256']}")
    # Group key = smallest *name* key in the component, so splits are stable as maps are added.
    comp_names: dict[str, str] = {}
    for h, key in name_key.items():
        root = find(f"map:{h}")
        comp_names[root] = min(comp_names.get(root, key), key)
    return {h: comp_names[find(f"map:{h}")] for h in name_key}


def split_for(group_key: str) -> str:
    bucket = int(hashlib.md5(group_key.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < 90 else "val" if bucket < 95 else "test"


def _build(paths, state, summary):
    maps = pl.read_parquet(paths.processed / "maps.parquet")
    groups = song_groups(maps)
    maps = maps.with_columns(
        pl.col("hash").replace_strict(groups, return_dtype=pl.String).alias("song_group_id"))
    maps = maps.with_columns(pl.col("song_group_id").map_elements(split_for, return_dtype=pl.String).alias("split"))
    maps.write_parquet(paths.processed / "maps.parquet")
    leaks = maps.group_by("song_group_id").agg(pl.col("split").n_unique().alias("n")).filter(pl.col("n") > 1)
    assert leaks.height == 0, "song group in more than one split"

    con = duckdb.connect(str(paths.query_db))
    for f in sorted(paths.processed.glob("*.parquet")):
        con.execute(f"CREATE OR REPLACE VIEW {f.stem} AS SELECT * FROM read_parquet('{f.as_posix()}')")
    for f in sorted(paths.lowrated.glob("*.parquet")):   # Phase 3 Tier D
        con.execute(f"CREATE OR REPLACE VIEW lowrated_{f.stem} AS SELECT * FROM read_parquet('{f.as_posix()}')")
    frames_glob = (paths.replay_frames / "*" / "*.parquet").as_posix()
    if any(paths.replay_frames.glob("*/*.parquet")):
        con.execute(f"""CREATE OR REPLACE VIEW replay_frames AS
            SELECT * FROM read_parquet('{frames_glob}', hive_partitioning = true, filename = true)""")
    con.close()
    summary["song_groups"] = len(set(groups.values()))
    summary["maps_by_split"] = dict(maps.group_by("split").len().sort("split").iter_rows())
    summary["views"] = sorted(f.stem for f in paths.processed.glob("*.parquet"))

build = _with_state("build", _build)


def _report(paths, state, summary, name="phase1"):
    from kata.data.report import write_report
    summary["path"] = str(write_report(paths, state, name))

report = _with_state("report", _report)


PILOT_STAGES = [
    ("select", select, {}),
    ("download-maps", download_maps, {"pilot": True}),
    ("parse-maps", parse_maps, {"pilot": True}),
    ("crawl-leaderboards", crawl_leaderboards, {"pilot": True}),
    ("select-replays", select_replays, {"pilot": True, "k": 3}),
    ("download-replays", download_replays, {"pilot": True}),
    ("parse-replays", parse_replays, {"pilot": True}),
    ("build", build, {}),
    ("report", report, {"name": "pilot"}),
]
