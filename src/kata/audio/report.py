"""Phase 2 Markdown report (docs/phases/phase-2-audio.md §12, criterion 10)."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from kata.audio.store import FeatureStore
from kata.audio.structure import boundary_density_check
from kata.config import Paths
from kata.data.report import dir_size, gb, md_table

TABLES = ("audio_features", "audio_beats", "audio_tempo", "audio_segments", "map_audio_sync",
          "beat_agreement", "maps", "difficulties", "notes")


def window_benchmark(store: FeatureStore, songs: pl.DataFrame, n: int = 200, seconds: float = 12.0) -> dict:
    """Median / p95 time of a random `seconds`-long window read (criterion 6)."""
    rng = np.random.default_rng(0)
    rows = songs.to_dicts()
    times = []
    for _ in range(n if rows else 0):
        r = rows[rng.integers(len(rows))]
        start = rng.uniform(0, max(r["duration_s"] - seconds, 0))
        t0 = time.perf_counter()
        store.window(r["audio_sha256"], start, start + seconds)
        times.append((time.perf_counter() - t0) * 1000)
    return {"n": len(times), "median_ms": float(np.median(times)) if times else None,
            "p95_ms": float(np.percentile(times, 95)) if times else None}


def structure_vs_density(con) -> pl.DataFrame:
    """Per tier: segment boundaries near a large note-density change, vs. chance (coverage)."""
    hardest = con.execute("""
        SELECT d.hash, d.characteristic, d.difficulty, d.tier, m.audio_sha256
        FROM difficulties d JOIN maps m USING (hash)
        WHERE d.parse_status = 'ok' AND m.audio_sha256 IN (SELECT audio_sha256 FROM audio_segments)
        QUALIFY row_number() OVER (PARTITION BY d.hash ORDER BY d.difficulty_rank DESC) = 1""").pl()
    out = []
    for r in hardest.iter_rows(named=True):
        times = con.execute("SELECT time_s FROM notes WHERE hash = ? AND characteristic = ? AND difficulty = ?",
                            [r["hash"], r["characteristic"], r["difficulty"]]).pl()["time_s"].to_numpy()
        segs = con.execute("SELECT start_s, end_s FROM audio_segments WHERE audio_sha256 = ? ORDER BY seg_idx",
                           [r["audio_sha256"]]).pl()
        if segs.height < 2 or len(times) == 0:
            continue
        c = boundary_density_check(segs["start_s"].to_numpy()[1:], times, float(segs["end_s"].max()))
        out.append({"tier": r["tier"], **c})
    if not out:
        return pl.DataFrame()
    return (pl.DataFrame(out).group_by("tier").agg(
        pl.len().alias("maps"), (pl.col("hits").sum() / pl.col("n").sum()).alias("boundary_hit_rate"),
        pl.col("coverage").mean().alias("chance_rate")).sort("tier"))


def write_report(paths: Paths, state, name: str) -> Path:
    P = paths.processed
    con = duckdb.connect()
    for t in TABLES:
        if (P / f"{t}.parquet").exists():
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{(P / f'{t}.parquet').as_posix()}')")
    has = lambda t: (P / f"{t}.parquet").exists()  # noqa: E731
    q = lambda sql: con.execute(sql).pl()  # noqa: E731
    store = FeatureStore(paths.features)
    out = [f"# Phase 2 audio report — {name}", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data root `{paths.root}`.", ""]

    runs = state.con.execute("""
        SELECT stage, args, round(seconds, 1) AS seconds, summary FROM runs WHERE stage LIKE 'audio-%'
        QUALIFY row_number() OVER (PARTITION BY stage, args ORDER BY started_at DESC) = 1
        ORDER BY started_at""").pl()
    last = {r["stage"]: json.loads(r["summary"] or "{}") for r in runs.iter_rows(named=True)}
    out += ["## 1. Stage timings (latest run of each)", "", md_table(runs.select("stage", "args", "seconds")), ""]

    if has("audio_features"):
        feats = q("SELECT * FROM audio_features")
        ok = feats.filter(pl.col("status") == "ok")
        size = dir_size(store.dir)
        frames = int(ok["n_frames"].sum()) if ok.height else 0
        out += ["## 2. Features", "",
                f"- Songs: {feats.height:,} · ok {ok.height:,} · errors {feats.height - ok.height:,}",
                f"- Audio: {ok['duration_s'].sum() / 3600 if ok.height else 0:.1f} h · {frames:,} frames",
                f"- Disk: {gb(size)} under `{store.dir}` ({size / max(frames, 1):.0f} B/frame)",
                f"- Window read (12 s, warm): `{window_benchmark(store, ok)}`", ""]
        if feats.height > ok.height:
            out += ["Errors:", "", md_table(q("""SELECT error, count(*) n FROM audio_features
                WHERE status <> 'ok' GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")), ""]
        if ok.height:
            out += [md_table(q("""SELECT source_sr, count(*) songs, sum(CAST(bandlimited AS INT)) bandlimited,
                    round(median(rms_dbfs), 1) median_rms_dbfs, round(avg(clip_ratio) * 100, 3) clip_pct
                    FROM audio_features WHERE status = 'ok' GROUP BY 1 ORDER BY 2 DESC""")), ""]

    if has("audio_tempo"):
        out += ["## 3. Tempo", "", md_table(q("""SELECT est_meter, count(*) songs,
                round(quantile_cont(est_bpm, 0.5), 1) median_bpm, round(quantile_cont(tempo_cv, 0.5), 4) median_cv,
                round(quantile_cont(tempo_cv, 0.95), 4) p95_cv FROM audio_tempo GROUP BY 1 ORDER BY 2 DESC""")), ""]

    if has("beat_agreement"):
        out += ["## 4. Map grid vs. audio beats", "",
                md_table(q("""SELECT tier, best_level, count(*) diffs, round(avg(best_f), 3) mean_f
                    FROM beat_agreement GROUP BY ALL ORDER BY tier, best_level""")),
                md_table(q("""SELECT tier, count(*) diffs,
                    round(avg(CAST(best_f >= 0.8 AS INT)), 3) share_best_f_ge_08,
                    round(quantile_cont(f_l1, 0.5), 3) median_f_level1
                    FROM beat_agreement GROUP BY 1 ORDER BY 1""")), ""]

    if has("map_audio_sync"):
        out += ["## 5. Map ↔ audio sync", "",
                f"Onset detector bias (subtracted): {last.get('audio-sync', {}).get('onset_bias_ms', float('nan')):+.2f} ms", "",
                md_table(q("""SELECT tier, src, count(*) diffs,
                    round(median(lag_ms), 2) median_lag_ms,
                    round(quantile_cont(lag_ms, 0.25), 2) p25, round(quantile_cont(lag_ms, 0.75), 2) p75,
                    round(avg(CAST(abs(lag_ms) <= 30 AS INT)), 3) share_within_30ms,
                    count(*) FILTER (WHERE abs(lag_ms) > 50) n_over_50ms,
                    round(median(peak_ratio), 2) median_peak_ratio
                    FROM map_audio_sync WHERE lag_ms IS NOT NULL AND NOT isnan(lag_ms)
                    GROUP BY ALL ORDER BY tier, src""")),
                "Tier A, mix onsets — lag histogram (5 ms buckets, clipped to ±50 ms):", "",
                md_table(q("""SELECT greatest(-50, least(50, floor(lag_ms / 5) * 5)) bucket_ms, count(*) diffs
                    FROM map_audio_sync WHERE tier = 'A' AND src = 'mix' AND NOT isnan(lag_ms)
                    GROUP BY 1 ORDER BY 1""")),
                "Per-map consistency (std of lag across a map's difficulties, mix):", "",
                md_table(q("""SELECT round(quantile_cont(s, 0.5), 2) median_std_ms, round(quantile_cont(s, 0.95), 2) p95_std_ms
                    FROM (SELECT hash, stddev_samp(lag_ms) s FROM map_audio_sync WHERE src = 'mix'
                          GROUP BY hash HAVING count(*) > 1)""")), ""]

    if has("audio_segments"):
        out += ["## 6. Structure", "",
                md_table(q("""SELECT count(DISTINCT audio_sha256) songs,
                    round(avg(n) , 1) mean_segments, round(quantile_cont(n, 0.5), 1) median_segments,
                    round(quantile_cont(len, 0.5), 1) median_seg_s, round(avg(nl), 2) mean_labels FROM (
                      SELECT audio_sha256, count(*) OVER w n, end_s - start_s len, n_labels nl
                      FROM audio_segments WINDOW w AS (PARTITION BY audio_sha256))""")),
                "Boundaries within 2 s of a large note-density change (hardest diff per map) vs. chance:", "",
                md_table(structure_vs_density(con)) if has("notes") else "_(no notes table)_" + chr(10), ""]

    if store.stats_path.exists():
        st = json.loads(store.stats_path.read_text())
        out += ["## 7. Normalisation stats", "",
                f"- From {st['n_songs']} {st['split']}-split songs; arrays {sorted(st['arrays'])}", ""]

    path = paths.reports / f"{name}-{datetime.now():%Y-%m-%d}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    return path
