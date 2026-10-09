"""Markdown data report: funnel counts, distributions, sizes, timings."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

from kata.config import Paths
from kata.data.state import State


def md_table(df: pl.DataFrame, floats: int = 3) -> str:
    if df.height == 0:
        return "_(none)_\n"
    cols = df.columns

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:,.{floats}f}" if abs(v) < 1e6 else f"{v:,.0f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join(lines) + "\n"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def gb(n: float) -> str:
    return f"{n / 1024**3:.2f} GB"


def write_report(paths: Paths, state: State, name: str) -> Path:
    P = paths.processed
    con = duckdb.connect()
    for f in P.glob("*.parquet"):
        con.execute(f"CREATE VIEW {f.stem} AS SELECT * FROM read_parquet('{f.as_posix()}')")
    q = lambda sql: con.execute(sql).pl()  # noqa: E731
    has = lambda t: (P / f"{t}.parquet").exists()  # noqa: E731
    pilot = name == "pilot"
    sfx = "_pilot" if pilot else ""
    out: list[str] = [f"# Phase 1 data report — {name}", "",
                      f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data root `{paths.root}`.", ""]

    runs = state.con.execute("""
        SELECT stage, args, round(seconds, 1) AS seconds, summary FROM runs
        QUALIFY row_number() OVER (PARTITION BY stage, args ORDER BY started_at DESC) = 1
        ORDER BY started_at""").pl()
    last = {r["stage"]: json.loads(r["summary"] or "{}") for r in runs.iter_rows(named=True)}

    out += ["## 1. Stage timings (latest run of each)", "",
            md_table(runs.select("stage", "args", "seconds")), ""]

    sel = last.get("select", {})
    if sel:
        out += ["## 2. Catalogue → selection funnel", "",
                md_table(pl.DataFrame({"step": list(sel["funnel"]), "maps": list(sel["funnel"].values())})),
                f"Maps by tier: `{sel['maps_by_tier']}` · Standard diffs by tier: `{sel['diffs_by_tier']}` · "
                f"pilot maps: {sel['pilot_maps']}", "",
                "### Tier B tuning grid (non-A maps passing hard filters; count by rating × min upvotes)", "",
                md_table(pl.DataFrame(sel["tier_b_tuning_grid"])), ""]

    if has("maps"):
        out += ["## 3. Maps", ""]
        out += [md_table(q(f"""
            SELECT map_tier, count(*) maps,
              count(*) FILTER (WHERE parse_status = 'ok') info_ok,
              count(*) FILTER (WHERE audio_status = 'ok') audio_ok,
              round(avg(audio_bytes) / 1048576, 2) avg_audio_mb,
              count(*) FILTER (WHERE abs(audio_duration_s - meta_duration_s) > 2) dur_mismatch_gt2s,
              count(*) FILTER (WHERE song_time_offset <> 0) nonzero_song_time_offset
            FROM maps {"WHERE pilot" if pilot else ""} GROUP BY 1 ORDER BY 1""")), ""]
        out += ["Info.dat versions:", "", md_table(q("SELECT info_format_version, count(*) n FROM maps GROUP BY 1 ORDER BY 2 DESC")), ""]
        bad_audio = q("SELECT hash, audio_error FROM maps WHERE audio_status <> 'ok' OR parse_status <> 'ok' LIMIT 10")
        if bad_audio.height:
            out += ["Map/audio errors (first 10):", "", md_table(bad_audio), ""]

        out += ["### Difficulties", "", md_table(q("""
            SELECT tier, parse_status, count(*) n FROM difficulties GROUP BY ALL ORDER BY ALL""")), ""]
        out += ["Format versions × tempo source (accepted):", "", md_table(q("""
            SELECT left(format_version, 3) fmt, tempo_source, count(*) n,
                   count(*) FILTER (WHERE n_bpm_changes > 0) with_bpm_changes
            FROM difficulties WHERE parse_status = 'ok' GROUP BY ALL ORDER BY ALL""")), ""]
        out += ["Reject reasons:", "", md_table(q("""
            SELECT regexp_replace(reject_reason, ':.*', '') reason, count(*) n, any_value(reject_reason) example
            FROM difficulties WHERE parse_status <> 'ok' GROUP BY 1 ORDER BY 2 DESC""")), ""]
        out += ["Parse warnings:", "", md_table(q("""
            SELECT w AS warning, count(*) n FROM (SELECT unnest(parse_warnings) w FROM difficulties) GROUP BY 1""")), ""]
        out += ["Distributions (accepted diffs; p10 / p50 / p90):", "", md_table(q("""
            SELECT tier, count(*) n,
              printf('%.1f / %.1f / %.1f', quantile_cont(nps, .1), quantile_cont(nps, .5), quantile_cont(nps, .9)) nps,
              printf('%.1f / %.1f / %.1f', quantile_cont(njs, .1), quantile_cont(njs, .5), quantile_cont(njs, .9)) njs,
              printf('%.1f / %.1f / %.1f', quantile_cont(bl_stars, .1), quantile_cont(bl_stars, .5), quantile_cont(bl_stars, .9)) bl_stars,
              printf('%.0f / %.0f / %.0f', quantile_cont(duration_s, .1), quantile_cont(duration_s, .5), quantile_cont(duration_s, .9)) duration_s,
              sum(n_arcs) arcs, sum(n_chains) chains
            FROM difficulties WHERE parse_status = 'ok' GROUP BY 1 ORDER BY 1""")), ""]
        out += ["Difficulty names (accepted):", "", md_table(q("""
            SELECT difficulty, count(*) FILTER (WHERE tier='A') a, count(*) FILTER (WHERE tier='B') b,
                   count(*) FILTER (WHERE tier='C') c
            FROM difficulties WHERE parse_status = 'ok' GROUP BY 1 ORDER BY min(difficulty_rank)""")), ""]

    lb_last = last.get("crawl-leaderboards", {})
    if has(f"leaderboards{sfx}"):
        out += ["## 4. BeatLeader coverage", "", md_table(q(f"""
            SELECT d.tier, count(*) accepted_diffs, count(l.leaderboard_id) with_leaderboard,
                   count(*) FILTER (WHERE l.bl_status = 3) bl_ranked_status
            FROM difficulties d LEFT JOIN leaderboards{sfx} l USING (hash, characteristic, difficulty)
            WHERE d.parse_status = 'ok' {"AND d.hash IN (SELECT hash FROM maps WHERE pilot)" if pilot else ""}
            GROUP BY 1 ORDER BY 1""")),
            f"Scores fetched: {lb_last.get('scores')} · errors: `{lb_last.get('errors')}`", "",
            "Scores fetched per leaderboard:", "", md_table(q(f"""
            SELECT d.tier, count(DISTINCT s.leaderboard_id) leaderboards,
                   quantile_cont(n, .5) median_scores, max(n) max_scores, quantile_cont(plays, .5) median_plays
            FROM (SELECT leaderboard_id, count(*) n, any_value(plays) plays FROM scores{sfx} GROUP BY 1) s
            JOIN leaderboards{sfx} USING (leaderboard_id)
            JOIN difficulties d USING (hash, characteristic, difficulty) GROUP BY 1 ORDER BY 1""")), ""]

    rs = last.get("select-replays", {})
    if rs:
        out += ["## 5. Replay selection", "",
                md_table(pl.DataFrame({"step": list(rs["funnel"]), "scores": list(rs["funnel"].values())})),
                f"Leaderboards with candidates: {rs['leaderboards_with_candidates']} · short of K: "
                f"{rs['leaderboards_short_of_k']} · distinct players: {rs['distinct_players']} · "
                f"per-player cap: {rs['player_cap']}", ""]

    if has(f"replays{sfx}"):
        R = f"replays{sfx}"
        out += ["## 6. Replays", "", md_table(q(f"""
            SELECT status, coalesce(reject_reason, '') reason, count(*) n FROM {R} GROUP BY ALL ORDER BY n DESC""")), ""]
        out += ["Alignment (right-handed replays):", "", md_table(q(f"""
            SELECT count(*) replays,
              printf('%.4f / %.4f / %.4f', min(align_match_rate), quantile_cont(align_match_rate, .1), quantile_cont(align_match_rate, .5)) match_rate_min_p10_p50,
              printf('%.3f / %.3f / %.3f', quantile_cont(align_p95_ms, .5), quantile_cont(align_p95_ms, .9), max(align_p95_ms)) p95_ms_p50_p90_max,
              printf('%.3f', quantile_cont(abs(align_p50_ms), .5)) median_abs_p50_ms
            FROM {R} WHERE NOT left_handed AND reject_reason IS DISTINCT FROM 'hash_mismatch'""")), ""]
        out += ["Worst-aligned right-handed replays (left-handed ones are mirrored and never align):", "", md_table(q(f"""
            SELECT r.score_id, r.hash, r.difficulty, d.format_version, d.tempo_source, d.n_bpm_changes,
                   m.song_time_offset, r.align_match_rate, r.align_p50_ms, r.align_p95_ms, r.game_version
            FROM {R} r JOIN difficulties d USING (hash, characteristic, difficulty) JOIN maps m USING (hash)
            WHERE NOT r.left_handed AND r.reject_reason IS DISTINCT FROM 'hash_mismatch'
            ORDER BY r.align_match_rate ASC, r.align_p95_ms DESC NULLS FIRST LIMIT 12""")), ""]
        out += ["Hardware / frame rate:", "", md_table(q(f"""
            SELECT hmd, count(*) n, round(quantile_cont(fps_median, .5), 1) fps_median,
                   round(quantile_cont(frame_dt_max_ms, .5), 1) median_max_dt_ms,
                   round(avg(bytes) / 1048576, 2) avg_mb
            FROM {R} GROUP BY 1 ORDER BY 2 DESC LIMIT 12""")), ""]
        out += ["Game versions:", "", md_table(q(f"""
            SELECT split_part(game_version, '_', 1) game, count(*) n FROM {R} GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")), ""]
        out += ["Replays per player:", "", md_table(q(f"""
            SELECT n_replays, count(*) players FROM (SELECT player_id, count(*) n_replays FROM {R} GROUP BY 1)
            GROUP BY 1 ORDER BY 1""")), ""]

    if has("maps"):
        out += ["## 7. Splits", "", md_table(q("""
            SELECT split, count(*) maps, count(DISTINCT song_group_id) song_groups FROM maps
            WHERE split IS NOT NULL GROUP BY 1 ORDER BY 1""")), ""]

    out += ["## 8. Disk usage", ""]
    sizes = {k: dir_size(v) for k, v in {"raw/maps": paths.maps, "raw/replays": paths.replays,
                                         "raw/beatsaver": paths.catalog_pages,
                                         "raw/beatleader": paths.raw / "beatleader",
                                         "processed": paths.processed}.items()}
    out += [md_table(pl.DataFrame({"dir": list(sizes), "size": [gb(v) for v in sizes.values()]})), ""]

    if pilot and has("maps") and sel:
        out += ["## 9. Projection to the full run", ""]
        n_maps = sum(v for k, v in sel["maps_by_tier"].items() if k in ("A", "B"))
        pm = q("SELECT count(*) n FROM maps WHERE pilot").item()
        map_bytes = sizes["raw/maps"] / max(1, pm)
        lines = [f"- Maps: {n_maps:,} × {map_bytes / 1048576:.1f} MB ≈ **{gb(n_maps * map_bytes)}**"]
        if has(f"replays{sfx}"):
            # Pilot is 50/50 A/B but the full set is not, so project per tier.
            per_tier = q(f"""
                SELECT m.map_tier, count(r.score_id) / count(DISTINCT m.hash) AS per_map
                FROM maps m LEFT JOIN replays{sfx} r USING (hash) WHERE m.pilot GROUP BY 1""")
            rate = dict(per_tier.iter_rows())
            n_rep = sum(rate.get(t, 0) * c for t, c in sel["maps_by_tier"].items())
            rep_bytes = q(f"SELECT avg(bytes) FROM replays{sfx}").item() or 0
            frames_bytes = dir_size(paths.replay_frames) / max(1, q(f"SELECT count(*) FROM replays{sfx} WHERE status='ok'").item())
            lines += [f"- Replays per map in pilot: `{ {k: round(v, 2) for k, v in rate.items()} }` ⇒ ~{n_rep:,.0f} replays × "
                      f"{rep_bytes / 1048576:.2f} MB ≈ **{gb(n_rep * rep_bytes)}** raw + "
                      f"**{gb(n_rep * frames_bytes)}** parsed frames"]
        out += lines + [""]

    path = paths.reports / f"{name}-{datetime.now():%Y-%m-%d}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
