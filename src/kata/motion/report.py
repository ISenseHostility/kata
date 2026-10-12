"""Phase 4 Markdown report (docs/phases/phase-4-motion.md §8, criterion 12)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from kata.config import Paths
from kata.data.report import md_table
from kata.motion.evaluate import QUANTITIES


def _tracks_section(paths: Paths, suffix: str) -> list[str]:
    f = paths.processed / f"motion_tracks{suffix}.parquet"
    if not f.exists():
        return ["_(no motion tracks yet)_", ""]
    t = pl.read_parquet(f)
    out = [md_table(t.group_by("status", "reject_reason").len().sort("status", "reject_reason")), ""]
    ok = t.filter(pl.col("status") == "ok")
    out += [f"Accepted: {ok.height} (left-handed recovered: {ok.filter(pl.col('mirrored')).height}; "
            f"world mirrored: {ok.filter(pl.col('world_mirrored')).height}; "
            f"hands swapped: {ok.filter(pl.col('hands_swapped')).height}).", ""]
    if ok.height:
        b = ok["blade_median_m"].to_numpy()
        out += ["**Blade check** (median cut point to swept blade plane, m): " +
                ", ".join(f"p{q} {np.percentile(b, q):.3f}" for q in (5, 50, 95, 99)), ""]
        rows = []
        for k in range(3):
            sub = ok.drop_nulls(["head_y_median_m", f"hand_y_l{k}"])
            if sub.height >= 3 and sub["head_y_median_m"].n_unique() >= 2:
                slope, icpt = np.polyfit(sub["head_y_median_m"].to_numpy(), sub[f"hand_y_l{k}"].to_numpy(), 1)
                rows.append({"layer": k, "n": sub.height, "slope_m_per_m": float(slope), "intercept_m": float(icpt)})
        out += ["**Hand height at hits vs. median head height** (raw, before normalising; spec §3.2):", "",
                md_table(pl.DataFrame(rows)) if rows else "_(too few tracks)_", ""]
    return out


def _runs_section(paths: Paths) -> list[str]:
    rows = []
    for d in sorted(p for p in paths.motion_runs.glob("*") if p.is_dir()):
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        log = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()] if (d / "log.jsonl").exists() else []
        evals = [r for r in log if "eval" in r]
        speed = [r["steps_per_s"] for r in log if "steps_per_s" in r]
        g = json.loads((d / "guidance.json").read_text())["guidance"] if (d / "guidance.json").exists() else None
        rows.append({"run": d.name, "pilot": cfg.get("pilot"), "window": cfg.get("window"),
                     "mirror_aug": cfg.get("mirror_aug"), "use_stems": cfg.get("use_stems"),
                     "steps": log[-1]["step"] if log else 0, "steps_per_s": speed[-1] if speed else None,
                     "last_val_f": evals[-1]["eval"]["val_timing_f"] if evals else None,
                     "best_val_f": max((e["eval"]["val_timing_f"] or 0) for e in evals) if evals else None,
                     "last_rate_ratio": evals[-1]["eval"]["val_rate_ratio"] if evals else None, "guidance": g})
    return [md_table(pl.DataFrame(rows)) if rows else "_(no runs yet)_", ""]


def _eval_section(paths: Paths, run: str) -> list[str]:
    out = []
    d = paths.motion_runs / run
    ev_path = paths.processed / "motion_eval.parquet"
    for split in ("val", "test"):
        f = d / f"eval_{split}.json"
        if not f.exists():
            continue
        e = json.loads(f.read_text())
        out += [f"### {split}", "", "**Acceptance criteria**", "",
                md_table(pl.DataFrame(e["criteria"]).select("criterion", "name", "value", "pass")), ""]
        q = [{"quantity": k, "gen_p50": (e["pooled"]["generated"][k] or [None, None])[0],
              "real_p50": (e["pooled"]["real"][k] or [None, None])[0],
              "gen_p95": (e["pooled"]["generated"][k] or [None, None])[1],
              "real_p95": (e["pooled"]["real"][k] or [None, None])[1]} for k in QUANTITIES]
        out += ["**Plausibility percentiles**", "", md_table(pl.DataFrame(q)), ""]
        if ev_path.exists():
            r = pl.read_parquet(ev_path).filter((pl.col("run") == run) & (pl.col("split") == split)
                                                 & pl.col("mode").is_in(["full", "replay"]))
            per = (r.group_by("label", "source").agg(pl.len().alias("n"), pl.col("timing_f").median(),
                                                     pl.col("hit_rate").median(), pl.col("nps").median())
                   .sort("label", "source"))
            out += ["**Per label**", "", md_table(per), ""]
        if e.get("speed"):
            out += [f"**Speed** (generate(), one song at one difficulty, scaled to 3 min): "
                    f"max {max(e['speed']):.1f} s over {len(e['speed'])} songs.", ""]
    return out or ["_(not evaluated yet)_", ""]


def write_report(paths: Paths, state, name: str, run: str | None = None, pilot: bool = False) -> Path:
    out = [f"# Phase 4 motion report — {name}", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data root `{paths.root}`.", ""]
    runs = state.con.execute("""
        SELECT stage, args, round(seconds, 1) AS seconds, summary FROM runs WHERE stage LIKE 'motion-%'
        QUALIFY row_number() OVER (PARTITION BY stage, args ORDER BY started_at DESC) = 1
        ORDER BY started_at""").pl()
    out += ["## 1. Stage timings (latest run of each)", "", md_table(runs.select("stage", "args", "seconds")), ""]
    out += ["## 2. Motion tracks (spec §3)", ""] + _tracks_section(paths, "_pilot" if pilot else "")
    out += ["## 3. Training runs and ablations (spec §6)", ""] + _runs_section(paths)
    if run is None and (paths.motion_runs / "final").exists():
        run = (paths.motion_runs / "final").read_text().strip()
    out += [f"## 4. Evaluation of run `{run}` (spec §7-§8)", ""] + (_eval_section(paths, run) if run else
                                                                   ["_(no run selected)_", ""])
    path = paths.reports / f"{name}-{datetime.now():%Y-%m-%d}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
