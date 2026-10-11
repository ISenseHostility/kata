"""Phase 3 Markdown report (spec §11, criterion 11)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from kata.config import Paths
from kata.data.report import md_table
from kata.validate.pipeline import corruption_summary
from kata.validate.rules import SCORED_RULES

YEAR_BINS = [(0, 2019, "≤2019"), (2020, 2021, "2020-21"), (2022, 2023, "2022-23"), (2024, 9999, "≥2024")]
NPS_BINS = [(0, 4, "0-4"), (4, 6, "4-6"), (6, 8, "6-8"), (8, 1e9, "8+")]
MIN_PER_SIDE = 20
# Section 4's comparison sets. The pilot report separates the tuning halves only (spec §6.2, §10,
# criterion 9); the held-out halves are reported once, after tuning.
CALIBRATION = {"holdout": (("val", "test"), "holdout", "Tier A val+test vs. Tier D holdout"),
               "fit": (("train",), "fit", "Tier A train vs. Tier D fit")}


def auc(pos, neg) -> float:
    """P(pos > neg) + ½ P(tie): Mann-Whitney U from average ranks."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def year_bin(year: int) -> str:
    return next(label for lo, hi, label in YEAR_BINS if lo <= year <= hi)


def nps_bin(nps: float) -> str:
    return next(label for lo, hi, label in NPS_BINS if lo <= nps < hi)


def stratified_auc(a: pl.DataFrame, d: pl.DataFrame, col: str = "score") -> tuple[pl.DataFrame, float]:
    """AUC within upload-year × NPS cells with >= MIN_PER_SIDE maps on each side, and the
    size-weighted mean over those cells (criterion 4)."""
    def cells(df):
        return df.with_columns(
            pl.col("year").map_elements(year_bin, return_dtype=pl.String).alias("years"),
            pl.col("nps").map_elements(nps_bin, return_dtype=pl.String).alias("nps_band"))
    a, d = cells(a), cells(d)
    rows = []
    for (years, band), ga in a.group_by("years", "nps_band"):
        gd = d.filter((pl.col("years") == years) & (pl.col("nps_band") == band))
        if ga.height >= MIN_PER_SIDE and gd.height >= MIN_PER_SIDE:
            rows.append({"years": years, "nps_band": band, "n_a": ga.height, "n_d": gd.height,
                         "auc": round(auc(ga[col], gd[col]), 3)})
    t = pl.DataFrame(rows, schema={"years": pl.String, "nps_band": pl.String, "n_a": pl.Int64,
                                   "n_d": pl.Int64, "auc": pl.Float64}).sort("years", "nps_band")
    w = (t["n_a"] + t["n_d"]).to_numpy()
    mean = float((t["auc"].to_numpy() * w).sum() / w.sum()) if t.height else float("nan")
    return t, mean


def write_report(paths: Paths, state, name: str, calibration: str = "holdout") -> Path:
    a_splits, d_split, title = CALIBRATION[calibration]
    P = paths.processed
    out = [f"# Phase 3 validator report — {name}", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data root `{paths.root}`.", ""]
    runs = state.con.execute("""
        SELECT stage, args, round(seconds, 1) AS seconds, summary FROM runs
        WHERE stage LIKE 'validate-%' OR stage LIKE 'baseline-%' OR stage LIKE '%lowrated%'
        QUALIFY row_number() OVER (PARTITION BY stage, args ORDER BY started_at DESC) = 1
        ORDER BY started_at""").pl()
    out += ["## 1. Stage timings (latest run of each)", "", md_table(runs.select("stage", "args", "seconds")), ""]
    last = {r["stage"]: json.loads(r["summary"] or "{}") for r in runs.iter_rows(named=True)}
    if "select-lowrated" in last:
        years = last["select-lowrated"].get("years", {})
        funnel = last["select-lowrated"].get("funnel", {})
        out += ["## 2. Tier D sample", "",
                md_table(pl.DataFrame({"step": list(funnel), "maps": list(funnel.values())})) if funnel else "", "",
                f"- Pool {last['select-lowrated'].get('pool_maps')} maps, sampled "
                f"{last['select-lowrated'].get('sampled_maps')} ({last['select-lowrated'].get('sampled_diffs')} diffs)", "",
                md_table(pl.DataFrame([{"year": y, **v} for y, v in years.items()])) if years else "", ""]

    if not (P / "validation.parquet").exists():
        return _write(paths, name, out)
    v = pl.read_parquet(P / "validation.parquet")
    if calibration == "fit":   # the tuning report never shows a held-out difficulty
        v = v.filter(~(((pl.col("tier") == "A") & pl.col("split").is_in(["val", "test"]))
                       | ((pl.col("tier") == "D") & (pl.col("split") == "holdout"))))
    out += ["## 3. Gate and score per tier", "", md_table(v.group_by("tier").agg(
        pl.len().alias("n"), pl.col("pass").mean().alias("pass_rate"),
        pl.col("score").quantile(0.1).alias("q10"), pl.col("score").median().alias("median"),
        *(pl.col(f"s_{r}").mean().alias(f"mean_s_{r}") for r in SCORED_RULES)).sort("tier")), "",
            "Score per stratum (difficulty label × NPS band):", "", md_table(v.with_columns(
                pl.col("stratum").str.split_exact("|", 1).struct.rename_fields(["label", "nps_band"]))
                .unnest("stratum").group_by("label", "nps_band", "tier").agg(
                pl.len().alias("n"), pl.col("pass").mean().alias("pass_rate"),
                *(pl.col("score").quantile(q).alias(f"q{round(q * 100)}") for q in (0.1, 0.5, 0.9)))
                .sort("label", "nps_band", "tier")), ""]

    in_a = (pl.col("tier") == "A") & pl.col("split").is_in(list(a_splits))
    in_d = (pl.col("tier") == "D") & (pl.col("split") == d_split)
    a, d = v.filter(in_a), v.filter(in_d)
    if a.height and d.height:
        cells, mean = stratified_auc(a, d)
        per_rule = {r: round(auc(a[f"s_{r}"], d[f"s_{r}"]), 3) for r in SCORED_RULES}
        per_label = v.filter(in_a | in_d)
        label_rows = [{"difficulty": lab, "n_a": g.filter(pl.col("tier") == "A").height,
                       "n_d": g.filter(pl.col("tier") == "D").height,
                       "auc": round(auc(g.filter(pl.col("tier") == "A")["score"],
                                        g.filter(pl.col("tier") == "D")["score"]), 3)}
                      for (lab,), g in per_label.group_by("difficulty")]
        out += [f"## 4. Separation: {title} (criterion 4)", "",
                f"- Gate pass: Tier A {a['pass'].mean():.4f} · Tier D {d['pass'].mean():.4f}",
                f"- **Overall AUC {auc(a['score'], d['score']):.3f}** (A {a.height}, D {d.height})",
                f"- **Per-stratum AUC, size-weighted mean {mean:.3f}** over {cells.height} cells "
                f"with ≥ {MIN_PER_SIDE} maps per side", "",
                md_table(cells), "", "Per-rule AUC (sub-scores):", "",
                md_table(pl.DataFrame([per_rule])), "", "Per difficulty label:", "",
                md_table(pl.DataFrame(label_rows).sort("difficulty")), ""]

    if (P / "corruption_results.parquet").exists():
        out += ["## 5. Synthetic corruptions (criterion 5)", "",
                md_table(corruption_summary(pl.read_parquet(P / "corruption_results.parquet"))), ""]

    if (P / "baseline_eval.parquet").exists():
        b = pl.read_parquet(P / "baseline_eval.parquet")
        q10 = (v.filter(pl.col("tier") == "A").group_by("stratum").agg(pl.col("score").quantile(0.1).alias("tier_a_q10")))
        bb = b.join(q10, on="stratum", how="left")
        out += ["## 6. Baseline (criterion 6)", "", md_table(bb.group_by("difficulty").agg(
            pl.len().alias("n"), pl.col("pass").mean().alias("pass_rate"), pl.col("score").median().alias("median_score"),
            pl.col("tier_a_q10").median().alias("tier_a_q10"), pl.col("nps").median().alias("median_nps"),
            pl.col("timing_f").median().alias("median_timing_f"), pl.col("round_trip").mean().alias("round_trip"))
            .sort("difficulty")), ""]

    v2 = v.filter(pl.col("format_version").str.starts_with("2") & pl.col("parity_errors").is_not_null())
    if v2.height:
        out += ["## 7. BeatSaver parity agreement (v2 only, diagnostic)", "", md_table(v2.group_by("tier").agg(
            pl.len().alias("n"), (pl.col("parity_errors") > 0).mean().alias("bs_errors"),
            (pl.col("E1") > 0).mean().alias("our_e1"),
            ((pl.col("parity_errors") > 0) == (pl.col("E1") > 0)).mean().alias("agree")).sort("tier")), ""]
    return _write(paths, name, out)


def _write(paths: Paths, name: str, out: list[str]) -> Path:
    path = paths.reports / f"{name}-{datetime.now():%Y-%m-%d}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
