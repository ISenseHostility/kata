"""Phase 3 validator stages (docs/phases/phase-3-validator.md §10). Same conventions as Phases 1-2:
every stage runs inside `_with_state` and rebuilds its tables deterministically from its inputs."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from kata.config import Paths
from kata.data.pipeline import _with_state
from kata.validate.corrupt import CORRUPTIONS, corrupt, evaluate
from kata.validate.reference import Reference, ValidatorDir
from kata.validate.rules import HARD_RULES, MapInput, analyse
from kata.validate.score import score_analysis, score_frame

KEY = ["hash", "characteristic", "difficulty"]
OBJECTS = ("notes", "bombs", "obstacles", "chains")
BATCH = 150
INJECTIONS_PER_KIND = 5
PILOT_CORRUPT_DIFFS = 100


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def difficulty_index(paths: Paths) -> pl.DataFrame:
    """Every parsed difficulty of Tiers A/B/C (processed/) and D (processed/lowrated/), with what
    the validator and the calibration need. Tier D's `split` is its calibration half. (The pilot
    needs no subset here: analysis is cheap, and only the pilot Tier D maps are parsed.)"""
    cols = ["hash", "characteristic", "difficulty", "tier", "njs", "njs_offset", "format_version",
            "parity_errors"]
    parts = []
    for source, folder, split_col in (("main", paths.processed, "split"), ("lowrated", paths.lowrated, "calib_half")):
        if not (folder / "difficulties.parquet").exists():
            continue
        maps = pl.read_parquet(folder / "maps.parquet").select(
            "hash", "info_bpm", pl.col(split_col).alias("split"), "pilot",
            pl.col("uploaded_at").str.slice(0, 4).cast(pl.Int64).alias("year"))
        d = (pl.read_parquet(folder / "difficulties.parquet").filter(pl.col("parse_status") == "ok")
             .select(cols).join(maps, on="hash"))
        parts.append(d.with_columns(pl.lit(source).alias("source")))
    return pl.concat(parts, how="vertical_relaxed").sort(KEY)


def _inputs(folders: dict[str, str], rows: list[dict], chains: bool = True) -> list[tuple[dict, MapInput]]:
    """MapInput per row, reading only this batch's difficulties from each source's tables."""
    out = []
    for source in sorted({r["source"] for r in rows}):   # sorted: set order varies per process
        mine = [r for r in rows if r["source"] == source]
        keys = pl.DataFrame({k: [r[k] for r in mine] for k in KEY}).lazy()
        tabs = {}
        for name in OBJECTS:
            f = Path(folders[source]) / f"{name}.parquet"
            tabs[name] = (pl.scan_parquet(f).join(keys, on=KEY, how="semi").collect().partition_by(KEY, as_dict=True)
                          if f.exists() else {})
        empty = pl.DataFrame(schema={"beat": pl.Float64, "time_s": pl.Float64, "x": pl.Int64, "y": pl.Int64,
                                     "color": pl.Int64, "cut_dir": pl.Int64, "angle_offset": pl.Int64})
        for r in mine:
            k = (r["hash"], r["characteristic"], r["difficulty"])
            out.append((r, MapInput(r["difficulty"], r["njs"] or 0.0, r["njs_offset"] or 0.0, r["info_bpm"],
                                    tabs["notes"].get(k, empty), tabs["bombs"].get(k), tabs["obstacles"].get(k),
                                    tabs["chains"].get(k) if chains else None)))
    return out


def _with_key(df: pl.DataFrame, r: dict) -> pl.DataFrame:
    return df.with_columns(*(pl.lit(r[k]).alias(k) for k in KEY)).select(KEY + df.columns)


def _analyse_batch(folders: dict[str, str], rows: list[dict]) -> dict:
    metrics, swings, issues, local = [], [], [], []
    for r, m in _inputs(folders, rows):
        base = {k: r[k] for k in KEY + ["tier", "split", "year", "source", "format_version", "parity_errors"]}
        try:
            a = analyse(m)
        except Exception as e:  # recorded per difficulty; the stage keeps going
            metrics.append({**base, "status": "error", "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        metrics.append({**base, "status": "ok", "error": None, **a.metrics})
        swings.append(_with_key(a.swings.drop("note_rows"), r))
        issues.append(_with_key(a.issues, r))
        local.append(_with_key(a.local, r))
    cat = lambda xs: pl.concat(xs, how="vertical_relaxed") if xs else None  # noqa: E731
    return {"metrics": metrics, "swings": cat(swings), "issues": cat(issues), "local": cat(local)}


def _folders(paths: Paths) -> dict[str, str]:
    return {"main": str(paths.processed), "lowrated": str(paths.lowrated)}


def _analyse(paths, state, summary):
    ValidatorDir(paths.validator).open_for_write()
    rows = difficulty_index(paths).to_dicts()
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    out = {"metrics": [], "swings": [], "issues": [], "local": []}
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        for i, res in enumerate(pool.map(_analyse_batch, [_folders(paths)] * len(batches), batches)):
            out["metrics"] += res["metrics"]
            for k in ("swings", "issues", "local"):
                if res[k] is not None:
                    out[k].append(res[k])
            if (i + 1) % 20 == 0:
                print(f"  [validate-analyse] {(i + 1) * BATCH}/{len(rows)}", flush=True)
    P = paths.processed
    metrics = pl.DataFrame(out["metrics"], infer_schema_length=None).sort(KEY)
    metrics.write_parquet(P / "validation_metrics.parquet")
    for name, order in (("swings", ["hand", "swing_idx"]), ("issues", ["time_s", "rule", "hand", "swing_idx"]),
                        ("local", ["window_start_beat"])):
        table = pl.concat(out[name], how="vertical_relaxed").sort(KEY + order)
        table.write_parquet(P / ("validation_local_metrics.parquet" if name == "local" else f"validation_{name}.parquet"))
    summary["difficulties"] = dict(metrics.group_by("tier", "status").len().sort("tier", "status")
                                   .select(pl.concat_str("tier", "status", separator="/"), "len").iter_rows())
    summary["errors"] = metrics.filter(pl.col("status") != "ok").select(KEY + ["error"]).head(10).to_dicts()

analyse_stage = _with_state("validate-analyse", _analyse)


def _reference(paths, state, summary):
    m = pl.read_parquet(paths.processed / "validation_metrics.parquet").filter(
        (pl.col("status") == "ok") & (pl.col("tier") == "A") & (pl.col("split") == "train"))
    assert set(m["split"].unique().to_list()) == {"train"}, "reference must use Tier A train only"
    ref = Reference.build(m)
    vd = ValidatorDir(paths.validator)
    vd.open_for_write()
    vd.write_reference(ref)
    summary.update(n_maps=m.height, stratum_sizes=ref.data["stratum_sizes"])

reference_stage = _with_state("validate-reference", _reference)


def _score(paths, state, summary):
    ref = ValidatorDir(paths.validator).load_reference()
    P = paths.processed
    m = pl.read_parquet(P / "validation_metrics.parquet").filter(pl.col("status") == "ok")
    m = m.with_columns(pl.struct("difficulty", "nps").map_elements(
        lambda r: ref.stratum(r["difficulty"], r["nps"]), return_dtype=pl.String).alias("stratum"))
    v = score_frame(m, ref).with_columns(
        pl.all_horizontal(*(pl.col(h) == 0 for h in HARD_RULES)).alias("pass"),
        pl.lit(ref.data["version"]).alias("validator_version"))
    v.write_parquet(P / "validation.parquet")
    local = pl.read_parquet(P / "validation_local_metrics.parquet").join(m.select(KEY + ["stratum"]), on=KEY)
    score_frame(local, ref).write_parquet(P / "validation_local.parquet")
    summary["by_tier"] = (v.group_by("tier").agg(pl.len().alias("n"), pl.col("pass").mean().round(4).alias("pass"),
                                                 pl.col("score").median().round(2).alias("median_score"))
                          .sort("tier").to_dicts())

score_stage = _with_state("validate-score", _score)


def _seed(key: tuple) -> int:
    return int(hashlib.md5("|".join(key).encode()).hexdigest()[:8], 16)


def _corrupt_batch(folders: dict[str, str], ref_data: dict, rows: list[dict]) -> list[dict]:
    ref = Reference(ref_data)
    out = []
    for r, m in _inputs(folders, rows, chains=False):
        a = analyse(m)
        clean = score_analysis(a, m.difficulty, ref)
        k = (r["hash"], r["characteristic"], r["difficulty"])
        rng = np.random.default_rng(_seed(k))
        q99 = ref.anchors("L1", clean.stratum)["q99"]
        for kind, rule in CORRUPTIONS.items():
            for i in range(INJECTIONS_PER_KIND):
                x = corrupt(kind, m, a, rng, q99)
                if x is None:
                    out.append({**dict(zip(KEY, k)), "corruption": kind, "rule": rule, "i": i, "eligible": False,
                                "detected": None, "false_alarms": None, "n_swings": clean.metrics["n_swings"]})
                    break
                inj, bad = x
                hit, fa = evaluate(inj, clean, score_analysis(analyse(bad), bad.difficulty, ref))
                out.append({**dict(zip(KEY, k)), "corruption": kind, "rule": rule, "i": i, "eligible": True,
                            "detected": hit, "false_alarms": fa, "n_swings": clean.metrics["n_swings"]})
    return out


def _corrupt(paths, state, summary, pilot=False):
    ref = ValidatorDir(paths.validator).load_reference()
    idx = difficulty_index(paths).filter((pl.col("tier") == "A") & pl.col("split").is_in(["val", "test"]))
    rows = (idx.head(PILOT_CORRUPT_DIFFS) if pilot else idx).to_dicts()
    batches = [rows[i:i + 20] for i in range(0, len(rows), 20)]
    res = []
    with ProcessPoolExecutor(max_workers=_workers()) as pool:
        for part in pool.map(_corrupt_batch, [_folders(paths)] * len(batches), [ref.data] * len(batches), batches):
            res += part
    df = pl.DataFrame(res, infer_schema_length=None).sort(KEY + ["corruption", "i"])
    df.write_parquet(paths.processed / "corruption_results.parquet")
    summary["by_corruption"] = corruption_summary(df).to_dicts()

corrupt_stage = _with_state("validate-corrupt", _corrupt)


def corruption_summary(df: pl.DataFrame) -> pl.DataFrame:
    ok = df.filter(pl.col("eligible"))
    return (ok.group_by("corruption", "rule").agg(
        pl.len().alias("injections"), pl.col("detected").mean().round(4).alias("recall"),
        (pl.col("false_alarms").sum() / pl.col("n_swings").sum() * 100).round(4).alias("false_alarm_pct"))
        .sort("corruption"))


def _report(paths, state, summary, name="phase3", calibration="holdout"):
    from kata.validate.report import write_report
    summary["path"] = str(write_report(paths, state, name, calibration))

report_stage = _with_state("validate-report", _report)
