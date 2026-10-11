"""Phase 3 validator stages on a tiny synthetic data root (Tier A train/val, Tier B, Tier D)."""
import polars as pl
import pytest

from conftest import stream
from kata.config import Paths
from kata.validate import pipeline as vp
from kata.validate.reference import ValidatorDir

KEY = ["hash", "characteristic", "difficulty"]
FLIP = {0: 1, 1: 0}


def _diff(h: str, broken: bool, n_per_hand: int = 48) -> pl.DataFrame:
    n = stream(n_per_hand=n_per_hand)
    if broken:   # every 4th note's arrow flipped: fast parity breaks
        n = n.with_columns(pl.when(pl.int_range(pl.len()) % 4 == 1).then(pl.col("cut_dir").replace(FLIP))
                           .otherwise(pl.col("cut_dir")).alias("cut_dir"))
    return n.with_columns(pl.lit(h).alias("hash"), pl.lit("Standard").alias("characteristic"),
                          pl.lit("ExpertPlus").alias("difficulty"), pl.int_range(pl.len()).alias("idx"))


def _write(folder, rows, split_col):
    maps = pl.DataFrame({"hash": [r[0] for r in rows], "info_bpm": [120.0] * len(rows),
                         split_col: [r[2] for r in rows], "pilot": [True] * len(rows),
                         "uploaded_at": ["2022-03-01T00:00:00Z"] * len(rows)})
    diffs = pl.DataFrame({"hash": [r[0] for r in rows], "characteristic": ["Standard"] * len(rows),
                          "difficulty": ["ExpertPlus"] * len(rows), "tier": [r[1] for r in rows],
                          "parse_status": ["ok"] * len(rows), "njs": [18.0] * len(rows), "njs_offset": [0.0] * len(rows),
                          "format_version": ["2.0.0"] * len(rows), "parity_errors": [0] * len(rows)})
    folder.mkdir(parents=True, exist_ok=True)
    maps.write_parquet(folder / "maps.parquet")
    diffs.write_parquet(folder / "difficulties.parquet")
    pl.concat([_diff(h, broken) for h, _, _, broken in rows]).write_parquet(folder / "notes.parquet")
    empty = {"hash": pl.String, "characteristic": pl.String, "difficulty": pl.String, "idx": pl.Int64,
             "beat": pl.Float64, "time_s": pl.Float64, "x": pl.Int64, "y": pl.Int64}
    pl.DataFrame(schema=empty).write_parquet(folder / "bombs.parquet")


@pytest.fixture
def paths(tmp_path) -> Paths:
    p = Paths(tmp_path).ensure()
    main = [(f"a{i}", "A", "train", False) for i in range(12)] + [("v0", "A", "val", False), ("v1", "A", "test", False),
                                                                   ("b0", "B", "train", False)]
    _write(p.processed, main, "split")
    _write(p.lowrated, [("d0", "D", "fit", True), ("d1", "D", "holdout", True), ("d2", "D", "holdout", True)],
           "calib_half")
    return p


def test_stages_end_to_end(paths):
    s = vp.analyse_stage(paths)
    assert s["difficulties"] == {"A/ok": 14, "B/ok": 1, "D/ok": 3}
    tables = ("validation_metrics", "validation_swings", "validation_issues", "validation_local_metrics")
    first = {t: pl.read_parquet(paths.processed / f"{t}.parquet") for t in tables}
    vp.analyse_stage(paths)                                    # rebuilds identical tables
    for t in tables:
        assert pl.read_parquet(paths.processed / f"{t}.parquet").equals(first[t]), t

    r = vp.reference_stage(paths)
    assert r["n_maps"] == 12                                   # Tier A train only
    assert ValidatorDir(paths.validator).reference_path.exists()

    vp.score_stage(paths)
    v = pl.read_parquet(paths.processed / "validation.parquet")
    a = v.filter(pl.col("tier") == "A")["score"]
    d = v.filter(pl.col("tier") == "D")["score"]
    assert a.min() > d.max()
    assert v.filter(pl.col("tier") == "A")["pass"].all() and not v.filter(pl.col("tier") == "D")["pass"].any()
    assert set(v["validator_version"]) == {"v1"}
    local = pl.read_parquet(paths.processed / "validation_local.parquet")
    assert {"score", "s_E1", "window_start_beat"} <= set(local.columns)

    c = vp.corrupt_stage(paths, pilot=True)
    res = pl.read_parquet(paths.processed / "corruption_results.parquet")
    assert set(res["hash"]) == {"v0", "v1"}                    # Tier A val + test only
    assert {row["corruption"] for row in c["by_corruption"]} >= {"C1", "C2", "C3", "C6", "C7"}

    out = vp.report_stage(paths, name="t")
    text = open(out["path"], encoding="utf-8").read()
    assert "Overall AUC" in text and "Synthetic corruptions" in text


def test_reference_refuses_a_changed_validator_config(paths):
    vp.analyse_stage(paths)
    ValidatorDir(paths.validator).config_path.write_text('{"version": "v1", "something": "else"}')
    with pytest.raises(ValueError, match="bump VALIDATOR_VERSION"):
        vp.reference_stage(paths)
