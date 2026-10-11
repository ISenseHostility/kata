"""`kata baseline generate` on one synthetic val song with stored features."""
import numpy as np
import polars as pl

from kata.audio.frames import FPS
from kata.audio.store import CURVE_COLUMNS, FeatureStore
from kata.baseline import pipeline as bp
from kata.config import Paths
from kata.validate.reference import ValidatorDir

SHA = "cd" * 32


def test_generate_stage_writes_maps_and_eval(tmp_path, toy_ref):
    p = Paths(tmp_path).ensure()
    dur = 60.0
    T = int(dur * FPS) + 1
    curves = np.zeros((T, 12), np.float32)
    curves[:, CURVE_COLUMNS.index("rms_mix")] = -20
    for k in range(int((dur - 1) / 0.125)):
        curves[int(round((0.5 + 0.125 * k) * FPS)), CURVE_COLUMNS.index("onset_drums")] = 1.0 if k % 4 == 0 else 0.5
    arrays = {"mel": np.zeros((T, 128)), "stem_mel": np.zeros((T, 4, 64)), "chroma": np.zeros((T, 12)), "curves": curves}
    beats = 0.5 + 0.5 * np.arange(int(dur / 0.5) - 1)
    FeatureStore(p.features).write(SHA, arrays, beats, beats[::4])
    (p.maps / "h1").mkdir(parents=True)
    (p.maps / "h1" / "song.egg").write_bytes(b"OggS fake")
    pl.DataFrame({"hash": ["h1"], "audio_status": ["ok"], "split": ["val"], "pilot": [True], "audio_sha256": [SHA],
                  "audio_file": ["song.egg"], "song_name": ["Song"], "song_sub_name": [""],
                  "song_author": ["Artist"]}).write_parquet(p.processed / "maps.parquet")
    pl.DataFrame({"audio_sha256": [SHA], "status": ["ok"], "duration_s": [dur]}).write_parquet(
        p.processed / "audio_features.parquet")
    pl.DataFrame({"hash": ["h1"], "tier": ["A"], "parse_status": ["ok"]}).write_parquet(
        p.processed / "difficulties.parquet")
    pl.DataFrame({"audio_sha256": [SHA, SHA], "seg_idx": [0, 1], "start_s": [0.0, 30.0], "end_s": [30.0, dur]}
                 ).write_parquet(p.processed / "audio_segments.parquet")
    human = 2.0 + 0.5 * np.arange(100)
    pl.DataFrame({"hash": ["h1"] * 100, "difficulty": ["Expert"] * 100, "time_s": human}).write_parquet(
        p.processed / "notes.parquet")
    vd = ValidatorDir(p.validator)
    vd.open_for_write()
    vd.write_reference(toy_ref)

    s = bp.generate_stage(p)
    ev = pl.read_parquet(p.processed / "baseline_eval.parquet")
    assert s["songs"] == 1 and ev.height == 5
    assert ev["pass"].all() and ev["round_trip"].all()
    assert ev.filter(pl.col("difficulty") == "Expert")["timing_f"][0] > 0.3
    assert ev.filter(pl.col("difficulty") == "Easy")["timing_f"][0] is None    # no human Easy map
    folder = p.generated / "baseline" / bp.BASELINE_VERSION / SHA
    assert (folder / "Info.dat").exists() and folder.with_suffix(".zip").exists()
