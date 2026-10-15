import polars as pl
from polars.testing import assert_frame_equal

from kata.config import Paths
from kata.data import bsor, pipeline
from kata.data.state import State
from motion_synth import synth_notes, synth_replay


def _write_inputs(p: Paths) -> None:
    """Five downloaded replays on two maps: three good, one hash mismatch, one corrupt file."""
    notes = synth_notes(40)
    pl.concat([pl.DataFrame({"hash": h, "characteristic": "Standard", "difficulty": "Expert", **notes})
               for h in ("h1", "h2")]).write_parquet(p.processed / "notes.parquet")
    replays = [("1", "h1", "h1"), ("2", "h1", "h1"), ("3", "h1", "hX"), ("4", "h2", "h2"), ("5", "h2", None)]
    state = State(p.state_db)
    for sid, h, info_hash in replays:
        (p.replays / h).mkdir(parents=True, exist_ok=True)
        path = p.replays / h / f"{sid}.bsor"
        if info_hash is None:
            path.write_bytes(b"not a replay")
        else:
            r = synth_replay(notes, seed=int(sid))
            r.info.update(hash=info_hash.upper(), difficulty="Expert", mode="Standard")
            path.write_bytes(bsor.write(r))
        state.mark("replay-download", sid, "ok")
    state.close()
    n = len(replays)
    pl.DataFrame({"score_id": [int(r[0]) for r in replays], "player_id": [f"p{r[0]}" for r in replays],
                  "leaderboard_id": [r[1] for r in replays], "hash": [r[1] for r in replays],
                  "characteristic": ["Standard"] * n, "difficulty": ["Expert"] * n, "accuracy": [0.9] * n,
                  "rank": [1] * n, "modifiers": [""] * n, "pauses": [0] * n, "full_combo": [True] * n,
                  "missed": [0] * n, "bad_cuts": [0] * n, "pick_reason": ["top_acc"] * n}
                 ).write_parquet(p.processed / "replay_selection.parquet")


def _parse(p: Paths, batch: int) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    state, summary = State(p.state_db), {}
    try:
        pipeline._parse_replays(p, state, summary, batch=batch, log=lambda _: None)
    finally:
        state.close()
    return (pl.read_parquet(p.processed / "replays.parquet"),
            pl.read_parquet(p.processed / "replay_note_events.parquet"), summary)


def test_parse_replays_in_batches_matches_one_batch(tmp_path):
    p = Paths(tmp_path).ensure()
    _write_inputs(p)
    one_replays, one_events, one_summary = _parse(p, batch=100)
    replays, events, summary = _parse(p, batch=2)

    assert summary == one_summary
    assert summary["status"] == {"ok": 3, "rejected": 2}
    assert_frame_equal(replays, one_replays.select(replays.columns))
    assert_frame_equal(events, one_events.select(events.columns))
    assert events["score_id"].n_unique() == 4   # the corrupt replay has no events
    assert len(list(p.replay_frames.glob("hash=*/*.parquet"))) == 3
    assert not (p.processed / "replay_parse.tmp").exists()
