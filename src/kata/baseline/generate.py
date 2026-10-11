"""One song's audio features -> five validated difficulties (spec §8)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import polars as pl

from kata.baseline import onsets
from kata.baseline.export import DifficultyOut, njs_offset_for
from kata.baseline.grid import Grid, fit_grid
from kata.baseline.patterns import place
from kata.validate.reference import Reference
from kata.validate.rules import LABELS, MapInput, analyse
from kata.validate.score import Result, score_analysis

REPAIR_ROUNDS = 3


@dataclass
class SongInput:
    sha: str
    duration_s: float
    curves: np.ndarray                    # (T, 12) Phase 2 curves
    beats: np.ndarray                     # Beat This beat times (s)
    downbeats: np.ndarray
    segments: list[tuple[float, float]]   # audio_segments (start_s, end_s)


def seed_for(sha: str, label: str) -> int:
    return int(hashlib.md5(f"{sha}|{label}".encode()).hexdigest()[:8], 16)


def _repair(m: MapInput, ref: Reference) -> tuple[MapInput, Result]:
    """Drop the notes of every swing with a hard or error issue, re-check, up to REPAIR_ROUNDS times."""
    res = score_analysis(analyse(m), m.difficulty, ref)
    for _ in range(REPAIR_ROUNDS):
        bad = res.issues.filter(pl.col("severity").is_in(["hard", "error"]) & (pl.col("hand") >= 0))
        if bad.height == 0:
            break
        keys = bad.select(pl.col("hand").cast(pl.Int8), pl.col("swing_idx").cast(pl.Int32)).unique()
        rows = res.swings.join(keys, on=["hand", "swing_idx"], how="semi")["note_rows"].explode().to_list()
        notes = m.notes.sort("time_s", "color", "x", "y", maintain_order=True)
        m = MapInput(m.difficulty, m.njs, m.njs_offset, m.bpm,
                     notes.filter(~pl.int_range(pl.len()).is_in(rows)))
        res = score_analysis(analyse(m), m.difficulty, ref)
    return m, res


def generate(song: SongInput, ref: Reference, labels=LABELS) -> tuple[Grid, dict[str, DifficultyOut], dict[str, Result]]:
    grid = fit_grid(song.beats, song.duration_s, song.segments)
    slot_t, slot_b = grid.slots()
    frame = onsets.onset_strength(song.curves) * onsets.loudness_boost(song.curves, song.segments)
    strength = onsets.slot_strengths(frame, slot_t)
    span = onsets.mapped_span(slot_t, strength, song.duration_s)
    slot_t, slot_b, strength = slot_t[span], slot_b[span], strength[span]
    span_s = float(slot_t[-1] - slot_t[0]) if len(slot_t) else 0.0
    out, results = {}, {}
    for label in labels:
        target_nps = ref.baseline["nps"][label]
        gap = ref.baseline["min_gap_s"][ref.stratum(label, target_nps)]
        idx = onsets.select(slot_t, strength, int(round(target_nps * span_s)), gap / 2.0)
        notes = place(slot_t[idx], slot_b[idx], strength[idx], song.downbeats, label, gap,
                      np.random.default_rng(seed_for(song.sha, label)))
        njs = float(ref.baseline["njs"][label])
        offset = njs_offset_for(njs, ref.baseline["reaction_time_s"][label], grid.bpm)
        m, res = _repair(MapInput(label, njs, offset, grid.bpm, notes), ref)
        out[label] = DifficultyOut(m.notes, njs, offset)
        results[label] = res
    return grid, out, results
