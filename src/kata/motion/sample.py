"""Whole-song generation from a trained run (docs/phases/phase-4-motion.md §5.6-5.7, §7.1)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from kata.audio.store import FeatureStore
from kata.config import Paths
from kata.motion.dataset import WindowSampler, decode_hits
from kata.motion.diffusion import Job, sample_long
from kata.motion.geometry import rot6_to_matrix
from kata.motion.prepare import FPS, HEAD_Y_REF_M, HIT, OFF, TRACK, MotionStore
from kata.motion.train import Run
from kata.validate.rules import LABELS

KEY = ["hash", "characteristic", "difficulty"]


@dataclass
class MotionTrack:
    """Generated motion on t = i / FPS from 0 s, in the play-space frame (§3.2)."""
    t: np.ndarray                       # (T,)
    head_pos: np.ndarray                # (T, 3) m
    head_rot: np.ndarray                # (T, 3, 3)
    left_pos: np.ndarray
    left_rot: np.ndarray
    right_pos: np.ndarray
    right_rot: np.ndarray
    hits_left: np.ndarray               # sorted hit times (s)
    hits_right: np.ndarray


def to_track(motion: np.ndarray, hits_left: np.ndarray, hits_right: np.ndarray,
             head_height: float | None = None) -> MotionTrack:
    """Heights are generated for a player whose median head height is HEAD_Y_REF_M (§3.2);
    head_height scales them to another player."""
    y_scale = 1.0 if head_height is None else head_height / HEAD_Y_REF_M
    parts = {}
    for part in ("head", "left", "right"):
        s = TRACK[part].start
        parts[f"{part}_pos"] = motion[:, s:s + 3].astype(np.float64) * [1.0, y_scale, 1.0]
        parts[f"{part}_rot"] = rot6_to_matrix(motion[:, s + 3:s + 9])
    return MotionTrack(np.arange(len(motion)) / FPS, **parts, hits_left=np.sort(hits_left),
                       hits_right=np.sort(hits_right))


def final_run(paths: Paths) -> str:
    return (paths.motion_runs / "final").read_text().strip()


def durations(paths: Paths) -> dict[str, float]:
    f = pl.read_parquet(paths.processed / "audio_features.parquet").filter(pl.col("status") == "ok")
    return dict(zip(f["audio_sha256"], f["duration_s"]))


def ckpt_fingerprint(run: Run, which: str = "best") -> str:
    """Identifies the checkpoint a sample came from; changes whenever ckpt/{which}.pt is rewritten."""
    st = (run.ckpt_dir / f"{which}.pt").stat()
    return f"{which}:{st.st_size}:{st.st_mtime_ns}"


class Generator:
    """A run's model, ready to sample whole songs in batches of jobs."""

    def __init__(self, paths: Paths, run_id: str, which: str = "best", device: str | None = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.run = Run(paths, run_id)
        self.config = self.run.config()
        self.ckpt = ckpt_fingerprint(self.run, which)
        self.model = self.run.load_model(which, self.device)
        self.diffusion = self.run.diffusion()
        self.norm = self.run.normaliser()
        self.audio = WindowSampler([], MotionStore(paths.motion), FeatureStore(paths.features), self.norm,
                                   self.config.window, self.config.ctx, use_stems=self.config.use_stems).audio

    def run_jobs(self, jobs: list[Job], guidance: float) -> list[dict]:
        """-> per job: motion (T, 27) de-normalised, hits_left, hits_right, joins."""
        import torch
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda")
               else torch.autocast("cpu", enabled=False))
        with amp:
            res = sample_long(self.model, self.diffusion, jobs, self.audio, self.config.window, self.config.ctx,
                              self.config.ddim_steps, guidance, self.device)
        return [{"motion": self.norm.denorm_motion(x).astype(np.float32),
                 "hits_left": decode_hits(x[:, HIT[0]], x[:, OFF[0]]),
                 "hits_right": decode_hits(x[:, HIT[1]], x[:, OFF[1]]), "joins": np.asarray(joins, np.int64)}
                for x, joins in res]


def generate(audio_sha256: str, label: str | None, nps: float | None, seed: int = 0, guidance: float | None = None,
             run: str | None = None, paths: Paths | None = None, device: str | None = None,
             head_height: float | None = None) -> MotionTrack:
    """Phase 5's entry point (§5.7): one whole song at one difficulty, for a player with this
    median head height (default HEAD_Y_REF_M, about the train median)."""
    paths = paths or Paths()
    gen = Generator(paths, run or final_run(paths), device=device)
    n = math.ceil(durations(paths)[audio_sha256] * FPS)
    job = Job(audio_sha256, None if label is None else LABELS.index(label), nps, seed, n)
    out = gen.run_jobs([job], gen.run.guidance() if guidance is None else guidance)[0]
    return to_track(out["motion"], out["hits_left"], out["hits_right"], head_height)


# --------------------------------------------------------------------------- split sampling (§7.1)

def eval_targets(paths: Paths, run: Run, split: str) -> pl.DataFrame:
    """One row per difficulty of the split with at least one accepted track: key, song, label, nps."""
    from kata.motion.train import tracks_table
    t = tracks_table(paths, run.config().pilot).filter((pl.col("status") == "ok") & (pl.col("split") == split))
    return (t.group_by(KEY).agg(pl.col("audio_sha256").first(), pl.col("label").first(), pl.col("nps").first(),
                                pl.col("hps").mean().alias("real_hps"), pl.col("score_id").sort().alias("score_ids"))
            .join(pl.DataFrame({"audio_sha256": list(durations(paths))}), on="audio_sha256", how="semi")
            .sort(KEY))


def spread(df: pl.DataFrame, n: int | None) -> pl.DataFrame:
    """n rows evenly spread over df (all when n is None or larger)."""
    if n is None or n >= df.height:
        return df
    return df[sorted(set(np.linspace(0, df.height - 1, n).round().astype(int).tolist()))]


def sample_path(out_dir: Path, key: tuple, mode: str, seed: int) -> Path:
    return out_dir / f"{key[0]}_{key[1]}_{key[2]}_{mode}_{seed}.npz"


def _save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _fresh(path: Path, guidance: float, ckpt: str) -> bool:
    """The sample exists and was made with this guidance from this checkpoint."""
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
    except (OSError, ValueError, KeyError):
        return False
    return meta.get("guidance") == guidance and meta.get("ckpt") == ckpt


def _modes(r: dict, seeds, label_only: bool) -> list[tuple[str, int, float | None]]:
    return [("full", s, r["nps"]) for s in seeds] + ([("label_only", 0, None)] if label_only else [])


def stale_samples(out_dir: Path, targets: pl.DataFrame, guidance: float, ckpt: str, seeds=(0, 1),
                  label_only: bool = True) -> list[Path]:
    """Sample files of the targets that are missing or were made with other guidance or weights."""
    return [p for r in targets.iter_rows(named=True) for mode, seed, _ in _modes(r, seeds, label_only)
            if not _fresh(p := sample_path(out_dir, (r["hash"], r["characteristic"], r["difficulty"]), mode, seed),
                          guidance, ckpt)]


def sample_targets(gen: Generator, targets: pl.DataFrame, out_dir: Path, guidance: float, seeds=(0, 1),
                   label_only: bool = True, batch_jobs: int = 16, log=print) -> dict:
    """Generate every target (mode 'full' per seed, plus 'label_only' seed 0) into out_dir. Files made
    with the same guidance from the same checkpoint are kept, so an interrupted run resumes; any
    other file is regenerated."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = durations(gen.run.paths)
    todo = []
    for r in targets.iter_rows(named=True):
        key = (r["hash"], r["characteristic"], r["difficulty"])
        n = math.ceil(dur[r["audio_sha256"]] * FPS)
        label = LABELS.index(r["label"])
        for mode, seed, nps in _modes(r, seeds, label_only):
            path = sample_path(out_dir, key, mode, seed)
            if not _fresh(path, guidance, gen.ckpt):
                todo.append((path, Job(r["audio_sha256"], label, nps, seed, n),
                             {"key": list(key), "audio_sha256": r["audio_sha256"], "label": r["label"],
                              "nps": nps, "seed": seed, "mode": mode, "guidance": guidance, "ckpt": gen.ckpt}))
    import time
    t0 = time.monotonic()
    for i in range(0, len(todo), batch_jobs):
        part = todo[i:i + batch_jobs]
        for (path, _, meta), out in zip(part, gen.run_jobs([j for _, j, _ in part], guidance)):
            _save_npz(path, meta=np.array(json.dumps(meta)), **out)
        log(f"  [motion-sample] {min(i + batch_jobs, len(todo))}/{len(todo)}")
    return {"jobs": len(todo), "seconds": time.monotonic() - t0}


def load_sample(path: Path) -> dict:
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out
