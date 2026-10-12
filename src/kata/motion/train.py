"""Training loop, runs, checkpoints and exact resume (docs/phases/phase-4-motion.md §6.3-6.4)."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch

from kata.audio.store import FeatureStore
from kata.config import Paths
from kata.motion.dataset import CTX, WINDOW, Normaliser, WindowSampler, load_tracks
from kata.motion.diffusion import Diffusion, LossWeights
from kata.motion.model import ModelConfig, MotionDenoiser
from kata.motion.prepare import MotionStore


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    window: int = WINDOW
    ctx: int = CTX
    batch: int = 64
    lr: float = 2e-4
    warmup: int = 2000
    total_steps: int = 80_000        # sets the lr schedule; a run may span nights (§6.4)
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    ema: float = 0.999
    seed: int = 0
    p_uncond: float = 0.1
    w_tip: float = 1.0
    w_tip_vel: float = 1.0
    ckpt_minutes: float = 30.0
    eval_every: int = 5000
    log_every: int = 100
    eval_diffs: int = 50
    eval_guidance: float = 2.0
    ddim_steps: int = 50
    mirror_aug: bool = False
    use_stems: bool = True
    pilot: bool = False
    workers: int = 4
    only_songs: tuple[str, ...] = ()  # overfit check (§6.5): train and quick-check on these songs only

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        d["model"] = ModelConfig(**d["model"])
        d["only_songs"] = tuple(d.get("only_songs", ()))
        return cls(**d)


def tracks_table(paths: Paths, pilot: bool) -> pl.DataFrame:
    return pl.read_parquet(paths.processed / f"motion_tracks{'_pilot' if pilot else ''}.parquet")


def songs_with_features(paths: Paths) -> set[str]:
    f = pl.read_parquet(paths.processed / "audio_features.parquet")
    return set(f.filter(pl.col("status") == "ok")["audio_sha256"])


class Run:
    """runs/motion/{run_id}/: config.json, the stats it was trained with, log.jsonl, ckpt/{last,best}.pt."""

    def __init__(self, paths: Paths, run_id: str):
        self.paths, self.id = paths, run_id
        self.dir = paths.motion_runs / run_id
        self.config_path = self.dir / "config.json"
        self.log_path = self.dir / "log.jsonl"
        self.ckpt_dir = self.dir / "ckpt"
        self.motion_stats_path = self.dir / "motion_stats.json"
        self.audio_stats_path = self.dir / "audio_stats.json"
        self.guidance_path = self.dir / "guidance.json"

    @classmethod
    def create(cls, paths: Paths, run_id: str, config: RunConfig) -> "Run":
        run = cls(paths, run_id)
        if run.config_path.exists():
            if json.loads(run.config_path.read_text()) != config.to_dict():
                raise ValueError(f"run {run_id} exists with a different config; use a new run id")
            return run
        run.ckpt_dir.mkdir(parents=True, exist_ok=True)
        run.config_path.write_text(json.dumps(config.to_dict(), indent=1, sort_keys=True))
        run.motion_stats_path.write_bytes(MotionStore(paths.motion).stats_path.read_bytes())
        run.audio_stats_path.write_bytes(FeatureStore(paths.features).stats_path.read_bytes())
        return run

    def config(self) -> RunConfig:
        return RunConfig.from_dict(json.loads(self.config_path.read_text()))

    def normaliser(self) -> Normaliser:
        return Normaliser(json.loads(self.motion_stats_path.read_text()), json.loads(self.audio_stats_path.read_text()))

    def diffusion(self) -> Diffusion:
        n = self.normaliser()
        return Diffusion(n.m_mean, n.m_std)

    def sampler(self, split: str, batch: int | None = None, seed: int | None = None) -> WindowSampler:
        c = self.config()
        songs = set(c.only_songs) if c.only_songs else songs_with_features(self.paths)
        tracks = load_tracks(tracks_table(self.paths, c.pilot), split, songs, c.window)
        return WindowSampler(tracks, MotionStore(self.paths.motion), FeatureStore(self.paths.features),
                             self.normaliser(), c.window, c.ctx, c.seed if seed is None else seed,
                             c.batch if batch is None else batch, c.mirror_aug and split == "train", c.use_stems)

    def load_model(self, which: str = "best", device: str = "cpu") -> MotionDenoiser:
        """The EMA weights of ckpt/{which}.pt, in eval mode."""
        state = torch.load(self.ckpt_dir / f"{which}.pt", map_location="cpu", weights_only=True)
        model = MotionDenoiser(self.config().model)
        model.load_state_dict(state["ema"])
        return model.to(device).eval()

    def guidance(self) -> float:
        if self.guidance_path.exists():
            return float(json.loads(self.guidance_path.read_text())["guidance"])
        return self.config().eval_guidance

    def log(self, row: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def lr_factor(c: RunConfig):
    def f(step: int) -> float:
        if step < c.warmup:
            return (step + 1) / c.warmup
        progress = min(1.0, (step - c.warmup) / max(1, c.total_steps - c.warmup))
        return c.min_lr_frac + (1 - c.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


class StepBatches(torch.utils.data.Dataset):
    """Item i is the whole batch of global step first + i (WindowSampler.batch is a function of step)."""

    def __init__(self, sampler: WindowSampler, first: int, last: int):
        self.sampler, self.first, self.last = sampler, first, last

    def __len__(self) -> int:
        return max(0, self.last - self.first)

    def __getitem__(self, i: int) -> dict:
        return {k: torch.from_numpy(v) for k, v in self.sampler.batch(self.first + i).items()}


def _save(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _rng_state(device: str) -> dict:
    s = {"cpu": torch.get_rng_state()}
    if device.startswith("cuda"):
        s["cuda"] = torch.cuda.get_rng_state(torch.device(device))
    return s


def _set_rng_state(s: dict, device: str) -> None:
    torch.set_rng_state(s["cpu"])
    if "cuda" in s and device.startswith("cuda"):
        torch.cuda.set_rng_state(s["cuda"], torch.device(device))


def train(paths: Paths, run_id: str, config: RunConfig | None = None, resume: bool = False,
          hours: float | None = None, device: str = "cuda", max_steps: int | None = None, log=print) -> dict:
    """Train run_id. Fresh (config given, no checkpoint yet) or --resume from ckpt/last.pt.

    Stops at total_steps, at max_steps, or after `hours` of wall time; always saves last.pt on exit.
    """
    from kata.motion.evaluate import quick_eval, quick_windows
    run = Run(paths, run_id)
    if resume:
        if not (run.ckpt_dir / "last.pt").exists():
            raise FileNotFoundError(f"no checkpoint to resume in {run.ckpt_dir}")
        c = run.config()
    else:
        if config is None:
            raise ValueError("a fresh run needs a config")
        if (run.ckpt_dir / "last.pt").exists():
            raise ValueError(f"run {run_id} already has a checkpoint; pass resume=True")
        run = Run.create(paths, run_id, config)
        c = config
    torch.manual_seed(c.seed)
    model = MotionDenoiser(c.model).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, betas=(c.beta1, c.beta2), weight_decay=c.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(c))
    step, best = 0, -math.inf
    if resume:
        state = torch.load(run.ckpt_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, best = state["step"], state["best"]
    diffusion = run.diffusion()
    weights = LossWeights(c.w_tip, c.w_tip_vel)
    sampler = run.sampler("train")
    val_sampler = run.sampler("train" if c.only_songs else "val", seed=0)    # overfit: check the song itself
    val_windows = quick_windows(val_sampler, c.eval_diffs)
    stop_at = min(c.total_steps, max_steps or c.total_steps)
    loader = torch.utils.data.DataLoader(StepBatches(sampler, step, stop_at), batch_size=None,
                                         num_workers=c.workers, persistent_workers=c.workers > 0,
                                         prefetch_factor=2 if c.workers > 0 else None)
    # The iterator draws its base seed from the global RNG, so create it before restoring the RNG:
    # a resumed run then continues from exactly the state a straight run would have.
    batches = iter(loader)
    if resume:
        _set_rng_state(state["rng"], device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast(
        "cpu", enabled=False)

    def checkpoint(name: str) -> None:
        _save(run.ckpt_dir / f"{name}.pt", {"model": model.state_dict(), "ema": ema.state_dict(),
                                             "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                                             "best": best, "rng": _rng_state(device)})

    t_start = t_ckpt = time.monotonic()
    first_step, reason = step, "total_steps"
    model.train()
    for batch in batches:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with amp:
            loss, parts = diffusion.loss(model, batch, weights, c.p_uncond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 1 - c.ema)
        step += 1
        now = time.monotonic()
        if step % c.log_every == 0:
            row = {"step": step, "loss": loss.item(), **parts, "lr": sched.get_last_lr()[0],
                   "steps_per_s": (step - first_step) / max(now - t_start, 1e-9)}
            run.log(row)
            log(f"  [motion-train] {json.dumps(row)}")
        if step % c.eval_every == 0 or step == stop_at:
            ev = quick_eval(ema, diffusion, val_sampler, val_windows, c.eval_guidance, c.ddim_steps, device)
            run.log({"step": step, "eval": ev})
            if ev["val_timing_f"] is not None and ev["val_timing_f"] > best:
                best = ev["val_timing_f"]
                checkpoint("best")
        if now - t_ckpt >= c.ckpt_minutes * 60:
            checkpoint("last")
            t_ckpt = now
        if hours is not None and now - t_start >= hours * 3600:
            reason = "time_limit"
            break
        if max_steps is not None and step >= max_steps:
            reason = "max_steps"
            break
    checkpoint("last")
    return {"run": run_id, "step": step, "stopped": reason, "best_val_timing_f": best,
            "seconds": time.monotonic() - t_start}
