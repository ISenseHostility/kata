"""Diffusion: cosine schedule, x0 losses, DDIM with classifier-free guidance, overlap inpainting
(docs/phases/phase-4-motion.md §5.4-5.6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from kata.motion.dataset import NONE_LABEL
from kata.motion.prepare import HIT, MOTION_DIM, OFF, TRACK

T_STEPS = 1000
P_UNCOND = 0.1


def cosine_alpha_bar(T: int = T_STEPS, s: float = 0.008) -> torch.Tensor:
    """ᾱ for steps 0 .. T-1 (Nichol & Dhariwal 2021), betas clipped at 0.999."""
    f = lambda x: math.cos((x / T + s) / (1 + s) * math.pi / 2) ** 2  # noqa: E731
    betas = torch.tensor([min(1 - f(i + 1) / f(i), 0.999) for i in range(T)], dtype=torch.float64)
    return torch.cumprod(1 - betas, 0)


def rot6_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3), Gram-Schmidt; the torch twin of geometry.rot6_to_matrix."""
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


def saber_tips(motion: torch.Tensor) -> torch.Tensor:
    """De-normalised (..., T, 27) motion -> saber tips (..., T, 2, 3): origin + 1 m along local +z."""
    from kata.motion.geometry import BLADE_AXIS, BLADE_LENGTH_M
    tips = []
    for part in ("left", "right"):
        s = TRACK[part].start
        rot = rot6_to_matrix(motion[..., s + 3:s + 9])
        tips.append(motion[..., s:s + 3] + BLADE_LENGTH_M * rot[..., :, BLADE_AXIS])
    return torch.stack(tips, -2)


@dataclass
class LossWeights:
    tip: float = 1.0
    tip_vel: float = 1.0


class Diffusion:
    def __init__(self, motion_mean, motion_std, T: int = T_STEPS):
        self.T = T
        self.alpha_bar = cosine_alpha_bar(T)
        self.m_mean = torch.as_tensor(np.asarray(motion_mean), dtype=torch.float32)
        self.m_std = torch.as_tensor(np.asarray(motion_std), dtype=torch.float32)

    def _ab(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(like.device)[t.to(like.device)].to(like.dtype).view(-1, *([1] * (like.dim() - 1)))

    def q_sample(self, x0, t, noise):
        ab = self._ab(t, x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :MOTION_DIM] * self.m_std.to(x.device) + self.m_mean.to(x.device)

    def loss(self, model, batch: dict, weights: LossWeights = LossWeights(),
             p_uncond: float = P_UNCOND) -> tuple[torch.Tensor, dict]:
        """batch: x (B,W,31), hit_mask (B,W,2), audio (B,W+2c,A), label (B,), nps (B,), all tensors."""
        x0 = batch["x"]
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        label = torch.where(torch.rand(B, device=x0.device) < p_uncond,
                            torch.full_like(batch["label"], NONE_LABEL), batch["label"])
        nps_drop = torch.rand(B, device=x0.device) < p_uncond
        pred = model(x_t, t, model.encode_audio(batch["audio"]), label, batch["nps"], nps_drop).float()
        dense = list(range(MOTION_DIM)) + list(HIT)
        l_simple = (pred[..., dense] - x0[..., dense]).square().mean()
        mask = batch["hit_mask"]
        l_off = ((pred[..., list(OFF)] - x0[..., list(OFF)]).square() * mask).sum() / mask.sum().clamp(min=1)
        tip_p, tip_x = saber_tips(self.denorm(pred)), saber_tips(self.denorm(x0))
        l_tip = (tip_p - tip_x).square().mean()
        l_vel = (tip_p.diff(dim=1) - tip_x.diff(dim=1)).square().mean()
        total = l_simple + l_off + weights.tip * l_tip + weights.tip_vel * l_vel
        return total, {"l_simple": l_simple.item(), "l_off": l_off.item(), "l_tip": l_tip.item(),
                       "l_vel": l_vel.item()}

    def _x0(self, model, x, t, audio_tokens, label, nps, guidance: float):
        """Guided x0: x0_u + w (x0_c - x0_u); w = 1 skips the unconditional pass."""
        B = x.shape[0]
        tt = torch.full((B,), int(t), device=x.device, dtype=torch.long)
        no_drop = torch.zeros(B, dtype=torch.bool, device=x.device)
        nps_drop = torch.isnan(nps)
        nps_val = torch.nan_to_num(nps, nan=1.0)
        if guidance == 1.0:
            return model(x, tt, audio_tokens, label, nps_val, nps_drop).float()
        both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([audio_tokens, audio_tokens]),
                     torch.cat([label, torch.full_like(label, NONE_LABEL)]), torch.cat([nps_val, nps_val]),
                     torch.cat([nps_drop, ~no_drop])).float()
        cond, uncond = both[:B], both[B:]
        return uncond + guidance * (cond - uncond)

    @torch.no_grad()
    def ddim(self, model, audio_tokens, label, nps, noise_fn: Callable[[], torch.Tensor], steps: int = 50,
             guidance: float = 1.0, known: torch.Tensor | None = None, known_mask: torch.Tensor | None = None):
        """DDIM (η = 0) from pure noise. nps NaN = dropped. known/known_mask (B, W, 31): values held
        fixed (noised to each step's level) at every step, the overlap-inpainting rule (§5.6).
        noise_fn() returns a fresh (B, W, 31) noise tensor (per-job generators, see sample_long)."""
        ts = torch.linspace(self.T - 1, 0, steps).round().long().unique_consecutive().tolist()
        x = noise_fn()
        if known is not None:
            x = torch.where(known_mask, self.q_sample(known, torch.full((x.shape[0],), ts[0]), noise_fn()), x)
        for i, t in enumerate(ts):
            x0 = self._x0(model, x, t, audio_tokens, label, nps, guidance)
            ab = self.alpha_bar[t].item()
            eps = (x - math.sqrt(ab) * x0) / math.sqrt(1 - ab)
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            ab_prev = self.alpha_bar[t_prev].item() if t_prev >= 0 else 1.0
            x = math.sqrt(ab_prev) * x0 + math.sqrt(1 - ab_prev) * eps
            if known is not None:
                held = (self.q_sample(known, torch.full((x.shape[0],), t_prev), noise_fn()) if t_prev >= 0
                        else known)
                x = torch.where(known_mask, held, x)
        return x


@dataclass
class Job:
    """One song at one difficulty: label index or None, NPS or None, a seed, and n_frames at 50 fps."""
    sha: str
    label: int | None
    nps: float | None
    seed: int
    n_frames: int


def n_windows(n_frames: int, window: int) -> int:
    half = window // 2
    return 1 if n_frames <= window else math.ceil((n_frames - window) / half) + 1


def sample_long(model, diffusion: Diffusion, jobs: list[Job], audio_fn: Callable[[str, int, int], np.ndarray],
                window: int, ctx: int, steps: int = 50, guidance: float = 1.0,
                device: str = "cpu") -> list[tuple[np.ndarray, list[int]]]:
    """Whole songs by overlapping windows (§5.6), all jobs in lockstep.

    audio_fn(sha, start, n) -> (n, AUDIO_DIM) model-space audio for 50 fps frames start .. start+n-1.
    Returns per job (model-space track (n_frames, 31), join frames). Each job draws its noise from its
    own generator, so a job's output does not depend on which other jobs share the batch.
    """
    half = window // 2
    gens = [torch.Generator().manual_seed(j.seed) for j in jobs]
    out = [np.zeros((n_windows(j.n_frames, window) * half + half, model.config.n_in), np.float32) for j in jobs]
    joins: list[list[int]] = [[] for _ in jobs]
    model.eval()
    for w in range(max(n_windows(j.n_frames, window) for j in jobs)):
        act = [k for k, j in enumerate(jobs) if w < n_windows(j.n_frames, window)]
        start = w * half
        audio = torch.from_numpy(np.stack([audio_fn(jobs[k].sha, start - ctx, window + 2 * ctx) for k in act]))
        label = torch.tensor([NONE_LABEL if jobs[k].label is None else jobs[k].label for k in act])
        nps = torch.tensor([float("nan") if jobs[k].nps is None else jobs[k].nps for k in act])
        known = mask = None
        if w > 0:
            known = torch.zeros(len(act), window, model.config.n_in)
            known[:, :half] = torch.from_numpy(np.stack([out[k][start:start + half] for k in act]))
            mask = torch.zeros_like(known, dtype=torch.bool)
            mask[:, :half] = True
            for k in act:
                joins[k].append(start + half)

        def noise_fn():
            return torch.stack([torch.randn(window, model.config.n_in, generator=gens[k]) for k in act]).to(device)

        with torch.inference_mode():
            tokens = model.encode_audio(audio.to(device))
            x = diffusion.ddim(model, tokens, label.to(device), nps.to(device), noise_fn, steps, guidance,
                               None if known is None else known.to(device), None if mask is None else mask.to(device))
        x = x.float().cpu().numpy()
        for i, k in enumerate(act):
            if w == 0:
                out[k][:window] = x[i]
            else:
                out[k][start + half:start + window] = x[i, half:]
    return [(o[:j.n_frames], [f for f in jn if f < j.n_frames]) for o, j, jn in zip(out, jobs, joins)]
