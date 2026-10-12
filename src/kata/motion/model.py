"""Audio encoder + denoiser transformer (docs/phases/phase-4-motion.md §5.2-5.3).

Self- and cross-attention use rotary positions. Motion frame k sits at audio position k + ctx, so
cross-attention sees the true time offset between a motion frame and every audio frame.
Conditioning (diffusion step, label, NPS) enters every denoiser layer as FiLM shift/scale/gate
(adaLN-Zero: gates start at zero).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d: int = 448                 # ~46M parameters with the layer counts below (spec §5.3)
    heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    n_in: int = 31
    audio_dim: int = 820
    n_labels: int = 6            # 5 labels + "none"


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary embedding on (B, H, L, Dh) at float positions (L,)."""
    half = x.shape[-1] // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.float()[:, None] * freqs[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, ctx, q_pos, k_pos):
        B, L, D = x.shape
        h = self.heads
        q = self.q(x).view(B, L, h, D // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, h, D // h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, q_pos), rope(k, k_pos)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, L, D))


def feed_forward(d: int, mult: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * d, d))


class EncoderBlock(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.attn = Attention(c.d, c.heads, c.dropout)
        self.ff = feed_forward(c.d, c.ff_mult, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, a, pos):
        h = self.n1(a)
        a = a + self.drop(self.attn(h, h, pos, pos))
        return a + self.drop(self.ff(self.n2(a)))


class DecoderBlock(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1 = nn.LayerNorm(c.d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(c.d)
        self.n3 = nn.LayerNorm(c.d, elementwise_affine=False)
        self.self_attn = Attention(c.d, c.heads, c.dropout)
        self.cross_attn = Attention(c.d, c.heads, c.dropout)
        self.ff = feed_forward(c.d, c.ff_mult, c.dropout)
        self.drop = nn.Dropout(c.dropout)
        self.mod = nn.Linear(c.d, 6 * c.d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond, audio, x_pos, a_pos):
        s1, sc1, g1, s2, sc2, g2 = self.mod(F.silu(cond))[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + sc1) + s1
        x = x + g1 * self.drop(self.self_attn(h, h, x_pos, x_pos))
        x = x + self.drop(self.cross_attn(self.n2(x), audio, x_pos, a_pos))
        h = self.n3(x) * (1 + sc2) + s2
        return x + g2 * self.drop(self.ff(h))


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], -1)


class MotionDenoiser(nn.Module):
    """x_t (B, W, n_in), t (B,), audio (B, W + 2·ctx, audio_dim) -> predicted clean x0 (B, W, n_in)."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.config = c
        self.audio_in = nn.Linear(c.audio_dim, c.d)
        self.encoder = nn.ModuleList(EncoderBlock(c) for _ in range(c.enc_layers))
        self.audio_norm = nn.LayerNorm(c.d)
        self.x_in = nn.Linear(c.n_in, c.d)
        self.t_mlp = nn.Sequential(nn.Linear(256, c.d), nn.SiLU(), nn.Linear(c.d, c.d))
        self.label = nn.Embedding(c.n_labels, c.d)
        self.nps_mlp = nn.Sequential(nn.Linear(1, c.d), nn.SiLU(), nn.Linear(c.d, c.d))
        self.nps_none = nn.Parameter(torch.zeros(c.d))
        self.blocks = nn.ModuleList(DecoderBlock(c) for _ in range(c.dec_layers))
        self.out_norm = nn.LayerNorm(c.d, elementwise_affine=False)
        self.out_mod = nn.Linear(c.d, 2 * c.d)
        self.out = nn.Linear(c.d, c.n_in)
        for m in (self.out_mod, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        a = self.audio_in(audio)
        pos = torch.arange(audio.shape[1], device=audio.device)
        for blk in self.encoder:
            a = blk(a, pos)
        return self.audio_norm(a)

    def cond(self, t, label, nps, nps_drop) -> torch.Tensor:
        n = self.nps_mlp(torch.log(nps.clamp(min=0.1)).float()[:, None])
        n = torch.where(nps_drop[:, None], self.nps_none.expand_as(n), n)
        return self.t_mlp(timestep_embedding(t)) + self.label(label) + n

    def forward(self, x_t, t, audio_tokens, label, nps, nps_drop) -> torch.Tensor:
        W = x_t.shape[1]
        ctx = (audio_tokens.shape[1] - W) // 2
        x_pos = torch.arange(W, device=x_t.device) + ctx
        a_pos = torch.arange(audio_tokens.shape[1], device=x_t.device)
        c = self.cond(t, label, nps, nps_drop)
        x = self.x_in(x_t)
        for blk in self.blocks:
            x = blk(x, c, audio_tokens, x_pos, a_pos)
        shift, scale = self.out_mod(F.silu(c))[:, None].chunk(2, -1)
        return self.out(self.out_norm(x) * (1 + scale) + shift)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
