"""Schedule, losses, DDIM with guidance and overlap inpainting (spec §5.4-5.6)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kata.motion import geometry as g  # noqa: E402
from kata.motion.diffusion import (Diffusion, Job, cosine_alpha_bar, n_windows, rot6_to_matrix,  # noqa: E402
                                   sample_long, saber_tips)
from kata.motion.model import ModelConfig, MotionDenoiser  # noqa: E402

TINY = ModelConfig(d=32, heads=4, enc_layers=1, dec_layers=1, dropout=0.0)


def _diffusion():
    return Diffusion(np.zeros(27), np.ones(27))


def test_cosine_schedule():
    ab = cosine_alpha_bar()
    assert ab.shape == (1000,) and (ab.diff() < 0).all()
    assert ab[0] > 0.999 and ab[-1] < 1e-3


def test_rot6_and_tips_match_numpy():
    r6 = np.random.default_rng(0).normal(size=(5, 6))
    assert np.allclose(rot6_to_matrix(torch.tensor(r6)).numpy(), g.rot6_to_matrix(r6), atol=1e-6)
    m = torch.zeros(1, 3, 27)
    m[..., 9:12] = torch.tensor([0.1, 1.0, 0.2])                       # left saber position
    m[..., 12:18] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])             # identity rotation
    m[..., 21:27] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
    tips = saber_tips(m)
    assert torch.allclose(tips[0, 0, 0], torch.tensor([0.1, 1.0, 1.2]))


def test_loss_is_finite_and_offsets_count_only_at_hits():
    torch.manual_seed(0)
    model = MotionDenoiser(TINY)
    B, W, C = 2, 16, 4
    batch = {"x": torch.randn(B, W, 31), "hit_mask": torch.zeros(B, W, 2), "audio": torch.randn(B, W + 2 * C, 820),
             "label": torch.tensor([0, 4]), "nps": torch.tensor([3.0, 8.0])}
    total, parts = _diffusion().loss(model, batch)
    assert torch.isfinite(total) and parts["l_off"] == 0.0
    batch["hit_mask"][0, 3, 1] = 1
    _, parts = _diffusion().loss(model, batch)
    assert parts["l_off"] > 0


def test_micro_batches_give_the_full_batch_loss_and_gradient():
    torch.manual_seed(0)
    model = MotionDenoiser(TINY)
    B, W, C = 4, 16, 4
    batch = {"x": torch.randn(B, W, 31), "hit_mask": torch.zeros(B, W, 2), "audio": torch.randn(B, W + 2 * C, 820),
             "label": torch.tensor([0, 4, 2, 5]), "nps": torch.tensor([3.0, 8.0, 5.0, 6.0])}
    batch["hit_mask"][0, 3, 1] = batch["hit_mask"][0, 9, 0] = batch["hit_mask"][3, 5, 1] = 1   # uneven hits per chunk
    d = _diffusion()
    torch.manual_seed(1)
    total, parts = d.loss(model, batch, p_uncond=0.5)
    total.backward()
    full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    torch.manual_seed(1)
    drawn = d.draw(batch, p_uncond=0.5)
    n_hits = batch["hit_mask"].sum()
    chunks = [d.chunk_loss(model, batch, drawn, sl=slice(i, i + 2), n_hits=n_hits) for i in (0, 2)]
    for loss, _ in chunks:
        loss.backward()
    assert torch.allclose(sum(c[0] for c in chunks), total, atol=1e-6)
    for k in parts:
        assert sum(c[1][k] for c in chunks) == pytest.approx(parts[k], abs=1e-6)
    assert all(torch.allclose(p.grad, g, atol=1e-6) for p, g in zip(model.parameters(), full))


class Oracle(torch.nn.Module):
    """Predicts x0 = the absolute frame index (audio column 0) in channel 0, zeros elsewhere."""

    def __init__(self):
        super().__init__()
        self.config = ModelConfig(n_in=31)

    def encode_audio(self, audio):
        return audio

    def forward(self, x_t, t, audio_tokens, label, nps, nps_drop):
        W = x_t.shape[1]
        ctx = (audio_tokens.shape[1] - W) // 2
        out = torch.zeros_like(x_t)
        out[..., 0] = audio_tokens[:, ctx:ctx + W, 0]
        return out


def _frame_audio(sha, start, n):
    a = np.zeros((n, 820), np.float32)
    a[:, 0] = start + np.arange(n)
    return a


def test_ddim_returns_the_predicted_x0_and_holds_known_frames():
    d = _diffusion()
    model = Oracle()
    audio = torch.from_numpy(_frame_audio("s", -4, 24))[None]
    gen = torch.Generator().manual_seed(0)
    x = d.ddim(model, audio, torch.tensor([1]), torch.tensor([5.0]), lambda: torch.randn(1, 16, 31, generator=gen),
               steps=10, guidance=2.5)
    assert torch.allclose(x[0, :, 0], torch.arange(16.0), atol=1e-4) and x[0, :, 1:].abs().max() < 1e-4
    known = torch.full((1, 16, 31), 7.0)
    mask = torch.zeros_like(known, dtype=torch.bool)
    mask[:, :5] = True
    x = d.ddim(model, audio, torch.tensor([1]), torch.tensor([float("nan")]),
               lambda: torch.randn(1, 16, 31, generator=gen), steps=10, known=known, known_mask=mask)
    assert torch.equal(x[0, :5], known[0, :5])


def test_long_songs_are_stitched_on_the_frame_grid():
    assert [n_windows(n, 16) for n in (10, 16, 17, 24, 25)] == [1, 1, 2, 2, 3]
    jobs = [Job("s", 1, 4.0, 0, 50), Job("t", None, None, 1, 12)]
    res = sample_long(Oracle(), _diffusion(), jobs, _frame_audio, window=16, ctx=4, steps=5)
    (a, joins_a), (b, joins_b) = res
    assert a.shape == (50, 31) and b.shape == (12, 31)
    assert np.allclose(a[:, 0], np.arange(50), atol=1e-3)
    assert joins_a == [16, 24, 32, 40, 48] and joins_b == []


def test_a_jobs_output_does_not_depend_on_its_batch_mates():
    torch.manual_seed(0)
    model = MotionDenoiser(TINY)
    for p in model.out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    d = _diffusion()

    def audio_fn(sha, start, n):
        return np.random.default_rng([ord(sha), start + 1000]).normal(size=(n, 820)).astype(np.float32)

    alone = sample_long(model, d, [Job("s", 2, 5.0, 3, 30)], audio_fn, window=16, ctx=4, steps=4, guidance=2.0)
    pair = sample_long(model, d, [Job("s", 2, 5.0, 3, 30), Job("t", 0, 2.0, 9, 40)], audio_fn, window=16, ctx=4,
                       steps=4, guidance=2.0)
    assert np.allclose(alone[0][0], pair[0][0], atol=1e-5)
    again = sample_long(model, d, [Job("s", 2, 5.0, 3, 30)], audio_fn, window=16, ctx=4, steps=4, guidance=2.0)
    assert np.array_equal(alone[0][0], again[0][0])
