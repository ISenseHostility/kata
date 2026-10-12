"""Audio encoder + denoiser (spec §5.2-5.3)."""
import pytest

torch = pytest.importorskip("torch")

from kata.motion.model import ModelConfig, MotionDenoiser, n_params, rope  # noqa: E402

TINY = ModelConfig(d=32, heads=4, enc_layers=1, dec_layers=2, dropout=0.0)


def _inputs(B=2, W=16, ctx=4, c=TINY):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(B, W, c.n_in, generator=g), torch.tensor([3, 900]),
            torch.randn(B, W + 2 * ctx, c.audio_dim, generator=g), torch.tensor([1, 5]),
            torch.tensor([4.0, 7.5]), torch.tensor([False, True]))


def test_full_size_model_is_about_45m_parameters():
    n = n_params(MotionDenoiser(ModelConfig()))
    assert 40e6 < n < 50e6


def test_shapes_and_zero_initialised_output():
    m = MotionDenoiser(TINY)
    x, t, audio, label, nps, drop = _inputs()
    out = m(x, t, m.encode_audio(audio), label, nps, drop)
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0                     # adaLN-Zero: the untrained model predicts 0


def test_rope_is_relative():
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    a = (rope(q, torch.tensor([3.0])) * rope(k, torch.tensor([1.0]))).sum()
    b = (rope(q, torch.tensor([12.0])) * rope(k, torch.tensor([10.0]))).sum()
    assert torch.allclose(a, b, atol=1e-5)


def test_conditioning_and_audio_change_the_output_after_a_step():
    torch.manual_seed(0)
    m = MotionDenoiser(TINY)
    x, t, audio, label, nps, drop = _inputs()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    target = torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    for _ in range(3):                                       # move the zero-initialised gates
        loss = (m(x, t, m.encode_audio(audio), label, nps, drop) - target).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    base = m(x, t, m.encode_audio(audio), label, nps, drop)
    assert not torch.allclose(base, m(x, t, m.encode_audio(audio), torch.tensor([2, 5]), nps, drop))
    assert not torch.allclose(base, m(x, t, m.encode_audio(audio + 1), label, nps, drop))
    assert not torch.allclose(base, m(x, t, m.encode_audio(audio), label, nps + 3, drop))
