"""Spectral features on the frame grid, in PyTorch so they run on the GPU (§4.1, §4.2)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from kata.audio.frames import HOP, SR

N_FFT = 2048
CHROMA_N_FFT = 4096
MEL_FMIN = 30.0
MEL_FMAX = 16000.0
CHROMA_FMIN = 55.0
CHROMA_FMAX = 5000.0
DB_FLOOR = -100.0
FLUX_MAX_BINS = 3  # SuperFlux max filter width over mel bins
RMS_FLOOR = 1e-5  # amplitude floor before dB

_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(f) -> np.ndarray:
    """Slaney mel scale (linear below 1 kHz, log above), as librosa's default."""
    f = np.asarray(f, dtype=np.float64)
    return np.where(f >= _MIN_LOG_HZ,
                    _MIN_LOG_MEL + np.log(np.maximum(f, 1e-10) / _MIN_LOG_HZ) / _LOGSTEP,
                    f / _F_SP)


def mel_to_hz(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.where(m >= _MIN_LOG_MEL, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), _F_SP * m)


def mel_centers(n_mels: int, fmin: float = MEL_FMIN, fmax: float = MEL_FMAX) -> np.ndarray:
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))[1:-1]


def mel_filterbank(n_mels: int, n_fft: int = N_FFT, sr: int = SR,
                   fmin: float = MEL_FMIN, fmax: float = MEL_FMAX) -> np.ndarray:
    """(n_mels, n_fft//2 + 1) triangular filters with Slaney area normalisation."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(pts)
    ramps = pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0.0, np.minimum(lower, upper))
    w *= (2.0 / (pts[2:] - pts[:-2]))[:, None]
    return w.astype(np.float32)


def chroma_filterbank(n_fft: int = CHROMA_N_FFT, sr: int = SR,
                      fmin: float = CHROMA_FMIN, fmax: float = CHROMA_FMAX) -> np.ndarray:
    """(12, n_fft//2 + 1): each FFT bin in [fmin, fmax] goes to its nearest pitch class (0 = C)."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((12, len(freqs)), np.float32)
    idx = np.nonzero((freqs >= fmin) & (freqs <= fmax))[0]
    midi = 69 + 12 * np.log2(freqs[idx] / 440.0)
    fb[np.round(midi).astype(np.int64) % 12, idx] = 1.0
    return fb


def spectral_flux(log_mel: torch.Tensor) -> torch.Tensor:
    """(T, B) dB -> (T,) SuperFlux-style onset strength: positive rise vs. the 3-bin max of the previous frame."""
    ref = F.max_pool1d(log_mel[None], kernel_size=FLUX_MAX_BINS, stride=1, padding=FLUX_MAX_BINS // 2)[0]  # max over bins
    flux = (log_mel[1:] - ref[:-1]).clamp_min(0).sum(dim=1)
    return torch.cat([flux.new_zeros(1), flux])


class SpectralExtractor:
    """Builds windows and filterbanks once; every method returns float32 numpy, time on axis 0."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.window = torch.hann_window(N_FFT, device=self.device)
        self.chroma_window = torch.hann_window(CHROMA_N_FFT, device=self.device)
        self.mel_fb = {n: torch.from_numpy(mel_filterbank(n)).to(self.device) for n in (64, 128)}
        self.chroma_fb = torch.from_numpy(chroma_filterbank()).to(self.device)

    def _tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32).to(self.device)

    def _power(self, x: torch.Tensor, n_fft: int, window: torch.Tensor) -> torch.Tensor:
        """(..., N) -> (..., n_fft//2 + 1, T) power spectrogram, centred frames."""
        s = torch.stft(x, n_fft, hop_length=HOP, window=window, center=True,
                       pad_mode="reflect", return_complex=True)
        return s.real.square() + s.imag.square()

    def _log_mel(self, x: torch.Tensor, n_mels: int) -> torch.Tensor:
        m = self.mel_fb[n_mels] @ self._power(x, N_FFT, self.window)
        return (10.0 * torch.log10(m.clamp_min(1e-10))).clamp_min(DB_FLOOR).transpose(-1, -2)

    def _rms_db(self, x: torch.Tensor) -> torch.Tensor:
        """(..., N) -> (..., T) frame RMS in dBFS over N_FFT-sample centred windows."""
        pad = N_FFT // 2
        xp = F.pad(x.reshape(-1, 1, x.shape[-1]), (pad, pad), mode="reflect")
        frames = xp[:, 0].unfold(-1, N_FFT, HOP)
        rms = frames.square().mean(dim=-1).sqrt().reshape(*x.shape[:-1], -1)
        return (20.0 * torch.log10(rms.clamp_min(RMS_FLOOR))).clamp_min(DB_FLOOR)

    def _chroma(self, x: torch.Tensor) -> torch.Tensor:
        c = self.chroma_fb @ self._power(x, CHROMA_N_FFT, self.chroma_window)
        return (c / c.amax(dim=0, keepdim=True).clamp_min(1e-10)).T

    @torch.inference_mode()
    def mix(self, mono) -> dict[str, np.ndarray]:
        """mono (N,) -> mel (T,128), chroma (T,12), onset (T,), rms (T,)."""
        x = self._tensor(mono)
        mel = self._log_mel(x, 128)
        out = {"mel": mel, "chroma": self._chroma(x), "onset": spectral_flux(mel), "rms": self._rms_db(x)}
        return {k: v.float().cpu().numpy() for k, v in out.items()}

    @torch.inference_mode()
    def stems(self, stems) -> dict[str, np.ndarray]:
        """stems (4, 2, N) -> stem_mel (T,4,64), onset (4,T), rms (4,T)."""
        x = self._tensor(stems).mean(dim=1)  # (4, N) mono per stem
        mel = self._log_mel(x, 64)  # (4, T, 64)
        onset = torch.stack([spectral_flux(m) for m in mel])
        return {"stem_mel": mel.permute(1, 0, 2).float().cpu().numpy(),
                "onset": onset.float().cpu().numpy(),
                "rms": self._rms_db(x).float().cpu().numpy()}
