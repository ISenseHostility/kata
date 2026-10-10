"""GPU model wrappers: Demucs stems (§4.2) and Beat This! beats (§4.3, §5.1)."""

from __future__ import annotations

import numpy as np
import torch

from kata.audio.frames import SR
from kata.audio.store import STEMS

DEMUCS_MODEL = "htdemucs"
BEAT_CHECKPOINT = "final0"
BEAT_FPS = 50


class StemSeparator:
    name = DEMUCS_MODEL

    def __init__(self, device: str = "cuda"):
        from demucs.api import Separator
        self.device = device
        self.sep = Separator(model=DEMUCS_MODEL, device=device, shifts=0, overlap=0.25,
                             split=True, progress=False)
        if tuple(self.sep.model.sources) != STEMS:
            raise RuntimeError(f"unexpected stem order {self.sep.model.sources}")

    def __call__(self, stereo: np.ndarray) -> torch.Tensor:
        """(2, N) float32 at SR -> (4, 2, N) on the model device, in STEMS order."""
        wav = torch.from_numpy(stereo).to(self.device)
        _, stems = self.sep.separate_tensor(wav, SR)
        return torch.stack([stems[s] for s in STEMS])


class BeatTracker:
    name = f"beat_this:{BEAT_CHECKPOINT}"

    def __init__(self, device: str = "cuda"):
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor
        self.a2f = Audio2Frames(checkpoint_path=BEAT_CHECKPOINT, device=device)
        self.post = Postprocessor(type="minimal", fps=BEAT_FPS)

    def __call__(self, mono: np.ndarray, n_frames: int) -> dict[str, np.ndarray]:
        """-> beat_act, downbeat_act (n_frames,) at 100 fps; beats, downbeats in seconds."""
        beat_logits, down_logits = self.a2f(mono, SR)
        beats, downbeats = self.post(beat_logits, down_logits)
        src = np.arange(len(beat_logits), dtype=np.float64)  # 50 fps frame j is centred at j / 50 s
        dst = np.arange(n_frames, dtype=np.float64) / 2.0
        act = lambda l: np.interp(dst, src, torch.sigmoid(l).cpu().numpy(), right=0.0)  # noqa: E731
        return {"beat_act": act(beat_logits).astype(np.float32),
                "downbeat_act": act(down_logits).astype(np.float32),
                "beats": np.asarray(beats, dtype=np.float64),
                "downbeats": np.asarray(downbeats, dtype=np.float64)}
