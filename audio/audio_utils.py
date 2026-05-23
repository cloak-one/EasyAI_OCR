# audio/audio_utils.py
import numpy as np


class PCMResampler:
    """有状态 PCM16 降采样器，跨 chunk 线性插值，避免拼接处 click/pop。"""

    def __init__(self, from_rate: int, to_rate: int):
        self.from_rate = from_rate
        self.to_rate = to_rate
        self._last_sample: int = 0

    def process(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes:
            return b""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        n_in = len(samples)
        n_out = max(1, int(round(n_in * self.to_rate / self.from_rate)))
        extended = np.concatenate([[self._last_sample], samples])
        x_in = np.arange(len(extended))
        x_out = np.linspace(0, len(extended) - 1, n_out + 1)[1:]
        resampled = np.interp(x_out, x_in, extended)
        self._last_sample = int(samples[-1])
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    def reset(self):
        self._last_sample = 0


def adjust_volume(pcm_bytes: bytes, factor: float) -> bytes:
    if not pcm_bytes:
        return b""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return np.clip(samples * factor, -32768, 32767).astype(np.int16).tobytes()