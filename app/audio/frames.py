"""PCM helpers: framing for Silero VAD, format conversion, in-memory WAV handling."""

import io
import wave

import numpy as np

VAD_FRAME_SAMPLES = 512  # Silero VAD operates on 512-sample windows at 16 kHz
PIPELINE_SAMPLE_RATE = 16000


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def wav_wrap(pcm16: bytes, sample_rate: int = PIPELINE_SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 mono in a WAV container (for STT upload / debug dumps)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def wav_unwrap(wav_bytes: bytes) -> bytes:
    """Extract raw PCM from a WAV container by locating its data chunk."""
    idx = wav_bytes.find(b"data")
    if wav_bytes[:4] != b"RIFF" or idx < 0:
        return wav_bytes  # not a WAV — assume already raw PCM
    return wav_bytes[idx + 8 :]


def read_wav_as_pipeline_pcm(path: str) -> bytes:
    """Read any PCM WAV file and return 16 kHz mono PCM16 (downmix + resample)."""
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported (got {width * 8}-bit)")
    samples = pcm16_to_float32(raw)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != PIPELINE_SAMPLE_RATE:
        n_out = int(len(samples) * PIPELINE_SAMPLE_RATE / rate)
        src_t = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        dst_t = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        samples = np.interp(dst_t, src_t, samples).astype(np.float32)
    return float32_to_pcm16(samples)


def pcm_duration_s(pcm16: bytes, sample_rate: int = PIPELINE_SAMPLE_RATE) -> float:
    return len(pcm16) / 2 / sample_rate


class FrameBuffer:
    """Re-chunks arbitrary-size PCM16 byte packets into exact VAD-size frames."""

    def __init__(self, frame_samples: int = VAD_FRAME_SAMPLES):
        self._frame_bytes = frame_samples * 2
        self._pending = b""

    def push(self, pcm: bytes) -> list[bytes]:
        self._pending += pcm
        frames = []
        while len(self._pending) >= self._frame_bytes:
            frames.append(self._pending[: self._frame_bytes])
            self._pending = self._pending[self._frame_bytes :]
        return frames

    def reset(self) -> None:
        self._pending = b""
