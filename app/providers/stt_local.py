"""Local STT via faster-whisper (guarded import - `uv sync --extra local-stt`)."""

import asyncio

import numpy as np
from faster_whisper import WhisperModel

from ..audio.frames import pcm16_to_float32
from .base import STTProvider


class LocalWhisperSTT(STTProvider):
    def __init__(self, model: str = "small", language: str = "en"):
        # faster-whisper uses different model names than Groq ("small", "medium", ...);
        # "whisper-large-v3-turbo" is Groq's hosted name, not a faster-whisper size
        size = "small" if model == "whisper-large-v3-turbo" else model
        self._model = WhisperModel(size, device="cpu", compute_type="int8")
        self._language = language

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        audio = pcm16_to_float32(pcm16)
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, language=self._language)
        return " ".join(seg.text.strip() for seg in segments).strip()
