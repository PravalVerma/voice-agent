"""Speech-to-text via Groq's hosted whisper-large-v3-turbo."""

from groq import AsyncGroq

from ..audio.frames import wav_wrap
from .base import STTProvider


class GroqSTT(STTProvider):
    def __init__(self, model: str = "whisper-large-v3-turbo", language: str = "en"):
        self._client = AsyncGroq()  # reads GROQ_API_KEY from the environment
        self._model = model
        self._language = language

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        wav = wav_wrap(pcm16, sample_rate)
        result = await self._client.audio.transcriptions.create(
            file=("utterance.wav", wav),
            model=self._model,
            language=self._language,
            temperature=0.0,
            response_format="text",
        )
        # response_format="text" returns a plain string; other formats return an object
        text = result if isinstance(result, str) else result.text
        return text.strip()
