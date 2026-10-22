"""Speech-to-text via the Gemini API's audio understanding: an inline audio
Part in a plain generateContent call, over REST (no SDK dependency, same
style as llm_gemini.py). Gemini isn't a dedicated ASR model like Whisper -
it's prompted to transcribe, so accuracy on a given language/accent should be
verified live before relying on it, the same way this project already
verified Whisper's accuracy per-language."""

import base64
import os

import httpx

from ..audio.frames import wav_wrap
from .base import STTProvider

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_TRANSCRIBE_PROMPT = (
    "Transcribe this audio exactly as spoken, in its original language and script "
    "(expected language code: {language}). Output only the transcript text - no "
    "translation, no commentary, no quotes, no markdown."
)


class GeminiSTT(STTProvider):
    def __init__(self, model: str = "gemini-2.5-flash", language: str = "en"):
        self._model = model
        self._language = language
        self._api_key = os.environ["GEMINI_API_KEY"]

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        wav_b64 = base64.b64encode(wav_wrap(pcm16, sample_rate)).decode("ascii")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": _TRANSCRIBE_PROMPT.format(language=self._language)},
                        {"inline_data": {"mime_type": "audio/wav", "data": wav_b64}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        url = f"{_BASE_URL}/models/{self._model}:generateContent"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, params={"key": self._api_key}, json=payload)
            resp.raise_for_status()
            data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        return text.strip()
