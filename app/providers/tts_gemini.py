"""Text-to-speech via the Gemini API's native speech generation
(generateContent with responseModalities=["AUDIO"]), over REST (no SDK
dependency, same style as llm_gemini.py). Unlike Google Cloud TTS this
returns raw PCM16 mono 24kHz directly - no WAV container to unwrap - and
there's no per-locale voice catalog: any of the 30 prebuilt voices can speak
any of Gemini's ~90 supported languages, auto-detected from the text itself."""

import base64
import os

import httpx

from .base import TTSProvider

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiTTS(TTSProvider):
    def __init__(self, voice_name: str = "Kore", model: str = "gemini-3.1-flash-tts-preview"):
        self._voice_name = voice_name
        self._model = model
        self._api_key = os.environ["GEMINI_API_KEY"]
        self.sample_rate = 24000  # Gemini TTS always returns 24kHz mono 16-bit PCM

    async def synthesize(self, text: str) -> bytes:
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice_name}}
                },
            },
        }
        url = f"{_BASE_URL}/models/{self._model}:generateContent"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, params={"key": self._api_key}, json=payload)
            resp.raise_for_status()
            data = resp.json()
        b64_audio = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        return base64.b64decode(b64_audio)
