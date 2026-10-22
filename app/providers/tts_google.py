"""Text-to-speech via Google Cloud TTS (LINEAR16 output)."""

from google.cloud import texttospeech

from ..audio.frames import wav_unwrap
from .base import TTSProvider


class GoogleTTS(TTSProvider):
    def __init__(
        self,
        voice: str = "en-IN-Neural2-A",
        language_code: str = "en-IN",
        speaking_rate: float = 1.0,
        sample_rate: int = 24000,
    ):
        # reads GOOGLE_APPLICATION_CREDENTIALS from the environment
        self._client = texttospeech.TextToSpeechAsyncClient()
        self._voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, name=voice
        )
        self._audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            speaking_rate=speaking_rate,
        )
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=self._voice,
            audio_config=self._audio_config,
        )
        # LINEAR16 responses arrive in a WAV container — strip to raw PCM
        return wav_unwrap(response.audio_content)
