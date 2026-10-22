"""Provider interfaces. All audio is 16-bit PCM mono; text is plain (no markdown)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    #: portion of input_tokens served from the provider's prompt cache, if it
    #: reports one (0 if the provider doesn't support/report caching)
    cached_tokens: int = 0


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        """Transcribe one utterance of raw PCM16 audio to text."""


class LLMProvider(ABC):
    #: usage of the most recent completed stream_chat call (None if unknown)
    last_usage: LLMUsage | None = None

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Yield text deltas for a chat completion. Must be cancellation-safe."""


class TTSProvider(ABC):
    #: sample rate of the PCM returned by synthesize()
    sample_rate: int = 24000

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to headerless PCM16 mono at self.sample_rate."""
