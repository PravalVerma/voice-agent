"""Provider interfaces. All audio is 16-bit PCM mono; text is plain (no markdown)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Sequence


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    #: portion of input_tokens served from the provider's prompt cache, if it
    #: reports one (0 if the provider doesn't support/report caching)
    cached_tokens: int = 0


@dataclass(frozen=True)
class TextDelta:
    """A chunk of reply text, to be spoken."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """A tool the model wants run.

    `arguments` stays a raw JSON string exactly as the provider emitted it and
    is parsed at dispatch, not here: providers stream it in fragments that are
    only valid JSON once complete, and a provider that emits malformed JSON
    should fail where the failure can be answered with a tool error rather than
    inside the stream.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Done:
    """End of a completion. Advisory - callers must also treat the generator
    running out as the end, because a stream cancelled by barge-in never
    reaches this."""

    usage: LLMUsage | None = None
    finish_reason: str = "stop"


LLMEvent = TextDelta | ToolCall | Done


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        """Transcribe one utterance of raw PCM16 audio to text."""


class LLMProvider(ABC):
    #: usage of the most recent completed stream_chat call (None if unknown).
    #: Still the source of truth rather than Done.usage: barge-in closes the
    #: generator mid-stream, so the caller reads usage at a point where no Done
    #: was ever yielded.
    last_usage: LLMUsage | None = None

    #: HTTP requests made since this provider was built, retries included.
    #: A call's budget is counted in requests, not turns, so this has to be
    #: incremented by whoever actually makes the request.
    request_count: int = 0

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: Sequence[dict] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Yield TextDelta / ToolCall / Done events. Must be cancellation-safe.

        When `tools` is None the request must be byte-identical to one made
        before tool calling existed - no empty `tools` array, no `tool_choice`.
        That keeps the no-tools path, which the check scripts and the second
        round of a tool turn both use, on exactly the shape that is already
        proven in production.
        """


class TTSProvider(ABC):
    #: sample rate of the PCM returned by synthesize()
    sample_rate: int = 24000

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to headerless PCM16 mono at self.sample_rate."""

    def supports_streaming(self) -> bool:
        """True if synthesize_stream() yields audio before the whole sentence is
        ready. Opt-in: a provider whose API is a single unary request (Google
        Cloud TTS, Gemini REST) gains nothing from the streaming path and keeps
        the simpler one."""
        return False

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Same audio as synthesize(), yielded in order as it becomes available.

        The default is the whole utterance in one chunk, so every provider
        satisfies this contract without implementing it - callers can branch on
        supports_streaming() for latency and still work either way.
        """
        yield await self.synthesize(text)
