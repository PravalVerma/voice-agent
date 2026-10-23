"""Text-to-speech via edge-tts.

DEVELOPMENT SCAFFOLDING, NOT A SHIPPING TIER. edge-tts is an unofficial client
of a read-aloud endpoint that publishes no SLA, documents no rate limits, and
can change or disappear without notice. It is here so the cascaded pipeline can
speak at all while the paid account is inactive. Everything a paid provider
would need is already config-driven (the voice map, the timeout, the breaker),
so replacing it is a config change plus one file, not a rewrite.

Two things about the upstream API shape the code below:

  * The output format is hardcoded to 24 kHz mono MP3 - Communicate has no
    output-format parameter, and the service returns nothing for formats it
    does not list. The transport wants headerless PCM16, so a decode is
    mandatory. PyAV is used rather than an ffmpeg subprocess because it ships
    its own FFmpeg in the wheel (no system prerequisite) and decodes
    in-process, packet by packet, so decoding overlaps synthesis instead of
    following it.

  * Each request opens its own WebSocket, and that setup dominates: first byte
    is ~630 ms for a three-word line and ~840 ms for a forty-word one, so the
    fixed cost, not the text, is what the caller waits for. Connection reuse is
    not reachable - Communicate.stream() wraps the session in `async with` and
    owns the connector it is handed, closing it after a single request. What is
    reachable is yielding audio as it decodes (~300 ms per sentence) and the
    existing on-disk cache for fixed script lines (the whole ~650 ms).
"""

import asyncio
import logging
import time
from typing import AsyncIterator

import av
import edge_tts

from .base import TTSProvider

log = logging.getLogger(__name__)


class SpeechSynthesisError(RuntimeError):
    """Synthesis failed. The message is for logs and the check scripts, never
    for a caller or an admin screen - session.py speaks a fixed neutral line
    instead, and nothing here is rendered in the UI."""


class _Breaker:
    """Opens after `threshold` consecutive failures, stays open for `cooldown_s`.

    Without it, an endpoint that has started refusing costs every turn its full
    timeout before the caller hears the apology - which turns one broken call
    into a very slow broken call.
    """

    def __init__(self, threshold: int, cooldown_s: float):
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at = 0.0

    @property
    def is_open(self) -> bool:
        if self._failures < self._threshold:
            return False
        if time.monotonic() - self._opened_at >= self._cooldown_s:
            self._failures = 0  # cooled down - let one through to re-test
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures == self._threshold:
            self._opened_at = time.monotonic()
            log.warning(
                "tts: %d consecutive failures - pausing synthesis attempts for %.0fs",
                self._failures, self._cooldown_s,
            )


def _rate_string(speaking_rate: float) -> str:
    """1.0 -> '+0%', 1.1 -> '+10%', 0.9 -> '-10%' - the percentage delta the
    upstream API takes, rather than the multiplier the rest of the app uses."""
    return f"{round((speaking_rate - 1.0) * 100):+d}%"


class EdgeTTS(TTSProvider):
    def __init__(
        self,
        voice: str = "en-IN-NeerjaNeural",
        speaking_rate: float = 1.0,
        sample_rate: int = 24000,
        timeout_s: float = 10.0,
        failure_threshold: int = 3,
        cooldown_s: float = 60.0,
    ):
        self._voice = voice
        self._rate = _rate_string(speaking_rate)
        self.sample_rate = sample_rate
        self._timeout_s = timeout_s
        self._breaker = _Breaker(failure_threshold, cooldown_s)

    def supports_streaming(self) -> bool:
        return True

    async def synthesize(self, text: str) -> bytes:
        """The whole utterance as one bytes, for the ABC and for the cache
        write path, which needs the complete audio before it can store it."""
        out = bytearray()
        async for chunk in self.synthesize_stream(text):
            out += chunk
        return bytes(out)

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        if self._breaker.is_open:
            raise SpeechSynthesisError("speech synthesis is temporarily unavailable")
        try:
            got_audio = False
            async for pcm in self._stream_pcm(text):
                got_audio = True
                yield pcm
            if not got_audio:
                raise SpeechSynthesisError("speech synthesis returned no audio")
        except GeneratorExit:
            raise  # barge-in cancelled the consumer; not a provider failure
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._breaker.record_failure()
            # Deliberately not logged here. session.py already logs a failed
            # turn with log.exception and the full traceback, and it knows
            # which call and turn this was - logging again would print the same
            # stack twice in the server and bury the one-line failure the check
            # scripts report. The cause is chained, so nothing is lost.
            raise SpeechSynthesisError(f"{type(e).__name__}: {e}".rstrip(": ")) from e
        else:
            self._breaker.record_success()

    async def _stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Drive the upstream stream, decoding each MP3 packet as it lands.

        The decoder is per-call: an MP3 CodecContext carries bit-reservoir state
        across packets, so sharing one between concurrent calls would interleave
        two streams into noise.
        """
        codec = av.CodecContext.create("mp3", "r")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=self.sample_rate)
        deadline = time.monotonic() + self._timeout_s
        stream = edge_tts.Communicate(text, self._voice, rate=self._rate).stream()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no audio within {self._timeout_s:.0f}s")
                try:
                    message = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                if message["type"] != "audio":
                    continue  # WordBoundary / SentenceBoundary metadata
                for pcm in _decode(codec, resampler, message["data"]):
                    yield pcm
            for pcm in _flush(codec, resampler):
                yield pcm
        finally:
            # stream() holds an aiohttp session; abandoning it mid-iteration
            # (a barge-in, a timeout) leaks the socket unless it is closed.
            await stream.aclose()


def _decode(codec, resampler, mp3: bytes) -> list[bytes]:
    out: list[bytes] = []
    for packet in codec.parse(mp3):
        for frame in codec.decode(packet):
            out.extend(_resample(resampler, frame))
    return out


def _flush(codec, resampler) -> list[bytes]:
    out: list[bytes] = []
    for frame in codec.decode(None):
        out.extend(_resample(resampler, frame))
    out.extend(_resample(resampler, None))
    return out


def _resample(resampler, frame) -> list[bytes]:
    out: list[bytes] = []
    for resampled in resampler.resample(frame):
        # s16 mono is packed, so plane 0 is the PCM. The plane buffer is padded
        # to its allocation, so it is sliced to the real sample count - without
        # that, every frame carries a little trailing silence.
        pcm = bytes(resampled.planes[0])[: resampled.samples * 2]
        if pcm:
            out.append(pcm)
    return out
