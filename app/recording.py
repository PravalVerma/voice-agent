"""Whole-call audio recording: both sides of the conversation mixed down to a
single mono WAV, saved next to the call's transcript and optionally archived to
S3.

Tapped at the transport (see transport/browser_ws.py) rather than in the
pipeline, so the standard STT->LLM->TTS path and the Gemini Live path are both
covered by the same code without either knowing recording exists.

Placement on the timeline is per-channel playhead-based, not append-based: the
caller's frames arrive in real time, but the agent's TTS audio is pushed down
the socket far faster than it plays back. Appending each side as it arrived
would slide the agent's speech earlier and earlier relative to the caller's.
Instead each channel tracks where its own audio has been laid down to; a chunk
that arrives after a gap pads silence up to "now" first, and chunks arriving
back-to-back within one utterance stay contiguous.

Known imprecision: when the caller barges in, the client flushes whatever agent
audio it hasn't played yet, but the server has already recorded the full
sentence. Interrupted agent turns therefore sound complete in the recording even
though the caller only heard the start of them.
"""

import logging
import time

import numpy as np

from .audio.frames import wav_wrap

log = logging.getLogger("agent")

# Speech-band mono is plenty for QA review and keeps a 10-minute call around
# 19 MB instead of 29 MB at the agent's native 24 kHz.
RECORDING_SAMPLE_RATE = 16000


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    n_out = int(round(samples.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.empty(0, dtype=np.float32)
    # linear interpolation is more than good enough for a review recording and
    # avoids pulling in scipy just for this
    src_idx = np.linspace(0.0, samples.size - 1, n_out, dtype=np.float64)
    return np.interp(src_idx, np.arange(samples.size), samples).astype(np.float32)


class _Channel:
    """One side of the call, laid out on the shared wall-clock timeline."""

    def __init__(self) -> None:
        self.samples: list[np.ndarray] = []
        self.n = 0  # samples written so far == playhead, in RECORDING_SAMPLE_RATE frames

    def add(self, audio: np.ndarray, elapsed_s: float) -> None:
        # pad with silence if this chunk starts after the last one ended
        target = int(elapsed_s * RECORDING_SAMPLE_RATE)
        gap = target - self.n
        if gap > 0:
            self.samples.append(np.zeros(gap, dtype=np.float32))
            self.n += gap
        self.samples.append(audio)
        self.n += audio.size

    def build(self, total: int) -> np.ndarray:
        track = np.concatenate(self.samples) if self.samples else np.empty(0, dtype=np.float32)
        if track.size < total:
            track = np.pad(track, (0, total - track.size))
        return track


class CallRecorder:
    """Accumulates both directions in memory, then mixes on demand.

    In-memory rather than streamed to disk because max_call_minutes caps a call
    at 10 minutes - roughly 19 MB per side at 16 kHz float32 - which is a fine
    trade for not having to mux two growing files at finalize time. If the call
    length cap is ever raised substantially this should become a spooled write.
    """

    def __init__(self, caller_rate: int = 16000, agent_rate: int = 24000):
        self.caller_rate = caller_rate
        self.agent_rate = agent_rate
        self._started = time.monotonic()
        self._caller = _Channel()
        self._agent = _Channel()

    def _elapsed(self) -> float:
        return time.monotonic() - self._started

    def _add(self, channel: _Channel, pcm: bytes, rate: int) -> None:
        if not pcm:
            return
        try:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            channel.add(_resample(audio, rate, RECORDING_SAMPLE_RATE), self._elapsed())
        except Exception:
            # a malformed frame must never take the call down
            log.exception("call recorder: dropped a bad audio chunk")

    def add_caller(self, pcm: bytes) -> None:
        self._add(self._caller, pcm, self.caller_rate)

    def add_agent(self, pcm: bytes) -> None:
        self._add(self._agent, pcm, self.agent_rate)

    @property
    def is_empty(self) -> bool:
        return self._caller.n == 0 and self._agent.n == 0

    def mix_wav(self) -> bytes | None:
        """Sum both sides into one mono track and wrap as a WAV. None if nothing
        was ever captured (e.g. the caller hung up before speaking)."""
        if self.is_empty:
            return None
        total = max(self._caller.n, self._agent.n)
        mixed = self._caller.build(total) + self._agent.build(total)
        # both sides talking at once can exceed full scale; scale the whole file
        # down rather than clip, so barge-ins don't turn into distortion
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 1.0:
            mixed /= peak
        pcm16 = (np.clip(mixed, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return wav_wrap(pcm16, RECORDING_SAMPLE_RATE)
