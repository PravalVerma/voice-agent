"""Silero VAD (ONNX, v5 interface) and the VADGate endpointing state machine."""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime

from ..config import VADConfig
from .frames import VAD_FRAME_SAMPLES, pcm16_to_float32

MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"

FRAME_MS = VAD_FRAME_SAMPLES * 1000 // 16000  # 32 ms per frame
_FRAME_BYTES = VAD_FRAME_SAMPLES * 2
_CONTEXT_SAMPLES = 64  # silero v5 carries 64 samples of context between frames
_TAIL_MS = 200  # trailing silence kept on each utterance (whisper context)


@lru_cache(maxsize=1)
def _get_session() -> onnxruntime.InferenceSession:
    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    names = {i.name for i in session.get_inputs()}
    if not {"input", "state", "sr"} <= names:
        raise RuntimeError(
            f"unexpected silero_vad.onnx inputs {names} - expected the v5 model"
        )
    return session


class SileroVAD:
    """Per-caller speech-probability model. The ONNX session is shared;
    the recurrent state is per instance and must never be shared across calls."""

    def __init__(self):
        self._session = _get_session()
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def process(self, frame_pcm16: bytes) -> float:
        """Speech probability for one 512-sample (1024-byte) PCM16 frame @ 16 kHz."""
        samples = pcm16_to_float32(frame_pcm16).reshape(1, -1)
        x = np.concatenate([self._context, samples], axis=1)
        out, self._state = self._session.run(
            ["output", "stateN"], {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -_CONTEXT_SAMPLES:]
        return float(out[0, 0])


@dataclass
class SpeechStart:
    pass


@dataclass
class SpeechEnd:
    utterance: bytes  # PCM16 @ 16 kHz: pre-roll + speech + ~200 ms tail
    duration_s: float
    forced: bool = False  # True when max_utterance_s cut it off


class VADGate:
    """Turns per-frame speech probabilities into utterance segments.

    Feed exact 512-sample frames via push(); it emits SpeechStart when speech
    has been sustained for min_speech_ms (prepending pre_roll_ms of audio) and
    SpeechEnd after silence_ms of quiet (trimmed to a short tail).

    Set .strict = True while the agent is speaking: barge-in detection then
    uses the stricter threshold/duration to resist acoustic echo.
    """

    def __init__(self, vad: SileroVAD, cfg: VADConfig):
        self._vad = vad
        self._cfg = cfg
        self.strict = False

        self._start_frames = max(1, round(cfg.min_speech_ms / FRAME_MS))
        self._strict_start_frames = max(1, round(cfg.barge_in_min_speech_ms / FRAME_MS))
        self._end_frames = max(1, round(cfg.silence_ms / FRAME_MS))
        self._tail_frames = max(1, round(_TAIL_MS / FRAME_MS))
        self._max_frames = max(1, round(cfg.max_utterance_s * 1000 / FRAME_MS))
        pre_roll_frames = max(1, round(cfg.pre_roll_ms / FRAME_MS))
        # pre-roll must cover the onset run that triggers SpeechStart
        self._pre_roll: deque[bytes] = deque(
            maxlen=pre_roll_frames + max(self._start_frames, self._strict_start_frames)
        )

        self.speech_active = False
        self._onset_run = 0
        self._silence_run = 0
        self._utterance: list[bytes] = []
        self.last_prob = 0.0

    def _start_threshold(self) -> float:
        return self._cfg.barge_in_threshold if self.strict else self._cfg.threshold

    def _required_onset(self) -> int:
        return self._strict_start_frames if self.strict else self._start_frames

    def push(self, frame_pcm16: bytes) -> list[SpeechStart | SpeechEnd]:
        if len(frame_pcm16) != _FRAME_BYTES:
            raise ValueError(f"expected {_FRAME_BYTES}-byte frames, got {len(frame_pcm16)}")
        prob = self._vad.process(frame_pcm16)
        self.last_prob = prob
        events: list[SpeechStart | SpeechEnd] = []

        if not self.speech_active:
            self._pre_roll.append(frame_pcm16)
            if prob >= self._start_threshold():
                self._onset_run += 1
                if self._onset_run >= self._required_onset():
                    self.speech_active = True
                    self._utterance = list(self._pre_roll)
                    self._silence_run = 0
                    events.append(SpeechStart())
            else:
                self._onset_run = 0
        else:
            self._utterance.append(frame_pcm16)
            if prob < self._cfg.end_threshold:
                self._silence_run += 1
            else:
                self._silence_run = 0
            if self._silence_run >= self._end_frames:
                events.append(self._finish(forced=False))
            elif len(self._utterance) >= self._max_frames:
                events.append(self._finish(forced=True))

        return events

    def _finish(self, forced: bool) -> SpeechEnd:
        frames = self._utterance
        if not forced and self._silence_run > self._tail_frames:
            frames = frames[: len(frames) - (self._silence_run - self._tail_frames)]
        pcm = b"".join(frames)
        self.speech_active = False
        self._onset_run = 0
        self._silence_run = 0
        self._utterance = []
        self._pre_roll.clear()
        return SpeechEnd(utterance=pcm, duration_s=len(pcm) / 2 / 16000, forced=forced)

    def reset(self) -> None:
        self.speech_active = False
        self._onset_run = 0
        self._silence_run = 0
        self._utterance = []
        self._pre_roll.clear()
        self._vad.reset()
