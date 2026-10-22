"""Per-call transcript log plus a rough cost estimate, per the design PDF's
target of <= Rs 30/call. Not exact billing - just enough to catch a runaway call."""

import json
import logging
import time

from . import s3_archive
from .audio.frames import wav_wrap
from .config import ROOT, AppConfig
from .tts_catalog import QUALITY_LABEL, tier_of

log = logging.getLogger("agent")


class CallLogger:
    def __init__(
        self,
        session_id: str,
        cfg: AppConfig,
        pipeline: str = "standard",
        user_id: int | None = None,
        username: str | None = None,
        recorder=None,
    ):
        self.session_id = session_id
        self.cfg = cfg
        self.pipeline = pipeline
        self.user_id = user_id
        self.username = username
        self.recorder = recorder
        self.started_at = time.monotonic()
        self.turn_count = 0
        self.language = cfg.stt.language
        self.tts_tier = (
            "Studio" if pipeline == "gemini_live"
            else "Gemini" if cfg.tts.provider == "gemini"
            else QUALITY_LABEL.get(tier_of(cfg.tts.voice), tier_of(cfg.tts.voice))
        )
        self.stt_requests = 0
        self.stt_billed_seconds = 0.0
        self.stt_true_seconds = 0.0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.llm_cached_tokens = 0
        self.tts_chars = 0
        self.tts_output_audio_s = 0.0
        # gemini_live only: a live session has no separate STT/LLM/TTS legs, just
        # continuous audio in and out - see estimate_cost_usd().
        self.live_input_audio_s = 0.0
        self.live_output_audio_s = 0.0

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = ROOT / cfg.call.log_dir / f"{ts}_{session_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.dir / "transcript.jsonl", "a", encoding="utf-8")
        # written immediately (not just in finalize()) so who made the call is
        # still known even if the process dies/restarts mid-call and meta.json
        # never gets written
        self.log_event({"type": "call_started", "user_id": user_id, "username": username})

    def log_event(self, event: dict) -> None:
        # a cancelled turn can still be unwinding (recording its partial reply)
        # after the call has already ended and finalize() closed the file -
        # logging is best-effort and must never be what breaks a call
        if self._fh.closed:
            return
        record = {"t": round(time.monotonic() - self.started_at, 3), **event}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def record_stt(self, duration_s: float) -> None:
        self.stt_requests += 1
        self.stt_true_seconds += duration_s
        # Groq (and most hosted STT) bill a minimum duration per request, so
        # rapid short utterances cost far more than their audio length implies -
        # Gemini has no such floor, so its cost math uses stt_true_seconds instead
        self.stt_billed_seconds += max(duration_s, 10.0)

    def record_llm(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> None:
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.llm_cached_tokens += cached_tokens

    def record_tts(self, n_chars: int, output_duration_s: float = 0.0) -> None:
        self.tts_chars += n_chars
        self.tts_output_audio_s += output_duration_s

    def record_live_audio(self, input_s: float = 0.0, output_s: float = 0.0) -> None:
        self.live_input_audio_s += input_s
        self.live_output_audio_s += output_s

    def dump_utterance_wav(self, pcm: bytes, sample_rate: int = 16000) -> None:
        if not self.cfg.call.dump_utterances:
            return
        name = f"utterance_{self.turn_count:03d}_{time.strftime('%H%M%S')}.wav"
        (self.dir / name).write_bytes(wav_wrap(pcm, sample_rate))

    def save_recording(self) -> dict | None:
        """Mix the call down to recording.wav in this call's log dir and, when
        S3 is configured, archive a copy. Returns the metadata block for
        meta.json, or None if there was nothing to save. Best-effort: a
        recording failure must not lose the transcript or the cost estimate."""
        if self.recorder is None:
            return None
        try:
            wav = self.recorder.mix_wav()
        except Exception:
            log.exception("call %s: failed to mix the recording", self.session_id)
            return None
        if wav is None:
            return None

        info: dict = {"file": "recording.wav", "bytes": len(wav)}
        try:
            (self.dir / "recording.wav").write_bytes(wav)
        except OSError:
            log.exception("call %s: failed to write recording.wav", self.session_id)
            return None

        if s3_archive.is_configured():
            key = s3_archive.build_key(self.dir.name, self.username, self.language, self.tts_tier)
            uri = s3_archive.upload(wav, key)
            if uri:
                info["s3_key"] = key
                info["s3_uri"] = uri
        return info

    def estimate_cost_usd(self) -> dict:
        c = self.cfg.cost

        if self.pipeline == "gemini_live":
            # 25 tok/s of audio each direction - same convention this codebase
            # already uses for Gemini TTS output (see the gemini_tts_usd_per_mtok_audio_out
            # comment above); no separate stt/llm/tts legs exist for this pipeline.
            live = (
                (self.live_input_audio_s * 25 / 1_000_000) * c.gemini_live_usd_per_mtok_audio_in
                + (self.live_output_audio_s * 25 / 1_000_000) * c.gemini_live_usd_per_mtok_audio_out
            )
            return {"stt": 0.0, "llm": 0.0, "tts": 0.0, "live": live, "total": live}

        if self.cfg.stt.provider == "gemini":
            # Gemini bills audio input as tokens (32 tok/s), no per-request
            # minimum-duration floor like Groq's, so this uses stt_true_seconds
            stt = (self.stt_true_seconds * 32 / 1_000_000) * c.gemini_stt_usd_per_mtok_audio_in
        else:
            stt = (self.stt_billed_seconds / 3600) * c.stt_usd_per_hour

        llm_in_rate = c.gemini_llm_usd_per_mtok_in if self.cfg.llm.provider == "gemini" else c.llm_usd_per_mtok_in
        llm_out_rate = c.gemini_llm_usd_per_mtok_out if self.cfg.llm.provider == "gemini" else c.llm_usd_per_mtok_out
        # cached input tokens are modeled at a flat 50% discount (Groq's documented
        # rate for gpt-oss models) - only meaningful once llm_cached_tokens is
        # nonzero, which requires the provider to actually report a cache hit
        # (GeminiLLM doesn't currently parse a cache-hit count, so this is always
        # 0 for the gemini pipeline - its cost estimate never reflects caching)
        uncached_input = self.llm_input_tokens - self.llm_cached_tokens
        llm = (
            (uncached_input / 1_000_000) * llm_in_rate
            + (self.llm_cached_tokens / 1_000_000) * llm_in_rate * 0.5
            + (self.llm_output_tokens / 1_000_000) * llm_out_rate
        )

        if self.cfg.tts.provider == "gemini":
            # Gemini bills TTS input as text tokens (~4 chars/token, a rough
            # estimate - Gemini doesn't expose an exact input-token count for
            # this call) and output as audio tokens (25 tok/s of generated speech)
            tts = (
                (self.tts_chars / 4 / 1_000_000) * c.gemini_tts_usd_per_mtok_in
                + (self.tts_output_audio_s * 25 / 1_000_000) * c.gemini_tts_usd_per_mtok_audio_out
            )
        else:
            tts = (self.tts_chars / 1_000_000) * c.tts_usd_per_mchar

        return {"stt": stt, "llm": llm, "tts": tts, "total": stt + llm + tts}

    def finalize(self) -> dict:
        duration_s = time.monotonic() - self.started_at
        cost_usd = self.estimate_cost_usd()
        recording = self.save_recording()
        meta = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "language": self.language,
            "tts_tier": self.tts_tier,
            "duration_s": round(duration_s, 1),
            "turn_count": self.turn_count,
            "usage": {
                "stt_requests": self.stt_requests,
                "stt_billed_seconds": round(self.stt_billed_seconds, 1),
                "stt_true_seconds": round(self.stt_true_seconds, 1),
                "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens,
                "llm_cached_tokens": self.llm_cached_tokens,
                "tts_chars": self.tts_chars,
                "tts_output_audio_s": round(self.tts_output_audio_s, 1),
                "live_input_audio_s": round(self.live_input_audio_s, 1),
                "live_output_audio_s": round(self.live_output_audio_s, 1),
            },
            "cost_usd": {k: round(v, 5) for k, v in cost_usd.items()},
            "cost_inr": round(cost_usd["total"] * self.cfg.cost.usd_to_inr, 2),
            "recording": recording,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._fh.close()
        return meta
