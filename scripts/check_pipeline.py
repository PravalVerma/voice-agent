"""Full pipeline offline: WAV question in -> agent reply WAV out, no server/browser.

Usage: uv run python scripts/check_pipeline.py question.wav [--out reply.wav]
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from _bootstrap import build_or_die, load_config_or_die, need_env

from app.audio.frames import read_wav_as_pipeline_pcm, wav_wrap
from app.pipeline import run_agent_turn
from app.providers import build_llm, build_stt, build_tts
from app.session import CallSession
from app.transport.base import AudioTransport


class CollectingTransport(AudioTransport):
    """Minimal AudioTransport that records everything instead of sending it."""

    def __init__(self, out_sample_rate: int):
        self.in_sample_rate = 16000
        self.out_sample_rate = out_sample_rate
        self.audio_by_turn: dict[int, bytearray] = {}
        self.controls: list[dict] = []

    async def recv(self):
        raise NotImplementedError("not used in this offline check")

    async def send_audio(self, turn_id: int, pcm: bytes) -> None:
        self.audio_by_turn.setdefault(turn_id, bytearray()).extend(pcm)

    async def send_control(self, data: dict) -> None:
        self.controls.append(data)
        if data["type"] == "transcript":
            print(f"  user: {data['text']}")
        elif data["type"] == "agent_text":
            print(f"  agent: {data['text']}")
        elif data["type"] == "state":
            print(f"  [state -> {data['state']}]")
        elif data["type"] == "error":
            print(f"  [error] {data['message']}")

    async def close(self) -> None:
        pass


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="path to a WAV file containing a spoken question")
    parser.add_argument("--out", default="reply.wav", help="output WAV path (default: reply.wav)")
    args = parser.parse_args()

    cfg = load_config_or_die()
    if cfg.stt.provider == "groq":
        need_env("GROQ_API_KEY", "Groq STT")
    if cfg.llm.provider == "groq":
        need_env("GROQ_API_KEY", "Groq LLM")
    if cfg.tts.provider == "google":
        need_env("GOOGLE_APPLICATION_CREDENTIALS", "Google TTS")

    pcm = read_wav_as_pipeline_pcm(args.wav)
    print(f"input: {len(pcm) / 2 / 16000:.2f}s @ 16 kHz mono")

    stt = build_or_die(build_stt, cfg)
    llm = build_or_die(build_llm, cfg)
    tts = build_or_die(build_tts, cfg)
    transport = CollectingTransport(out_sample_rate=cfg.tts.sample_rate)
    session = CallSession(transport, cfg, stt, llm, tts)

    t0 = time.perf_counter()
    await run_agent_turn(session, turn_id=1, utterance_pcm=pcm)
    total_ms = (time.perf_counter() - t0) * 1000
    meta = session.call_logger.finalize()

    reply_pcm = bytes(transport.audio_by_turn.get(1, b""))
    if not reply_pcm:
        sys.exit("no agent audio was produced - check the logs above for an error")

    out = Path(args.out)
    out.write_bytes(wav_wrap(reply_pcm, cfg.tts.sample_rate))
    print(f"\nwrote {out} ({len(reply_pcm) / 2 / cfg.tts.sample_rate:.2f}s of audio)")
    print(f"total turn time: {total_ms:.0f} ms")
    print(f"estimated cost: Rs {meta['cost_inr']} ({meta['usage']})")


if __name__ == "__main__":
    asyncio.run(main())
