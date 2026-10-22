"""Transcribe a WAV file through the configured STT provider and print timing.

Usage: uv run python scripts/check_stt.py path\\to\\sample.wav [--local]
"""

import argparse
import asyncio
import time

from _bootstrap import build_or_die, load_config_or_die, need_env

from app.audio.frames import pcm_duration_s, read_wav_as_pipeline_pcm
from app.providers import build_stt


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="path to a WAV file with speech")
    parser.add_argument("--local", action="store_true", help="use local faster-whisper")
    args = parser.parse_args()

    cfg = load_config_or_die()
    if args.local:
        cfg.stt.provider = "local"
    if cfg.stt.provider == "groq":
        need_env("GROQ_API_KEY", "Groq STT")

    pcm = read_wav_as_pipeline_pcm(args.wav)
    print(f"audio: {pcm_duration_s(pcm):.2f}s @ 16 kHz mono")

    stt = build_or_die(build_stt, cfg)
    t0 = time.perf_counter()
    text = await stt.transcribe(pcm)
    dt = (time.perf_counter() - t0) * 1000
    print(f"latency: {dt:.0f} ms  ({cfg.stt.provider}:{cfg.stt.model})")
    print(f"transcript: {text}")


if __name__ == "__main__":
    asyncio.run(main())
