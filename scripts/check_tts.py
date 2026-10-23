"""Synthesize text through the configured TTS provider and write a WAV file.

Usage: uv run python scripts/check_tts.py "Hello there, how are you today?" [--out out.wav]
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from _bootstrap import build_or_die, load_config_or_die, need_env

from app.audio.frames import pcm_duration_s, wav_wrap
from app.providers import build_tts


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="text to speak")
    parser.add_argument("--out", default="out.wav", help="output WAV path (default: out.wav)")
    args = parser.parse_args()

    cfg = load_config_or_die()
    if cfg.tts.provider == "google":
        need_env("GOOGLE_APPLICATION_CREDENTIALS", "Google TTS")

    tts = build_or_die(build_tts, cfg)
    t0 = time.perf_counter()
    try:
        pcm = await tts.synthesize(args.text)
    except Exception as e:
        # One clear line, like every other failure these scripts report.
        # A provider that cannot reach its endpoint is the expected case
        # here, and a traceback buries which of the four it was.
        sys.exit(f"error: speech synthesis failed - {type(e).__name__}: {e}")
    dt = (time.perf_counter() - t0) * 1000

    out = Path(args.out)
    out.write_bytes(wav_wrap(pcm, tts.sample_rate))
    print(f"latency: {dt:.0f} ms  ({cfg.tts.provider}:{cfg.tts.voice})")
    print(
        f"wrote {out} - {pcm_duration_s(pcm, tts.sample_rate):.2f}s of audio "
        f"@ {tts.sample_rate} Hz, {len(args.text)} chars"
    )


if __name__ == "__main__":
    asyncio.run(main())
