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
from app import tts_catalog
from app.providers import build_tts


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="text to speak")
    parser.add_argument("--out", default="out.wav", help="output WAV path (default: out.wav)")
    parser.add_argument("--language", help="language code (default: config default_language)")
    parser.add_argument("--gender", default="female", choices=("female", "male"))
    args = parser.parse_args()

    cfg = load_config_or_die()
    if cfg.tts.provider == "google":
        need_env("GOOGLE_APPLICATION_CREDENTIALS", "Google TTS")

    # Resolve the voice the way a real call does rather than using TTSConfig's
    # static default, which only ever named a voice for one provider.
    language = args.language or cfg.default_language
    lang = next((l for l in cfg.languages if l.code == language), None)
    if lang is None:
        sys.exit(f"error: unknown language '{language}' - check [[languages]] in config.toml")
    tts_catalog.init_catalog([lang.locale])  # no-op unless the provider has a live catalog
    try:
        voice, tier, gender = tts_catalog.resolve_voice(lang.locale, None, args.gender, None, cfg)
    except ValueError as e:
        sys.exit(f"error: {e}")
    cfg = cfg.model_copy(update={"tts": cfg.tts.model_copy(update={"voice": voice, "tier": tier})})

    tts = build_or_die(build_tts, cfg)
    t0 = time.perf_counter()
    try:
        pcm = await tts.synthesize(args.text)
    except Exception as e:
        # One clear line, like every other failure these scripts report.
        # A provider that cannot reach its endpoint is the expected case
        # here, and a traceback buries which of the four it was.
        sys.exit(f"error: speech synthesis failed - {e}")
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
