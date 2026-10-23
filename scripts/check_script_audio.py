"""Pre-synthesize the fixed collections script lines into the audio cache.

Usage: uv run python scripts/check_script_audio.py [--language en] [--tier Neural2] [--gender female] [--stats]

Run this after editing any [collections.scripts] line. A cold cache costs the
caller four to six serial synthesis round trips before they hear anything,
which lands precisely on the regulatory disclosure at the start of the call.

Lines containing {customer} or {lender} are reported but not warmed: they
differ per call, and splicing cached fragments around a name produces audibly
wrong intonation at every seam.
"""

import argparse
import asyncio
import sys

from _bootstrap import build_or_die, load_config_or_die, need_env

from app import tts_cache
from app.providers import build_tts

_PER_CALL_TOKENS = ("{customer}", "{lender}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en", help="language code to warm (default: en)")
    parser.add_argument("--tier", default=None, help="voice tier, e.g. Neural2 (default: the catalog's first)")
    parser.add_argument("--gender", default="female", help="voice gender (default: female)")
    parser.add_argument("--stats", action="store_true", help="only report what is already cached")
    args = parser.parse_args()

    cfg = load_config_or_die()

    if args.stats:
        count, size = tts_cache.stats()
        print(f"cache: {count} files, {size / 1024:.0f} KB at {tts_cache.CACHE_DIR}")
        return

    scripts = cfg.collections.scripts.get(args.language)
    if not scripts:
        known = ", ".join(sorted(cfg.collections.scripts)) or "none"
        sys.exit(f"error: no [collections.scripts.{args.language}] in config.toml (have: {known})")

    if cfg.tts.provider == "google":
        need_env("GOOGLE_APPLICATION_CREDENTIALS", "Google TTS")

    # Resolve the voice the same way a real call does, so the cache key matches.
    from app import tts_catalog

    tts_catalog.init_catalog([f"{args.language}-IN"])
    try:
        voice, tier, gender = tts_catalog.resolve_voice(f"{args.language}-IN", args.tier, args.gender, None)
    except ValueError as e:
        sys.exit(f"error: {e} - check GOOGLE_APPLICATION_CREDENTIALS and that the Text-to-Speech API is enabled")

    cfg.tts.voice = voice
    tts = build_or_die(build_tts, cfg)
    print(f"voice: {voice} ({tier}, {gender})  rate={cfg.tts.speaking_rate}  sr={cfg.tts.sample_rate}")

    warmed = hits = skipped = chars = 0
    for key, text in sorted(scripts.items()):
        if any(tok in text for tok in _PER_CALL_TOKENS):
            print(f"  skip  {key:22} (per-call text, not cacheable)")
            skipped += 1
            continue
        _, hit = await tts_cache.synthesize_cached(
            tts, text, voice=voice, speaking_rate=cfg.tts.speaking_rate, sample_rate=cfg.tts.sample_rate
        )
        if hit:
            hits += 1
            print(f"  cached {key:22} ({len(text)} chars)")
        else:
            warmed += 1
            chars += len(text)
            print(f"  WARMED {key:22} ({len(text)} chars)")

    rate = cfg.cost.tts_usd_per_mchar
    cost_inr = (chars / 1_000_000) * rate * cfg.cost.usd_to_inr
    print(f"\n{len(scripts)} lines: {hits} already cached, {warmed} synthesized, {skipped} per-call")
    print(f"synthesis cost this run: Rs {cost_inr:.4f} ({chars} chars at ${rate}/M)")


if __name__ == "__main__":
    asyncio.run(main())
