"""On-disk cache for fixed-script speech.

A collections call opens with the same disclosure every time and asks the same
verification question every time. Synthesising those afresh on every call costs
money per character and, worse, puts four to six serial TTS round trips between
answering and the caller hearing anything.

Only *fixed* lines are cached. Model-generated replies are unique by nature, so
caching them would grow the directory without ever hitting.

The key is content-addressed over everything that changes the audio - the text,
the voice, the speaking rate, the sample rate - so an admin editing a script
line invalidates it implicitly. There is nothing to purge and no version to
bump by hand. The one thing a content hash cannot notice is a change in how the
bytes are encoded, which is what CACHE_VERSION is for.
"""

import asyncio
import hashlib
import logging
from pathlib import Path

from .config import ROOT

log = logging.getLogger("agent")

#: Bump only if the stored byte format changes (it is raw PCM16 today). The
#: text, voice, provider and rate are already in the key and need no version.
#: Bumped to 2 when the provider joined the key - see cache_key().
CACHE_VERSION = "2"

CACHE_DIR = ROOT / "data" / "tts_cache"

# Serialises synthesis per key, so N concurrent calls opening with the same
# greeting perform one synthesis between them rather than N.
_locks: dict[str, asyncio.Lock] = {}


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:64]


def cache_key(text: str, *, provider: str, voice: str, speaking_rate: float,
              sample_rate: int) -> str:
    """Content-addressed over everything that changes the audio.

    `provider` is in the key even though a voice name usually implies one,
    because the failure it prevents is silent and specifically nasty: the
    fixed script lines (greeting, disclosure, the tool filler) come from the
    cache while the conversation around them is synthesised live, so a stale
    entry means the caller hears the disclosure in one voice and the reply in
    another. That is obvious to a listener and invisible to ?mode=vad, which
    plays no audio at all.
    """
    raw = f"{CACHE_VERSION}|{provider}|{voice}|{speaking_rate}|{sample_rate}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def cache_path(text: str, *, provider: str, voice: str, speaking_rate: float,
               sample_rate: int) -> Path:
    key = cache_key(text, provider=provider, voice=voice,
                    speaking_rate=speaking_rate, sample_rate=sample_rate)
    return CACHE_DIR / _safe(provider) / _safe(voice) / f"{key}.pcm"


async def synthesize_cached(tts, text: str, *, provider: str, voice: str,
                            speaking_rate: float, sample_rate: int) -> tuple[bytes, bool]:
    """Return (pcm, was_cache_hit). Raw PCM16 at sample_rate, exactly what
    TTSProvider.synthesize returns, so there is no conversion on the read path.

    A cache failure is never fatal: on any filesystem error this falls back to
    synthesising normally and reports a miss. Losing the cache costs latency;
    raising here would cost the call.
    """
    path = cache_path(text, provider=provider, voice=voice,
                      speaking_rate=speaking_rate, sample_rate=sample_rate)
    try:
        if path.is_file():
            return path.read_bytes(), True
    except OSError:
        log.warning("tts cache: unreadable %s - synthesizing instead", path.name)

    lock = _locks.setdefault(path.name, asyncio.Lock())
    async with lock:
        # another waiter may have written it while this one queued
        try:
            if path.is_file():
                return path.read_bytes(), True
        except OSError:
            pass
        pcm = await tts.synthesize(text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pcm)
            # Sidecar so the cache is greppable. Without it the directory is
            # opaque hashes, and "why did it say that?" has no answer short of
            # playing every file.
            path.with_suffix(".txt").write_text(text, encoding="utf-8")
        except OSError:
            log.warning("tts cache: could not write %s", path.name)
        return pcm, False


#: Script lines carrying one of these differ per call, so they are never
#: cached - splicing cached fragments around a name produces audibly wrong
#: intonation at every seam.
PER_CALL_TOKENS = ("{customer}", "{lender}")


def cacheable_lines(cfg, language: str) -> dict[str, str]:
    """The fixed script lines for a language that are worth caching."""
    scripts = cfg.collections.scripts.get(language) or {}
    return {k: v for k, v in scripts.items()
            if v and not any(tok in v for tok in PER_CALL_TOKENS)}


async def prewarm(cfg, build_tts) -> None:
    """Synthesize every fixed script line for every enabled language and
    gender, so the first real call does not pay for them.

    A cold cache costs four to six serial round trips before the caller hears
    anything, landing precisely on the regulatory disclosure that opens the
    call. On a provider whose per-request setup dominates, that is most of the
    delay.

    Never fatal and never blocking: this runs as a background task, and a
    provider that is down at boot must not stop the server starting. It logs
    loudly instead, and the lines are simply synthesized on demand later.
    """
    from . import tts_catalog

    warmed = failed = 0
    for lang in (l for l in cfg.languages if l.enabled):
        lines = cacheable_lines(cfg, lang.code)
        if not lines:
            continue
        for gender in ("female", "male"):
            try:
                voice, _, resolved = tts_catalog.resolve_voice(
                    lang.locale, None, gender, None, cfg
                )
            except ValueError:
                continue  # no voices for this locale/gender - nothing to warm
            if resolved != gender:
                continue  # fell back to a gender already warmed
            call_cfg = cfg.model_copy(
                update={"tts": cfg.tts.model_copy(update={"voice": voice})}
            )
            try:
                tts = build_tts(call_cfg)
                for text in lines.values():
                    _, hit = await synthesize_cached(
                        tts, text,
                        provider=cfg.tts.provider, voice=voice,
                        speaking_rate=cfg.tts.speaking_rate,
                        sample_rate=tts.sample_rate,
                    )
                    warmed += 0 if hit else 1
            except Exception:
                failed += 1
                log.exception(
                    "tts cache: could not pre-warm %s/%s - the first call will "
                    "synthesize these lines itself", lang.code, gender
                )
    if warmed or failed:
        log.info("tts cache: pre-warmed %d line(s), %d voice(s) failed", warmed, failed)


def stats() -> tuple[int, int]:
    """(file count, total bytes) for the whole cache."""
    if not CACHE_DIR.exists():
        return 0, 0
    files = list(CACHE_DIR.rglob("*.pcm"))
    return len(files), sum(f.stat().st_size for f in files)
