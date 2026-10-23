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
#: text, voice and rate are already in the key and need no version.
CACHE_VERSION = "1"

CACHE_DIR = ROOT / "data" / "tts_cache"

# Serialises synthesis per key, so N concurrent calls opening with the same
# greeting perform one synthesis between them rather than N.
_locks: dict[str, asyncio.Lock] = {}


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:64]


def cache_key(text: str, *, voice: str, speaking_rate: float, sample_rate: int) -> str:
    raw = f"{CACHE_VERSION}|{voice}|{speaking_rate}|{sample_rate}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def cache_path(text: str, *, voice: str, speaking_rate: float, sample_rate: int) -> Path:
    key = cache_key(text, voice=voice, speaking_rate=speaking_rate, sample_rate=sample_rate)
    return CACHE_DIR / _safe(voice) / f"{key}.pcm"


async def synthesize_cached(tts, text: str, *, voice: str, speaking_rate: float,
                            sample_rate: int) -> tuple[bytes, bool]:
    """Return (pcm, was_cache_hit). Raw PCM16 at sample_rate, exactly what
    TTSProvider.synthesize returns, so there is no conversion on the read path.

    A cache failure is never fatal: on any filesystem error this falls back to
    synthesising normally and reports a miss. Losing the cache costs latency;
    raising here would cost the call.
    """
    path = cache_path(text, voice=voice, speaking_rate=speaking_rate, sample_rate=sample_rate)
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


def stats() -> tuple[int, int]:
    """(file count, total bytes) for the whole cache."""
    if not CACHE_DIR.exists():
        return 0, 0
    files = list(CACHE_DIR.rglob("*.pcm"))
    return len(files), sum(f.stat().st_size for f in files)
