"""Google Cloud TTS voice catalog: queries the real voice inventory instead of
hand-typed voice names, so tier/gender/language options shown to the caller are
whatever Google's API actually has - not a guess."""

import logging

from google.cloud import texttospeech

log = logging.getLogger("agent")

# Google's current published per-million-character rates. "Chirp-HD" (pre-Chirp3,
# English-only) is deliberately not offered - Chirp3-HD supersedes it.
TIER_RATES: dict[str, float] = {
    "Standard": 4.0,
    "Neural2": 16.0,
    "Chirp3-HD": 30.0,
}

# The single cascaded tier. One internal name, deliberately not a voice-family
# word, so it cannot collide with tier_of()'s substring match against a vendor's
# voice names the way "Standard" would.
CASCADED_TIER = "cascaded"

# Caller-facing quality tag shown in the UI - callers never see the internal
# tier name or which vendor/model it maps to, only this label.
QUALITY_LABEL: dict[str, str] = {
    CASCADED_TIER: "Standard",
    "gemini_live": "Studio",
}

# Character word per gender for a provider that offers one voice per gender
# (the config voice map). Same vocabulary as _LETTER_CHARACTER below, so the
# dropdown reads the same whichever provider is behind it.
_GENDER_CHARACTER = {"female": "Warm", "male": "Deep"}

_TIER_ORDER = ["Standard", "Neural2", "Chirp3-HD"]  # preferred default order


def all_tier_names() -> list[str]:
    """Every Voice Quality tier name the app can offer - the universe main.py
    validates a per-assignment allow-list against, and the Manage Bots modal
    renders as pills. Locale-independent, so it needs no populated catalog."""
    return [CASCADED_TIER, "gemini_live"]

_catalog: dict[str, dict[str, dict[str, str]]] = {}

# Google's own published one-word character for each Chirp3-HD voice (same 30
# names/genders as the Gemini-TTS voice set - verified against
# ai.google.dev/gemini-api/docs/speech-generation, cross-checked against this
# app's live en-IN/hi-IN Chirp3-HD catalog, all 30 names match exactly).
_CHIRP3_CHARACTER: dict[str, str] = {
    "Achernar": "Soft", "Achird": "Friendly", "Algenib": "Gravelly", "Algieba": "Smooth",
    "Alnilam": "Firm", "Aoede": "Breezy", "Autonoe": "Bright", "Callirrhoe": "Easy-going",
    "Charon": "Informative", "Despina": "Smooth", "Enceladus": "Breathy", "Erinome": "Clear",
    "Fenrir": "Excitable", "Gacrux": "Mature", "Iapetus": "Clear", "Kore": "Firm",
    "Laomedeia": "Upbeat", "Leda": "Youthful", "Orus": "Firm", "Puck": "Upbeat",
    "Pulcherrima": "Forward", "Rasalgethi": "Informative", "Sadachbia": "Lively",
    "Sadaltager": "Knowledgeable", "Schedar": "Even", "Sulafat": "Warm", "Umbriel": "Easy-going",
    "Vindemiatrix": "Gentle", "Zephyr": "Bright", "Zubenelgenubi": "Casual",
}

# Standard/Neural2 voices aren't individually documented by Google, but real
# measurement shows each letter is literally the same speaker across both
# tiers within a locale (e.g. en-IN-Standard-A/-Neural2-A have identical
# pitch - same recording, different synthesis quality). These
# labels are relative pitch character (low to high), computed once from real
# synthesized audio via autocorrelation-based pitch estimation, not guessed.
_LETTER_CHARACTER: dict[tuple[str, str], str] = {
    ("en-IN", "E"): "Deep", ("en-IN", "D"): "Warm", ("en-IN", "A"): "Bright",
    ("en-IN", "F"): "Deep", ("en-IN", "B"): "Warm", ("en-IN", "C"): "Bright",
    ("hi-IN", "E"): "Deep", ("hi-IN", "D"): "Warm", ("hi-IN", "A"): "Bright",
    ("hi-IN", "B"): "Deep", ("hi-IN", "F"): "Warm", ("hi-IN", "C"): "Bright",
}


def tier_of(voice_name: str) -> str | None:
    for tier in TIER_RATES:
        if tier in voice_name:
            return tier
    return None


def build_catalog(language_codes: list[str]) -> dict[str, dict[str, dict[str, list[str]]]]:
    """{locale: {tier: {"female": [voice_name, ...], "male": [voice_name, ...]}}}
    for each given locale (e.g. "en-IN"), built from a single live list_voices()
    call. Every voice Google offers for that locale+tier+gender is kept (not just
    the first alphabetically) so callers can pick a specific one, not just a gender."""
    client = texttospeech.TextToSpeechClient()
    all_voices = client.list_voices().voices

    catalog: dict[str, dict[str, dict[str, list[str]]]] = {}
    for locale in language_codes:
        by_tier: dict[str, dict[str, list[str]]] = {}
        matching = [v for v in all_voices if locale in v.language_codes]
        for voice in sorted(matching, key=lambda v: v.name):
            tier = tier_of(voice.name)
            if tier is None:
                continue
            gender = "female" if voice.ssml_gender.name == "FEMALE" else "male"
            by_tier.setdefault(tier, {}).setdefault(gender, []).append(voice.name)
        catalog[locale] = by_tier
    return catalog


def voice_character(locale: str, tier: str, voice_name: str) -> str:
    """Returns a human-meaningful character word for a specific voice (e.g.
    "Warm", "Firm") - never the raw letter/star-name Google uses internally."""
    prefix = f"{locale}-{tier}-"
    suffix = voice_name[len(prefix):] if voice_name.startswith(prefix) else voice_name
    if tier == "Chirp3-HD":
        return _CHIRP3_CHARACTER.get(suffix, suffix)
    return _LETTER_CHARACTER.get((locale, suffix), suffix)


def _dedupe_by_label(voices: list[dict]) -> list[dict]:
    """Some character words repeat across different underlying voices within
    the same gender (e.g. both "Zephyr" and "Autonoe" are "Bright") - keep
    only the first one per label so the caller never sees two identical-
    looking options. `voices` is already alphabetical by Google's internal
    voice name (see build_catalog), so "first" is a stable, deterministic
    pick - not a claim that one voice sounds objectively better."""
    seen: set[str] = set()
    kept = []
    for v in voices:
        if v["label"] in seen:
            continue
        seen.add(v["label"])
        kept.append(v)
    return kept


def init_catalog(language_codes: list[str]) -> None:
    """Fetches the Google voice catalog at startup.

    Only the google provider has a live catalog to fetch; for the others this
    is a no-op, so it does not spend a doomed API call (and a confusing
    credentials warning) on every boot.

    A failure is not fatal either way: without credentials there are simply no
    cascaded voices to offer and the dropdown falls back to the Studio tier
    alone, which runs on GEMINI_API_KEY and never touches this catalog. Letting
    the exception escape would take the whole server down at boot over a
    credential the Studio pipeline does not need.
    """
    global _catalog
    from .config import get_config

    if get_config().tts.provider != "google":
        _catalog = {}
        return
    try:
        _catalog = build_catalog(language_codes)
    except Exception as e:
        _catalog = {}
        log.warning(
            "tts catalog unavailable (%s) - the cascaded tier will have no "
            "voices to offer; the Studio tier is unaffected", e
        )
        return
    for locale, tiers in _catalog.items():
        log.info("tts catalog: %s -> %s", locale, sorted(tiers.keys()))


def get_catalog() -> dict[str, dict[str, dict[str, str]]]:
    return _catalog


def _google_voices(locale: str) -> dict[str, list[tuple[str, str]]]:
    """{gender: [(voice_name, label)]} for the best Google tier this locale has.

    The three quality levels are no longer separate dropdown entries, so one of
    them has to represent the cascaded tier; _TIER_ORDER's preference applies.
    The ladder itself is kept intact above, because flipping tts.provider back
    to "google" has to restore the old behaviour.
    """
    by_tier = _catalog.get(locale, {})
    if not by_tier:
        return {}
    tier = next((t for t in _TIER_ORDER if t in by_tier), next(iter(by_tier)))
    return {
        gender: [(v, voice_character(locale, tier, v)) for v in voices]
        for gender, voices in by_tier[tier].items()
    }


def _edge_voices(locale: str, cfg) -> dict[str, list[tuple[str, str]]]:
    """{gender: [(opaque_id, label)]} from the config voice map.

    The id handed to the browser is "en-female", not the vendor's voice short
    name. The dropdown only ever displays the label, but the id travels to the
    client and back over the WebSocket, and a short name is exactly the kind of
    vendor tell the rest of the app keeps server-side.
    """
    code = locale.split("-")[0]
    pair = cfg.tts.edge.voices.get(code)
    if pair is None:
        return {}
    return {
        gender: [(f"{code}-{gender}", _GENDER_CHARACTER[gender])]
        for gender in ("female", "male")
        if getattr(pair, gender, None)
    }


def _cascaded_voices(locale: str, cfg) -> dict[str, list[tuple[str, str]]]:
    """{gender: [(id, label)]} for whichever provider is configured."""
    if cfg.tts.provider == "edge":
        return _edge_voices(locale, cfg)
    if cfg.tts.provider == "gemini":
        tier = gemini_voice_tier()
        return {g: [(v["name"], v["label"]) for v in voices]
                for g, voices in tier["voices"].items()}
    return _google_voices(locale)


def _real_voice(voice_id: str, cfg) -> str:
    """The id the UI uses -> the voice name the provider is given. Only the edge
    provider indirects; the others hand out real names already."""
    if cfg.tts.provider != "edge":
        return voice_id
    code, _, gender = voice_id.rpartition("-")
    pair = cfg.tts.edge.voices.get(code)
    if pair is None or gender not in ("female", "male"):
        raise ValueError(f"no voice configured for {voice_id!r}")
    return getattr(pair, gender)


def rate_for(cfg, voice: str) -> float:
    """USD per million characters for the voice this call resolved to.

    Kept as a rate rather than deleted for a provider that does not bill per
    character: the character counting and the cost field stay live, so a call
    on a free provider reports a real zero instead of a missing number, and a
    paid provider needs only its rate filled in.
    """
    if cfg.tts.provider == "edge":
        return 0.0
    if cfg.tts.provider == "gemini":
        return 0.0  # billed per token in calllog, not per character
    return TIER_RATES.get(tier_of(voice) or "", 0.0)


def cascaded_tier(locale: str, cfg) -> dict:
    """The one cascaded entry in the Voice Quality dropdown, in the same shape
    gemini_voice_tier() returns."""
    voices = _cascaded_voices(locale, cfg)
    first = next((v[0][0] for v in voices.values() if v), "")
    return {
        "name": CASCADED_TIER,
        "quality": QUALITY_LABEL[CASCADED_TIER],
        "usd_per_mchar": rate_for(cfg, _real_voice(first, cfg)) if first else None,
        "voices": {
            gender: _dedupe_by_label([{"name": n, "label": lbl} for n, lbl in entries])
            for gender, entries in voices.items()
        },
    }


def resolve_voice(
    locale: str, tier: str | None, gender: str | None, voice_name: str | None, cfg
) -> tuple[str, str, str]:
    """Returns (voice_name, tier_actually_used, gender_actually_used).

    An unrecognised tier raises rather than falling back. The old fallback
    silently downgraded any unknown name to the cheapest tier, which meant a
    retired or forged tier name still placed a call - and billed at a rate
    nobody picked. Gender and specific voice still fall back, since those
    genuinely can go missing for a locale.
    """
    if tier not in (None, CASCADED_TIER):
        raise ValueError(f"unknown Voice Quality tier {tier!r}")
    voices = _cascaded_voices(locale, cfg)
    if not voices:
        raise ValueError(f"no TTS voices known for locale {locale!r}")

    chosen_gender = gender if gender in voices else next(iter(voices))
    entries = voices[chosen_gender]
    chosen_id = next((n for n, _ in entries if n == voice_name), entries[0][0])
    return _real_voice(chosen_id, cfg), CASCADED_TIER, chosen_gender


# ---- Gemini native TTS: same 30 voices as Chirp3-HD, but flat (no per-locale
# catalog - any voice can speak any of Gemini's ~90 supported languages, and
# there's no per-tier quality ladder like Standard/Neural2/Chirp3-HD -
# it's a single offering). Reuses the gender data this app already fetches
# live for Chirp3-HD (same underlying voice roster) instead of hand-typing a
# second copy that could drift out of sync.

def _gemini_voice_genders() -> dict[str, str]:
    genders: dict[str, str] = {}
    for by_tier in _catalog.values():
        chirp3 = by_tier.get("Chirp3-HD", {})
        for gender, voices in chirp3.items():
            for v in voices:
                name = v.rsplit("-", 1)[-1]  # "en-IN-Chirp3-HD-Achernar" -> "Achernar"
                genders[name] = gender
    return genders


def gemini_voice_tier() -> dict:
    """A single synthetic "tier" in the same shape tiers_for() returns, appended
    to every /tts-options response alongside the real quality ladder - picking
    it in the Voice Quality dropdown *is* how a caller opts into the Gemini Live
    pipeline for this call (see main.run_call), not a per-bot setting."""
    genders = _gemini_voice_genders()
    by_gender: dict[str, list[str]] = {}
    for name in _CHIRP3_CHARACTER:
        gender = genders.get(name, "female")  # only missing if no Chirp3-HD locale was ever cataloged
        by_gender.setdefault(gender, []).append(name)
    return {
        "name": "gemini_live",
        "quality": "Studio",
        "usd_per_mchar": None,
        "voices": {
            gender: _dedupe_by_label(
                sorted(
                    [{"name": name, "label": _CHIRP3_CHARACTER[name]} for name in names],
                    key=lambda v: v["name"],
                )
            )
            for gender, names in by_gender.items()
        },
    }


def resolve_gemini_voice(gender: str | None, voice_name: str | None = None) -> tuple[str, str]:
    """Returns (voice_name, gender_actually_used) - same fallback shape as
    resolve_voice(), just without a locale/tier axis."""
    tier = gemini_voice_tier()
    genders = tier["voices"]
    if not genders:
        return "Kore", "female"  # only if no Chirp3-HD locale was ever cataloged
    chosen_gender = gender if gender in genders else next(iter(genders))
    voices = genders[chosen_gender]
    chosen = next((v for v in voices if v["name"] == voice_name), voices[0])
    return chosen["name"], chosen_gender
