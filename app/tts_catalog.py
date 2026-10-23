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

# Caller-facing quality tag shown in the UI - callers never see the
# underlying tier name ("Neural2", "Chirp3-HD") or which vendor/model it maps
# to, only this label.
QUALITY_LABEL: dict[str, str] = {
    "Standard": "Core",
    "Neural2": "Enhanced",
    "Chirp3-HD": "Premium",
}

_TIER_ORDER = ["Standard", "Neural2", "Chirp3-HD"]  # preferred default order


def all_tier_names() -> list[str]:
    """Every Voice Quality tier name the app can ever offer, real ones plus the
    synthetic "gemini_live" one - the universe main.py validates a per-
    assignment allow-list against and the Manage Bots modal renders as
    checkboxes. Locale-independent (TIER_RATES/QUALITY_LABEL aren't per-locale),
    so this doesn't need a populated catalog."""
    return [*_TIER_ORDER, "gemini_live"]

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
    """Fetches the catalog at startup. A failure here is not fatal: without
    Google credentials there are simply no Standard/Neural2/Chirp3-HD voices to
    offer, and the Voice Quality dropdown falls back to the Studio tier alone
    (which runs on GEMINI_API_KEY and never touches this catalog). Letting the
    exception escape would take the whole server down at boot over a credential
    the standard pipeline needs but the Studio one does not."""
    global _catalog
    try:
        _catalog = build_catalog(language_codes)
    except Exception as e:
        _catalog = {}
        log.warning(
            "tts catalog unavailable (%s) - Core/Enhanced/Premium will not be "
            "offered; the Studio tier is unaffected", e
        )
        return
    for locale, tiers in _catalog.items():
        log.info("tts catalog: %s -> %s", locale, sorted(tiers.keys()))


def get_catalog() -> dict[str, dict[str, dict[str, str]]]:
    return _catalog


def tiers_for(locale: str) -> list[dict]:
    """Tier options for the frontend dropdown, in a sensible default order. Each
    tier lists every specific voice available per gender, so the UI can offer a
    (tier -> gender -> specific voice) drill-down instead of picking a gender's
    single hardcoded voice."""
    by_tier = _catalog.get(locale, {})
    ordered = [t for t in _TIER_ORDER if t in by_tier] + [t for t in by_tier if t not in _TIER_ORDER]
    return [
        {
            "name": tier,
            "quality": QUALITY_LABEL.get(tier, tier),
            "usd_per_mchar": TIER_RATES.get(tier),
            "voices": {
                gender: _dedupe_by_label(
                    [{"name": v, "label": voice_character(locale, tier, v)} for v in voices]
                )
                for gender, voices in by_tier[tier].items()
            },
        }
        for tier in ordered
    ]


def resolve_voice(
    locale: str, tier: str | None, gender: str | None, voice_name: str | None = None
) -> tuple[str, str, str]:
    """Returns (voice_name, tier_actually_used, gender_actually_used). Falls back
    to the first available tier/gender/voice if the requested combination doesn't
    exist - the caller should use the returned gender (not the requested one)
    when telling the LLM which voice it has, since they can differ on fallback."""
    by_tier = _catalog.get(locale, {})
    if not by_tier:
        raise ValueError(f"no TTS voices known for locale {locale!r}")

    chosen_tier = tier if tier in by_tier else next(
        (t for t in _TIER_ORDER if t in by_tier), next(iter(by_tier))
    )
    genders = by_tier[chosen_tier]
    chosen_gender = gender if gender in genders else next(iter(genders))
    voices = genders[chosen_gender]
    chosen_voice = voice_name if voice_name in voices else voices[0]
    return chosen_voice, chosen_tier, chosen_gender


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
