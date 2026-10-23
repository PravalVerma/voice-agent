"""Configuration loading: config.toml for tunables, .env for secrets."""

import os
import tomllib
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"


class PersonaConfig(BaseModel):
    agent_name: str = "Asha"
    system_prompt: str = "You are a friendly, concise voice assistant on a phone call."


class VADConfig(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    end_threshold: float = Field(0.35, ge=0.0, le=1.0)
    silence_ms: int = Field(600, ge=100)
    min_speech_ms: int = Field(120, ge=0)
    pre_roll_ms: int = Field(300, ge=0)
    max_utterance_s: int = Field(20, ge=1)
    barge_in_min_speech_ms: int = Field(200, ge=0)
    barge_in_threshold: float = Field(0.6, ge=0.0, le=1.0)


class STTConfig(BaseModel):
    provider: str = "groq"
    model: str = "whisper-large-v3-turbo"
    language: str = "en"


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    max_tokens: int = Field(200, ge=1)
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    #: exchanges (caller turn + reply), not messages - see trim_history
    history_max_turns: int = Field(4, ge=1)
    #: hard ceiling on LLM requests per call. A runaway tool loop or a retry
    #: storm would otherwise spend the daily cap on one call; past this the
    #: agent speaks a fixed line instead of asking the model again.
    max_requests_per_call: int = Field(10, ge=1)
    #: log a warning above this many prompt tokens in one request
    warn_prompt_tokens: int = Field(1000, ge=1)


class EdgeVoicePair(BaseModel):
    """The voice to use for each gender in one language."""
    female: str
    male: str


class EdgeConfig(BaseModel):
    """Settings for the edge provider. Kept provider-shaped rather than folded
    into [tts] so a paid replacement can add its own block beside this one and
    be selected by `provider` alone."""
    #: per-language gender -> voice, keyed by language code ("en", "hi")
    voices: dict[str, EdgeVoicePair] = {}
    #: give up on one request after this long and speak the fallback line
    timeout_s: float = Field(10.0, gt=0)
    #: consecutive failures before synthesis is paused rather than retried
    failure_threshold: int = Field(3, ge=1)
    #: how long that pause lasts before one request is let through to re-test
    cooldown_s: float = Field(60.0, gt=0)


class TTSConfig(BaseModel):
    provider: str = "google"
    voice: str = "en-IN-Neural2-A"
    #: the Voice Quality tier this call resolved to, set by apply_language().
    #: Carried here so CallLogger can label the call without re-deriving the
    #: tier by pattern-matching the voice name, which only ever worked for one
    #: provider's naming scheme.
    tier: str = "cascaded"
    speaking_rate: float = Field(1.0, gt=0.25, lt=4.0)
    sample_rate: int = 24000
    edge: EdgeConfig = EdgeConfig()

    @property
    def language_code(self) -> str:
        # Google voice names start with the language code: "en-IN-Standard-A" -> "en-IN"
        return "-".join(self.voice.split("-")[:2])


class LanguageConfig(BaseModel):
    code: str
    label: str
    stt_language: str
    enabled: bool = True

    @property
    def locale(self) -> str:
        # Google TTS locale for this language, e.g. "hi" -> "hi-IN". Every
        # language this app supports is an Indian-region variant.
        return f"{self.code}-IN"


class CallConfig(BaseModel):
    max_call_minutes: int = Field(10, ge=1)
    log_dir: str = "logs/calls"
    dump_utterances: bool = False
    record_calls: bool = True


class ComplianceConfig(BaseModel):
    """Collections-mode enforcement. Read by app/compliance.py; ignored entirely
    for ordinary assistant bots.

    Patterns are compiled here, at config load, so a broken one fails at startup
    with a clear line instead of raising inside a live call."""

    call_window_start: str = "09:00"
    call_window_end: str = "19:00"
    allowed_weekdays: list[int] = [0, 1, 2, 3, 4, 5]
    holidays: list[str] = []
    max_attempts_per_day: int = Field(1, ge=1)
    max_attempts_per_week: int = Field(3, ge=1)
    max_promise_days: int = Field(30, ge=1)
    settlement_floor_pct: int = Field(75, ge=1, le=100)
    amount_tolerance_paise: int = Field(100, ge=0)
    recovery_line: dict[str, str] = {}
    threats: dict = {}
    inbound: dict[str, list[str]] = {}
    numerals: dict = {}

    @field_validator("call_window_start", "call_window_end")
    @classmethod
    def _valid_time(cls, v: str) -> str:
        try:
            hh, mm = v.split(":")
            if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError(f"must be HH:MM, got {v!r}")
        return v

    @field_validator("allowed_weekdays")
    @classmethod
    def _valid_weekdays(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("at least one weekday must be allowed, or no call can ever be placed")
        if any(d < 0 or d > 6 for d in v):
            raise ValueError("weekdays are 0 (Monday) to 6 (Sunday)")
        return v

    @field_validator("threats", "inbound")
    @classmethod
    def _compilable(cls, v, info):
        """Every regex must compile AND be free of control characters.

        The control-character check is not paranoia. TOML basic strings (double
        quotes) process escapes, so a pattern written "\b(stop calling)\b"
        decodes to a backspace rather than a word boundary - it compiles
        perfectly and then matches nothing, silently disabling that screen.
        Patterns must use TOML literal strings (single quotes, or triple single
        quotes when the pattern contains an apostrophe)."""
        import re as _re

        def check(patterns, where):
            for pat in patterns:
                if any(ch in pat for ch in ""):
                    raise ValueError(
                        f"{where}: pattern contains a control character - it was written with "
                        f"double quotes, which makes TOML read \b as a backspace. Use single "
                        f"quotes (or  if it contains an apostrophe)."
                    )
                try:
                    _re.compile(pat)
                except _re.error as e:
                    raise ValueError(f"{where}: invalid regex {pat!r} - {e}")

        for key, val in v.items():
            if isinstance(val, list) and val and isinstance(val[0], str):
                if key.startswith("hedge_markers"):
                    continue  # plain words, not patterns
                check(val, f"{info.field_name}.{key}")
        return v


class CollectionsConfig(BaseModel):
    """Scripted-stage settings. Scripts are per language, keyed by stage."""

    verify_strikes: int = Field(3, ge=1)
    max_tokens: int = Field(700, ge=1)
    scripts: dict[str, dict[str, str]] = {}

    def script(self, language: str, key: str, **fields) -> str:
        """One script line, falling back to English when a language lacks it.

        Falling back is deliberate: a missing translation should degrade to a
        line in the wrong language, not to silence at the exact moment the
        regulatory disclosure is due.
        """
        table = self.scripts.get(language) or self.scripts.get("en") or {}
        text = table.get(key) or (self.scripts.get("en") or {}).get(key) or ""
        for name, value in fields.items():
            text = text.replace("{" + name + "}", str(value))
        return text


class LLMPostConfig(BaseModel):
    """Post-call jobs. See the [llm_post] comment in config.toml for why the
    audit is mostly deterministic rather than batched through a model."""

    enabled: bool = True
    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    in_process: bool = True
    poll_seconds: int = Field(60, ge=5)
    settle_seconds: int = Field(30, ge=0)
    max_attempts: int = Field(3, ge=1)
    rpm: int = Field(3, ge=1)
    tpm: int = Field(2000, ge=1)
    batch_token_budget: int = Field(1800, ge=1)


class RetentionConfig(BaseModel):
    enabled: bool = False
    recording_keep_days: int = Field(180, ge=1)
    record_keep_days: int = Field(2555, ge=1)


class CostConfig(BaseModel):
    usd_to_inr: float = 88.0
    stt_usd_per_hour: float = 0.04
    llm_usd_per_mtok_in: float = 0.075
    llm_usd_per_mtok_out: float = 0.30
    tts_usd_per_mchar: float = 16.0
    # Gemini pipeline rates - real, published rates for gemini-2.5-flash /
    # gemini-2.5-flash-preview-tts (verified against ai.google.dev/gemini-api/docs/pricing,
    # 2026-07). Gemini bills audio as tokens (even TTS output), not per-hour/
    # per-character like the rates above, so a gemini-pipeline call is costed
    # with these instead - see CallLogger.estimate_cost_usd().
    gemini_llm_usd_per_mtok_in: float = 0.30
    gemini_llm_usd_per_mtok_out: float = 2.50
    gemini_stt_usd_per_mtok_audio_in: float = 1.00  # audio input tokens (32 tok/s of audio)
    gemini_tts_usd_per_mtok_in: float = 0.50  # the text being spoken
    gemini_tts_usd_per_mtok_audio_out: float = 10.00  # generated speech (25 tok/s of audio)
    # Gemini Live pipeline rates (ai.google.dev/gemini-api/docs/pricing, verified
    # 2026-07) - a live session has no separate STT/LLM/TTS legs to cost, just
    # continuous audio in and out, so it's billed on these two rates alone (see
    # CallLogger.estimate_cost_usd()).
    gemini_live_usd_per_mtok_audio_in: float = 3.00
    gemini_live_usd_per_mtok_audio_out: float = 12.00


class AppConfig(BaseModel):
    persona: PersonaConfig = PersonaConfig()
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    call: CallConfig = CallConfig()
    cost: CostConfig = CostConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    collections: CollectionsConfig = CollectionsConfig()
    llm_post: LLMPostConfig = LLMPostConfig()
    retention: RetentionConfig = RetentionConfig()
    default_language: str = "en"
    languages: list[LanguageConfig] = []


class ConfigError(RuntimeError):
    pass


def _required_env(cfg: AppConfig, pipeline: str = "standard") -> dict[str, str]:
    """Map of env var -> reason, for the providers selected in config.toml.

    "gemini_live" bypasses that entirely - config.apply_language() leaves
    cfg.stt/llm/tts untouched for that pipeline (no provider objects are built
    for it - see app.live_session), so the requirement can't be read off them
    the way it is for every other pipeline; only GEMINI_API_KEY applies."""
    if pipeline == "gemini_live":
        return {"GEMINI_API_KEY": "pipeline 'gemini_live'"}
    required: dict[str, str] = {}
    if cfg.stt.provider == "groq" or cfg.llm.provider == "groq":
        required["GROQ_API_KEY"] = "stt/llm provider 'groq'"
    if cfg.tts.provider == "google":
        required["GOOGLE_APPLICATION_CREDENTIALS"] = "tts provider 'google'"
    if cfg.llm.provider == "openai":
        required["OPENAI_API_KEY"] = "llm provider 'openai'"
    if cfg.llm.provider == "gemini" or cfg.stt.provider == "gemini" or cfg.tts.provider == "gemini":
        required["GEMINI_API_KEY"] = "stt/llm/tts provider 'gemini'"
    return required


def missing_secrets(cfg: AppConfig, pipeline: str = "standard") -> list[str]:
    """Human-readable list of missing env vars for the selected providers."""
    problems = []
    for var, reason in _required_env(cfg, pipeline).items():
        value = os.environ.get(var, "").strip()
        if not value:
            problems.append(f"{var} (needed for {reason}) is not set - add it to .env")
        elif var == "GOOGLE_APPLICATION_CREDENTIALS" and not Path(value).is_file():
            problems.append(f"{var} points to '{value}' which does not exist")
    return problems


def require_secrets(cfg: AppConfig) -> None:
    problems = missing_secrets(cfg)
    if problems:
        raise ConfigError(
            "Missing credentials:\n  - " + "\n  - ".join(problems)
            + "\n(copy .env.example to .env and fill in the values)"
        )


# Per-language instruction override, when a language needs something other than
# the generic "reply only in {label}" rule below. English has one to stop it
# code-switching into Hindi on its own; Hindi has one for script (see below).
# An earlier attempt at a *language* override for Hindi - telling it to speak
# natural code-switched Hinglish - was reverted because gpt-oss-20b kept
# drifting back to pure English; the rule here is about script only, and leaves
# the Hinglish wording to the bot prompt.
#
# Script matters as much as language here: Google's voices read roman-letter
# Hindi with English pronunciation rules, so "main" comes out as the English
# word "mayn" rather than "मैं" (measured via TTS -> STT round-trip on every
# tier). A bot prompt written in romanized Hinglish makes the model copy that
# script verbatim into its scripted lines, so the script rule lives here in
# code, where it can override whatever script the admin-edited prompt uses.
_LANGUAGE_INSTRUCTIONS = {
    "en": (
        "Always reply in pure English only, even if the caller speaks Hindi or "
        "Hinglish to you - do not mix in any Hindi words, never switch languages."
    ),
    "hi": (
        "Always reply only in Hindi, even if you can't understand what the caller said - "
        "never switch languages. SCRIPT RULE, overriding every example above: every Hindi "
        "word must be written in Devanagari, never in roman letters. The example lines "
        "above are written in roman Hindi - re-spell them in Devanagari before saying "
        "them: 'main Arya bol raha hoon' -> 'मैं Arya बोल रहा हूँ'; 'Bas aapko batana tha' "
        "-> 'बस आपको बताना था'; 'do lakh painsath hazaar rupaye' -> 'दो लाख पैंसठ हज़ार "
        "रुपये'. The genuinely English words inside those lines stay English in Latin "
        "letters - account, settlement, credit score, EMI, legal, quality purposes, "
        "personal loan, and every personal and company name - never translated into Hindi "
        "and never respelled in Devanagari: write 'quality purposes के लिए', not "
        "'गुणवत्ता उद्देश्यों के लिए' and not 'क्वालिटी पर्पज़ेस के लिए'. If you are about "
        "to write a Hindi word in roman letters, stop and write it in Devanagari instead - "
        "the voice system reads roman Hindi with English pronunciation, so 'main' comes "
        "out as 'mayn' rather than मैं."
    ),
}
_DEFAULT_LANGUAGE_INSTRUCTION = (
    "Always reply only in {label}, even if you can't understand what the caller said - "
    "never switch languages. Write {label} words in that language's own script, never "
    "transliterated into roman letters - the voice system mispronounces romanized text. "
    "Genuinely English terms stay in Latin script."
)

_LANGUAGE_SUFFIX_TAIL = (
    " Never use markdown, bullet points, numbered lists, or emoji; everything you "
    "say is read aloud. The voice speaking for you on this call is {gender} - if "
    "your instructions above give you different names or descriptions for "
    "different voice genders, use the one matching {gender} for this call. "
    "When an amount has decimal rupees (paisa), say it as '<rupees> rupees and "
    "<paisa> paisa' - e.g. two lakh sixty-five thousand rupees and ten paisa - "
    "never as '<rupees> point <paisa> rupees'."
)

# Per-call template tokens a base prompt can optionally reference (e.g. a
# collections script naming a specific customer/company). A prompt that doesn't
# use either is unaffected - substitution is a plain string replace, not a
# format() call, so stray "{"/"}" in an unrelated prompt can't break it.
_CALL_VAR_KEYS = ("customer_name", "company_name")


def build_call_vars(customer_name: str | None = None, company_name: str | None = None) -> dict:
    """Turn the required per-call fields from the Voice Assistant page's "Call
    details" section into the {{token}} values apply_language() substitutes into
    the base prompt."""
    return {
        "customer_name": (customer_name or "").strip(),
        "company_name": (company_name or "").strip(),
    }


def _apply_call_vars(text: str, call_vars: dict) -> str:
    for key in _CALL_VAR_KEYS:
        text = text.replace("{{" + key + "}}", str(call_vars.get(key) or ""))
    return text


def apply_language(
    cfg: AppConfig,
    code: str | None,
    gender: str | None = None,
    tier: str | None = None,
    call_vars: dict | None = None,
    bot_id: int | None = None,
    voice_name: str | None = None,
) -> AppConfig:
    """Return a per-call variant of cfg with STT/LLM/TTS/persona swapped to the
    given language, voice gender, and TTS tier (e.g. "Neural2"). Only considers
    enabled languages; falls back to default_language, then the first enabled
    entry, then leaves cfg untouched if none are enabled. The system prompt is
    the admin-edited prompt for the selected bot+language (bots_store), with any
    {{customer_name}}-style call_vars tokens substituted in, plus a code-generated
    language-lock/formatting instruction.

    Every bot supports every pipeline - which one runs is purely a per-call
    choice, picked via the Voice Quality dropdown's synthetic "gemini_live"
    tier (tts_catalog.gemini_voice_tier()), not a bot property. When `tier` is
    "gemini_live", no provider objects are built at all (app.live_session.
    run_live_call talks to the Gemini Live API directly instead) - only
    persona.system_prompt (and stt.language, for CallLogger's display label)
    are updated here; the caller resolves the actual voice name itself via
    tts_catalog.resolve_gemini_voice()."""
    # deferred imports: db.py imports config.ROOT, so importing these at module
    # level here would be circular - safe once config.py has finished loading.
    from . import bots_store, tts_catalog

    enabled = [l for l in cfg.languages if l.enabled]
    if not enabled:
        return cfg
    lang = next((l for l in enabled if l.code == code), None)
    if lang is None:
        lang = next((l for l in enabled if l.code == cfg.default_language), enabled[0])

    base_prompt = _apply_call_vars(bots_store.resolve_prompt_for_call(bot_id, lang.code), call_vars or {})
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(
        lang.code, _DEFAULT_LANGUAGE_INSTRUCTION.format(label=lang.label)
    )

    if tier == "gemini_live":
        _, resolved_gender = tts_catalog.resolve_gemini_voice(gender, voice_name)
        system_prompt = (
            base_prompt + "\n" + lang_instruction + _LANGUAGE_SUFFIX_TAIL.format(gender=resolved_gender)
        )
        return cfg.model_copy(
            update={
                "stt": cfg.stt.model_copy(update={"language": lang.stt_language}),
                "persona": cfg.persona.model_copy(update={"system_prompt": system_prompt}),
            }
        )

    voice, resolved_tier, resolved_gender = tts_catalog.resolve_voice(
        lang.locale, tier, gender, voice_name, cfg
    )
    system_prompt = (
        base_prompt + "\n" + lang_instruction + _LANGUAGE_SUFFIX_TAIL.format(gender=resolved_gender)
    )
    return cfg.model_copy(
        update={
            "stt": cfg.stt.model_copy(update={"language": lang.stt_language}),
            "tts": cfg.tts.model_copy(update={"voice": voice, "tier": resolved_tier}),
            "persona": cfg.persona.model_copy(update={"system_prompt": system_prompt}),
            "cost": cfg.cost.model_copy(
                update={"tts_usd_per_mchar": tts_catalog.rate_for(cfg, voice)}
            ),
        }
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    load_dotenv(ROOT / ".env")
    if not CONFIG_PATH.is_file():
        raise ConfigError(f"config.toml not found at {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"config.toml is not valid TOML: {e}") from e
    try:
        return AppConfig(**data)
    except ValidationError as e:
        lines = [
            f"  - [{'.'.join(str(p) for p in err['loc'])}]: {err['msg']}"
            for err in e.errors()
        ]
        raise ConfigError("config.toml has invalid values:\n" + "\n".join(lines)) from e
