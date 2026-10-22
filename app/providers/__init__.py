"""Config-driven provider factories."""

from ..config import AppConfig, ConfigError
from .base import LLMProvider, STTProvider, TTSProvider


def build_stt(cfg: AppConfig) -> STTProvider:
    p = cfg.stt.provider
    if p == "groq":
        from .stt_groq import GroqSTT

        return GroqSTT(model=cfg.stt.model, language=cfg.stt.language)
    if p == "local":
        try:
            from .stt_local import LocalWhisperSTT
        except ModuleNotFoundError as e:
            raise ConfigError(
                "stt.provider 'local' needs faster-whisper: uv sync --extra local-stt"
            ) from e
        return LocalWhisperSTT(model=cfg.stt.model, language=cfg.stt.language)
    if p == "gemini":
        from .stt_gemini import GeminiSTT

        return GeminiSTT(model=cfg.stt.model, language=cfg.stt.language)
    raise ConfigError(f"unknown stt.provider '{p}' (valid: groq, local, gemini)")


def build_llm(cfg: AppConfig) -> LLMProvider:
    p = cfg.llm.provider
    if p == "groq":
        from .llm_groq import GroqLLM

        return GroqLLM(model=cfg.llm.model)
    if p == "ollama":
        from .llm_ollama import OllamaLLM

        return OllamaLLM(model=cfg.llm.model)
    if p == "openai":
        try:
            from .llm_openai import OpenAILLM
        except ModuleNotFoundError as e:
            raise ConfigError(
                "llm.provider 'openai' needs the openai SDK: uv sync --extra openai"
            ) from e
        return OpenAILLM(model=cfg.llm.model)
    if p == "gemini":
        from .llm_gemini import GeminiLLM

        return GeminiLLM(model=cfg.llm.model)
    raise ConfigError(f"unknown llm.provider '{p}' (valid: groq, ollama, openai, gemini)")


def build_tts(cfg: AppConfig) -> TTSProvider:
    p = cfg.tts.provider
    if p == "google":
        from .tts_google import GoogleTTS

        return GoogleTTS(
            voice=cfg.tts.voice,
            language_code=cfg.tts.language_code,
            speaking_rate=cfg.tts.speaking_rate,
            sample_rate=cfg.tts.sample_rate,
        )
    if p == "gemini":
        from .tts_gemini import GeminiTTS

        # cfg.tts.voice is a bare Gemini voice name here (e.g. "Kore"), not a
        # locale-prefixed Google Cloud TTS name - apply_language() sets it
        # accordingly for a gemini-pipeline bot.
        return GeminiTTS(voice_name=cfg.tts.voice)
    raise ConfigError(f"unknown tts.provider '{p}' (valid: google, gemini)")
