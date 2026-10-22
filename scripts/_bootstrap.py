"""Shared setup for the check scripts: import path, config, credential checks."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

from app.config import AppConfig, ConfigError, get_config  # noqa: E402


def load_config_or_die() -> AppConfig:
    try:
        return get_config()
    except ConfigError as e:
        sys.exit(f"config error: {e}")


def need_env(var: str, why: str) -> None:
    value = os.environ.get(var, "").strip()
    if not value:
        sys.exit(f"error: {var} is not set (needed for {why}) - add it to .env")
    if var == "GOOGLE_APPLICATION_CREDENTIALS" and not Path(value).is_file():
        sys.exit(f"error: {var} points to '{value}' which does not exist")


def build_or_die(builder, cfg: AppConfig):
    """Run a provider factory (build_stt/build_llm/build_tts) and exit cleanly
    on ConfigError (unknown provider name, missing optional dependency, ...)."""
    try:
        return builder(cfg)
    except ConfigError as e:
        sys.exit(f"error: {e}")
