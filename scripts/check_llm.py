"""Stream a chat completion from the configured LLM and print TTFT/timings.

Usage: uv run python scripts/check_llm.py "Tell me a joke" [--provider groq|ollama|openai|gemini]
"""

import argparse
import asyncio
import time

from _bootstrap import build_or_die, load_config_or_die, need_env

from app.providers import build_llm
from app.providers.base import TextDelta, ToolCall


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="user message to send")
    parser.add_argument("--provider", help="override llm.provider from config.toml")
    args = parser.parse_args()

    cfg = load_config_or_die()
    if args.provider:
        cfg.llm.provider = args.provider
    if cfg.llm.provider == "groq":
        need_env("GROQ_API_KEY", "Groq LLM")
    elif cfg.llm.provider == "openai":
        need_env("OPENAI_API_KEY", "OpenAI LLM")
    elif cfg.llm.provider == "gemini":
        need_env("GEMINI_API_KEY", "Gemini LLM")

    llm = build_or_die(build_llm, cfg)
    messages = [
        {"role": "system", "content": cfg.persona.system_prompt},
        {"role": "user", "content": args.prompt},
    ]

    print(f"model: {cfg.llm.provider}:{cfg.llm.model}")
    t0 = time.perf_counter()
    ttft_ms = None
    n_chars = 0
    async for event in llm.stream_chat(
        messages, max_tokens=cfg.llm.max_tokens, temperature=cfg.llm.temperature
    ):
        if isinstance(event, ToolCall):
            print(f"\n[tool call] {event.name}({event.arguments})", flush=True)
            continue
        if not isinstance(event, TextDelta):
            continue  # Done
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000
        n_chars += len(event.text)
        print(event.text, end="", flush=True)
    total_ms = (time.perf_counter() - t0) * 1000

    print()
    print(f"ttft: {ttft_ms:.0f} ms | total: {total_ms:.0f} ms | {n_chars} chars")
    if llm.last_usage:
        print(f"usage: {llm.last_usage.input_tokens} in / {llm.last_usage.output_tokens} out tokens")


if __name__ == "__main__":
    asyncio.run(main())
