"""LLM via a local Ollama server (e.g. llama3.2). No extra install needed -
Ollama is used over plain REST via httpx, which groq already pulls in."""

import json
import os

import httpx

from .base import LLMProvider, LLMUsage


def _parse_line(line: str) -> tuple[str | None, LLMUsage | None]:
    """Parse one NDJSON line from Ollama's /api/chat stream."""
    chunk = json.loads(line)
    content = chunk.get("message", {}).get("content") or None
    usage = None
    if chunk.get("done"):
        usage = LLMUsage(
            input_tokens=chunk.get("prompt_eval_count", 0) or 0,
            output_tokens=chunk.get("eval_count", 0) or 0,
        )
    return content, usage


class OllamaLLM(LLMProvider):
    def __init__(self, model: str = "llama3.2"):
        self._model = model
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    async def stream_chat(self, messages, *, max_tokens, temperature):
        self.last_usage = None
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    content, usage = _parse_line(line)
                    if content:
                        yield content
                    if usage:
                        self.last_usage = usage
