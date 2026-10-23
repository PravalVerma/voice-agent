"""LLM via a local Ollama server (e.g. llama3.2). No extra install needed -
Ollama is used over plain REST via httpx, which groq already pulls in."""

import json
import os
from typing import AsyncIterator, Sequence

import httpx

from .base import Done, LLMEvent, LLMProvider, LLMUsage, TextDelta, ToolCall


def _parse_line(line: str) -> tuple[str | None, list[dict], LLMUsage | None]:
    """Parse one NDJSON line from Ollama's /api/chat stream.

    Returns (content, tool_calls, usage). Ollama emits tool calls as complete
    objects rather than fragments, so unlike the OpenAI-shaped providers there
    is nothing to accumulate.
    """
    chunk = json.loads(line)
    message = chunk.get("message", {}) or {}
    content = message.get("content") or None
    tool_calls = message.get("tool_calls") or []
    usage = None
    if chunk.get("done"):
        usage = LLMUsage(
            input_tokens=chunk.get("prompt_eval_count", 0) or 0,
            output_tokens=chunk.get("eval_count", 0) or 0,
        )
    return content, tool_calls, usage


class OllamaLLM(LLMProvider):
    def __init__(self, model: str = "llama3.2"):
        self._model = model
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.request_count = 0

    async def stream_chat(
        self,
        messages,
        *,
        max_tokens,
        temperature,
        tools: Sequence[dict] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.last_usage = None
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = list(tools)

        emitted = 0
        self.request_count += 1
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    content, tool_calls, usage = _parse_line(line)
                    if content:
                        yield TextDelta(content)
                    for call in tool_calls:
                        fn = call.get("function", {}) or {}
                        name = fn.get("name")
                        if not name:
                            continue
                        args = fn.get("arguments")
                        emitted += 1
                        yield ToolCall(
                            # Ollama supplies no call id. One is synthesised
                            # here because the id exists only to pair the tool
                            # result back to its call, and both sides of that
                            # pairing are written by this process.
                            id=f"call_{emitted}",
                            name=name,
                            # arguments arrive already decoded; re-encode so
                            # ToolCall.arguments is a JSON string everywhere.
                            arguments=json.dumps(args or {}, ensure_ascii=False)
                            if not isinstance(args, str) else args,
                        )
                    if usage:
                        self.last_usage = usage
        yield Done(usage=self.last_usage)
