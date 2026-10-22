"""LLM via the Gemini API, over plain REST + SSE (no google-genai SDK dependency)."""

import json
import os

import httpx

from .base import LLMProvider, LLMUsage

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-style {role, content} messages to Gemini's systemInstruction + contents."""
    system_instruction = None
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_instruction = m["content"]
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return system_instruction, contents


def _parse_sse_line(line: str) -> tuple[str | None, LLMUsage | None]:
    """Parse one 'data: {...}' line from Gemini's streamGenerateContent SSE stream."""
    if not line.startswith("data: "):
        return None, None
    chunk = json.loads(line[len("data: ") :])
    usage = None
    meta = chunk.get("usageMetadata")
    if meta:
        usage = LLMUsage(
            input_tokens=meta.get("promptTokenCount", 0) or 0,
            output_tokens=meta.get("candidatesTokenCount", 0) or 0,
        )
    text = "".join(
        part.get("text", "")
        for cand in chunk.get("candidates", [])
        for part in cand.get("content", {}).get("parts", [])
        if part.get("text")
    )
    return (text or None), usage


class GeminiLLM(LLMProvider):
    def __init__(self, model: str = "gemini-3.1-flash-lite"):
        # gemini-2.0-flash/-lite were shut down 2026-06-01. gemini-3.1-flash-lite
        # measured ~40-50% lower time-to-first-token than gemini-2.5-flash for
        # this voice-agent call shape (live-tested 2026-07) - current default.
        self._model = model
        self._api_key = os.environ["GEMINI_API_KEY"]

    async def stream_chat(self, messages, *, max_tokens, temperature):
        self.last_usage = None
        system_instruction, contents = _to_gemini_contents(messages)
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                # thinking is on by default for most Gemini models and eats into
                # maxOutputTokens with invisible reasoning tokens, adding latency
                # a voice call can't afford - a voice reply doesn't need deep
                # reasoning, so it's disabled outright rather than budgeted.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{_BASE_URL}/models/{self._model}:streamGenerateContent"
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", url, params={"alt": "sse", "key": self._api_key}, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    text, usage = _parse_sse_line(line)
                    if text:
                        yield text
                    if usage:
                        self.last_usage = usage
