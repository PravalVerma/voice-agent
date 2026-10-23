"""LLM via the Gemini API, over plain REST + SSE (no google-genai SDK dependency)."""

import json
import os
from typing import AsyncIterator, Sequence

import httpx

from .base import Done, LLMEvent, LLMProvider, LLMUsage, TextDelta, ToolCall

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

#: Gemini rejects a function declaration carrying either of these.
_SCHEMA_STRIP = ("$schema", "additionalProperties")


def _clean_schema(schema: dict) -> dict:
    out = {k: v for k, v in (schema or {}).items() if k not in _SCHEMA_STRIP}
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: _clean_schema(v) if isinstance(v, dict) else v
                             for k, v in props.items()}
    return out


def _to_function_declarations(tools: Sequence[dict]) -> list[dict]:
    """Unwrap OpenAI-shaped tool definitions into Gemini's flatter form."""
    decls = []
    for tool in tools:
        fn = tool.get("function", tool)
        decls.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": _clean_schema(fn.get("parameters") or {"type": "object", "properties": {}}),
        })
    return decls


def _to_gemini_contents(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-style {role, content} messages to Gemini's systemInstruction + contents.

    Only the LAST system message becomes systemInstruction, which is why the
    cascaded pipeline keeps its fixed prefix and its per-call block in one
    system message plus a user message, rather than two system messages - the
    first of two would be silently discarded here.
    """
    system_instruction = None
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""  # may be None on a tool-call message
        if role == "system":
            system_instruction = content
        elif role == "tool":
            # Gemini carries tool output as a user-role functionResponse part.
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": m.get("name") or m.get("tool_call_id") or "tool",
                "response": {"result": content},
            }}]})
        elif role == "assistant" and m.get("tool_calls"):
            parts = []
            if content:
                parts.append({"text": content})
            for call in m["tool_calls"]:
                fn = call.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": content}],
            })
    return system_instruction, contents


def _parse_sse_line(line: str) -> tuple[list[LLMEvent], LLMUsage | None]:
    """Parse one 'data: {...}' line from Gemini's streamGenerateContent SSE stream."""
    if not line.startswith("data: "):
        return [], None
    chunk = json.loads(line[len("data: ") :])
    usage = None
    meta = chunk.get("usageMetadata")
    if meta:
        usage = LLMUsage(
            input_tokens=meta.get("promptTokenCount", 0) or 0,
            output_tokens=meta.get("candidatesTokenCount", 0) or 0,
        )

    events: list[LLMEvent] = []
    text_parts = []
    for cand in chunk.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if part.get("text"):
                text_parts.append(part["text"])
            call = part.get("functionCall")
            if call:
                events.append(ToolCall(
                    id=call.get("id") or f"call_{call.get('name', 'fn')}",
                    name=call.get("name", ""),
                    arguments=json.dumps(call.get("args") or {}, ensure_ascii=False),
                ))
    if text_parts:
        events.insert(0, TextDelta("".join(text_parts)))
    return events, usage


class GeminiLLM(LLMProvider):
    def __init__(self, model: str = "gemini-3.1-flash-lite"):
        # gemini-2.0-flash/-lite were shut down 2026-06-01. gemini-3.1-flash-lite
        # measured ~40-50% lower time-to-first-token than gemini-2.5-flash for
        # this voice-agent call shape (live-tested 2026-07) - current default.
        self._model = model
        self._api_key = os.environ["GEMINI_API_KEY"]
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
        if tools:
            payload["tools"] = [{"functionDeclarations": _to_function_declarations(tools)}]

        url = f"{_BASE_URL}/models/{self._model}:streamGenerateContent"
        self.request_count += 1
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", url, params={"alt": "sse", "key": self._api_key}, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    events, usage = _parse_sse_line(line)
                    for event in events:
                        yield event
                    if usage:
                        self.last_usage = usage
        yield Done(usage=self.last_usage)
