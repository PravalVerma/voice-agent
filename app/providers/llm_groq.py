"""LLM via Groq chat completions (default model: openai/gpt-oss-20b)."""

import logging
from typing import AsyncIterator

from groq import APIError, AsyncGroq

from .base import LLMProvider, LLMUsage

log = logging.getLogger("agent")

# gpt-oss is trained with a "harmony" chat format that has a built-in tool-call
# channel - even though this app never declares any `tools`, the model
# occasionally hallucinates a tool-call token anyway, which Groq's server then
# rejects since none were permitted. It's a per-generation model quirk, not
# anything wrong with the request, so a fresh retry almost always succeeds.
_TOOL_CHOICE_ERROR = "Tool choice is none, but model called a tool"
_MAX_ATTEMPTS = 3


class GroqLLM(LLMProvider):
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        # max_retries=0: the SDK's default (2) honours Groq's Retry-After on a
        # 429, which on the free tier means sleeping 8-16s inside the request.
        # On a live call that's dead air the caller reads as a dropped line -
        # measured up to 13.7s time-to-first-token, after which they start
        # talking again and barge-in cancels the turn anyway. Failing fast
        # instead lets CallSession speak its apology within ~1s. Retrying the
        # gpt-oss tool-call quirk is handled below, where it's cheap.
        self._client = AsyncGroq(max_retries=0)  # reads GROQ_API_KEY from the environment
        self._model = model

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        self.last_usage = None
        kwargs: dict = {}
        if "gpt-oss" in self._model:
            # gpt-oss is a reasoning model; keep thinking minimal for voice latency
            kwargs["reasoning_effort"] = "low"

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            yielded_any = False
            try:
                stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    **kwargs,
                )
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        details = getattr(chunk.usage, "prompt_tokens_details", None)
                        self.last_usage = LLMUsage(
                            input_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
                        )
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yielded_any = True
                            yield delta.content
                return
            except APIError as e:
                # only safe to retry on a clean slate - if part of a reply
                # already went out (and got queued for TTS), re-running the
                # request would speak a second, overlapping reply on top
                if yielded_any or _TOOL_CHOICE_ERROR not in str(e) or attempt == _MAX_ATTEMPTS:
                    raise
                log.warning(
                    "groq: model hallucinated a tool call (attempt %d/%d) - retrying",
                    attempt, _MAX_ATTEMPTS,
                )
