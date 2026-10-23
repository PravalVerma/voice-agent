"""LLM via Groq chat completions (default model: openai/gpt-oss-20b)."""

import logging
from typing import AsyncIterator, Sequence

from groq import APIError, AsyncGroq

from ._tool_stream import ToolCallAccumulator
from .base import Done, LLMEvent, LLMProvider, LLMUsage, TextDelta

log = logging.getLogger("agent")

# gpt-oss is trained with a "harmony" chat format that has a built-in tool-call
# channel, and it occasionally emits a tool-call token even when no tools were
# declared - which Groq's server then rejects, since none were permitted. It's
# a per-generation model quirk, not anything wrong with the request, so a fresh
# retry almost always succeeds.
#
# Declaring tools makes this error unreachable for that request, but the
# no-tools path is still live: the check scripts use it, and so does the second
# round of a tool-calling turn. So the retry stays.
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
        self.request_count = 0

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        tools: Sequence[dict] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.last_usage = None
        kwargs: dict = {}
        if "gpt-oss" in self._model:
            # gpt-oss is a reasoning model; keep thinking minimal for voice latency
            kwargs["reasoning_effort"] = "low"
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # True once anything with downstream consequences has been emitted:
            # spoken text, or a tool call that will be executed. Re-running the
            # request after either would speak a second reply over the first,
            # or run the tool twice.
            committed = False
            accumulator = ToolCallAccumulator()
            finish_reason = "stop"
            try:
                self.request_count += 1
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
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if not delta:
                        continue
                    if getattr(delta, "tool_calls", None):
                        accumulator.feed(delta.tool_calls)
                    if delta.content:
                        committed = True
                        yield TextDelta(delta.content)

                for call in accumulator.drain():
                    yield call
                yield Done(usage=self.last_usage, finish_reason=finish_reason)
                return
            except APIError as e:
                if committed or _TOOL_CHOICE_ERROR not in str(e) or attempt == _MAX_ATTEMPTS:
                    raise
                log.warning(
                    "groq: model emitted a tool call with none declared (attempt %d/%d) - retrying",
                    attempt, _MAX_ATTEMPTS,
                )
