"""LLM via the OpenAI API (needs `uv sync --extra openai`)."""

from typing import AsyncIterator, Sequence

from openai import AsyncOpenAI

from ._tool_stream import ToolCallAccumulator
from .base import Done, LLMEvent, LLMProvider, LLMUsage, TextDelta


class OpenAILLM(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        self._client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
        self._model = model
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
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"

        self.request_count += 1
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        accumulator = ToolCallAccumulator()
        finish_reason = "stop"
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                self.last_usage = LLMUsage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
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
                yield TextDelta(delta.content)

        for call in accumulator.drain():
            yield call
        yield Done(usage=self.last_usage, finish_reason=finish_reason)
