"""LLM via the OpenAI API (guarded import - `uv sync --extra openai`)."""

from openai import AsyncOpenAI

from .base import LLMProvider, LLMUsage


class OpenAILLM(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        self._client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
        self._model = model

    async def stream_chat(self, messages, *, max_tokens, temperature):
        self.last_usage = None
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                self.last_usage = LLMUsage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
