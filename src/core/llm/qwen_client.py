"""Qwen LLM client — 阿里云百炼 Qwen2.5-7B-Instruct via OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from src.config import get_settings


class QwenLLMClient:
    """Async streaming client for 阿里云百炼 Qwen models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.bailian_api_key
        self.base_url = base_url or settings.bailian_base_url
        self.model = model or settings.bailian_model
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=2,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion, yielding one token string at a time."""
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            choices = chunk.choices
            if choices:
                delta = choices[0].delta
                if delta and delta.content:
                    yield delta.content

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming chat completion."""
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
