"""LLM abstraction over an OpenAI-compatible completion endpoint."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from openai import OpenAI

from backend.models import Message
from backend.config import Settings, get_settings


class LLM(Protocol):
    """Minimal contract for a chat completion backend."""

    def chat_completion(self, messages: Sequence[Message]) -> str: ...


class OpenAILLM:
    """OpenAI-compatible VLM adapter backed by the official SDK."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: OpenAI | None = None

    def chat_completion(self, messages: Sequence[Message]) -> str:
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Provide it via environment variable.")

        client = self._client
        if client is None:
            client = OpenAI(
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url,
            )
            self._client = client

        response = client.chat.completions.create(
            model=self._settings.model,
            messages=_langchain_messages_to_openai(messages),
            response_format={"type": "json_object"},
            temperature=self._settings.temperature
        )
        return response.choices[0].message.content or ""


def _langchain_messages_to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [
        {"role": message.role, "content": _normalize_content(message.content)}
        for message in messages
    ]


def _normalize_content(content: Any) -> Any:
    """Return content in the shape the OpenAI chat API expects."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        normalized = [_normalize_block(block) for block in content]
        return normalized if len(normalized) > 1 else normalized[0]
    return content


def _normalize_block(block: Any) -> Any:
    if isinstance(block, dict):
        if block.get("type") == "image_url":
            image_url = block.get("image_url")
            if isinstance(image_url, dict):
                return {"type": "image_url", "image_url": {"url": image_url.get("url")}}
            return {"type": "image_url", "image_url": str(image_url)}
        if block.get("type") == "text":
            return {"type": "text", "text": block.get("text", "")}
    return block
