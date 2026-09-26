"""High-level document extraction orchestration."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any

from PIL import Image

from backend.config import Settings, get_settings
from backend.models import FieldSchema, Schema, Message
from backend.prompts import get_prompt
from backend.services.document import Document
from backend.services.extract_response import (
    ExtractResponseError,
    ParsedContent,
    parse_extracted,
)
from backend.services.llm import LLM
from backend.services.pdf_image_conversion_service import PdfToImageConversionService

logger = logging.getLogger("app")


class DocumentParserService:
    """Converts a PDF to images, builds the VLM prompt, and parses the response."""

    def __init__(
        self,
        llm: LLM,
        pdf_to_image_conversion_service: PdfToImageConversionService,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.pdf_to_image_conversion_service = pdf_to_image_conversion_service
        self._settings = settings or get_settings()

    async def parse(self, doc: Document) -> ParsedContent:
        images = self.pdf_to_image_conversion_service.convert(doc)
        messages = self._build_prompt(doc.schema, images)
        return await self._extract_with_retry(doc.schema, messages)

    async def _extract_with_retry(self, schema: Schema, messages: list[Message]) -> ParsedContent:
        max_attempts = max(1, self._settings.llm_retries + 1)
        last_error: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                raw_content = self.llm.chat_completion(messages)
                return parse_extracted(raw_content, schema)
            except ExtractResponseError as exc:
                logger.info("[parser] Extract: response could not be parsed; attempt %d/%d: %s",
                            attempt, max_attempts, exc)
                last_error = exc
            except Exception as exc:
                logger.info("[parser] Extract: LLM call failed; attempt %d/%d: %s",
                            attempt, max_attempts, exc)
                last_error = exc
            if attempt < max_attempts and self._settings.llm_retry_backoff > 0:
                await asyncio.sleep(self._settings.llm_retry_backoff)
        assert last_error is not None
        raise last_error

    def _build_prompt(self, schema: Schema, images: list[Image.Image]) -> list[Message]:
        system_message = Message(role="developer", content=self._build_system_text(schema))
        user_message = Message(role="user", content=self._build_user_content(schema, images))
        return [system_message, user_message]

    def _build_system_text(self, schema: Schema) -> str:
        return (
            get_prompt("parser_system_prompt")
            + "\n\nThe output JSON must contain exactly the following fields:"
            + _format_fields(schema.fields)
        )

    def _build_user_content(
        self, schema: Schema, images: list[Image.Image]
    ) -> list[dict[str, Any]]:
        keys = ", ".join(field.name for field in schema.fields)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Extract the requested data from the attached document image(s) "
                    "and return JSON only, with no commentary.\n"
                    f"Keys to return: {keys}."
                ),
            }
        ]
        for image in images:
            data_url = f"data:image/jpeg;base64,{_image_to_base64jpeg(image, self._settings.image_quality)}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        return content


def _image_to_base64jpeg(image: Image.Image, quality: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _format_fields(fields: list[FieldSchema]) -> str:
    lines = [""]
    for field in fields:
        lines.append(f"- {field.name} ({field.type.value}): {field.description}")
        if field.items:
            lines.append(
                "    -> return a JSON array of objects; each object has these fields:"
            )
            for item in field.items:
                lines.append(f"      - {item.name} ({item.type.value}): {item.description}")
    return "\n".join(lines)
