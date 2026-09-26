"""High-level document extraction orchestration."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from PIL import Image

from backend.config import Settings, get_settings
from backend.models import FieldSchema, Schema, Message
from backend.prompts import get_prompt
from backend.services.document import Document
from backend.services.llm import LLM
from backend.services.pdf_image_conversion_service import PdfToImageConversionService

ParsedContent = dict[str, Any]


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

    def parse(self, doc: Document) -> ParsedContent:
        images = self.pdf_to_image_conversion_service.convert(doc)
        messages = self._build_prompt(doc.schema, images)
        raw_content = self.llm.chat_completion(messages)
        return _parse_json(raw_content)

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
    return "\n".join(lines)


def _parse_json(content: str) -> ParsedContent:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fall back to extracting a JSON object if the model wrapped it in a block.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise
