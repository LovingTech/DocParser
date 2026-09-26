"""Extraction routes: parse a PDF against a schema and return structured data."""

from base64 import b64decode
import json
import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.config import get_settings
from backend.models import ExtractionResponse, FieldSchema, Schema
from backend.services import Document, DocumentParserService, PdfToImageConversionService
from backend.services.llm import OpenAILLM

logger = logging.getLogger("app")

router = APIRouter(tags=["extract"])

_settings = get_settings()
pdf_image_conversion_service = PdfToImageConversionService(_settings)
parser_service = DocumentParserService(
    llm=OpenAILLM(_settings),
    pdf_to_image_conversion_service=pdf_image_conversion_service,
    settings=_settings,
)


class JSONSchemaBody(BaseModel):
    """Body for the JSON submission endpoint (PDF provided as base64)."""

    file_base64: str
    schema: Schema


@router.post("/extract", response_model=ExtractionResponse)
async def extract(
    file: UploadFile = File(..., media_type="application/pdf"),
    schema: str = Form(...),
) -> ExtractionResponse:
    """Multipart endpoint: send a PDF file and a schema (as a JSON string)."""
    schema_obj = _parse_schema(schema)
    data = await file.read()
    logger.info(
        "[route] Extract: %r (%d bytes) schema=%r",
        file.filename,
        len(data),
        schema_obj.name,
    )
    if not data[:4] == b"%PDF":
        logger.info("[route] Extract: rejected non-PDF payload (%d bytes)", len(data))
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")

    try:
        logger.info("[route] Extract: rendering + parsing with model=%s", _settings.model)
        extracted = await parser_service.parse(Document(pdf_bytes=data, schema=schema_obj))
    except HTTPException:
        raise
    except ValueError as exc:
        logger.info("[route] Extract: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("[route] Extract: unexpected failure")
        raise HTTPException(status_code=500, detail="Internal error; see server logs")

    logger.info("[route] Extract: ok fields=%s", sorted(extracted.keys()))
    return ExtractionResponse(
        request_id=uuid.uuid4().hex,
        schema_name=schema_obj.name,
        data=extracted,
    )


@router.post("/extract/json", response_model=ExtractionResponse)
async def extract_json(body: JSONSchemaBody) -> ExtractionResponse:
    """JSON endpoint: PDF provided as a base64 string (handy for non-multipart clients)."""
    data = b64decode(body.file_base64)
    logger.info("[route] Extract: schema=%r payload=%d bytes", body.schema.name, len(data))
    if not data[:4] == b"%PDF":
        logger.info("[route] Extract: rejected non-PDF payload (%d bytes)", len(data))
        raise HTTPException(status_code=400, detail="file_base64 is not a valid PDF.")

    try:
        logger.info("[route] Extract: rendering + parsing with model=%s", _settings.model)
        extracted = await parser_service.parse(Document(pdf_bytes=data, schema=body.schema))
    except HTTPException:
        raise
    except ValueError as exc:
        logger.info("[route] Extract: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("[route] Extract: unexpected failure")
        raise HTTPException(status_code=500, detail="Internal error; see server logs")

    logger.info("[route] Extract: ok fields=%s", sorted(extracted.keys()))
    return ExtractionResponse(
        request_id=uuid.uuid4().hex,
        schema_name=body.schema.name,
        data=extracted,
    )


def _parse_schema(schema_str: str) -> Schema:
    try:
        parsed = json.loads(schema_str)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"schema is not valid JSON: {exc}")
    if "fields" in parsed and not parsed.get("schema"):
        parsed.setdefault("name", "")
        return Schema(**parsed)
    # Allow a bare list of fields too.
    if isinstance(parsed, list):
        return Schema(name="", fields=[FieldSchema(**f) for f in parsed])
    return Schema(**parsed)
