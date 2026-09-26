"""Document parser service entrypoint."""

from __future__ import annotations

import logging
import time
import warnings

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from .routes.extract import pdf_image_conversion_service, parser_service, router as extract_router
from .routes.health import router as health_router

__all__ = [
    "app",
    "log_requests",
    "parser_service",
    "pdf_image_conversion_service",
]

# The `schema` field name is intentional (a document schema); silence pydantic's
# benign shadow warning for it specifically so logs stay clean.
warnings.filterwarnings("ignore", message='.*shadows an attribute in parent "BaseModel".*')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

app = FastAPI(
    title="Document Parser (VLM)",
    version="0.1.0",
    description="Extract structured fields from a PDF using an OpenAI-compatible VLM.",
)

app.include_router(health_router)
app.include_router(extract_router)


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    """Log every request/response so request flow is visible in the server logs."""
    method, path = request.method, request.url.path
    if method not in ("GET", "HEAD") or path not in ("/", "/favicon.ico"):
        logger.info("[http] -> %s %s", method, path)
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[http] Unhandled error for %s %s", method, path)
        raise
    duration_ms = (time.monotonic() - start) * 1000
    logger.info("[http] <- %s %s -> %s (%.0fms)", method, path, response.status_code, duration_ms)
    return response
