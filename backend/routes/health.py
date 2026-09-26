"""Health and configuration endpoint."""

from typing import Any, Dict

from fastapi import APIRouter

from backend.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.model,
        "base_url": settings.openai_base_url,
        "api_key_configured": bool(settings.openai_api_key),
    }
