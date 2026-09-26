"""Configuration for the document parser service."""

import os
from functools import lru_cache


class Settings:
    def __init__(self) -> None:
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        # Defaults to the real OpenAI endpoint; override for OpenAI-compatible
        # proxies (OpenRouter, vLLM, DeepSeek, etc.).
        self.openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.pdf_dpi: int = int(os.getenv("PDF_DPI", "200"))
        self.max_pages: int = int(os.getenv("MAX_PAGES", "10"))
        self.image_max_side: int = int(os.getenv("IMAGE_MAX_SIDE", "2000"))
        self.image_quality: int = int(os.getenv("IMAGE_QUALITY", "75"))

    @property
    def api_key_required(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
