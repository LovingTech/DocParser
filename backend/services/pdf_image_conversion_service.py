"""Convert PDF documents into compressed images for vision models."""

from __future__ import annotations

import io

from PIL import Image
import pymupdf as fitz

from backend.config import Settings, get_settings
from backend.services.document import Document


class PdfToImageConversionService:
    """Render each PDF page to a compressed image ready for a VLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def convert(self, doc: Document) -> list[Image.Image]:
        pdf_buffer = io.BytesIO(doc.pdf_bytes)
        document = fitz.open(stream=pdf_buffer, filetype="pdf")
        try:
            zoom = self._settings.pdf_dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            images: list[Image.Image] = []
            for page in document[: self._settings.max_pages]:
                pixel = page.get_pixmap(matrix=matrix)
                image = Image.frombytes("RGB", (pixel.width, pixel.height), pixel.samples)
                images.append(self._compress(image))
            if not images:
                raise ValueError("PDF contained no pages.")
            return images
        finally:
            document.close()

    @staticmethod
    def _compress(image: Image.Image) -> Image.Image:
        max_side = 2000
        width, height = image.size
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.LANCZOS,
            )
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return image
