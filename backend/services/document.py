"""Domain model for a document awaiting extraction."""

import warnings

from pydantic import BaseModel

from backend.models import Schema


class Document(BaseModel):
    """A PDF document plus the schema used to extract fields from it."""

    pdf_bytes: bytes
    schema: Schema
